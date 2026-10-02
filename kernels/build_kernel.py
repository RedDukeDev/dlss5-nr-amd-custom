#!/usr/bin/env python3
"""Builds one native kernel into a standalone code object.

    python build_kernel.py <name> [--arch gfx1100 ...] [--flags "..."]

compiles <name>.hip into <name>.hsaco, then writes the disassembly for this
machine's card to <name>.isa and prints each kernel's registers, spills and
LDS, and the number of WMMA instructions.

--genco emits a code object rather than an executable, which is what the
runtime wants: it loads the file with hipModuleLoadData, takes the kernel with
hipModuleGetFunction and launches it on the caller's own stream. Nothing here
has to be a DLL, and nothing has to link against the HIP host runtime.

Needs the AMD HIP SDK (ROCM_PATH, default C:\\Program Files\\AMD\\ROCm\\7.2)
and Visual Studio (VS_PATH, default its 2026 Community edition): hipcc on
Windows compiles against MSVC's headers.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROCM = os.environ.get("ROCM_PATH", r"C:\Program Files\AMD\ROCm\7.2")
VS = os.environ.get("VS_PATH", r"C:\Program Files\Microsoft Visual Studio\18\Community")

# Every RDNA3 and RDNA3.5 part, named individually, in one code object: the
# runtime picks the right one at load time. Naming them beats a generic target
# -- gfx11-generic costs the 1.5x VGPR allocation these kernels live on -- and
# it keeps the kernel from being an RX 7900 XT curiosity.
#
# RDNA4 (gfx1200, gfx1201) has its own WMMA layout, handled in dlssnr_hip.h;
# that path is checked on RDNA3 through DLSSNR_EMULATE_GFX12 (README, "RDNA4").
ARCHES = ["gfx1100", "gfx1101", "gfx1102", "gfx1103",
          "gfx1150", "gfx1151", "gfx1152", "gfx1153",
          "gfx1200", "gfx1201"]

# The card in this machine, the one disassembled.
DISASSEMBLE = "gfx1100"

# These switch off the HIP math headers whose device overloads collide with
# MSVC's constexpr <cmath>; without them the compile dies on 'isgreater'
# before reaching any of our code.
DEFINES = ["-D__CLANG_HIP_CMATH_H__", "-D__CLANG__CUDA_MATH_FORWARD_DECLARES_H__",
           "-D__CLANG_CUDA_COMPLEX_BUILTINS"]


def tool(name):
    return os.path.join(ROCM, "bin", name)


_msvc = None


def msvc_environment():
    """The environment vcvars64.bat sets up, read once and reused."""
    global _msvc
    if _msvc is None:
        vcvars = os.path.join(VS, "VC", "Auxiliary", "Build", "vcvars64.bat")
        if not os.path.isfile(vcvars):
            raise SystemExit("Visual Studio not found at %s (set VS_PATH)" % VS)
        out = subprocess.run('"%s" >nul && set' % vcvars, shell=True, stdout=subprocess.PIPE,
                             text=True, errors="replace", check=True).stdout
        _msvc = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    return _msvc


def kernel_flags(name):
    # A kernel whose best build needs options keeps them in <name>.flags, one
    # line -- measured per kernel, since the same scheduler strategy helps one
    # fused block and slows another.
    path = os.path.join(HERE, name + ".flags")
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        return f.read().split()


def figures(elf):
    """Per kernel: VGPRs, VGPR spills, SGPRs and LDS bytes, from the code
    object's metadata note."""
    notes = subprocess.run([tool("llvm-readobj.exe"), "--notes", elf], stdout=subprocess.PIPE,
                           text=True, errors="replace").stdout
    kernels, current = [], None
    for line in notes.splitlines():
        m = re.match(r"\s*\.(\w+):\s+(\S+)", line)
        if not m:
            continue
        key, value = m.groups()
        if key == "group_segment_fixed_size":   # the first field of each kernel's entry
            current = {"lds": value}
            kernels.append(current)
        elif current is not None and key in ("name", "vgpr_count", "vgpr_spill_count", "sgpr_count"):
            current[key] = value
    return kernels


def build(name, arches=None, flags=None, quiet=False):
    """Builds <name>.hip. Returns (ok, report): the report is the compiler's
    output on failure, the figures on success."""
    source = os.path.join(HERE, name + ".hip")
    if not os.path.isfile(source):
        return False, "no source: %s" % source
    arches = arches or ARCHES
    flags = kernel_flags(name) if flags is None else flags
    env = msvc_environment()
    output = os.path.join(HERE, name + ".hsaco")

    command = [tool("hipcc.exe"), "--genco", "-O3"] + ["--offload-arch=" + a for a in arches]
    command += flags + DEFINES + [source, "-o", output]
    compiled = subprocess.run(command, cwd=HERE, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, errors="replace")
    if compiled.returncode != 0:
        return False, compiled.stdout

    report = []
    if DISASSEMBLE in arches:
        # --genco writes a bundle, not a bare ELF: the runtime reads it happily
        # but llvm-objdump refuses it, so one architecture is unbundled first.
        with tempfile.TemporaryDirectory() as scratch:
            elf = os.path.join(scratch, name + ".elf")
            subprocess.run([tool("clang-offload-bundler.exe"), "--type=o", "--unbundle",
                            "--input=" + output, "--targets=hipv4-amdgcn-amd-amdhsa--" + DISASSEMBLE,
                            "--output=" + elf], check=True)
            isa = subprocess.run([tool("llvm-objdump.exe"), "--disassemble", elf], stdout=subprocess.PIPE,
                                 text=True, errors="replace").stdout
            with open(os.path.join(HERE, name + ".isa"), "w", encoding="utf-8") as f:
                f.write(isa)
            # The register count decides how many waves fit, and it is the
            # first number to look at when a kernel is slower than the
            # arithmetic says it should be.
            for k in figures(elf):
                report.append("%s: vgpr %s (spilled %s), sgpr %s, lds %s" % (
                    k.get("name", "?"), k.get("vgpr_count", "?"), k.get("vgpr_spill_count", "?"),
                    k.get("sgpr_count", "?"), k["lds"]))
            report.append("wmma instructions on %s: %d" % (
                DISASSEMBLE, sum(1 for line in isa.splitlines() if "v_wmma" in line)))
    return True, "\n".join(report)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="the kernel: <name>.hip")
    ap.add_argument("--arch", action="append",
                    help="build for this target only (repeatable): a test build for the card at hand")
    ap.add_argument("--flags", help="compiler options instead of <name>.flags, to try one without editing it")
    args = ap.parse_args()
    ok, report = build(args.name, args.arch, args.flags.split() if args.flags is not None else None)
    print(report)
    if ok:
        print("built %s.hsaco" % args.name)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
