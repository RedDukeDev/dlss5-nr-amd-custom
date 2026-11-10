#!/usr/bin/env python3
"""Builds the native super resolution layers (DLSS presets K and L/M) from d4r's sources.

    python build_sr_kernels.py [--d4r DIR] [--out DIR] [--arch gfx1100 ...]

The eleven `dltss_pwin_*_layer` kernels (preset K) and the five `rrlite_*_4x4` Swin blocks (presets L
and M: enc1, enc2, enc3_tube, dec2, dec1, which are nine tenths of those networks' time) are d4r's (https://github.com/countervolts/d4r,
GPL-3): hand-written WMMA code that takes the place of NVIDIA's translated PTX kernel of
the same name, with the same launch shape and parameter block, reading the network's
weights out of the library's own buffers. Nothing of NVIDIA's is stored here.

Each kernel goes into one bundle holding every architecture, the way the neural
rendering kernels do (build_kernel.py): the runtime loads the file and HIP picks the
code object for the card. Named targets rather than a generic one, because these
kernels live on 170-230 registers and gfx11-generic costs them the larger allocation.
The code objects are built one target at a time and bundled afterwards, as d4r's kernels/build.sh
does for its own: the sources carry compiler flags per target ("// d4r-build-flags-gfx11:" and so on,
the same rule as there: the line of the exact target, else of its family, else the plain one), and
one clang run for all of them could give them only one set.

Writes <out>/<kernel>.hsaco and <out>/kernels_sr.txt, the manifest the gate reads
(NVCUDA_PROXY_NATIVE accepts several manifests joined by ';'). A layer that runs as several kernels
(dec5 of preset K: the main kernel and dec5_layer_post1..3) is one code object with all of them; the gate
launches the later ones after the first (runtime/gate/native_hook.h).

The sources are read from a d4r checkout (--d4r, or the D4R_DIR environment variable,
or third_party/d4r next to this repository's root) and are never modified: they are
copied to a scratch directory first, because some of them are built differently here.

Why: the decoder dec4 and the encoder enc4 are written for 32 waves per
workgroup (NG = 8, a 1024 thread workgroup). On an RX 7900 XT that launch never
completes: 170 to 192 registers per wave do not fit when the whole workgroup has to sit
on one compute unit. NG = 4 (16 waves) runs; --wide-groups sets it, and 8 gives d4r's
own tuning back for a card where it is known to work (gfx1200 per d4r's measurements).

Needs the AMD HIP SDK (ROCM_PATH, default C:\\Program Files\\AMD\\ROCm\\7.2). The flags
are those of d4r's kernels/build.sh, which includes HIP's own headers (the sources need them now) plus
-DD4R_WINDOWS_GFX11_LANE_EXCHANGE, d4r's Windows RDNA3 workaround (scripts/windows/build-native-k-experiment.ps1
there). K is built as d4r's "accuracy" set (-DD4R_ACCURACY: a rounding to f16 after every MMA step and
NVIDIA's handling of denormals), which is what the translated PTX computes; M and L are its normal set. d4r's
kernels/windows-flags.txt (flags measured on one RDNA4 card) is not applied.
"""
import argparse
import concurrent.futures
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROCM = os.environ.get("ROCM_PATH", r"C:\Program Files\AMD\ROCm\7.2")

# Every RDNA3 and RDNA3.5 part, and RDNA4 (its WMMA layout is d4r's kernels/common/wmma_layout.h;
# not run on real RDNA4 hardware yet).
ARCHES = ["gfx1100", "gfx1101", "gfx1102", "gfx1103",
          "gfx1150", "gfx1151", "gfx1152", "gfx1153",
          "gfx1200", "gfx1201"]

# The block shape DLSS launches each layer with: the gate checks it, the native code is
# written for it (d4r_block_z in the code object says how many waves it really uses).
LAYERS = {
    "enc0": (32, 1, 1), "enc1": (32, 1, 2), "enc2": (32, 1, 2), "enc3": (32, 1, 4), "enc4": (32, 1, 4),
    "dec5": (32, 1, 8), "dec4": (32, 1, 4), "dec3": (32, 1, 4), "dec2": (32, 1, 2), "dec1": (32, 1, 2),
    "dec0": (32, 1, 1),
}
PARAMETER_BYTES = 176

# The presets L and M share the `rrlite` Swin blocks. d4r writes five of them natively (the rest are small):
# the block shape DLSS launches each with and the size of its parameter block (CommonParams: 56 bytes,
# the tube encoder's TubeParams: 88).
MID_LAYERS = {
    "rrlite_enc1_4x4": ((32, 1, 2), 56), "rrlite_enc2_4x4": ((32, 1, 4), 56),
    "rrlite_enc3_tube_4x4": ((32, 1, 4), 88), "rrlite_dec2_4x4": ((32, 1, 4), 56),
    "rrlite_dec1_4x4": ((32, 1, 2), 56),
}

# What every build shares
# The three -D__CLANG... switch off HIP's cmath wrappers, whose device overloads of isless and the like
# collide with the Microsoft C++ library's host and device ones (the same fix as in ZLUDA's ptx/lib/build_bc_msvc.sh)
COMMON_FLAGS = ["-x", "hip", "-std=c++17", "--offload-device-only", "--no-gpu-bundle-output", "-mno-wavefrontsize64",
                "-O3", "-DD4R_WINDOWS_GFX11_LANE_EXCHANGE", "-D__CLANG_HIP_CMATH_H__",
                "-D__CLANG__CUDA_MATH_FORWARD_DECLARES_H__", "-D__CLANG_CUDA_COMPLEX_BUILTINS"]
K_FLAGS = ["-DD4R_ACCURACY"]
# The wide layers (enc4, dec4 and the phase kernels of dec5) are built as d4r's normal set instead: in its accuracy mode
# they get the windows at the left and right edge of the image wrong on RDNA3 (checked on a gfx1100 against d4r's own
# numpy model of the layer and against the translated PTX: the interior windows match, those edge windows are off by
# hundreds of f16 ulps), and in the normal set they agree with the translation as well as the older kernels did.
K_FAST_LAYERS = ("enc4", "dec4", "dec5")
K_FAST_FLAGS = []
MID_FLAGS = []
# v_permlanex16 on RDNA3 under Windows copies the even value into the odd slot (d4r's note in
# kernels/common/wave32_exchange.h). K's layers reach it through two helpers its workaround does not cover
# (wm_other_half in wmma_layout.h, other_half in pwin_common.h), so on gfx11 the half exchange goes through
# ds_bpermute (WM_HALF_BPERMUTE; the layers whose flag lines already ask for it were right) and other_half
# is made to use the helper that has the workaround.
K_GFX11_FLAGS = ["-DWM_HALF_BPERMUTE"]

HOST_TARGET = "host-x86_64-unknown-linux-gnu-"


def find_d4r(argument):
    candidates = [argument, os.environ.get("D4R_DIR"),
                  os.path.join(HERE, "..", "third_party", "d4r")]
    for candidate in candidates:
        if candidate and os.path.isdir(os.path.join(candidate, "kernels", "k")):
            return os.path.abspath(candidate)
    raise SystemExit("no d4r checkout: pass --d4r DIR or set D4R_DIR (it needs kernels/k and kernels/common)")


def source_flags(path, arch):
    """The per-kernel flags d4r keeps in '// d4r-build-flags:' lines of the source. A line for the target
    ('// d4r-build-flags-gfx1101:', or '-gfx12:' for any gfx12 target) replaces the plain one."""
    lines = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"//\s*d4r-build-flags(?:-([a-z0-9]+))?:\s*(.*)", line)
            if m:
                lines[m.group(1)] = m.group(2).split()
    for tag in (arch, arch[:-2]):
        if tag in lines:
            return lines[tag]
    return lines.get(None, [])


def device_libs():
    for candidate in (os.path.join(ROCM, "amdgcn", "bitcode"), os.path.join(ROCM, "lib", "llvm", "amdgcn", "bitcode")):
        if os.path.isdir(candidate):
            return candidate
    raise SystemExit("no device libraries under %s (set ROCM_PATH)" % ROCM)


def with_wide_groups(text, groups):
    """The wide layers end their macro call with NG: WIDE_ENCODER(name, 4, 128, 160, 8)."""
    return re.sub(r"(WIDE_(?:ENCODER|DECODER|PLAIN)\([^)]*,\s*)\d+(\s*\))", r"\g<1>%d\2" % groups, text)


def compile_one(clang, libs, source, include, arch, flags, output):
    if arch.startswith("gfx11") and os.path.basename(source).startswith("dltss_pwin_"):
        flags = flags + K_GFX11_FLAGS
    command = [clang] + COMMON_FLAGS + ["--offload-arch=" + arch, "--rocm-path=" + ROCM,
                                        "--rocm-device-lib-path=" + libs, "-I" + os.path.join(ROCM, "include")]
    command += flags + source_flags(source, arch) + ["-I" + include, source, "-o", output]
    done = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    errors = [l for l in done.stdout.splitlines() if "error" in l]
    return done.returncode == 0, errors


def bundle(bundler, parts, scratch, output):
    """One file with a code object per target, the layout clang itself writes for several --offload-arch."""
    empty = os.path.join(scratch, "host.empty")
    open(empty, "wb").close()
    targets = [HOST_TARGET] + ["hipv4-amdgcn-amd-amdhsa--" + arch for arch, _ in parts]
    command = [bundler, "--type=o", "--targets=" + ",".join(targets), "--input=" + empty]
    command += ["--input=" + path for _, path in parts] + ["--output=" + output]
    done = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    if done.returncode != 0:
        print(done.stdout)
    return done.returncode == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d4r", help="a d4r checkout (the windows branch)")
    ap.add_argument("--out", default=os.path.join(HERE, "sr"), help="where the code objects and the manifest go")
    ap.add_argument("--arch", action="append", help="build for this target only (repeatable)")
    ap.add_argument("--wide-groups", type=int, default=4, help="NG of enc4 and dec4 (default 4)")
    ap.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 4), help="compilations at once")
    args = ap.parse_args()

    d4r = find_d4r(args.d4r)
    clang = os.path.join(ROCM, "bin", "clang++.exe")
    bundler = os.path.join(ROCM, "bin", "clang-offload-bundler.exe")
    for tool in (clang, bundler):
        if not os.path.isfile(tool):
            raise SystemExit("no %s (set ROCM_PATH)" % tool)
    libs = device_libs()
    arches = args.arch or ARCHES
    os.makedirs(args.out, exist_ok=True)

    manifest = ["# Native super resolution layers (d4r, GPL-3): see build_sr_kernels.py.",
                "# <ptx kernel>  <code object>  block=x,y,z  args=<bytes>"]
    failures = 0
    with tempfile.TemporaryDirectory() as scratch:
        for part in ("k", "m", "common"):
            shutil.copytree(os.path.join(d4r, "kernels", part), os.path.join(scratch, "kernels", part))
        sources = os.path.join(scratch, "kernels", "k")
        mid_sources = os.path.join(scratch, "kernels", "m")

        # d4r builds with a newer clang than the HIP SDK's: where it asks for IEEE maximumNumber and
        # minimumNumber (the clamp of the patch merge), the SDK's clang has the builtins that return the
        # number when the other operand is not one (maxnum, minnum): the same, but for the sign of a zero.
        common = os.path.join(mid_sources, "swin_common.h")
        with open(common, encoding="utf-8") as f:
            text = f.read()
        patched = text.replace("__builtin_elementwise_maximumnum", "__builtin_elementwise_max")
        patched = patched.replace("__builtin_elementwise_minimumnum", "__builtin_elementwise_min")
        if patched != text:
            with open(common, "w", encoding="utf-8") as f:
                f.write(patched)

        # K's other_half goes through the helper that has d4r's Windows workaround for permlanex16
        pwin_common = os.path.join(sources, "pwin_common.h")
        with open(pwin_common, encoding="utf-8") as f:
            text = f.read()
        builtin = "return __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xfedcba98u, false, false);"
        if text.count(builtin) != 1:
            raise SystemExit("pwin_common.h no longer has the other_half this script patches: check it")
        include = '#include "../common/wmma_layout.h"'
        if text.count(include) != 1:
            raise SystemExit("pwin_common.h includes wmma_layout.h differently: check it")
        text = text.replace(builtin, "return d4r_wave32_other_half(v);")
        with open(pwin_common, "w", encoding="utf-8") as f:
            f.write(text.replace(include, include + '\n#include "../common/wave32_exchange.h"'))

        # (kernel name, source, include directory, flags, parameter bytes, block)
        jobs = []
        for layer, block in LAYERS.items():
            name = "dltss_pwin_%s_layer" % layer
            source = os.path.join(sources, name + ".hip")
            with open(source, encoding="utf-8") as f:
                text = f.read()
            patched = with_wide_groups(text, args.wide_groups) if layer in ("enc4", "dec4") else text
            if patched != text:
                with open(source, "w", encoding="utf-8") as f:
                    f.write(patched)
            jobs.append((name, source, sources, K_FAST_FLAGS if layer in K_FAST_LAYERS else K_FLAGS, PARAMETER_BYTES,
                         block))
        for name, (block, parameter_bytes) in MID_LAYERS.items():
            jobs.append((name, os.path.join(mid_sources, name + ".hip"), mid_sources, MID_FLAGS, parameter_bytes, block))

        objects = os.path.join(scratch, "objects")
        os.makedirs(objects)
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            futures = {}
            for name, source, include, flags, _, _ in jobs:
                for arch in arches:
                    output = os.path.join(objects, "%s.%s.hsaco" % (name, arch))
                    futures[pool.submit(compile_one, clang, libs, source, include, arch, flags, output)] = (
                        name, arch, output)
            for future in concurrent.futures.as_completed(futures):
                name, arch, output = futures[future]
                ok, errors = future.result()
                results[(name, arch)] = output if ok else None
                if not ok:
                    print("FAILED %s for %s\n%s" % (name, arch, "\n".join(errors)))

        for name, _, _, _, parameter_bytes, block in jobs:
            parts = [(arch, results[(name, arch)]) for arch in arches if results.get((name, arch))]
            output = os.path.join(args.out, name + ".hsaco")
            if len(parts) != len(arches) or not bundle(bundler, parts, scratch, output):
                failures += 1
                continue
            print("built %s (%d bytes)" % (name, os.path.getsize(output)))
            manifest.append("%s %s block=%d,%d,%d args=%d" % (name, name + ".hsaco", *block, parameter_bytes))
    with open(os.path.join(args.out, "kernels_sr.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(manifest) + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
