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

Writes <out>/<kernel>.hsaco and <out>/kernels_sr.txt, the manifest the gate reads
(NVCUDA_PROXY_NATIVE accepts several manifests joined by ';').

The sources are read from a d4r checkout (--d4r, or the D4R_DIR environment variable,
or third_party/d4r next to this repository's root) and are never modified: they are
copied to a scratch directory first, because three of them are built differently here.

Why: the decoders dec4/dec5 and the encoder enc4 are written for 32 waves per
workgroup (NG = 8, a 1024 thread workgroup). On an RX 7900 XT that launch never
completes: 170 to 192 registers per wave do not fit when the whole workgroup has to sit
on one compute unit. NG = 4 (16 waves) runs; --wide-groups sets it, and 8 gives d4r's
own tuning back for a card where it is known to work (gfx1200 per d4r's measurements).

Needs the AMD HIP SDK (ROCM_PATH, default C:\\Program Files\\AMD\\ROCm\\7.2). The flags
are d4r's Windows ones (kernels/tools and scripts/windows/build-native-k-experiment.ps1
there): -DD4R_DEVICE_ONLY_MINIMAL keeps HIP's own headers out of the build, which is what
spares it the collision with MSVC's <cmath>. -DD4R_K_FP16_BASELINE is d4r's strict K
arithmetic (a rounding to f16 after every MMA step); it already is what d4r's separate
"accuracy" set builds for K, so there is only one set here.
"""
import argparse
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

COMMON_FLAGS = ["-x", "hip", "-std=c++17", "--offload-device-only", "-mno-wavefrontsize64", "-nogpuinc",
                "-nogpulib", "-O3", "-DD4R_DEVICE_ONLY_MINIMAL", "-DD4R_K_FP16_BASELINE",
                "-DD4R_WINDOWS_GFX11_LANE_EXCHANGE"]


MID_FLAGS = [flag for flag in COMMON_FLAGS if flag != "-DD4R_K_FP16_BASELINE"]


def find_d4r(argument):
    candidates = [argument, os.environ.get("D4R_DIR"),
                  os.path.join(HERE, "..", "third_party", "d4r")]
    for candidate in candidates:
        if candidate and os.path.isdir(os.path.join(candidate, "kernels", "k")):
            return os.path.abspath(candidate)
    raise SystemExit("no d4r checkout: pass --d4r DIR or set D4R_DIR (it needs kernels/k and kernels/common)")


def source_flags(path):
    """The per-kernel flags d4r keeps in a '// d4r-build-flags:' line of the source."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"//\s*d4r-build-flags:\s*(.*)", line)
            if m:
                return m.group(1).split()
    return []


def with_wide_groups(text, groups):
    """The wide layers end their macro call with NG: WIDE_ENCODER(name, 4, 128, 160, 8)."""
    return re.sub(r"(WIDE_(?:ENCODER|DECODER|PLAIN)\([^)]*,\s*)\d+(\s*\))", r"\g<1>%d\2" % groups, text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d4r", help="a d4r checkout (the windows branch)")
    ap.add_argument("--out", default=os.path.join(HERE, "sr"), help="where the code objects and the manifest go")
    ap.add_argument("--arch", action="append", help="build for this target only (repeatable)")
    ap.add_argument("--wide-groups", type=int, default=4, help="NG of enc4, dec5 and dec4 (default 4)")
    args = ap.parse_args()

    d4r = find_d4r(args.d4r)
    clang = os.path.join(ROCM, "bin", "clang++.exe")
    if not os.path.isfile(clang):
        raise SystemExit("no clang++ at %s (set ROCM_PATH)" % clang)
    arches = args.arch or ARCHES
    os.makedirs(args.out, exist_ok=True)

    manifest = ["# Native super resolution layers (d4r, GPL-3): see build_sr_kernels.py.",
                "# <ptx kernel>  <code object>  block=x,y,z  args=<bytes>"]
    failures = 0
    with tempfile.TemporaryDirectory() as scratch:
        for part in ("k", "m", "common"):
            shutil.copytree(os.path.join(d4r, "kernels", part), os.path.join(scratch, "kernels", part))
        sources = os.path.join(scratch, "kernels", "k")
        for layer, block in LAYERS.items():
            name = "dltss_pwin_%s_layer" % layer
            source = os.path.join(sources, name + ".hip")
            with open(source, encoding="utf-8") as f:
                text = f.read()
            patched = with_wide_groups(text, args.wide_groups) if layer in ("enc4", "dec5", "dec4") else text
            if patched != text:
                with open(source, "w", encoding="utf-8") as f:
                    f.write(patched)
            output = os.path.join(args.out, name + ".hsaco")
            command = [clang] + COMMON_FLAGS + ["--offload-arch=" + a for a in arches]
            command += source_flags(source) + ["-I" + sources, source, "-o", output]
            done = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                  errors="replace")
            if done.returncode != 0:
                failures += 1
                print("FAILED %s\n%s" % (name, "\n".join(l for l in done.stdout.splitlines() if "error" in l)))
                continue
            print("built %s (%d bytes)" % (name, os.path.getsize(output)))
            manifest.append("%s %s block=%d,%d,%d args=%d" % (name, name + ".hsaco", *block, PARAMETER_BYTES))
        # The Swin blocks of presets L and M. Their numerics are d4r's shipped ones (intermediate activations in
        # f16 where NVIDIA re-quantises to e4m3, f32 accumulation): the sources say so with SWIN_NO_Q8 and
        # SWIN_F32ACC, and the build adds nothing to that.
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
        for name, (block, parameter_bytes) in MID_LAYERS.items():
            source = os.path.join(mid_sources, name + ".hip")
            output = os.path.join(args.out, name + ".hsaco")
            command = [clang] + MID_FLAGS + ["--offload-arch=" + a for a in arches]
            command += source_flags(source) + ["-I" + mid_sources, source, "-o", output]
            done = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
            if done.returncode != 0:
                failures += 1
                errors = [l for l in done.stdout.splitlines() if "error" in l]
                print("FAILED %s\n%s" % (name, "\n".join(errors)))
                continue
            print("built %s (%d bytes)" % (name, os.path.getsize(output)))
            manifest.append("%s %s block=%d,%d,%d args=%d" % (name, name + ".hsaco", *block, parameter_bytes))
    with open(os.path.join(args.out, "kernels_sr.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(manifest) + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
