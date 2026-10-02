#!/usr/bin/env python3
"""The RDNA4 code path of every native kernel, checked on RDNA3.

No RDNA4 card is at hand, so the RDNA4 path is run here instead: each kernel is
built for gfx1100 with DLSSNR_EMULATE_GFX12, which sends every operand and
accumulator through the RDNA4 layouts and emulates only the RDNA4 WMMA
instruction itself (dlssnr_hip.h). The network then runs on those builds and
must give the reference image with as many native launches as the normal
build: everything the kernels do differently on RDNA4 has then been executed
and found exact. What this cannot check is the card itself -- that its WMMA
follows AMD's published layout, and how it rounds inside an accumulation.

Usage: check_gfx12.py [--only name,name]   (sources without .hip)

With --only, just those sources run emulated and the rest run their normal
builds: the way to find which kernel a difference comes from.
"""
import argparse
import os
import subprocess
import sys

import ab_native
import workspace  # noqa: E402

NATIVE = workspace.kernels()
OUT = os.path.join(NATIVE, "gfx12_emulated")
HIPCC = r"C:\Program Files\AMD\ROCm\7.2\bin\hipcc.exe"
DEFINES = ["-D__CLANG_HIP_CMATH_H__", "-D__CLANG__CUDA_MATH_FORWARD_DECLARES_H__",
           "-D__CLANG_CUDA_COMPLEX_BUILTINS"]


def manifest_lines():
    for line in open(os.path.join(NATIVE, "kernels.txt"), encoding="utf-8"):
        body = line.split("#", 1)[0].split()
        if len(body) >= 2:
            yield body


def build(source):
    flags_file = os.path.join(NATIVE, source + ".flags")
    extra = open(flags_file).read().split() if os.path.exists(flags_file) else []
    cmd = [HIPCC, "--genco", "--offload-arch=gfx1100", "-O3", "-DDLSSNR_EMULATE_GFX12"] + extra + \
        DEFINES + [os.path.join(NATIVE, source + ".hip"), "-o", os.path.join(OUT, source + ".hsaco")]
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        sys.exit("%s does not build:\n%s" % (source, r.stderr[-2000:]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help="emulate only these sources")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    lines = list(manifest_lines())
    sources = sorted({os.path.splitext(os.path.basename(b[1]))[0] for b in lines})
    only = [s for s in args.only.split(",") if s]
    for source in only or sources:
        print("building %s with the RDNA4 path emulated" % source, flush=True)
        build(source)
    # The same manifest, pointing at the emulated builds.
    with open(os.path.join(OUT, "kernels.txt"), "w", encoding="utf-8") as f:
        for body in lines:
            f.write(" ".join([body[0], os.path.basename(body[1])] + body[2:]) + "\n")

    normal = ab_native.run_one("-", "gfx12_normal", os.path.join(NATIVE, "kernels.txt"))
    emulated = ab_native.run_one("-", "gfx12_emulated", os.path.join(OUT, "kernels.txt"))
    print("normal build    img=%s  native launches=%d" % (normal["digest"], normal["native"]))
    print("RDNA4 emulated  img=%s  native launches=%d" % (emulated["digest"], emulated["native"]))
    if emulated["native"] != normal["native"] or not emulated["native"]:
        print("FAILED: the emulated builds did not take every native launch; the log says why")
        return 1
    if emulated["digest"] != ab_native.REFERENCE:
        print("FAILED: the RDNA4 path gives a different image (reference %s)" % ab_native.REFERENCE)
        return 1
    print("RESULT: the RDNA4 path of every native kernel reproduces the reference image")
    return 0


if __name__ == "__main__":
    sys.exit(main())
