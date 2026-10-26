#!/usr/bin/env python3
"""Builds the native kernels of NVIDIA's frame generation network.

    python build_fg_kernels.py [--arch gfx1100 ...]

fg_conv.hip stands in for the 16 variants of the network's k_conv_fp16_nhwc and fg_custom.hip for its eleven custom_*
convolutions: WMMA code written for this card, byte for byte what the translated PTX computes, 2 to 15 times faster.
They are loaded by the NVAPI stand-in (runtime/fg), which takes the kernel of that name out of the library and
launches these instead; one it refuses to launch is translated after all.

Each is one bundle holding every architecture, like the neural rendering kernels (build_kernel.py).
Writes fg_conv.hsaco and fg_custom.hsaco beside the sources.
"""
import argparse
import sys

import build_kernel

KERNELS = ["fg_conv", "fg_custom"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", action="append", help="build for this target only (repeatable)")
    args = ap.parse_args()
    build_kernel.msvc_environment()
    failed = False
    for name in KERNELS:
        ok, report = build_kernel.build(name, args.arch)
        print("%s %s" % ("built" if ok else "FAILED", name))
        print("".join("    %s\n" % line for line in report.splitlines()), end="")
        failed |= not ok
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
