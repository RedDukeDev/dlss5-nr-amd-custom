#!/usr/bin/env python3
"""Builds the native kernels of the ray reconstruction network (preset F).

    python build_rr_kernels.py [nvngx_dlssd.dll] [--arch gfx1100 ...] [--no-generated]

rr2_stage.hip is the stage that fourteen of the network's kernels share (encoders 1-5, decoders 4-0): one code object,
rr2_stage.hsaco, which kernels/rr/kernels_rr.txt names.

Two more kernels -- the first encoder and the last decoder -- have stretches that are translated from the library's own
PTX (tools/build_rr_generated.py), so they are made from the copy of nvngx_dlssd.dll given here (default
third_party\\nvngx_dlssd.dll) and are never in the repository. Without the library, or with --no-generated, only the
stage is built and those two kernels stay on the translation: the package works, a good deal slower.

Each is one bundle holding every architecture, like the neural rendering kernels (build_kernel.py).
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "tools"))
import build_kernel  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("library", nargs="?", default=os.path.join(os.path.dirname(HERE), "third_party", "nvngx_dlssd.dll"))
    ap.add_argument("--arch", action="append", help="build for this target only (repeatable)")
    ap.add_argument("--no-generated", action="store_true", help="only the shared stage")
    args = ap.parse_args()
    build_kernel.msvc_environment()
    ok, report = build_kernel.build("rr2_stage", args.arch)
    print("%s rr2_stage" % ("built" if ok else "FAILED"))
    print("".join("    %s\n" % line for line in report.splitlines()), end="")
    if not ok:
        return 1
    fragment = os.path.join(HERE, "rr", "kernels_rr_generated.txt")
    if args.no_generated or not os.path.isfile(args.library):
        if os.path.isfile(fragment):
            os.remove(fragment)   # a fragment left by an earlier library would name code objects that no longer match
        print("encoder 0 and the last decoder are not built: %s" %
              ("--no-generated" if args.no_generated else "no library at %s" % args.library))
        return 0
    import build_rr_generated
    sys.argv = [sys.argv[0], args.library]
    for a in args.arch or []:
        sys.argv += ["--arch", a]
    build_rr_generated.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
