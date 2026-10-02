#!/usr/bin/env python3
"""Builds every native kernel the manifest names, several at a time.

    python build_kernels.py [name ...] [--jobs N]

Without names, the kernels are the code objects kernels.txt lists; with names,
only those. Each is built as build_kernel.py builds it, <name>.flags included.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import build_kernel

HERE = os.path.dirname(os.path.abspath(__file__))


def manifest_kernels():
    names = []
    with open(os.path.join(HERE, "kernels.txt"), encoding="utf-8") as manifest:
        for line in manifest:
            fields = line.split("#", 1)[0].split()
            if len(fields) >= 2:
                name = os.path.splitext(fields[1])[0]
                if name not in names:
                    names.append(name)
    return names


def timed(name, arches):
    began = time.monotonic()
    ok, report = build_kernel.build(name, arches)
    return name, ok, time.monotonic() - began, report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*")
    # Each build compiles for ten architectures in one process, and the fused
    # blocks take a few GB each: one per core would run out of memory first.
    ap.add_argument("--jobs", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    ap.add_argument("--arch", action="append", help="build for this target only (repeatable)")
    args = ap.parse_args()

    names = args.names or manifest_kernels()
    build_kernel.msvc_environment()   # once, before the threads need it
    print("building %d kernels, %d at a time" % (len(names), args.jobs))
    began = time.monotonic()
    failed = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for future in as_completed([pool.submit(timed, n, args.arch) for n in names]):
            name, ok, seconds, report = future.result()
            print("%s %s, %.0f s" % ("built" if ok else "FAILED", name, seconds))
            if not ok:
                failed.append(name)
            print("".join("    %s\n" % line for line in report.splitlines()), end="")
    print("%d of %d built in %.0f s" % (len(names) - len(failed), len(names), time.monotonic() - began))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
