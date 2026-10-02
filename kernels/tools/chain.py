#!/usr/bin/env python3
"""The definition chain of one register, printed as instructions.

The layer graph (analyze_flow.py) says where a value comes from in one word;
when that word is not enough -- a scale applied somewhere, a bias added, a
rounding in an unexpected place -- this prints the actual instructions that
produced it, breadth first, so the arithmetic can be read directly.

Usage: chain.py <kernel.ptx> <%register> [--depth N]
"""
import argparse
import re
import sys

sys.path.insert(0, __file__.rsplit("\\", 1)[0] if "\\" in __file__ else ".")
from analyze_flow import parse  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ptx")
    ap.add_argument("reg")
    ap.add_argument("--depth", type=int, default=6)
    args = ap.parse_args()
    instrs, defs = parse(args.ptx)

    seen, level = set(), [args.reg]
    for d in range(args.depth):
        nxt = []
        for reg in level:
            if reg in seen or reg not in defs:
                continue
            seen.add(reg)
            for i in defs[reg]:
                op, dests, srcs, rest = instrs[i]
                print("%s%-40s %s" % ("  " * d, op, rest[:110]))
                nxt.extend(s for s in srcs if s not in seen)
        if not nxt:
            break
        level = nxt
    return 0


if __name__ == "__main__":
    sys.exit(main())
