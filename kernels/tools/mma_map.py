#!/usr/bin/env python3
"""A kernel's layer structure, read off the multiplies it actually executes.

analyze_flow.py reads the PTX as text, which is enough when the text is
straight-line. These kernels are not: they loop, so the same instruction runs
more than once, and the registers cannot be read afterwards because they are
reused. What can be trusted is ptxsim's log of every multiply as it ran, with
its operands as matrices -- and the weight blob, in which each B operand's
bytes can be found, which is what names the region it came from.

The output is one line per group of consecutive multiplies that share a weight
region and an accumulator kind, in execution order: the layer's shape.

Usage: mma_map.py <capture dir> <kernel.ptx> --grid X,Y,Z --block X,Y,Z
                    [--block-index N] [--weights 0x10] [--runs N]
"""
import argparse
import glob
import io
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ptxsim  # noqa: E402
import run_ptx  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("ptx")
    ap.add_argument("--grid", required=True)
    ap.add_argument("--block", required=True)
    ap.add_argument("--block-index", type=int, default=0, dest="index")
    ap.add_argument("--weights", default="0x10", help="parameter offset of the weight pointer")
    ap.add_argument("--runs", type=int, default=0, help="how many groups to print (0 = all)")
    args = ap.parse_args()
    grid = tuple(int(v) for v in args.grid.split(","))
    block = tuple(int(v) for v in args.block.split(","))
    w_off = int(args.weights, 0)

    params, pre, post = run_ptx.load_capture(args.capture)
    text = io.open(args.ptx, encoding="utf-8").read()
    if ".version" not in text:
        text = ".version 9.4\n.target sm_120\n.address_size 64\n" + text
    words = struct.unpack_from("<%dQ" % (len(params) // 8), params)
    mem = ptxsim.GlobalMemory()
    for off, data in sorted(pre.items()):
        mem.add(words[off // 8], data)
    m = ptxsim.Machine(ptxsim.Kernel(text), grid, block, params, mem,
                       blocks=(args.index, args.index + 1))
    m.mma_log = []
    m.run()

    weights = pre[w_off]
    index = {}
    for off in range(0, len(weights) - 4):
        index.setdefault(weights[off:off + 4], off)
    codes = {}
    for c in range(256):
        codes.setdefault(float(ptxsim.E4M3[c]), c)

    def where(entry, wave=0):
        """The weight offset a multiply's B operand came from, by its bytes."""
        b = entry["b"][wave]
        try:
            key = bytes(codes[float(b[k, 0])] for k in range(4))
        except KeyError:
            return None
        return index.get(key)

    groups = []
    for n, e in enumerate(m.mma_log):
        off = where(e)
        tag = (None if off is None else off & ~0x1FF, not e["c"][0].any())
        if groups and groups[-1][0] == tag:
            groups[-1][2] = n
        else:
            groups.append([tag, n, n])
    print("%d multiplies in %d groups (weight piece, accumulator)" % (len(m.mma_log), len(groups)))
    for (region, czero), first, last in (groups if not args.runs else groups[:args.runs]):
        print("  runs %4d..%4d : %-9s C = %s"
              % (first, last, hex(region) if region is not None else "computed",
                 "0" if czero else "the one before"))


if __name__ == "__main__":
    main()
