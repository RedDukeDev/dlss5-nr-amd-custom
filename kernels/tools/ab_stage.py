#!/usr/bin/env python3
"""A native kernel on a gate capture (runtime/gate/capture.h), compared with what the translated kernel wrote.

The pointers in the capture's argument buffer are rebased onto fresh allocations holding the "pre" memory, the
kernel runs on the captured grid and block, and every byte the translated kernel changed is compared. The
differences are reported per output pointer of the argument buffer (the offset of the pointer there), with the
first few differing positions, which is what tells an index error from a rounding one.

Usage: ab_stage.py <capture dir> <code object> <entry> [--time N] [--blocks X0:X1,Y0:Y1]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hip_run import Hip  # noqa: E402
from replay_gpu import read_info  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("hsaco")
    ap.add_argument("entry")
    ap.add_argument("--time", type=int, default=0)
    ap.add_argument("--waves", type=int, default=0, help="launch with this many waves per block (block z)")
    ap.add_argument("--shift-check", action="store_true", help="also list differing offsets")
    args = ap.parse_args()
    info = read_info(args.capture)
    params = bytearray(open(os.path.join(args.capture, "args.bin"), "rb").read())
    h = Hip()
    fn = h.module(args.hsaco, args.entry)
    dev = {}
    for base in info["regions"]:
        dev[base] = h.upload(np.fromfile(os.path.join(args.capture, "pre_%x.bin" % base), np.uint8))
    for offset, base, delta in info["pointers"]:
        params[offset:offset + 8] = (dev[base] + delta).to_bytes(8, "little")
    block = info["block"] if not args.waves else (info["block"][0], info["block"][1], args.waves)
    h.launch(fn, info["grid"], block, bytes(params))
    total = 0
    for base, size in info["regions"].items():
        captured = np.fromfile(os.path.join(args.capture, "post_%x.bin" % base), np.uint8)
        before = np.fromfile(os.path.join(args.capture, "pre_%x.bin" % base), np.uint8)
        mine = h.download(dev[base], size)
        changed = (captured != before) | (mine != before)
        diff = changed & (captured != mine)
        print("region %x: translated changed %d, native changed %d, differing %d"
              % (base, int((captured != before).sum()), int((mine != before).sum()), int(diff.sum())))
        total += int(diff.sum())
        # attribute the differences to the nearest pointer at or below them
        pointers = sorted((d, off) for off, b, d in info["pointers"] if b == base)
        if diff.any():
            where = np.nonzero(diff)[0]
            starts = np.array([d for d, _ in pointers])
            owner = np.searchsorted(starts, where, side="right") - 1
            for k, (d, off) in enumerate(pointers):
                mask = owner == k
                if mask.any():
                    sel = where[mask]
                    print("   output at argument +%d: %d differing bytes; first at +%d" % (off, len(sel), sel[0] - d))
                    if args.shift_check:
                        print("      ", [(int(i - d), int(captured[i]), int(mine[i])) for i in sel[:8]])
    if args.time:
        ms = h.time(fn, info["grid"], block, bytes(params), repeat=args.time)
        print("%.3f ms per launch" % ms)
    print("IDENTICAL" if total == 0 else "DIFFERENT (%d bytes)" % total)
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
