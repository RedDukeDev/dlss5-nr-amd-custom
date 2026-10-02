#!/usr/bin/env python3
"""Two builds of a native kernel on the same captured launch: are their outputs
the same, byte for byte, and how long does each take?

The regression check for optimisations. A capture compares a kernel against the
translation only as far as its dumped windows reach; two builds of the same
kernel compare everywhere, because both run on the same reconstructed pool.

Usage: same_output.py <capture> <entry> <grid x,y,z> <block x,y,z> <hsaco A> <hsaco B> [--time N]
"""
import glob
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hip_run import Hip  # noqa: E402


def run(h, hsaco, entry, grid, block, params0, pre, repeat):
    fn = h.module(hsaco, entry)
    params = bytearray(params0)
    ptrs = {o: struct.unpack_from("<Q", params, o)[0] for o in pre}
    low = min(ptrs.values())
    pool = np.zeros(max(ptrs[o] + len(pre[o]) for o in pre) - low + (128 << 20), np.uint8)
    at = {}
    for o, d in sorted(pre.items(), key=lambda kv: ptrs[kv[0]]):
        at[o] = ptrs[o] - low
        pool[at[o]:at[o] + len(d)] = np.frombuffer(d, np.uint8)
    base = h.upload(pool)
    for o in pre:
        struct.pack_into("<Q", params, o, base + at[o])
    h.launch(fn, grid, block, bytes(params))
    out = h.download(base, len(pool))
    ms = None
    if repeat:
        h.write(base, pool)
        ms = h.time(fn, grid, block, bytes(params), repeat=repeat)
    return out, ms


def main():
    cap, entry, grid, block, a, b = sys.argv[1:7]
    repeat = int(sys.argv[sys.argv.index("--time") + 1]) if "--time" in sys.argv else 0
    grid = tuple(int(v) for v in grid.split(","))
    block = tuple(int(v) for v in block.split(","))
    base = glob.glob(os.path.join(cap, "*_args.bin"))[0][:-9]
    params = open(base + "_args.bin", "rb").read()
    pre = {int(f[-6:-4], 16): open(f, "rb").read() for f in glob.glob(base + "_pre_+*.bin")}
    h = Hip()
    out_a, ms_a = run(h, a, entry, grid, block, params, pre, repeat)
    out_b, ms_b = run(h, b, entry, grid, block, params, pre, repeat)
    differ = int((out_a != out_b).sum())
    print("%s: %d bytes differ" % ("SAME" if differ == 0 else "DIFFERENT", differ))
    if repeat:
        print("%s %.4f ms   %s %.4f ms   (%+.1f%%)" % (os.path.basename(a), ms_a, os.path.basename(b), ms_b,
                                                   100.0 * (ms_b - ms_a) / ms_a))


if __name__ == "__main__":
    main()
