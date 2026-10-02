#!/usr/bin/env python3
"""A native kernel on a captured launch, compared with the translated one.

Generic form of ab_ffwd.py: every buffer the capture dumped is uploaded on its
own, the argument buffer is rewritten to point at the uploads, the native
kernel runs on the given grid and block, and every buffer the translated kernel
changed is compared byte for byte (e4m3 differences are reported as values).

The capture's windows onto one memory pool become separate buffers here, so a
kernel that reads what it writes in the same launch cannot be checked this way.
--pool instead rebuilds a single allocation that keeps the distances between the
captured pointers, which is what a kernel whose surface is larger than the
dumped window needs: separate buffers would have it write past the end of one
and into the next.

Usage: ab_capture.py <capture dir> <code object> <entry> --grid X,Y,Z --block X,Y,Z [--time N] [--pool]
"""
import argparse
import glob
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ptxsim  # noqa: E402
from hip_run import Hip  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("hsaco")
    ap.add_argument("entry")
    ap.add_argument("--grid", required=True)
    ap.add_argument("--block", required=True)
    ap.add_argument("--time", type=int, default=0)
    ap.add_argument("--pool", action="store_true",
                    help="one allocation holding every window at its captured distance")
    args = ap.parse_args()
    grid = tuple(int(v) for v in args.grid.split(","))
    block = tuple(int(v) for v in args.block.split(","))

    base = glob.glob(os.path.join(args.capture, "*_args.bin"))[0][:-9]
    params = bytearray(open(base + "_args.bin", "rb").read())
    pre = {int(f[-6:-4], 16): open(f, "rb").read() for f in glob.glob(base + "_pre_+*.bin")}
    post = {int(f[-6:-4], 16): open(f, "rb").read() for f in glob.glob(base + "_post_+*.bin")}

    h = Hip()
    fn = h.module(args.hsaco, args.entry)
    dev, at = {}, {}
    if args.pool:
        # The dumped window of a surface can be smaller than the surface, so the
        # pool gets a margin past the last one; anything the kernel writes there
        # stays inside the allocation instead of landing on another buffer.
        MARGIN = 128 << 20
        ptrs = {off: struct.unpack_from("<Q", params, off)[0] for off in pre}
        low = min(ptrs.values())
        span = max(ptrs[off] + len(pre[off]) for off in pre) - low + MARGIN
        pool = np.zeros(span, np.uint8)
        for off, data in pre.items():
            at[off] = ptrs[off] - low
            pool[at[off]:at[off] + len(data)] = np.frombuffer(data, np.uint8)
        base = h.upload(pool)
        for off in pre:
            dev[off] = base + at[off]
            struct.pack_into("<Q", params, off, dev[off])
    else:
        for off, data in pre.items():
            dev[off] = h.upload(np.frombuffer(data, np.uint8))
            struct.pack_into("<Q", params, off, dev[off])
    h.launch(fn, grid, block, bytes(params))

    for off, after in sorted(post.items()):
        before = np.frombuffer(pre[off], np.uint8)
        gpu = np.frombuffer(after, np.uint8)
        mine = h.download(dev[off], len(after))
        region = (gpu != before) | (mine != before)
        if not region.any():
            continue
        same = gpu[region] == mine[region]
        diff = np.abs(ptxsim.E4M3[gpu[region][~same]] - ptxsim.E4M3[mine[region][~same]]) if (~same).any() else [0]
        print("+0x%02X: translated changed %d bytes, native %d; %.4f%% of the union identical, max |e4m3 diff| %.4g"
              % (off, (gpu != before).sum(), (mine != before).sum(), 100.0 * same.mean(), np.max(diff)))
    if args.time:
        for off, data in pre.items():            # restore inputs the kernel may have touched
            h.write(dev[off], np.frombuffer(data, np.uint8))
        ms = h.time(fn, grid, block, bytes(params), repeat=args.time)
        print("native kernel: %.4f ms per launch (%d launches)" % (ms, args.time))


if __name__ == "__main__":
    main()
