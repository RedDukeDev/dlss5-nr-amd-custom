#!/usr/bin/env python3
"""Run a translated kernel on a captured launch, through ZLUDA, on this card.

Two jobs, both wanted while a native kernel is being written:

  --out   saves what the translation writes into the capture's output buffer,
          which is the reference a native kernel has to match byte for byte.
          It is not the same as the capture's own post dump: a capture only
          dumps a window of each buffer, so blocks whose input lies past the
          window read zeros here and write something else. Comparing native
          against this run removes that, and leaves only real differences.
  --probe saves what a probe put in by probe_ptx.py wrote, one word per lane
          per block, which is how an intermediate value is pinned down.

The capture's buffers are rebuilt as one allocation that keeps the distances
between the captured pointers (ab_capture.py --pool does the same), because a
kernel whose surface is larger than the dumped window would otherwise write
past the end of one buffer and into the next.

Usage: probe_zluda.py <capture dir> <ptx> <entry> --grid X,Y,Z [--block X,Y,Z]
                      [--out file.npy] [--probe file.npy] [--stride 640] [--slot 64]
"""
import argparse
import glob
import io
import os
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from zluda_run import Zluda  # noqa: E402

MARGIN = 128 << 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("ptx")
    ap.add_argument("entry")
    ap.add_argument("--grid", required=True)
    ap.add_argument("--block", default="32,1,1")
    ap.add_argument("--out", help="save the output buffer here")
    ap.add_argument("--probe", help="save the probe buffer here")
    ap.add_argument("--stride", type=int, default=640)
    ap.add_argument("--slot", type=int, default=64)
    ap.add_argument("--output-slot", type=int, default=8, dest="output_slot")
    args = ap.parse_args()
    grid = tuple(int(v) for v in args.grid.split(","))
    block = tuple(int(v) for v in args.block.split(","))

    base = glob.glob(os.path.join(args.capture, "*_args.bin"))[0][:-9]
    params = bytearray(open(base + "_args.bin", "rb").read())
    pre = {int(f[-6:-4], 16): open(f, "rb").read() for f in glob.glob(base + "_pre_+*.bin")}

    text = io.open(args.ptx, encoding="utf-8").read()
    if not text.lstrip().startswith(".version"):
        text = ".version 9.4\n.target sm_120\n.address_size 64\n" + text
    z = Zluda()
    started = time.time()
    fn = z.module(text, args.entry)
    print("translation: %.1f s" % (time.time() - started))

    ptrs = {off: struct.unpack_from("<Q", params, off)[0] for off in pre}
    low = min(ptrs.values())
    span = max(ptrs[off] + len(pre[off]) for off in pre) - low + MARGIN
    pool = np.zeros(span, np.uint8)
    for off, data in pre.items():
        pool[ptrs[off] - low: ptrs[off] - low + len(data)] = np.frombuffer(data, np.uint8)
    gpu = z.upload(pool)
    for off in pre:
        struct.pack_into("<Q", params, off, gpu + ptrs[off] - low)

    probe_bytes = grid[0] * grid[1] * grid[2] * args.stride * 4
    if args.probe:
        probe = z.upload(np.zeros(probe_bytes, np.uint8))
        struct.pack_into("<Q", params, args.slot, probe)
    z.launch(fn, grid, block, bytes(params))

    if args.out:
        out = z.download(gpu + ptrs[args.output_slot] - low, len(pre[args.output_slot]))
        np.save(args.out, out)
        print("output saved: %s" % args.out)
    if args.probe:
        np.save(args.probe, z.download(probe, probe_bytes))
        print("probe saved: %s" % args.probe)


if __name__ == "__main__":
    main()
