#!/usr/bin/env python3
"""Any kernel's PTX, run by ptxsim on any captured launch, checked buffer by buffer.

The generic form of run_ffwd.py: every pre dump of the capture is mapped at the
address its parameter word held, the kernel runs on the launch's grid and block,
and every post dump is compared with what the emulation left in memory.

Usage: run_ptx.py <capture dir> <kernel.ptx> --grid X,Y,Z --block X,Y,Z [--save emu.npz]
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


def load_capture(capture):
    base = glob.glob(os.path.join(capture, "*_args.bin"))[0][:-9]
    params = open(base + "_args.bin", "rb").read()
    pre, post = {}, {}
    for f in glob.glob(base + "_pre_+*.bin"):
        pre[int(f[-6:-4], 16)] = open(f, "rb").read()
    for f in glob.glob(base + "_post_+*.bin"):
        post[int(f[-6:-4], 16)] = open(f, "rb").read()
    return params, pre, post


def emulate(ptx_text, params, pre, grid, block, z_serial=False, blocks=None, batch=0):
    words = struct.unpack_from("<%dQ" % (len(params) // 8), params)
    mem = ptxsim.GlobalMemory()
    for off, data in sorted(pre.items()):
        mem.add(words[off // 8], data)
    kernel = ptxsim.Kernel(ptx_text)
    if z_serial:
        for z in range(grid[2]):
            ptxsim.Machine(kernel, grid, block, params, mem, z_slice=z).run()
        return mem, words
    total = grid[0] * grid[1] * grid[2]
    first, last = blocks if blocks else (0, total)
    last = min(last, total)
    step = batch or (last - first)
    for start in range(first, last, step):
        ptxsim.Machine(kernel, grid, block, params, mem,
                       blocks=(start, min(start + step, last))).run()
    return mem, words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("ptx")
    ap.add_argument("--grid", required=True)
    ap.add_argument("--block", required=True)
    ap.add_argument("--save")
    ap.add_argument("--blocks", help="FIRST:LAST of the flattened grid")
    ap.add_argument("--batch", type=int, default=0, help="blocks per machine, for big grids")
    ap.add_argument("--z-serial", action="store_true",
                    help="run the z slices one after the other (split K chained through flags)")
    args = ap.parse_args()
    grid = tuple(int(v) for v in args.grid.split(","))
    block = tuple(int(v) for v in args.block.split(","))
    text = open(args.ptx, encoding="utf-8").read()
    if ".version" not in text:
        text = ".version 9.4\n.target sm_120\n.address_size 64\n" + text
    params, pre, post = load_capture(args.capture)
    rng = tuple(int(v) for v in args.blocks.split(":")) if args.blocks else None
    mem, words = emulate(text, params, pre, grid, block, args.z_serial, rng, args.batch)
    saved = {}
    for off, after in sorted(post.items()):
        before = np.frombuffer(pre[off], np.uint8)
        gpu = np.frombuffer(after, np.uint8)
        emu = mem.read(words[off // 8], len(after))
        changed = gpu != before
        emu_changed = emu != before
        region = changed | emu_changed
        # When only part of the grid was run, what matters is whether the bytes
        # the emulation wrote are the bytes the GPU wrote.
        wrote = emu_changed
        agree = 100.0 * (gpu[wrote] == emu[wrote]).mean() if wrote.any() else 100.0
        print("+0x%02X: GPU changed %d bytes, emulation %d, of which %.3f%% match; over the union %.3f%%"
              % (off, changed.sum(), emu_changed.sum(), agree,
                 100.0 * (gpu[region] == emu[region]).mean() if region.any() else 100.0))
        saved["emu_%02X" % off] = emu
    if args.save:
        np.savez(args.save, **saved)


if __name__ == "__main__":
    main()
