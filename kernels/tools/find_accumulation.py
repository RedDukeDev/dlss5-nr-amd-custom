#!/usr/bin/env python3
"""Which model of the matrix unit's summation reproduces the hardware.

Runs the hidden-layer variant of the ffwd kernel through ptxsim under each
candidate model and reports how many f16 values come out identical to what the
GPU wrote. The model that reaches 100% is the one a native kernel has to match
-- or at least the one to measure a native kernel against.

Usage: find_accumulation.py
"""
import glob
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ptxsim  # noqa: E402
import workspace  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
E = 120 * 16384


def load(capture):
    b = glob.glob(os.path.join(HERE, capture, "*_args.bin"))[0][:-9]
    params = open(b + "_args.bin", "rb").read()
    p = struct.unpack_from("<7Q", params)
    dumps = {off: open("%s_pre_+%02X.bin" % (b, off), "rb").read() for off in (0, 8, 0x10, 0x20, 0x30)}
    gpu = np.frombuffer(open(b + "_post_+08.bin", "rb").read()[:E], dtype=np.uint8).view(np.uint16)
    return params, p, dumps, gpu


def run(params, p, dumps, kernel):
    mem = ptxsim.GlobalMemory()
    for off, addr in {0x00: p[0], 0x08: p[1], 0x10: p[2], 0x20: p[4], 0x30: p[6]}.items():
        mem.add(addr, dumps[off])
    m = ptxsim.Machine(kernel, (10, 6, 2), (32, 8, 1), params, mem)
    m.run()
    return mem.read(p[1], E).view(np.uint16)


def main():
    params, p, dumps, gpu = load("capture_hidden")
    kernel = ptxsim.Kernel(open(os.path.join(workspace.captures(), "ffwd_hidden.ptx"), encoding="utf-8").read())
    models = [("exact", None, None)]
    for trunc in ("rz", "rd"):
        for bits in (18, 19, 20, 21, 22, 23, 24, 25, 26):
            models.append(("align", bits, trunc))
    for mode, bits, trunc in models:
        ptxsim.Machine.mma_mode = mode
        if bits:
            ptxsim.Machine.mma_align_bits = bits
            ptxsim.Machine.mma_align_trunc = trunc
        emu = run(params, p, dumps, kernel)
        same = 100.0 * (emu == gpu).mean()
        print("%-6s %4s %-3s %8.3f%% identical" % (mode, bits or "", trunc or "", same), flush=True)


if __name__ == "__main__":
    main()
