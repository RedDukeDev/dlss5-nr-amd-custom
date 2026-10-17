#!/usr/bin/env python3
"""The pre block's time at a game's resolution, native, and its output for comparison.

Binds what the network's own launch binds at 2560x1440 (the colour texture alone,
flags at one: see DLSS5NR_GATE_DUMP in runtime/gate/native_hook.h), with the weights
of the captures directory, and times the native kernel of 310.9.16 (_native2) with HIP
events. --hsaco=A,B times two builds and checks their outputs are the same bytes:
what a change meant to be faster is held to.

Usage: pre_block_bench.py [--hsaco=PATH[,PATH]] [--size=HxW] [--repeat=N] [--translated]
"""
import glob
import io
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hip_run import Hip  # noqa: E402
from zluda_run import Zluda  # noqa: E402
import workspace  # noqa: E402

ENTRY = "cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8"
H, W, REPEAT = 1440, 2560, 20
SHARED = 0
HSACOS = [os.path.join(os.path.dirname(HERE), "swin_1h.hsaco")]
for a in sys.argv[1:]:
    if a.startswith("--hsaco="):
        HSACOS = a[8:].split(",")
    if a.startswith("--size="):
        H, W = (int(v) for v in a[7:].split("x"))
    if a.startswith("--repeat="):
        REPEAT = int(a[9:])
    if a.startswith("--shared="):   # extra dynamic LDS per block: fewer blocks resident at once
        SHARED = int(a[9:])
TOK_Y, TOK_X = (H + 31) // 32 * 32, (W + 31) // 32 * 32
HALF_Y, HALF_X = TOK_Y // 2, TOK_X // 2


def params(color, out, weights, half):
    p = struct.pack("<5Q", color, 0, 0, 0, 0)
    p += struct.pack("<i", 0)                                    # mask blend mode
    p += struct.pack("<6f", 0, 0, 0, 0, 0, 0) * 4                # history, motion, depth, extra
    p += struct.pack("<6f", 0.0, 0.0, float(W), float(H), 1.0 / W, 1.0 / H)   # colour
    p += struct.pack("<2f", 1.0 / W, 1.0 / H)                    # motion vector scale
    p += struct.pack("<i", 0)
    p += struct.pack("<f", 1.0)
    p += struct.pack("<2f", 1.0, 0.0)
    p += struct.pack("<2f", 1.0, 1.0)                            # flags
    p += struct.pack("<2i", 15, 1)                               # channels, flag mode
    p += struct.pack("<f", 0.0625)
    p += struct.pack("<2i", 0, 0)
    p += struct.pack("<2i", H, W)
    p += struct.pack("<3Q", out, weights, 0)
    p += struct.pack("<2i", TOK_Y, TOK_X)
    p += struct.pack("<Q", half)
    p += struct.pack("<2i", HALF_Y, HALF_X)
    assert len(p) == 272
    return p


def main():
    rng = np.random.default_rng(1)
    z = Zluda()
    h = Hip()
    color = z.texture(z.array2d(rng.uniform(0, 1, (H, W, 4)).astype(np.float16)))
    base = glob.glob(os.path.join(workspace.captures(), "capture_pre16", "*_args.bin"))[0][:-9]
    weights = z.upload(np.fromfile(base + "_pre_+E0.bin", np.uint8)[:1 << 16])
    out_bytes = TOK_Y * TOK_X * 32
    half_bytes = 2 * HALF_Y * HALF_X * 16
    grid, block = (TOK_X // 8, TOK_Y // 8, 1), (32, 1, 1)
    results = []
    for path in HSACOS:
        fn = h.module(path, ENTRY + "_native2")
        out = z.upload(np.zeros(out_bytes, np.uint8))
        half = z.upload(np.zeros(half_bytes, np.uint8))
        p = params(color, out, weights, half)
        ms = h.time(fn, grid, block, p, repeat=REPEAT, shared=SHARED)
        results.append((z.download(out, out_bytes), z.download(half, half_bytes)))
        print("%-60s %7.3f ms at %dx%d" % (path, ms, W, H))
    if "--translated" in sys.argv:
        text = ".version 9.4\n.target sm_120\n.address_size 64\n" + io.open(
            os.path.join(workspace.captures(), "pre_block_32.ptx"), encoding="utf-8").read()
        fn = z.module(text, ENTRY)
        out = z.upload(np.zeros(out_bytes, np.uint8))
        half = z.upload(np.zeros(half_bytes, np.uint8))
        p = params(color, out, weights, half)
        z.launch(fn, grid, block, p)
        results.append((z.download(out, out_bytes), z.download(half, half_bytes)))
        print("translated: output kept for comparison")
    for i in range(1, len(results)):
        same = all((a == b).all() for a, b in zip(results[0], results[i]))
        print("output %d against the first: %s" % (i, "identical" if same else "DIFFERENT"))


if __name__ == "__main__":
    main()
