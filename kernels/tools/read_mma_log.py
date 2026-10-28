#!/usr/bin/env python3
"""The debug buffer instrument_ptx.py's replacement leaves (runtime/gate/swap.h), as matrices.

    from read_mma_log import load
    log = load("enc0_80_40.bin")     # arrays over mma number: a (n,16,32) codes, b (n,32,8) codes, c, d (n,16,8) f16

Layout per mma k and lane l, at 256 + k * 1280 + l * 40: D0 D1 A0 A1 A2 A3 B0 B1 C0 C1 (u32 each), the
m16n8k32 fragments of PTX: A row g (+8) holds k = 4t .. 4t + 3 (+16), B column g holds k = 4t .. (+16), C and D
hold row g (+8), columns 2t, 2t + 1 as f16 pairs; g = lane // 4, t = lane % 4.
"""
import numpy as np


def load(path):
    raw = np.fromfile(path, np.uint8)
    n = (len(raw) - 256) // 1280
    words = raw[256:256 + n * 1280].view(np.uint32).reshape(n, 32, 10)
    a = np.zeros((n, 16, 32), np.uint8)
    b = np.zeros((n, 32, 8), np.uint8)
    c = np.zeros((n, 16, 8), np.float16)
    d = np.zeros((n, 16, 8), np.float16)
    for lane in range(32):
        g, t = lane // 4, lane % 4
        for r in range(4):
            row = g + (8 if r in (1, 3) else 0)
            col0 = 4 * t + (16 if r >= 2 else 0)
            w = words[:, lane, 2 + r]
            for byte in range(4):
                a[:, row, col0 + byte] = (w >> (8 * byte)) & 0xFF
        for r in range(2):
            w = words[:, lane, 6 + r]
            for byte in range(4):
                b[:, 4 * t + 16 * r + byte, g] = (w >> (8 * byte)) & 0xFF
        for r in range(2):
            row = g + 8 * r
            for src, dst in ((8, c), (0, d)):
                w = words[:, lane, src + r]
                dst[:, row, 2 * t] = (w & 0xFFFF).astype(np.uint16).view(np.float16)
                dst[:, row, 2 * t + 1] = (w >> 16).astype(np.uint16).view(np.float16)
    return {"a": a, "b": b, "c": c, "d": d, "n": n}
