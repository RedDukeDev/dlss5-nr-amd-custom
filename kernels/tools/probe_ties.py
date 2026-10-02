#!/usr/bin/env python3
"""Hand-built mma cases on the GPU, to see how ZLUDA's m16n8k32 e4m3 rounds ties.

Each case is a full 16x32 A, 32x8 B and 16x8 C chosen so that the exact result
of a few elements lands on a midpoint between two f16 values. The lane
registers are built from those matrices with the PTX fragment layout, so what
comes back can be read off matrix element by matrix element.
"""
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ptxsim  # noqa: E402
from test_mma import PTX  # noqa: E402
from zluda_run import Zluda  # noqa: E402

L = np.arange(32)
G, T = L >> 2, L & 3


def e4m3(value):
    """The e4m3 code of an exactly representable value."""
    hit = np.nonzero(ptxsim.E4M3 == value)[0]
    if len(hit) == 0:
        raise ValueError("%r is not an e4m3 value" % value)
    return int(hit[0])


def to_lanes(A, B, C):
    """Matrices (W,16,32 codes), (W,32,8 codes), (W,16,8 f16) -> register images."""
    W = A.shape[0]
    a = np.zeros((W, 32, 4, 4), np.uint8)
    b = np.zeros((W, 32, 2, 4), np.uint8)
    c = np.zeros((W, 32, 2, 2), np.float16)
    for r in range(4):
        for byte in range(4):
            a[:, :, r, byte] = A[:, G + (8 if r in (1, 3) else 0), 4 * T + byte + (16 if r >= 2 else 0)]
    for r in range(2):
        for byte in range(4):
            b[:, :, r, byte] = B[:, 4 * T + byte + (16 if r == 1 else 0), G]
    for r in range(2):
        for h in range(2):
            c[:, :, r, h] = C[:, G + (8 if r == 1 else 0), 2 * T + h]
    return a.ravel(), b.ravel(), c.view(np.uint8).ravel()


def from_lanes(d, W):
    d = d.view(np.float16).reshape(W, 32, 2, 2)
    D = np.zeros((W, 16, 8))
    for r in range(2):
        for h in range(2):
            D[:, G + (8 if r == 1 else 0), 2 * T + h] = d[:, :, r, h]
    return D


def run(A, B, C, gpu=None):
    gpu = gpu or Zluda()
    W = A.shape[0]
    a, b, c = to_lanes(A, B, C)
    fn = gpu.module(PTX, "mma_probe")
    pa, pb, pc = gpu.upload(a), gpu.upload(b), gpu.upload(c)
    pd = gpu.upload(np.zeros(W * 256, np.uint8))
    gpu.launch(fn, (W, 1, 1), (32, 1, 1), struct.pack("<4Q", pa, pb, pc, pd))
    return from_lanes(gpu.download(pd, W * 256), W)
