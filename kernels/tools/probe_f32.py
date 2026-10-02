#!/usr/bin/env python3
"""The WMMA ZLUDA builds e4m3 mma from, read out in f32 before any f16 rounding.

On gfx11 ZLUDA computes m16n8k32 e4m3 as two m16n8k16 f16 -> f32 steps, then
converts D to f16. Running those two steps as plain PTX f16 mma with an f32
accumulator gives the same hardware operation, but hands back the f32 sums
themselves, so what the matrix unit really computes can be compared with the
exact value, not just after rounding.

Usage (as a module): run_f32(A, B, C) with A (W,16,32), B (W,32,8) float
matrices of f16-representable values and C (W,16,8) f32.
"""
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zluda_run import Zluda  # noqa: E402

PTX = r"""
.version 9.4
.target sm_120
.address_size 64

.visible .entry wmma_f32_probe(
.param .align 8 .b8 wmma_f32_probe_param_0[32]
)
{
.reg .b32 %r<40>;
.reg .f32 %f<16>;
.reg .b64 %rd<16>;
ld.param.b64 %rd1, [wmma_f32_probe_param_0];
ld.param.b64 %rd2, [wmma_f32_probe_param_0+8];
ld.param.b64 %rd3, [wmma_f32_probe_param_0+16];
ld.param.b64 %rd4, [wmma_f32_probe_param_0+24];
mov.u32 %r0, %ctaid.x;
mov.u32 %r1, %tid.x;
shl.b32 %r2, %r0, 5;
add.s32 %r3, %r2, %r1;
mul.wide.u32 %rd5, %r3, 32;
add.s64 %rd6, %rd1, %rd5;
ld.global.v4.u32 {%r4, %r5, %r6, %r7}, [%rd6];
ld.global.v4.u32 {%r8, %r9, %r10, %r11}, [%rd6+16];
mul.wide.u32 %rd7, %r3, 16;
add.s64 %rd8, %rd2, %rd7;
ld.global.v4.u32 {%r12, %r13, %r14, %r15}, [%rd8];
add.s64 %rd9, %rd3, %rd7;
ld.global.v4.f32 {%f0, %f1, %f2, %f3}, [%rd9];
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%f4, %f5, %f6, %f7}, {%r4, %r5, %r6, %r7}, {%r12, %r13}, {%f0, %f1, %f2, %f3};
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%f8, %f9, %f10, %f11}, {%r8, %r9, %r10, %r11}, {%r14, %r15}, {%f4, %f5, %f6, %f7};
add.s64 %rd10, %rd4, %rd7;
st.global.v4.f32 [%rd10], {%f8, %f9, %f10, %f11};
ret;
}
"""

PTX_SINGLE = PTX.replace("wmma_f32_probe", "wmma_f32_single").replace(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%f8, %f9, %f10, %f11}, "
    "{%r8, %r9, %r10, %r11}, {%r14, %r15}, {%f4, %f5, %f6, %f7};", "").replace(
    "st.global.v4.f32 [%rd10], {%f8, %f9, %f10, %f11};", "st.global.v4.f32 [%rd10], {%f4, %f5, %f6, %f7};")

L = np.arange(32)
G, T = L >> 2, L & 3


def to_lanes(A, B, C):
    W = A.shape[0]
    a = np.zeros((W, 32, 2, 8), np.float16)     # two halves of k, 8 f16 each
    b = np.zeros((W, 32, 2, 4), np.float16)
    c = np.zeros((W, 32, 4), np.float32)
    for half in range(2):
        k0 = 16 * half
        for j in range(8):
            row = G + (8 if j in (2, 3, 6, 7) else 0)
            col = 2 * T + (j & 1) + (8 if j >= 4 else 0)
            a[:, :, half, j] = A[:, row, k0 + col]
        for j in range(4):
            k = 2 * T + (j & 1) + (8 if j >= 2 else 0)
            b[:, :, half, j] = B[:, k0 + k, G]
    for j in range(4):
        c[:, :, j] = C[:, G + (8 if j >= 2 else 0), 2 * T + (j & 1)]
    return a.view(np.uint8).ravel(), b.view(np.uint8).ravel(), c.view(np.uint8).ravel()


def from_lanes(d, W):
    d = d.view(np.float32).reshape(W, 32, 4)
    D = np.zeros((W, 16, 8))
    for j in range(4):
        D[:, G + (8 if j >= 2 else 0), 2 * T + (j & 1)] = d[:, :, j]
    return D


def run_f32(A, B, C, gpu=None, single=False):
    """single: only the first mma (k 0..15), so the second step cannot add anything."""
    gpu = gpu or Zluda()
    W = A.shape[0]
    a, b, c = to_lanes(A, B, C)
    fn = gpu.module(PTX_SINGLE, "wmma_f32_single") if single else gpu.module(PTX, "wmma_f32_probe")
    pa, pb, pc = gpu.upload(a), gpu.upload(b), gpu.upload(c)
    pd = gpu.upload(np.zeros(W * 512, np.uint8))
    gpu.launch(fn, (W, 1, 1), (32, 1, 1), struct.pack("<4Q", pa, pb, pc, pd))
    return from_lanes(gpu.download(pd, W * 512), W)
