#!/usr/bin/env python3
"""One mma.sync.m16n8k32 e4m3 -> f16, on the GPU and in ptxsim, same inputs.

Each warp loads its A, B and C registers straight from memory -- the lane's own
words, no layout of ours in between -- runs one mma, and stores D. Both sides
see the same registers, so any element that differs is a difference in how the
instruction itself computes, and its exact inputs are on hand to study.

Usage: test_mma.py [warps] [seed] [--kind random|network]
"""
import argparse
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ptxsim  # noqa: E402
from zluda_run import Zluda  # noqa: E402

PTX = r"""
.version 9.4
.target sm_120
.address_size 64

.visible .entry mma_probe(
.param .align 8 .b8 mma_probe_param_0[32]
)
{
.reg .b32 %r<16>;
.reg .b64 %rd<16>;
ld.param.b64 %rd1, [mma_probe_param_0];
ld.param.b64 %rd2, [mma_probe_param_0+8];
ld.param.b64 %rd3, [mma_probe_param_0+16];
ld.param.b64 %rd4, [mma_probe_param_0+24];
mov.u32 %r0, %ctaid.x;
mov.u32 %r1, %tid.x;
shl.b32 %r2, %r0, 5;
add.s32 %r3, %r2, %r1;
mul.wide.u32 %rd5, %r3, 16;
add.s64 %rd6, %rd1, %rd5;
ld.global.v4.u32 {%r4, %r5, %r6, %r7}, [%rd6];
mul.wide.u32 %rd7, %r3, 8;
add.s64 %rd8, %rd2, %rd7;
ld.global.v2.u32 {%r8, %r9}, [%rd8];
add.s64 %rd9, %rd3, %rd7;
ld.global.v2.u32 {%r10, %r11}, [%rd9];
mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%r12, %r13}, {%r4, %r5, %r6, %r7}, {%r8, %r9}, {%r10, %r11};
add.s64 %rd10, %rd4, %rd7;
st.global.v2.u32 [%rd10], {%r12, %r13};
ret;
}
"""


def e4m3_codes(rng, n, kind):
    """Codes that look like the network's: mostly small magnitudes, no NaN."""
    if kind == "random":
        codes = rng.integers(0, 256, n, dtype=np.uint8)
        return np.where((codes & 0x7F) == 0x7F, codes & 0xF8, codes).astype(np.uint8)
    # Values drawn around the network's scale, then encoded.
    v = rng.normal(0, 1.0, n) * np.exp(rng.normal(0, 1.2, n))
    return ptxsim.f16_to_e4m3(v.astype(np.float16).astype(np.float64)).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("warps", nargs="?", type=int, default=4096)
    ap.add_argument("seed", nargs="?", type=int, default=1)
    ap.add_argument("--kind", default="network")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    lanes = args.warps * 32

    a = e4m3_codes(rng, lanes * 16, args.kind)
    b = e4m3_codes(rng, lanes * 8, args.kind)
    c16 = (rng.normal(0, 2.0, lanes * 4)).astype(np.float16)
    c = c16.view(np.uint8)

    gpu = Zluda()
    fn = gpu.module(PTX, "mma_probe")
    pa, pb, pc = gpu.upload(a), gpu.upload(b), gpu.upload(c)
    pd = gpu.upload(np.zeros(lanes * 8, dtype=np.uint8))
    params = struct.pack("<4Q", pa, pb, pc, pd)
    gpu.launch(fn, (args.warps, 1, 1), (32, 1, 1), params)
    d_gpu = gpu.download(pd, lanes * 8).view(np.uint16)

    mem = ptxsim.GlobalMemory()
    for addr, data in ((pa, a), (pb, b), (pc, c), (pd, np.zeros(lanes * 8, np.uint8))):
        mem.add(addr, data.tobytes())
    m = ptxsim.Machine(ptxsim.Kernel(PTX), (args.warps, 1, 1), (32, 1, 1), params, mem)
    m.run()
    d_emu = mem.read(pd, lanes * 8).view(np.uint16)

    bad = np.nonzero(d_gpu != d_emu)[0]
    print("%d mma results, %d differ (%.4f%%)" % (len(d_gpu), len(bad), 100.0 * len(bad) / len(d_gpu)))
    np.savez("prova_mma_%d.npz" % args.seed, a=a, b=b, c=c, gpu=d_gpu, emu=d_emu)
    for i in bad[:8]:
        g, e = ptxsim.f16_bits_to_f64(d_gpu[i]), ptxsim.f16_bits_to_f64(d_emu[i])
        print("  element %6d: gpu %-12g emu %-12g" % (i, g, e))


if __name__ == "__main__":
    main()
