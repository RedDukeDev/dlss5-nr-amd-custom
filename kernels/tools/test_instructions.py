#!/usr/bin/env python3
"""The lane-crossing PTX instructions, on the GPU and in ptxsim, same inputs.

shfl, movmatrix and prmt decide where values end up in the attention kernels,
and a wrong model of any of them makes an emulation that runs and produces
nonsense. Each is checked here against the hardware, through ZLUDA, on random
data: one warp's worth of values in, the instruction's result out.

Usage: test_instructions.py
"""
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

.visible .entry probe(
.param .align 8 .b8 probe_param_0[16]
)
{
.reg .b32 %r<32>;
.reg .b64 %rd<8>;
ld.param.b64 %rd1, [probe_param_0];
ld.param.b64 %rd2, [probe_param_0+8];
mov.u32 %r1, %ctaid.x;
mov.u32 %r2, %tid.x;
shl.b32 %r3, %r1, 5;
add.s32 %r4, %r3, %r2;
mul.wide.u32 %rd3, %r4, 4;
add.s64 %rd4, %rd1, %rd3;
ld.global.b32 %r5, [%rd4];
mov.b32 %r6, 31;
mov.b32 %r7, -1;
mov.b32 %r8, 1;
shfl.sync.bfly.b32 %r9, %r5, %r8, %r6, %r7;
mov.b32 %r10, 4;
shfl.sync.bfly.b32 %r11, %r5, %r10, %r6, %r7;
mov.b32 %r12, 2;
shfl.sync.down.b32 %r13, %r5, %r12, %r6, %r7;
shfl.sync.up.b32 %r14, %r5, %r12, %r6, %r7;
mov.b32 %r15, 7;
shfl.sync.idx.b32 %r16, %r5, %r15, %r6, %r7;
movmatrix.sync.trans.aligned.m8n8.b16 %r17, %r5;
prmt.b32 %r18, %r5, %r9, 0x5410U;
prmt.b32 %r19, %r5, %r9, 0x7632U;
mul.wide.u32 %rd5, %r4, 32;
add.s64 %rd6, %rd2, %rd5;
st.global.b32 [%rd6], %r9;
st.global.b32 [%rd6+4], %r11;
st.global.b32 [%rd6+8], %r13;
st.global.b32 [%rd6+12], %r14;
st.global.b32 [%rd6+16], %r16;
st.global.b32 [%rd6+20], %r17;
st.global.b32 [%rd6+24], %r18;
st.global.b32 [%rd6+28], %r19;
ret;
}
"""

NAMES = ["bfly 1", "bfly 4", "down 2", "up 2", "idx 7", "movmatrix", "prmt 5410", "prmt 7632"]


def main():
    warps = 64
    rng = np.random.default_rng(3)
    src = rng.integers(0, 2 ** 32, warps * 32, dtype=np.uint64).astype(np.uint32)

    gpu = Zluda()
    fn = gpu.module(PTX, "probe")
    pin = gpu.upload(src)
    pout = gpu.upload(np.zeros(warps * 32 * 8, np.uint32))
    gpu.launch(fn, (warps, 1, 1), (32, 1, 1), struct.pack("<2Q", pin, pout))
    got = gpu.download(pout, warps * 32 * 32).view(np.uint32).reshape(-1, 8)

    mem = ptxsim.GlobalMemory()
    mem.add(pin, src.tobytes())
    mem.add(pout, bytes(warps * 32 * 32))
    m = ptxsim.Machine(ptxsim.Kernel(PTX), (warps, 1, 1), (32, 1, 1), struct.pack("<2Q", pin, pout), mem)
    m.run()
    mine = mem.read(pout, warps * 32 * 32).view(np.uint32).reshape(-1, 8)

    for i, name in enumerate(NAMES):
        same = (got[:, i] == mine[:, i]).mean()
        print("%-10s %s" % (name, "matches the hardware" if same == 1.0 else
                            "DIFFERS: %.1f%% equal, first lane %d: gpu %08X emu %08X" % (
                                100 * same, np.nonzero(got[:, i] != mine[:, i])[0][0],
                                got[np.nonzero(got[:, i] != mine[:, i])[0][0], i],
                                mine[np.nonzero(got[:, i] != mine[:, i])[0][0], i])))


if __name__ == "__main__":
    main()
