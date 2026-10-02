#!/usr/bin/env python3
"""The PTX approximate f32 operations, as ZLUDA translates them, against the
HIP expressions a native kernel uses in their place.

Each operation runs over the same inputs twice: once as a PTX kernel through
ZLUDA, once as a HIP kernel, in the same process. The results are compared
bit for bit.

Usage: approx_check.py
"""
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hip_run import Hip  # noqa: E402
from zluda_run import Zluda  # noqa: E402

OPS = {
    # name: (PTX body using %f1 [, %f2] -> %f0, number of inputs)
    "div":   ("div.approx.ftz.f32 %f0, %f1, %f2;", 2),
    "rcp":   ("rcp.approx.ftz.f32 %f0, %f1;", 1),
    "lg2":   ("lg2.approx.ftz.f32 %f0, %f1;", 1),
    "sqrt":  ("sqrt.approx.ftz.f32 %f0, %f1;", 1),
    "rsqrt": ("rsqrt.approx.ftz.f32 %f0, %f1;", 1),
    "sin":   ("sin.approx.ftz.f32 %f0, %f1;", 1),
    "cos":   ("cos.approx.ftz.f32 %f0, %f1;", 1),
    "floor": ("cvt.rmi.ftz.f32.f32 %f0, %f1;", 1),
    "fma":   ("fma.rn.ftz.f32 %f0, %f1, %f2, %f1;", 2),
}

PTX = """.version 8.0
.target sm_80
.address_size 64
.visible .entry k_%(name)s(.param .align 8 .b8 p[24])
{
    .reg .b64 %%rd<8>;
    .reg .b32 %%r<4>;
    .reg .f32 %%f<3>;
    ld.param.u64 %%rd1, [p];
    ld.param.u64 %%rd2, [p+8];
    ld.param.u64 %%rd3, [p+16];
    mov.u32 %%r1, %%ctaid.x;
    mov.u32 %%r2, %%ntid.x;
    mov.u32 %%r3, %%tid.x;
    mad.lo.s32 %%r1, %%r1, %%r2, %%r3;
    mul.wide.s32 %%rd4, %%r1, 4;
    add.s64 %%rd5, %%rd1, %%rd4;
    add.s64 %%rd6, %%rd2, %%rd4;
    add.s64 %%rd7, %%rd3, %%rd4;
    ld.global.f32 %%f1, [%%rd5];
    ld.global.f32 %%f2, [%%rd6];
    %(body)s
    st.global.f32 [%%rd7], %%f0;
    ret;
}
"""


def inputs(name, n, rng):
    if name == "div":
        return rng.uniform(0.5, 4000, n).astype(np.float32), rng.uniform(1, 4000, n).astype(np.float32)
    if name in ("rcp",):
        return rng.uniform(1e-3, 1e4, n).astype(np.float32), None
    if name == "lg2":
        return (rng.integers(1, 1 << 24, n) * np.float32(2.0 ** -24)).astype(np.float32), None
    if name in ("sqrt", "rsqrt"):
        return rng.uniform(1e-6, 40, n).astype(np.float32), None
    if name in ("sin", "cos"):
        return (rng.integers(1, 1 << 24, n) * np.float32(2.0 ** -24) * np.float32(6.2831855)).astype(np.float32), None
    if name == "floor":
        return rng.uniform(-10, 3000, n).astype(np.float32), None
    return rng.uniform(-4, 4, n).astype(np.float32), rng.uniform(-4, 4, n).astype(np.float32)


def main():
    n = 1 << 20
    rng = np.random.default_rng(1)
    z = Zluda()
    h = Hip()
    native = h.module(os.path.join(os.path.dirname(HERE), "approx_check.hsaco"), "approx_ops")
    names = list(OPS)
    for index, name in enumerate(names):
        body, arity = OPS[name]
        a, b = inputs(name, n, rng)
        if b is None:
            b = np.zeros(n, np.float32)
        fn = z.module(PTX % {"name": name, "body": body}, "k_" + name)
        da, db = z.upload(a), z.upload(b)
        dout = z.upload(np.zeros(n, np.float32))
        z.launch(fn, (n // 256, 1, 1), (256, 1, 1), struct.pack("<3Q", da, db, dout))
        ref = z.download(dout, 4 * n).view(np.float32)
        ha, hb, hout = h.upload(a), h.upload(b), h.upload(np.zeros(n, np.float32))
        h.launch(native, (n // 256, 1, 1), (256, 1, 1), struct.pack("<3Qi4x", ha, hb, hout, index))
        got = np.frombuffer(h.download(hout, 4 * n), np.float32)
        same = (ref.view(np.uint32) == got.view(np.uint32)).mean()
        worst = np.nanmax(np.abs(ref.astype(np.float64) - got))
        print("%-6s %8.4f%% bit-identical, worst difference %.3g" % (name, 100 * same, worst))


if __name__ == "__main__":
    main()
