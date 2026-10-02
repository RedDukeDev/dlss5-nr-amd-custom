#!/usr/bin/env python3
"""tex.2d through ZLUDA against sample2d in a native kernel, on the same
texture object: random float16 RGBA texels, bilinear filtering, clamped
normalised coordinates, points spread over and past the edges.

Usage: texture_check.py
"""
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hip_run import Hip  # noqa: E402
from zluda_run import Zluda  # noqa: E402

PTX = """.version 8.0
.target sm_80
.address_size 64
.visible .entry sample_points(.param .align 8 .b8 p[32])
{
    .reg .b64 %rd<10>;
    .reg .b32 %r<4>;
    .reg .f32 %f<8>;
    ld.param.u64 %rd1, [p];
    ld.param.u64 %rd2, [p+8];
    ld.param.u64 %rd3, [p+16];
    ld.param.u64 %rd4, [p+24];
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.s32 %r1, %r1, %r2, %r3;
    mul.wide.s32 %rd5, %r1, 4;
    add.s64 %rd6, %rd2, %rd5;
    add.s64 %rd7, %rd3, %rd5;
    ld.global.f32 %f1, [%rd6];
    ld.global.f32 %f2, [%rd7];
    tex.2d.v4.f32.f32 {%f3, %f4, %f5, %f6}, [%rd1, {%f1, %f2}];
    mul.wide.s32 %rd8, %r1, 16;
    add.s64 %rd9, %rd4, %rd8;
    st.global.v4.f32 [%rd9], {%f3, %f4, %f5, %f6};
    ret;
}
"""


def main():
    rng = np.random.default_rng(2)
    z = Zluda()
    h = Hip()
    texels = rng.uniform(-2, 2, (97, 131, 4)).astype(np.float16)
    tex = z.texture(z.array2d(texels))
    n = 1 << 18
    u = rng.uniform(-0.05, 1.05, n).astype(np.float32)
    v = rng.uniform(-0.05, 1.05, n).astype(np.float32)
    fn = z.module(PTX, "sample_points")
    du, dv, dout = z.upload(u), z.upload(v), z.upload(np.zeros(4 * n, np.float32))
    z.launch(fn, (n // 256, 1, 1), (256, 1, 1), struct.pack("<4Q", tex, du, dv, dout))
    ref = z.download(dout, 16 * n).view(np.float32)
    native = h.module(os.path.join(os.path.dirname(HERE), "texture_check.hsaco"), "sample_points")
    hu, hv, hout = h.upload(u), h.upload(v), h.upload(np.zeros(4 * n, np.float32))
    h.launch(native, (n // 256, 1, 1), (256, 1, 1), struct.pack("<4Q", tex, hu, hv, hout))
    got = np.frombuffer(h.download(hout, 16 * n), np.float32)
    same = (ref.view(np.uint32) == got.view(np.uint32)).mean()
    print("tex.2d vs sample2d: %.4f%% bit-identical over %d points, values in [%.3f, %.3f]"
          % (100 * same, n, ref.min(), ref.max()))


if __name__ == "__main__":
    main()
