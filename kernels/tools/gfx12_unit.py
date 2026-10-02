#!/usr/bin/env python3
"""The RDNA4 building blocks of dlssnr_hip.h against the RDNA3 ones, on RDNA3.

gfx12_unit.hip is built twice for this card, normally and with the RDNA4 path
emulated (DLSSNR_EMULATE_GFX12), fed the same random operands, and the outputs
compared byte for byte. check_gfx12.py does the same on the whole network;
this is where a difference is small enough to read.

Usage: gfx12_unit.py
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hip_run import Hip  # noqa: E402

HIPCC = r"C:\Program Files\AMD\ROCm\7.2\bin\hipcc.exe"
DEFINES = ["-D__CLANG_HIP_CMATH_H__", "-D__CLANG__CUDA_MATH_FORWARD_DECLARES_H__",
           "-D__CLANG_CUDA_COMPLEX_BUILTINS"]


def build(extra, out):
    cmd = [HIPCC, "--genco", "--offload-arch=gfx1100", "-O3"] + extra + DEFINES + \
        [os.path.join(os.path.dirname(HERE), "gfx12_unit.hip"), "-o", out]
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if r.returncode:
        sys.exit(r.stderr[-2000:])


def e4m3_codes(rng, n):
    # finite codes only (0x7F and 0xFF are NaN), small enough not to saturate
    c = rng.integers(0, 0x60, n, dtype=np.uint8)
    return c | (rng.integers(0, 2, n, dtype=np.uint8) << 7)


def widen(codes):
    """e4m3 codes to their f16 values."""
    sign = np.where(codes & 0x80, -1.0, 1.0)
    e, m = (codes >> 3) & 15, codes & 7
    v = np.where(e == 0, m / 8.0 * 2.0 ** -6, (1 + m / 8.0) * 2.0 ** (e.astype(np.float64) - 7))
    return (sign * v).astype(np.float16)


def main():
    rng = np.random.default_rng(1)
    # gfx11 view: lane l holds row l % 16, the same in both halves. Operands are
    # e4m3 values and accumulators f16 values of mixed size, as in the network,
    # where the WMMA's truncating accumulation shows in the last bit.
    tiles = 2048
    # half the tiles e4m3 values, half f16 over a wide range (the f16 products
    # of the pre and post blocks)
    wide = lambda: (rng.standard_normal((tiles // 2, 16, 16)) *
                    2.0 ** rng.integers(-10, 6, (tiles // 2, 16, 16))).astype(np.float16)
    a = np.concatenate([widen(e4m3_codes(rng, tiles * 128)).reshape(tiles // 2, 16, 16), wide()])
    b = np.concatenate([widen(e4m3_codes(rng, tiles * 128)).reshape(tiles // 2, 16, 16), wide()])
    a32 = np.concatenate([a, a], axis=1); b32 = np.concatenate([b, b], axis=1)
    c = (rng.standard_normal((tiles, 32, 8)) * 2.0 ** rng.integers(-4, 8, (tiles, 32, 8))
         ).astype(np.float16).astype(np.float32)
    codes = e4m3_codes(rng, tiles * 4 * 16 * 16).reshape(tiles, 4, 16, 16)
    codes32 = np.concatenate([codes, codes], axis=2)            # rows duplicated in both halves

    results = {}
    h = Hip()
    for tag, extra in (("rdna3", []), ("rdna4-emulated", ["-DDLSSNR_EMULATE_GFX12"])):
        path = os.path.join(os.path.dirname(HERE), "gfx12_unit_%s.hsaco" % tag)
        build(extra, path)
        out = {}
        fn = h.module(path, "unit_wmma")
        pa, pb, pc = h.upload(a32), h.upload(b32), h.upload(c)
        pd = h.alloc(tiles * 32 * 32)
        h.launch(fn, (tiles, 1, 1), (32, 1, 1), np.array([pa, pb, pc, pd], np.uint64).tobytes())
        out["wmma"] = h.download(pd, tiles * 32 * 32)
        fn = h.module(path, "unit_slice8")
        pcodes = h.upload(np.ascontiguousarray(codes32))
        po = h.alloc(tiles * 64 * 16)
        h.launch(fn, (tiles, 1, 1), (32, 1, 1), np.array([pcodes, po], np.uint64).tobytes())
        out["slice8"] = h.download(po, tiles * 64 * 16)
        results[tag] = out
        os.remove(path)

    failed = 0
    tiles = len(results["rdna3"]["wmma"]) // 1024
    for name in ("wmma", "slice8"):
        x = np.frombuffer(bytes(results["rdna3"][name]), np.uint32)
        y = np.frombuffer(bytes(results["rdna4-emulated"][name]), np.uint32)
        bad = np.nonzero(x != y)[0]
        if len(bad) and name == "wmma":
            per_tile = np.bincount(bad // 256, minlength=tiles)
            print("        tiles with differences: %d of %d (e4m3 half: %d)"
                  % ((per_tile > 0).sum(), tiles, (per_tile[:tiles // 2] > 0).sum()))
        print("%-7s %s" % (name, "SAME" if not len(bad) else "%d of %d words differ, first at %s"
                           % (len(bad), len(x), list(bad[:8]))))
        failed += bool(len(bad))
    print("RESULT: %s" % ("the RDNA4 path matches" if not failed else "FAILED"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
