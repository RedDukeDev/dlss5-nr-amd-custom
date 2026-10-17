#!/usr/bin/env python3
"""A fused Swin block's time on its own, native, on synthetic data.

Times cc_tinlayout_fused_swin_<n>h_..._chained_fp8_native at the token grid it runs
at for a 2560x1440 frame (the 2H block at half that, 1280x736 tokens), on random
tokens and random weights -- the time does not depend on the values -- and, with
two code objects, says whether their outputs are the same bytes. --shared adds
dynamic LDS to every block, which shows how much the kernel depends on how many
blocks fit at once.

Usage: swin_bench.py <1h|2h|4h|8h> [--hsaco=PATH[,PATH]] [--shared=BYTES] [--repeat=N]
"""
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hip_run import Hip  # noqa: E402

# heads -> (kernel, block, token grid at 2560x1440, channels, bytes of weights)
BLOCKS = {
    "1h": ("cc_tinlayout_fused_swin_1h_32_1_chained_fp8", (32, 1, 1), (736, 1280), 32, 1 << 16),
    "2h": ("cc_tinlayout_fused_swin_2h_64_2_chained_fp8", (32, 2, 1), (368, 640), 64, 1 << 17),
    "4h": ("cc_tinlayout_fused_swin_4h_128_4_chained_fp8", (32, 4, 1), (184, 320), 128, 1 << 18),
    "8h": ("cc_tinlayout_fused_swin_8h_256_8_chained_fp8", (32, 8, 1), (92, 160), 256, 1 << 19),
}


def main():
    which = sys.argv[1]
    name, block, (dim_y, dim_x), channels, wbytes = BLOCKS[which]
    hsacos = [os.path.join(os.path.dirname(HERE), "swin_%s.hsaco" % which)]
    shared, repeat = 0, 20
    for a in sys.argv[2:]:
        if a.startswith("--hsaco="):
            hsacos = a[8:].split(",")
        if a.startswith("--shared="):
            shared = int(a[9:])
        if a.startswith("--repeat="):
            repeat = int(a[9:])
    h = Hip()
    rng = np.random.default_rng(3)
    tokens = dim_y * dim_x
    # e4m3 codes kept small and finite: sign, exponent 1..6, any mantissa
    codes = lambda n: ((rng.integers(8, 56, n) | (rng.integers(0, 2, n) << 7)).astype(np.uint8))
    inp = h.upload(codes(tokens * channels))
    # every byte from 0x20 to 0x37: as e4m3 codes, values from 2^-3 to 2^0; read in
    # pairs as the f16 tables (scales, biases), values from 0.008 to 0.5 -- nothing
    # that overflows, so the comparison is between numbers, not NaNs
    weights = rng.integers(0x20, 0x38, wbytes).astype(np.uint8)
    wptr = h.upload(weights)
    out_bytes = tokens * channels
    grid = ((dim_x + 7) // 8, (dim_y + 7) // 8, 1)
    results = []
    for path in hsacos:
        fn = h.module(path, name + "_native")
        out = h.upload(np.zeros(out_bytes, np.uint8))
        p = struct.pack("<3Q", inp, out, wptr) + struct.pack("<Q", 0)
        p += struct.pack("<2i", dim_y, dim_x) + struct.pack("<2i", 0, 0)
        p += struct.pack("<Q", 0) + struct.pack("<2i", 0, 0) + struct.pack("<Q", 0) + struct.pack("<Q", 0)
        p += struct.pack("<2i", 0, 0)
        p = p.ljust(88 if which != "1h" else 96, b"\0")
        ms = h.time(fn, grid, block, p, repeat=repeat, shared=shared)
        results.append(h.download(out, out_bytes))
        print("%-60s %7.3f ms  (%s, %dx%d tokens, +%d bytes of LDS)" % (path, ms, which, dim_x, dim_y, shared))
    for i in range(1, len(results)):
        d = results[i] != results[0]
        print("output %d against the first: %s" % (i, "identical" if not d.any() else
              "DIFFERENT, %d of %d bytes" % (d.sum(), d.size)))


if __name__ == "__main__":
    main()
