#!/usr/bin/env python3
"""The ViT stage's q, k, v projection, cc_vit_1d_qkv_chained_fp8, in numpy.

Used to find the layer's arithmetic and its channel map against a captured
launch before writing it for the hardware.

Usage: vit_qkv_ref.py <capture dir>
"""
import glob
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ffwd_ref as R  # noqa: E402

TOKENS, K, N = 960, 1024, 3072


def load(cap):
    base = glob.glob(os.path.join(cap, "*_args.bin"))[0][:-9]
    pre = {int(f[-6:-4], 16): open(f, "rb").read() for f in glob.glob(base + "_pre_+*.bin")}
    post = {int(f[-6:-4], 16): open(f, "rb").read() for f in glob.glob(base + "_post_+*.bin")}
    return open(base + "_args.bin", "rb").read(), pre, post


def partials(pre):
    x = R.value(R.decode_activations(pre[0x00], TOKENS // 16, K))
    w = R.value(R.decode_matrix(pre[0x20], 128, N, K // 32))
    zero = np.zeros((TOKENS, N))
    p0 = R.chained(zero, x[:, :512], w[:, :512])
    p1 = R.chained(zero, x[:, 512:], w[:, 512:])
    return p0, p1


def main():
    params, pre, post = load(sys.argv[1])
    p0, p1 = partials(pre)
    s = R.f16(p0 + p1)
    codes = R.e4m3(s)
    for slot, name in ((0x08, "q"), (0x10, "k"), (0x18, "v")):
        out = R.decode_activations(post[slot], TOKENS // 16, 1024)
        found = []
        for c in range(32):
            blk = out[:, 32 * c:32 * c + 32]
            best, where = 0.0, -1
            for b in range(N // 32):
                m = (codes[:, 32 * b:32 * b + 32] == blk).mean()
                if m > best:
                    best, where = m, b
            found.append((where, round(best, 3)))
        print(name, found)


if __name__ == "__main__":
    main()
