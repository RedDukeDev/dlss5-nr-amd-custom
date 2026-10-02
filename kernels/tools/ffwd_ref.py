#!/usr/bin/env python3
"""cc_split_swin_16h_ffwd_512_chained_fp8 as matrices, not as lanes.

ptxsim runs the kernel instruction by instruction; this is the same computation
written the way a native kernel would be designed from: what each token's 512
channels go through, where each weight byte sits, and where every rounding
happens. It is derived from the PTX and checked against ptxsim bit for bit, so
a native kernel can be checked against it -- per stage, on any input -- without
running the translation at all.

The layer, per token (tokens are independent):

    h   = e4m3(X W1^T)                       512 -> 512, all channels
    for each group g of 64 hidden channels (8 groups):
        for i in 0..7:                        32 intermediate channels at a time
            p_i = W2[g] applied to h[g]       64 -> 32
            s_i = e4m3(silu(p_i))
            o_g += W3[g][:, i] s_i            32 -> 64, accumulated
    out = e4m3(o)                            channel g*64+j comes from group g

Every m16n8k32 multiply rounds its result to f16, so a sum over K is a chain
of f16 roundings, one per 32-wide slice, in a fixed order. That order is part
of the definition here.

Memory layouts (all e4m3, one byte per value):

  activations  patches of 4x4 tokens, 8 KB each, row-major over the (12, 20)
               patch grid; inside a patch, 16 chunks of 32 channels, 512 bytes
               each; inside a chunk, the A fragment of m16n8k32 lane by lane:
               byte L*16 + r*4 + b holds token g + 8(r&1), fragment column
               k = 16(r>>1) + 4t + b, with g = L>>2, t = L&3.
  weights      B fragments: for a matrix of n_rows outputs, chunk kc of 32
               inputs occupies n_rows*32 bytes; inside it, 512-byte pieces of
               16 outputs; byte L*16 + w*4 + b of piece nb holds output
               16nb + 8(w>>1) + g and fragment column k = 16(w&1) + 4t + b.
               W1 at +0 (512 x 512), W2 at +0x40000 (8 groups of 256 x 64),
               W3 at +0x60000 (8 groups of 64 x 256), 16 KB per group.

The fragment column k is not the channel. An m16n8k32 result comes out as C
fragments, and the kernel feeds it to the next multiply as an A fragment by
packing two neighbouring n-tiles into each A register, so fragment column k
carries channel PERM32[k] = 16h + 8(b>>1) + 2t + (b&1) for k = 16h + 4t + b.
The same packing writes the output and, upstream, wrote the input, and the
weights were stored to match, so in channel order the whole layer is plain
matrix algebra.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ptxsim  # noqa: E402

PERM32 = np.array([16 * (k // 16) + 8 * ((k % 4) >> 1) + 2 * ((k % 16) // 4) + (k % 2)
                   for k in range(32)])

# Inside one 512-byte chunk: which token and which channel each byte holds.
_pos = np.arange(512)
_L, _r, _b = _pos // 16, (_pos // 4) % 4, _pos % 4
_g, _t = _L >> 2, _L & 3
CHUNK_TOKEN = _g + 8 * (_r & 1)
CHUNK_CHANNEL = PERM32[16 * (_r >> 1) + 4 * _t + _b]
# Inside one 512-byte weight piece: which output and which input channel.
_w = _r
PIECE_ROW = 8 * (_w >> 1) + _g
PIECE_COL = PERM32[16 * (_w & 1) + 4 * _t + _b]

PATCH_BYTES = 16 * 512
W2_BASE, W3_BASE, GROUP_BYTES = 0x40000, 0x60000, 0x4000


def decode_activations(buf, patches, channels=512):
    """bytes -> (patches*16, channels) e4m3 codes, token-major, channels in order.

    A patch is 16 tokens; its channels are stored in 512-byte chunks of 32, each
    in mma fragment order. Stages differ only in how many chunks a patch has
    (512 channels in the Swin stages, 1024 or 4096 in the ViT one, 32 in the
    fused Swin blocks)."""
    chunks = channels // 32
    raw = np.frombuffer(buf, np.uint8, patches * 16 * channels).reshape(patches, chunks, 512)
    codes = np.zeros((patches, 16, channels), np.uint8)
    for c in range(chunks):
        codes[:, CHUNK_TOKEN, 32 * c + CHUNK_CHANNEL] = raw[:, c, :]
    return codes.reshape(patches * 16, channels)


def encode_activations(codes, channels=None):
    """(patches*16, channels) e4m3 codes -> bytes, the inverse of decode_activations."""
    channels = channels or codes.shape[1]
    chunks = channels // 32
    patches = codes.shape[0] // 16
    c3 = codes.reshape(patches, 16, channels)
    raw = np.zeros((patches, chunks, 512), np.uint8)
    for c in range(chunks):
        raw[:, c, :] = c3[:, CHUNK_TOKEN, 32 * c + CHUNK_CHANNEL]
    return raw.tobytes()


def decode_matrix(buf, offset, n_rows, k_chunks):
    """B-fragment bytes -> (n_rows, 32*k_chunks) e4m3 codes, [output, input]."""
    raw = np.frombuffer(buf, np.uint8, n_rows * 32 * k_chunks, offset)
    raw = raw.reshape(k_chunks, n_rows // 16, 512)
    m = np.zeros((n_rows, 32 * k_chunks), np.uint8)
    for kc in range(k_chunks):
        for nb in range(n_rows // 16):
            m[16 * nb + PIECE_ROW, 32 * kc + PIECE_COL] = raw[kc, nb]
    return m


def decode_weights(buf):
    w1 = decode_matrix(buf, 0, 512, 16)
    w2 = [decode_matrix(buf, W2_BASE + g * GROUP_BYTES, 256, 2) for g in range(8)]
    w3 = [decode_matrix(buf, W3_BASE + g * GROUP_BYTES, 64, 8) for g in range(8)]
    return w1, w2, w3


def value(codes):
    return ptxsim.E4M3[codes]


def f16(x):
    return np.asarray(x, np.float64).astype(np.float16).astype(np.float64)


def e4m3(x):
    return ptxsim.f16_to_e4m3(x).astype(np.uint8)


def chained(acc, a, b):
    """acc + a @ b.T one 32-wide K slice at a time, rounding to f16 after each:
    what a chain of m16n8k32 with an f16 accumulator computes, if every single
    multiply sums exactly (the PTX semantics)."""
    for k in range(0, a.shape[1], 32):
        acc = f16(acc + a[:, k:k + 32] @ b[:, k:k + 32].T)
    return acc


def silu(x):
    """MpCubicSiluActivation exactly as the kernel evaluates it, in f16."""
    xc = np.clip(x, -4.0, 4.0)
    inner = f16(-0.055908203125 * np.abs(xc) + 0.447265625)
    return f16(x * f16(xc * inner + 0.89453125))


def forward(x_codes, weights, stages=False):
    """(tokens, 512) e4m3 codes -> (tokens, 512) e4m3 codes."""
    w1, w2, w3 = weights
    x = value(x_codes)
    h = e4m3(chained(np.zeros((x.shape[0], 512)), x, value(w1)))
    hv = value(h)
    out = np.zeros((x.shape[0], 512))
    for g in range(8):
        hg = hv[:, 64 * g:64 * g + 64]
        w2g, w3g = value(w2[g]), value(w3[g])
        o = np.zeros((x.shape[0], 64))
        for i in range(8):
            p = chained(np.zeros((x.shape[0], 32)), hg, w2g[32 * i:32 * i + 32])
            s = value(e4m3(silu(p)))
            o = f16(o + s @ w3g[:, 32 * i:32 * i + 32].T)
        out[:, 64 * g:64 * g + 64] = o
    result = e4m3(out)
    return (result, h) if stages else result


def main():
    import glob
    here = os.path.dirname(os.path.abspath(__file__))
    capture = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "capture_l8")
    base = glob.glob(os.path.join(capture, "*_args.bin"))[0][:-9]
    patches = 240
    x = decode_activations(open(base + "_pre_+00.bin", "rb").read(), patches)
    weights = decode_weights(open(base + "_pre_+10.bin", "rb").read())
    out = forward(x, weights)
    mine = np.frombuffer(encode_activations(out), np.uint8)
    gpu = np.frombuffer(open(base + "_post_+08.bin", "rb").read(), np.uint8, len(mine))
    print("reference vs GPU capture: %.3f%% bytes identical" % (100 * (mine == gpu).mean()))
    emu = os.path.join(here, "emu_out.npy")
    if os.path.exists(emu):
        e = np.load(emu).view(np.uint8).ravel()[:len(mine)]
        print("reference vs ptxsim:      %.3f%% bytes identical" % (100 * (mine == e).mean()))


if __name__ == "__main__":
    main()
