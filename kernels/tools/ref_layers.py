#!/usr/bin/env python3
"""Matrix-level references for the layers ported so far, beyond ffwd_ref.py.

Each function states a layer's arithmetic in channel order, with the same
helpers and the same rounding points as ffwd_ref.py, and main() checks it
against a capture of the translated kernel. A native kernel is written from
these, and a mismatch between one of these and ptxsim is a misunderstanding
of the PTX, found before any HIP is written.

Usage: ref_layers.py <conv1x1_512|swin_2h|qkv_512|swin_4h|swin_8h> <capture dir>
"""
import glob
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ffwd_ref as R  # noqa: E402
import ptxsim  # noqa: E402


def conv1x1_512(x_codes, res_codes, weights):
    """tin3::Conv2d1x1Layer<512, 512> (proj_512, ffwd_proj_512).

    out[n] = e4m3( chain_k( f16(r[n] * s[n]), x, W ) ): the f16 accumulator
    starts from the residual times a per-channel scale, rounded once, and takes
    the 16 slices of 32 input channels in order. W is 512x512 in the same
    B-fragment layout as ffwd's W1; the 512 f16 scales follow it at +0x40000,
    in plain channel order.
    """
    w = R.decode_matrix(weights, 0, 512, 16)
    s = ptxsim.f16_bits_to_f64(np.frombuffer(weights, np.uint16, 512, 0x40000))
    acc = R.f16(R.value(res_codes) * s[None, :])
    return R.e4m3(R.chained(acc, R.value(x_codes), R.value(w)))


# The fused Swin 2H block's state before the attention. Verified against the
# emulator's own multiplies (ptxsim's mma_log, because the kernel loops and
# reuses registers, so reading them after the run reports a later stage):
# every branch product, the chain across branches, both passes, and the
# expansion that adds the scaled residual, all within one f16 ulp of the
# emulation on a captured launch.
#
# Layout of the weight blob, in bytes (the kernel reads it at +0x10):
#
#   0x0000 + 0x400k   branch k of pass 1, input channels 0..31  (32 hidden)
#   0x1000 + 0x400k   the same branch, channels 32..63
#   0x4000 + 0x400k   that branch's second matrix, 32 outputs
#   0x6000            the expansion of pass 1, 64 outputs from those 32
#   0x2000/0x3000/0x5000/0x6800   the same four for pass 2
#   0x7010            64 f16, the per-channel scale on the residual
#   0x70A0 + 0xC00h   q, k and v of head h, input channels 0..31 (96 outputs)
#   0x88A0 + 0xC00h   the same, channels 32..63
SILU_2H = (-4.0, 4.0, 0.89453125, 0.447265625, -0.055908203125)


def silu_2h(x):
    lo, hi, c0, c1, c2 = SILU_2H
    xc = np.clip(x, lo, hi)
    return R.f16(x * R.f16(xc * R.f16(c2 * np.abs(xc) + c1) + c0))


def swin_2h_branches(x, weights, base_a, base_b):
    """One pass: four branches, each 64 channels in and 32 hidden, chained."""
    acc = np.zeros((x.shape[0], 32))
    for k in range(4):
        wa = np.concatenate([R.value(R.decode_matrix(weights, base_a + 0x400 * k, 32, 1)),
                             R.value(R.decode_matrix(weights, base_a + 0x1000 + 0x400 * k, 32, 1))],
                            axis=1)
        hidden = R.chained(np.zeros((x.shape[0], 32)), x, wa)
        top = R.value(R.e4m3(silu_2h(hidden)))
        acc = R.chained(acc, top, R.value(R.decode_matrix(weights, base_b + 0x400 * k, 32, 1)))
    return acc


def swin_2h_state(x, weights):
    """The window's 64 tokens of 64 channels, as the attention receives them."""
    scale = ptxsim.f16_bits_to_f64(np.frombuffer(weights, np.uint16, 64, 0x7010))
    state = R.f16(x * scale[None, :])
    for base_a, base_b, expand in ((0x0000, 0x4000, 0x6000), (0x2000, 0x5000, 0x6800)):
        acc = swin_2h_branches(x, weights, base_a, base_b)
        state = R.chained(state, R.value(R.e4m3(acc)),
                          R.value(R.decode_matrix(weights, expand, 64, 1)))
    return state


def swin_2h_qkv(state_codes, weights, head):
    """q, k and v of one head: 64 tokens x 96, from the whole window's state."""
    w = np.concatenate([R.value(R.decode_matrix(weights, 0x70A0 + 0xC00 * head, 96, 1)),
                        R.value(R.decode_matrix(weights, 0x88A0 + 0xC00 * head, 96, 1))], axis=1)
    return R.chained(np.zeros((state_codes.shape[0], 96)), state_codes, w)


#   0xA0A0 + 0x2000h  the attention bias of head h, 64 queries x 64 keys, f16
#   0xE0A0 + 4h       the query scale of head h, one f32
#   0xE0B0            the projection, 64 outputs x 64 inputs: the two heads'
#                     attention concatenated, head 0 first
#   0xF0B0            64 f16, the per-channel scale on the output residual
#
# The attention repeats the 1H block's arithmetic with the same constants:
# cosine normalisation with a floor of 2^-14 * (1 + 1/64), the query scaled
# after the normalisation, the exponential done on the f16 bits, and the row
# normalised by its own sum.
NORM_FLOOR = 6.1988831e-05
QK_CLAMP = (1.03125, 1.5693359375)
EXP_MUL, EXP_ADD = 0.044921875, 1.30078125


def swin_2h_normalise(v, scale=1.0):
    n = np.maximum(R.f16(np.sum(R.f16(v * v), axis=1)), NORM_FLOOR)
    unit = R.f16(v * R.f16(1.0 / np.sqrt(n))[:, None])
    return R.f16(unit * scale) if scale != 1.0 else unit


def swin_2h_bias(weights, head):
    """The 64x64 bias of one head, gathered the way the kernel gathers it."""
    out = np.zeros((64, 64))
    for q in range(64):
        for kk in range(64):
            piece = (q >> 4) * 4 + (kk >> 4)
            lane = 4 * (q & 7) + ((kk & 15) & 7) // 2
            word = 2 * ((kk & 15) >> 3) + ((q & 15) >> 3)
            off = 0xA0A0 + 0x2000 * head + piece * 512 + lane * 16 + word * 4 + (kk & 1) * 2
            out[q, kk] = ptxsim.f16_bits_to_f64(np.frombuffer(weights, np.uint16, 1, off))[0]
    return out


def swin_2h_attention(qkv, weights, head):
    """One head's attention over the window's 64 tokens: 64 x 32, e4m3."""
    scale = float(np.float16(np.frombuffer(weights, np.float32, 1, 0xE0A0 + 4 * head)[0]))
    qhat = R.value(R.e4m3(swin_2h_normalise(qkv[:, 0:32], scale)))
    khat = R.value(R.e4m3(swin_2h_normalise(qkv[:, 32:64])))
    scores = R.chained(swin_2h_bias(weights, head), qhat, khat)
    clamped = np.clip(R.f16(scores * EXP_MUL + EXP_ADD), QK_CLAMP[0], QK_CLAMP[1])
    bits = np.asarray(clamped, np.float64).astype(np.float16).view(np.uint16).astype(np.uint32)
    p = ptxsim.f16_bits_to_f64((((bits << 5) + 0x8000) & 0xFFFF).astype(np.uint16))
    pn = R.f16(p * R.f16(1.0 / R.f16(np.sum(p, axis=1)))[:, None])
    return R.chained(np.zeros((64, 32)), R.value(R.e4m3(pn)), R.value(R.e4m3(qkv[:, 64:96])).T)


def swin_2h_layer(x, weights):
    """The whole fused Swin 2H block on one window: 64 tokens of 64 channels.

    Checked against a captured launch through ptxsim: 97.9% of the output
    values identical, the rest a rounding apart -- which is what a reference in
    plain f16 can be, since the order of an f16 sum and the matrix unit's own
    accumulation are what a native kernel has to reproduce, not this.
    """
    state = swin_2h_state(x, weights)
    codes = R.value(R.e4m3(state))                 # what crosses shared memory
    attn = [swin_2h_attention(swin_2h_qkv(codes, weights, h), weights, h) for h in (0, 1)]
    both = np.concatenate([R.value(R.e4m3(attn[0])), R.value(R.e4m3(attn[1]))], axis=1)
    scale = ptxsim.f16_bits_to_f64(np.frombuffer(weights, np.uint16, 64, 0xF0B0))
    proj = R.value(R.decode_matrix(weights, 0xE0B0, 64, 2))
    return R.value(R.e4m3(R.chained(R.f16(codes * scale[None, :]), both, proj)))


def check_swin_2h(capture, bx=37, by=23):
    """The reference on one window of a capture, against what the GPU wrote."""
    base = glob.glob(os.path.join(capture, "*_args.bin"))[0][:-9]
    pre = open("%s_pre_+00.bin" % base, "rb").read()
    weights = open("%s_pre_+10.bin" % base, "rb").read()
    post = open("%s_post_+08.bin" % base, "rb").read()
    cols = 160                                     # patches of 16 tokens, 1024 bytes each
    prow0, pcol0 = 2 * by - 1, 2 * bx - 1
    window, wrote = b"", b""
    for pr in range(2):
        for pc in range(2):
            q = (prow0 + pr) * cols + pcol0 + pc
            window += pre[q * 1024: q * 1024 + 1024]
            wrote += post[q * 1024: q * 1024 + 1024]
    x = R.value(R.decode_activations(window, 4, 64))
    ours = swin_2h_layer(x, weights)
    theirs = R.value(R.decode_activations(wrote, 4, 64))
    same = ours == theirs
    print("swin 2H, window (%d, %d): %.2f%% of %d values identical, worst %.4g"
          % (bx, by, 100 * same.mean(), same.size, np.abs(ours - theirs).max()))


# The 4H block, 128 channels and four heads on four warps. Same arithmetic as
# the 2H block, split differently, and the split is what the kernel has to
# follow: a warp takes a head through the branches (all 64 tokens of the
# window), the four results cross shared memory, and then a warp takes 32 of
# the 128 output channels and sums the four heads' contributions into them.
# Verified against the emulator's own multiplies: the q,k,v stage reads exactly
# the state this produces.
#
#   0x00000 + 0x4000h + 0x1000c + 0x400k   head h, branch k, input channels
#                                          32c..32c+31 -> 32 hidden
#   0x10000 + 0x1000h + 0x400k             that branch's second matrix
#   0x14000 + 0x1000h + 0x400g             head h's expansion into output
#                                          channels 32g..32g+31
#   0x18010                                128 f16, the scale on the residual
#   0x18120 + 0x3000c + 0xC00h             q, k and v of head h, input
#                                          channels 32c..32c+31 (96 outputs)
#   0x24120 + 0x2000h                      that head's attention bias, 64 x 64
#   0x2C120 + 4h                           that head's query scale, one f32
#   0x2C130 + 0x1000c + 0x400w             the projection: output channels
#                                          32w..32w+31 from input channels
#                                          32c..32c+31, the four heads'
#                                          attention concatenated
#   0x30130                                128 f16, the scale on the output
#                                          residual
def swin_4h_branches(x, weights, head):
    """One head's four chained branches: 64 tokens x 32 hidden."""
    acc = np.zeros((x.shape[0], 32))
    for k in range(4):
        wa = np.concatenate(
            [R.value(R.decode_matrix(weights, 0x0000 + 0x4000 * head + 0x1000 * c + 0x400 * k, 32, 1))
             for c in range(4)], axis=1)
        hidden = R.chained(np.zeros((x.shape[0], 32)), x, wa)
        top = R.value(R.e4m3(silu_2h(hidden)))
        acc = R.chained(acc, top, R.value(R.decode_matrix(weights, 0x10000 + 0x1000 * head + 0x400 * k, 32, 1)))
    return acc


def swin_4h_state(x, weights):
    """The window's 64 tokens of 128 channels, as the attention receives them."""
    scale = ptxsim.f16_bits_to_f64(np.frombuffer(weights, np.uint16, 128, 0x18010))
    state = R.f16(x * scale[None, :])
    for head in range(4):
        mid = R.value(R.e4m3(swin_4h_branches(x, weights, head)))
        wexp = np.concatenate([R.value(R.decode_matrix(weights, 0x14000 + 0x1000 * head + 0x400 * g, 32, 1))
                               for g in range(4)], axis=0)
        state = R.chained(state, mid, wexp)
    return state


def swin_4h_qkv(state_codes, weights, head):
    """q, k and v of one head: 64 tokens x 96, from the whole window's state."""
    w = np.concatenate([R.value(R.decode_matrix(weights, 0x18120 + 0x3000 * c + 0xC00 * head, 96, 1))
                        for c in range(4)], axis=1)
    return R.chained(np.zeros((state_codes.shape[0], 96)), state_codes, w)


def swin_4h_attention(qkv, weights, head):
    """One head's attention, with the 2H block's arithmetic and its own tables."""
    scale = float(np.float16(np.frombuffer(weights, np.float32, 1, 0x2C120 + 4 * head)[0]))
    qhat = R.value(R.e4m3(swin_2h_normalise(qkv[:, 0:32], scale)))
    khat = R.value(R.e4m3(swin_2h_normalise(qkv[:, 32:64])))
    bias = np.zeros((64, 64))
    for q in range(64):
        for kk in range(64):
            piece = (q >> 4) * 4 + (kk >> 4)
            lane = 4 * (q & 7) + ((kk & 15) & 7) // 2
            word = 2 * ((kk & 15) >> 3) + ((q & 15) >> 3)
            off = 0x24120 + 0x2000 * head + piece * 512 + lane * 16 + word * 4 + (kk & 1) * 2
            bias[q, kk] = ptxsim.f16_bits_to_f64(np.frombuffer(weights, np.uint16, 1, off))[0]
    scores = R.chained(bias, qhat, khat)
    clamped = np.clip(R.f16(scores * EXP_MUL + EXP_ADD), QK_CLAMP[0], QK_CLAMP[1])
    bits = np.asarray(clamped, np.float64).astype(np.float16).view(np.uint16).astype(np.uint32)
    p = ptxsim.f16_bits_to_f64((((bits << 5) + 0x8000) & 0xFFFF).astype(np.uint16))
    pn = R.f16(p * R.f16(1.0 / R.f16(np.sum(p, axis=1)))[:, None])
    return R.chained(np.zeros((64, 32)), R.value(R.e4m3(pn)), R.value(R.e4m3(qkv[:, 64:96])).T)


# cc_split_swin_16h_qkv_512_chained_fp8: the whole attention of a 16-head Swin
# stage, one 8x8 window per block. Grid 10x6x4, block 32x4: the z slice and the
# warp pick the head (head = 4z + warp), so a warp carries one head from the
# projection through to the output, and the window's 64 tokens are shared by
# all four warps. Parameter of 56 bytes: the activations, the output, the
# weights, then (dim_y, dim_x) and the two flag buffers.
#
# Activations and output share the fused blocks' layout: patches of 4x4 tokens,
# each token's channels in 512-byte chunks of 32 in mma fragment order, so a
# patch is 16 * channels bytes and the window is the 2x2 of them that
# swin_window gathers. The output of head h lands in chunk h.
#
# Layout of the weight blob:
#
#   0x00000 + 0xC000c + 0xC00h   q, k and v of head h, input channels
#                                32c..32c+31 (96 outputs, q then k then v)
#   0xC0000 + 0x2000h            head h's attention bias, 64 queries x 64 keys
#   0xE0000 + 4h                 head h's query scale, one f32
#
# The arithmetic is the fused block's, on 512 channels instead of 64: cosine
# normalisation of q and k, the query scaled after it, the exponential on the
# f16 bits, the row divided by its own sum, and v taken in the same fragment
# order -- which is why the reference may use plain channel order, a
# permutation of both sides of q.k leaving the product alone.
def attention_bias(weights, base, head):
    """One head's 64x64 bias, gathered the way the kernels gather it."""
    out = np.zeros((64, 64))
    for q in range(64):
        for kk in range(64):
            piece = (q >> 4) * 4 + (kk >> 4)
            lane = 4 * (q & 7) + ((kk & 15) & 7) // 2
            word = 2 * ((kk & 15) >> 3) + ((q & 15) >> 3)
            off = base + 0x2000 * head + piece * 512 + lane * 16 + word * 4 + (kk & 1) * 2
            out[q, kk] = ptxsim.f16_bits_to_f64(np.frombuffer(weights, np.uint16, 1, off))[0]
    return out


def qkv_512_qkv(x_codes, weights):
    """q, k and v of all sixteen heads: 64 tokens x 1536, head h at 96h."""
    w = np.concatenate([R.value(R.decode_matrix(weights, 0xC000 * c, 1536, 1))
                        for c in range(16)], axis=1)
    return R.chained(np.zeros((x_codes.shape[0], 1536)), R.value(x_codes), w)


def qkv_512_attention(qkv, weights, head):
    """One head's attention over the window: 64 tokens x 32."""
    scale = float(np.float16(np.frombuffer(weights, np.float32, 1, 0xE0000 + 4 * head)[0]))
    qhat = R.value(R.e4m3(swin_2h_normalise(qkv[:, 0:32], scale)))
    khat = R.value(R.e4m3(swin_2h_normalise(qkv[:, 32:64])))
    scores = R.chained(attention_bias(weights, 0xC0000, head), qhat, khat)
    clamped = np.clip(R.f16(scores * EXP_MUL + EXP_ADD), QK_CLAMP[0], QK_CLAMP[1])
    bits = np.asarray(clamped, np.float64).astype(np.float16).view(np.uint16).astype(np.uint32)
    p = ptxsim.f16_bits_to_f64((((bits << 5) + 0x8000) & 0xFFFF).astype(np.uint16))
    pn = R.f16(p * R.f16(1.0 / R.f16(np.sum(p, axis=1)))[:, None])
    return R.chained(np.zeros((64, 32)), R.value(R.e4m3(pn)), R.value(R.e4m3(qkv[:, 64:96])).T)


def qkv_512_layer(x_codes, weights):
    """The whole layer on one window: 64 tokens x 512 e4m3 codes out."""
    qkv = qkv_512_qkv(x_codes, weights)
    out = np.zeros((64, 512))
    for h in range(16):
        out[:, 32 * h: 32 * h + 32] = qkv_512_attention(qkv[:, 96 * h: 96 * h + 96], weights, h)
    return R.e4m3(out)


def check_qkv_512(capture, bx=3, by=2):
    base = glob.glob(os.path.join(capture, "*_args.bin"))[0][:-9]
    params = open(base + "_args.bin", "rb").read()
    dim_y, dim_x = struct.unpack_from("<2i", params, 0x18)
    off_x, off_y = struct.unpack_from("<2i", params, 0x20)
    pre = {off: open("%s_pre_+%02X.bin" % (base, off), "rb").read() for off in (0, 0x10)}
    post = open(base + "_post_+08.bin", "rb").read()
    pc, pr = dim_x // 4, dim_y // 4
    x = swin_window(pre[0], bx, by, off_x, off_y, pc, pr, 512)
    ours = qkv_512_layer(x, pre[0x10])
    gpu = swin_window(post, bx, by, off_x, off_y, pc, pr, 512)
    rows = np.concatenate([np.arange(16 * p, 16 * p + 16)
                           for p in swin_patches(bx, by, off_x, off_y, pc, pr)])
    print("qkv_512 reference vs translated kernel, block (%d,%d): %.4f%% codes identical"
          % (bx, by, 100 * (ours[rows] == gpu[rows]).mean()))


def swin_4h_layer(x, weights):
    """The whole fused Swin 4H block on one window: 64 tokens of 128 channels."""
    state = swin_4h_state(x, weights)
    codes = R.value(R.e4m3(state))                 # what crosses shared memory
    attn = [R.value(R.e4m3(swin_4h_attention(swin_4h_qkv(codes, weights, h), weights, h)))
            for h in range(4)]
    both = np.concatenate(attn, axis=1)
    proj = np.zeros((128, 128))
    for w in range(4):
        for c in range(4):
            proj[32 * w: 32 * w + 32, 32 * c: 32 * c + 32] =                 R.value(R.decode_matrix(weights, 0x2C130 + 0x1000 * c + 0x400 * w, 32, 1))
    scale = ptxsim.f16_bits_to_f64(np.frombuffer(weights, np.uint16, 128, 0x30130))
    return R.e4m3(R.chained(R.f16(codes * scale[None, :]), both, proj))


def swin_patches(bx, by, off_x, off_y, patch_cols, patch_rows):
    """Which of the window's four patches are inside the image."""
    prow0, pcol0 = 2 * by + (off_y >> 2), 2 * bx + (off_x >> 2)
    return [p for p in range(4)
            if 0 <= prow0 + (p >> 1) < patch_rows and 0 <= pcol0 + (p & 1) < patch_cols]


def swin_window(buf, bx, by, off_x, off_y, patch_cols, patch_rows, channels):
    """One block's 64 tokens, the patches outside the image read as zeros."""
    stride = 16 * channels
    prow0, pcol0 = 2 * by + (off_y >> 2), 2 * bx + (off_x >> 2)
    parts = []
    for p in range(4):
        pr, pc = prow0 + (p >> 1), pcol0 + (p & 1)
        if 0 <= pr < patch_rows and 0 <= pc < patch_cols:
            o = (pr * patch_cols + pc) * stride
            parts.append(R.decode_activations(buf[o:o + stride], 1, channels))
        else:
            parts.append(np.zeros((16, channels), np.uint8))
    return np.concatenate(parts, axis=0)


def check_swin_4h(capture, bx=20, by=12):
    base = glob.glob(os.path.join(capture, "*_args.bin"))[0][:-9]
    params = open(base + "_args.bin", "rb").read()
    dim_y, dim_x = struct.unpack_from("<2i", params, 0x20)
    off_x, off_y = struct.unpack_from("<2i", params, 0x28)
    pre_in = open(base + "_pre_+00.bin", "rb").read()
    weights = open(base + "_pre_+10.bin", "rb").read()
    post = open(base + "_post_+08.bin", "rb").read()
    pc, pr = dim_x // 4, dim_y // 4
    x = swin_window(pre_in, bx, by, off_x, off_y, pc, pr, 128)
    ours = swin_4h_layer(R.value(x), weights)
    got = swin_window(post, bx, by, off_x, off_y, pc, pr, 128)
    # Only the patches inside the image are written, so only those are compared.
    rows = np.concatenate([np.arange(16 * p, 16 * p + 16)
                           for p in swin_patches(bx, by, off_x, off_y, pc, pr)])
    print("swin_4h reference vs translated kernel, block (%d,%d): %.4f%% codes identical"
          % (bx, by, 100 * (ours[rows] == got[rows]).mean()))


# The fused Swin block in its three sizes at once. 2H, 4H and 8H are the same
# layer -- what the 2H's decoding called a "pass" is a head -- and their weight
# blobs follow one formula, which the three known maps confirm to the byte:
#
#   chunks = channels / 32                     the 32-channel slices of a token
#   branch A  0x0000  + 0x1000*chunks*h + 0x1000c + 0x400k     32 hidden
#   branch B  A_end   + 0x1000h + 0x400k                       32 outputs
#   expand    B_end   + 0x400*chunks*h + 0x400g                32 of `channels`
#   scale1    exp_end + 0x10                                   `channels` f16
#   qkv       s1_end  + 0x10 + 0xC00*heads*c + 0xC00h          96 outputs
#   bias      qkv_end + 0x2000h                                64 x 64 f16
#   qk scale  bias_end + 4h                                    one f32 a head
#   proj      scales rounded up to 16 + 0x400*heads*c + 0x400g 32 outputs
#   scale2    proj_end                                         `channels` f16
#
# so a fourth size, if one ever appears, is decoded by filling in two numbers.
class FusedSwin:
    def __init__(self, channels, heads):
        self.channels, self.heads = channels, heads
        c = self.chunks = channels // 32
        self.BRANCH_A = 0
        self.BRANCH_B = 0x1000 * c * heads
        self.EXPAND = self.BRANCH_B + 0x1000 * heads
        # One head needs no expansion: its branches are already the whole
        # channel count, so they chain straight into the scaled residual and
        # the region is not there at all.
        self.SCALE1 = (self.EXPAND if heads > 1 else self.BRANCH_B + 0x1000 * heads)             + (0x400 * c * heads if heads > 1 else 0) + 0x10
        self.QKV = self.SCALE1 + 2 * channels + 0x10
        self.BIAS = self.QKV + 0xC00 * heads * c
        self.QK_SCALE = self.BIAS + 0x2000 * heads
        self.PROJ = self.QK_SCALE + (4 * heads + 15) // 16 * 16
        self.SCALE2 = self.PROJ + 0x400 * heads * c
        # The downsampling siblings put one more matrix after the scales: 2C
        # outputs from the 2x2 mean of C channels.
        self.DS = self.SCALE2 + 2 * channels

    def branches(self, x, weights, head):
        """One head's four chained branches: 64 tokens x 32 hidden."""
        acc = np.zeros((x.shape[0], 32))
        for k in range(4):
            wa = np.concatenate(
                [R.value(R.decode_matrix(weights, self.BRANCH_A + 0x1000 * self.chunks * head
                                         + 0x1000 * c + 0x400 * k, 32, 1))
                 for c in range(self.chunks)], axis=1)
            top = R.value(R.e4m3(silu_2h(R.chained(np.zeros((x.shape[0], 32)), x, wa))))
            acc = R.chained(acc, top, R.value(R.decode_matrix(
                weights, self.BRANCH_B + 0x1000 * head + 0x400 * k, 32, 1)))
        return acc

    def state(self, x, weights):
        """The window's 64 tokens, as the attention receives them."""
        scale = ptxsim.f16_bits_to_f64(np.frombuffer(weights, np.uint16, self.channels, self.SCALE1))
        out = R.f16(x * scale[None, :])
        if self.heads == 1:
            for k in range(4):
                wa = np.concatenate(
                    [R.value(R.decode_matrix(weights, self.BRANCH_A + 0x1000 * c + 0x400 * k, 32, 1))
                     for c in range(self.chunks)], axis=1)
                top = R.value(R.e4m3(silu_2h(R.chained(np.zeros((x.shape[0], 32)), x, wa))))
                out = R.chained(out, top, R.value(R.decode_matrix(
                    weights, self.BRANCH_B + 0x400 * k, 32, 1)))
            return out
        for head in range(self.heads):
            mid = R.value(R.e4m3(self.branches(x, weights, head)))
            wexp = np.concatenate([R.value(R.decode_matrix(
                weights, self.EXPAND + 0x400 * self.chunks * head + 0x400 * g, 32, 1))
                for g in range(self.chunks)], axis=0)
            out = R.chained(out, mid, wexp)
        return out

    def qkv(self, state_codes, weights, head):
        """q, k and v of one head: 64 tokens x 96, from the whole window's state."""
        w = np.concatenate([R.value(R.decode_matrix(
            weights, self.QKV + 0xC00 * self.heads * c + 0xC00 * head, 96, 1))
            for c in range(self.chunks)], axis=1)
        return R.chained(np.zeros((state_codes.shape[0], 96)), state_codes, w)

    def attention(self, qkv, weights, head):
        """One head's attention over the window's 64 tokens: 64 x 32."""
        scale = float(np.float16(np.frombuffer(weights, np.float32, 1, self.QK_SCALE + 4 * head)[0]))
        qhat = R.value(R.e4m3(swin_2h_normalise(qkv[:, 0:32], scale)))
        khat = R.value(R.e4m3(swin_2h_normalise(qkv[:, 32:64])))
        scores = R.chained(attention_bias(weights, self.BIAS, head), qhat, khat)
        clamped = np.clip(R.f16(scores * EXP_MUL + EXP_ADD), QK_CLAMP[0], QK_CLAMP[1])
        bits = np.asarray(clamped, np.float64).astype(np.float16).view(np.uint16).astype(np.uint32)
        p = ptxsim.f16_bits_to_f64((((bits << 5) + 0x8000) & 0xFFFF).astype(np.uint16))
        pn = R.f16(p * R.f16(1.0 / R.f16(np.sum(p, axis=1)))[:, None])
        return R.chained(np.zeros((64, 32)), R.value(R.e4m3(pn)), R.value(R.e4m3(qkv[:, 64:96])).T)

    def layer(self, x, weights, raw=False):
        """The whole block on one window: 64 tokens of `channels` e4m3 codes.

        raw=True returns the f16 result before it is narrowed, which is what a
        downsampling sibling averages."""
        codes = R.value(R.e4m3(self.state(x, weights)))
        both = np.concatenate([R.value(R.e4m3(self.attention(self.qkv(codes, weights, h), weights, h)))
                               for h in range(self.heads)], axis=1)
        proj = np.zeros((self.channels, self.channels))
        for g in range(self.chunks):
            for c in range(self.chunks):
                proj[32 * g: 32 * g + 32, 32 * c: 32 * c + 32] = R.value(R.decode_matrix(
                    weights, self.PROJ + 0x400 * self.heads * c + 0x400 * g, 32, 1))
        scale = ptxsim.f16_bits_to_f64(np.frombuffer(weights, np.uint16, self.channels, self.SCALE2))
        out = R.chained(R.f16(codes * scale[None, :]), both, proj)
        return out if raw else R.e4m3(out)

    def downsample(self, out_f16, weights):
        """The ds sibling's second output: the window's 8x8 result averaged 2x2
        and put through one matrix, 16 tokens of 2 * channels."""
        win = out_f16.reshape(4, 4, 2, 2, self.channels)      # patch, token rows
        spatial = np.zeros((8, 8, self.channels))
        for p in range(4):
            for t in range(16):
                spatial[4 * (p >> 1) + t // 4, 4 * (p & 1) + t % 4] = out_f16[16 * p + t]
        merged = spatial.reshape(4, 2, 4, 2, self.channels).mean(axis=(1, 3))
        w = R.value(R.decode_matrix(weights, self.DS, 2 * self.channels, self.chunks))
        return R.e4m3(R.chained(np.zeros((16, 2 * self.channels)),
                                R.value(R.e4m3(merged.reshape(16, self.channels))), w))


def check_fused_swin(capture, channels, heads, bx, by):
    base = glob.glob(os.path.join(capture, "*_args.bin"))[0][:-9]
    params = open(base + "_args.bin", "rb").read()
    # The 88-byte parameter has a slot before the dimensions that the 96-byte
    # one does not, so the family's two ABIs put them eight bytes apart.
    at = 0x20 if len(params) == 88 else 0x18
    dim_y, dim_x = struct.unpack_from("<2i", params, at)
    off_x, off_y = struct.unpack_from("<2i", params, at + 8)
    weights = open(base + "_pre_+10.bin", "rb").read()
    pc, pr = dim_x // 4, dim_y // 4
    x = swin_window(open(base + "_pre_+00.bin", "rb").read(), bx, by, off_x, off_y, pc, pr, channels)
    got = swin_window(open(base + "_post_+08.bin", "rb").read(), bx, by, off_x, off_y, pc, pr, channels)
    ours = FusedSwin(channels, heads).layer(R.value(x), weights)
    rows = np.concatenate([np.arange(16 * p, 16 * p + 16)
                           for p in swin_patches(bx, by, off_x, off_y, pc, pr)])
    print("fused swin %dH reference vs translated kernel, block (%d,%d): %.4f%% codes identical"
          % (heads, bx, by, 100 * (ours[rows] == got[rows]).mean()))


def check_conv1x1_512(capture):
    base = glob.glob(os.path.join(capture, "*_args.bin"))[0][:-9]
    pre = {off: open("%s_pre_+%02X.bin" % (base, off), "rb").read() for off in (0, 8, 0x18)}
    patches = 240
    x = R.decode_activations(pre[0], patches)
    res = R.decode_activations(pre[8], patches)
    out = np.frombuffer(R.encode_activations(conv1x1_512(x, res, pre[0x18])), np.uint8)
    gpu = np.frombuffer(open(base + "_post_+10.bin", "rb").read(), np.uint8, len(out))
    print("conv1x1_512 reference vs translated kernel: %.4f%% bytes identical" % (100 * (out == gpu).mean()))


def main():
    checks = {"conv1x1_512": check_conv1x1_512, "swin_2h": check_swin_2h,
              "qkv_512": check_qkv_512, "swin_4h": check_swin_4h,
              "swin_8h": lambda c: check_fused_swin(c, 256, 8, 10, 6)}
    if len(sys.argv) < 3 or sys.argv[1] not in checks:
        print(__doc__)
        return 2
    checks[sys.argv[1]](sys.argv[2])
    return 0


if __name__ == "__main__":
    sys.exit(main())
