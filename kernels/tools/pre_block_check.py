#!/usr/bin/env python3
"""The pre block, translated and native, on synthetic textures.

The network's harness binds the colour texture alone, so half of the pre
block's input stage -- history reprojection, the depth search, the bicubic
history, the fifth texture -- never runs there. This builds every texture the
kernel can read, with ZLUDA's own driver API, and runs the translated PTX (through
ZLUDA) and the native kernel (through HIP, in the same process, on the same
texture objects) on the same parameter, once per branch combination; then
compares both outputs byte for byte.

The weights are a captured launch's (capture_pre16).

Usage: pre_block_check.py [variant ...] [--hsaco=PATH]
"""
import glob
import io
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hip_run import Hip  # noqa: E402
from zluda_run import Zluda  # noqa: E402
import workspace  # noqa: E402

ENTRY = "cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8"
H, W = 60, 100
HSACO = os.path.join(os.path.dirname(HERE), "swin_1h.hsaco")   # --hsaco=PATH: another build of it
for _arg in sys.argv[1:]:
    if _arg.startswith("--hsaco="):
        HSACO = _arg[8:]
    if _arg.startswith("--size="):
        H, W = (int(v) for v in _arg[7:].split("x"))
TOK_Y, TOK_X = (H + 31) // 32 * 32, (W + 31) // 32 * 32   # rounded up to 32, as the network's are
HALF_Y, HALF_X = TOK_Y // 2, TOK_X // 2
SEED = 3
for _arg in sys.argv[1:]:
    if _arg.startswith("--seed="):
        SEED = int(_arg[7:])

VARIANTS = {
    "all":        dict(),
    "far":        dict(depth_far=1),
    "no_depth":   dict(depth=False),
    "no_history": dict(history=False),
    "no_extra":   dict(extra=False),
    "no_extra_flags_off": dict(extra=False, flag_mode=0),
    "flags_negative": dict(extra=False, flag_x=-0.5, flag_y=-0.25),
    "half_padded": dict(half_pad=8),
}


def build_params(tex, out, weights, half, depth_far=0, flag_mode=1, flag_x=0.3, flag_y=-0.2, half_pad=0):
    def trio(off):
        return [off[0], off[1], float(W), float(H), 1.0 / W, 1.0 / H]
    p = struct.pack("<5Q", tex["color"], tex["history"], tex["motion"], tex["depth"], tex["extra"])
    p += struct.pack("<6f", *trio((0.25, -0.125)))          # history
    p += struct.pack("<6f", *trio((0.0, 0.0)))              # motion
    p += struct.pack("<6f", *trio((0.0, 0.0)))              # depth
    p += struct.pack("<6f", *trio((0.5, 0.0)))              # extra
    p += struct.pack("<6f", *trio((0.0, 0.0)))              # colour
    p += struct.pack("<2f", 1.0, 1.0)                       # motion vector scale
    p += struct.pack("<if", depth_far, 1.0)
    p += struct.pack("<2f", 0.5, 0.25)
    p += struct.pack("<2f", flag_x, flag_y)
    p += struct.pack("<if", flag_mode, 0.0625)
    p += struct.pack("<2i", 12345 + SEED, 0)
    p += struct.pack("<2i", H, W)
    p += struct.pack("<3Q", out, weights, 0)
    p += struct.pack("<2i", TOK_Y, TOK_X)
    p += struct.pack("<Q", half)
    p += struct.pack("<2i", HALF_Y + half_pad, HALF_X + half_pad)
    assert len(p) == 264, len(p)
    return p


def main():
    wanted = [a for a in sys.argv[1:] if not a.startswith('--')] or list(VARIANTS)
    rng = np.random.default_rng(SEED)
    z = Zluda()
    h = Hip()
    textures = {
        "color": z.texture(z.array2d(rng.uniform(0, 1, (H, W, 4)).astype(np.float16))),
        "history": z.texture(z.array2d(rng.uniform(0, 1, (H, W, 4)).astype(np.float16))),
        "motion": z.texture(z.array2d(rng.uniform(-0.03, 0.03, (H, W, 4)).astype(np.float16))),
        "depth": z.texture(z.array2d(rng.uniform(0, 1, (H, W)).astype(np.float32))),
        "extra": z.texture(z.array2d(rng.uniform(0, 2, (H, W, 4)).astype(np.float16))),
    }
    base = glob.glob(os.path.join(workspace.captures(), "capture_pre16", "*_args.bin"))[0][:-9]
    blob = np.fromfile(base + "_pre_+E0.bin", np.uint8)[:1 << 16]
    weights = z.upload(blob)

    text = ".version 9.4\n.target sm_120\n.address_size 64\n" + io.open(
        os.path.join(workspace.captures(), "pre_block_32.ptx"), encoding="utf-8").read()
    translated = z.module(text, ENTRY)
    native = h.module(HSACO, ENTRY + "_native")
    out_bytes = TOK_Y * TOK_X * 32
    half_bytes = 4 * (HALF_Y + 8) * (HALF_X + 8) * 16
    grid, block = (TOK_X // 8, TOK_Y // 8, 1), (32, 1, 1)

    for name in wanted:
        cfg = dict(VARIANTS[name])
        tex = dict(textures)
        for t in ("history", "depth", "extra"):
            if cfg.pop(t, True) is False:
                tex[t] = 0
        results = []
        for run in ("translated", "native"):
            out = z.upload(np.full(out_bytes, 0x77, np.uint8))
            half = z.upload(np.full(half_bytes, 0x77, np.uint8))
            params = build_params(tex, out, weights, half, **cfg)
            if run == "translated":
                z.launch(translated, grid, block, params)
            else:
                h.launch(native, grid, block, params)
            results.append((z.download(out, out_bytes), z.download(half, half_bytes)))
        (to, th), (no, nh) = results
        written = (to != 0x77).mean()
        print("%-20s output %8.4f%% identical (%.1f%% written)   halved %8.4f%% identical"
              % (name, 100 * (to == no).mean(), 100 * written, 100 * (th == nh).mean()))
        if "--detail" in sys.argv:
            import ffwd_ref as R
            for idx in np.nonzero(to != no)[0][:12]:
                patch, byte = divmod(int(idx), 512)
                py, px = divmod(patch, TOK_X // 4)
                t, ch = R.CHUNK_TOKEN[byte], R.CHUNK_CHANNEL[byte]
                print("   out  token (%d, %d) channel %2d: translated %s native %s"
                      % (4 * py + t // 4, 4 * px + t % 4, ch, R.value(to[idx]), R.value(no[idx])))
            for idx in np.nonzero(th != nh)[0][:6]:
                plane, rest = divmod(int(idx), HALF_Y * HALF_X * 16)
                tok, byte = divmod(rest, 16)
                print("   half token (%d, %d) plane %d byte %d: translated %s native %s"
                      % (tok // HALF_X, tok % HALF_X, plane, byte, R.value(th[idx]), R.value(nh[idx])))


if __name__ == "__main__":
    main()
