#!/usr/bin/env python3
"""The post block, translated and native, on synthetic inputs.

Like pre_block_check.py: every texture and buffer the kernel can read is built
here -- the halved surface and the full-resolution features as random e4m3,
colour, history and motion textures, the blend weight -- and the translated PTX
(through ZLUDA) and the native kernel (through HIP, same process, same objects)
each write their own output surface, compared texel for texel.

Usage: post_block_check.py [variant ...] [--size=HxW] [--seed=N] [--hsaco=PATH]
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

ENTRY = "cc_tinlayout_fused_post_block_swin_1h_32_fp8"
H, W, SEED = 60, 100, 3
HSACO = os.path.join(os.path.dirname(HERE), "swin_1h.hsaco")   # --hsaco=PATH: another build of it
for _arg in sys.argv[1:]:
    if _arg.startswith("--hsaco="):
        HSACO = _arg[8:]
    if _arg.startswith("--size="):
        H, W = (int(v) for v in _arg[7:].split("x"))
    if _arg.startswith("--seed="):
        SEED = int(_arg[7:])
TOK_Y, TOK_X = (H + 31) // 32 * 32, (W + 31) // 32 * 32

VARIANTS = {
    "all":          dict(),
    "no_clamp":     dict(clamp_mode=0),
    "no_history":   dict(history=False),
    "no_motion":    dict(motion=False),
    "motion_off":   dict(use_motion=0),
    "no_weight":    dict(weight=None),
    "weight_inf":   dict(weight=np.inf),
    "weight_nan":   dict(weight=np.nan),
    "weight_half":  dict(weight=0.5),
    "no_color":     dict(color=False),
}


def random_e4m3(rng, n):
    codes = rng.integers(0, 256, n).astype(np.uint8)
    codes[(codes & 0x7F) == 0x7F] = 0x3C                     # no NaN codes
    codes[(codes & 0x78) >= 0x50] &= 0xC7                    # keep magnitudes modest
    return codes


def build_params(tex, low, inp, surface, weights, weight_ptr, clamp_mode=1, use_motion=1):
    p = struct.pack("<3Q", low, inp, surface)
    p += struct.pack("<Q", weights)
    p += struct.pack("<2i", TOK_Y, TOK_X)
    p += struct.pack("<2i", -4, -4)
    p += struct.pack("<fi", 0.03125, clamp_mode)
    p += struct.pack("<Q", tex["color"])
    p += struct.pack("<6f", 0.0, 0.0, float(W), float(H), 1.0 / W, 1.0 / H)
    p += struct.pack("<2Q", tex["history"], tex["motion"])
    p += struct.pack("<Q", weight_ptr)
    p += struct.pack("<i", use_motion)
    p += struct.pack("<6f", 0.25, -0.125, float(W), float(H), 1.0 / W, 1.0 / H)   # history
    p += struct.pack("<6f", 0.0, 0.0, float(W), float(H), 1.0 / W, 1.0 / H)       # motion
    p += struct.pack("<2f", 1.0, 1.0)
    p += struct.pack("<2i", W, H)
    p += struct.pack("<i", 0)
    assert len(p) == 184, len(p)
    return p


def main():
    wanted = [a for a in sys.argv[1:] if not a.startswith("--")] or list(VARIANTS)
    rng = np.random.default_rng(SEED)
    z = Zluda()
    h = Hip()
    textures = {
        "color": z.texture(z.array2d(rng.uniform(0, 1, (H, W, 4)).astype(np.float16))),
        "history": z.texture(z.array2d(rng.uniform(0, 1, (H, W, 4)).astype(np.float16))),
        "motion": z.texture(z.array2d(rng.uniform(-0.03, 0.03, (H, W, 4)).astype(np.float16))),
    }
    hy, hx = TOK_Y // 2, TOK_X // 2
    low = z.upload(random_e4m3(rng, 2 * hy * hx * 16))
    inp = z.upload(random_e4m3(rng, TOK_Y * TOK_X * 32))
    base = glob.glob(os.path.join(workspace.captures(), "capture_post16", "*_args.bin"))[0][:-9]
    weights = z.upload(np.fromfile(base + "_pre_+18.bin", np.uint8)[:1 << 16])

    text = ".version 9.4\n.target sm_120\n.address_size 64\n" + io.open(
        os.path.join(workspace.captures(), "post_block_32.ptx"), encoding="utf-8").read()
    translated = z.module(text, ENTRY)
    native = h.module(HSACO, ENTRY + "_native")
    grid, block = (TOK_X // 8 + 1, TOK_Y // 8 + 1, 1), (32, 1, 1)

    for name in wanted:
        cfg = dict(VARIANTS[name])
        tex = dict(textures)
        for t in ("history", "motion", "color"):
            if cfg.pop(t, True) is False:
                tex[t] = 0
        weight = cfg.pop("weight", 0.75)
        weight_ptr = 0 if weight is None else z.upload(np.array([weight], np.float16))
        results = []
        for run in ("translated", "native"):
            arr = z.array2d(np.zeros((H, W, 4), np.float16), surface=True)
            surf = z.surface(arr)
            params = build_params(tex, low, inp, surf, weights, weight_ptr, **cfg)
            if run == "translated":
                z.launch(translated, grid, block, params)
            else:
                h.launch(native, grid, block, params)
            results.append(z.read_array2d(arr, (H, W, 4), np.float16))
        a, b = results
        same = (a.view(np.uint16) == b.view(np.uint16))
        print("%-12s %8.4f%% of texels identical   (%.1f%% non-zero)" % (name, 100 * same.mean(), 100 * (a != 0).mean()))
        if "--detail" in sys.argv:
            for y, x, c in list(zip(*np.nonzero(~same)))[:8]:
                print("   pixel (%d, %d) channel %d: translated %r native %r" % (y, x, c, float(a[y, x, c]), float(b[y, x, c])))


if __name__ == "__main__":
    main()
