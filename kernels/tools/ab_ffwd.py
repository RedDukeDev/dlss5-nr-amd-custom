#!/usr/bin/env python3
"""The native ffwd kernel on a captured launch, compared byte for byte.

Loads the buffers a capture recorded before the translated kernel ran, runs
ffwd_512.hsaco on them through HIP, and compares what it writes with what the
translated kernel wrote (the capture's post dump) and with ffwd_ref.py. Also
times it, and checks the ready flags it raises.

Usage: ab_ffwd.py [capture dir] [--time N]
"""
import argparse
import glob
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ffwd_ref  # noqa: E402
import ptxsim  # noqa: E402
from hip_run import Hip  # noqa: E402

EXTENT = 120 * 16384
KERNEL = "cc_split_swin_16h_ffwd_512_chained_fp8_native"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", nargs="?", default=os.path.join(HERE, "capture_l8"))
    ap.add_argument("--hsaco", default=os.path.join(os.path.dirname(HERE), "ffwd_512.hsaco"))
    ap.add_argument("--time", type=int, default=0, help="launches to average over")
    args = ap.parse_args()

    base = glob.glob(os.path.join(args.capture, "*_args.bin"))[0][:-9]
    params = open(base + "_args.bin", "rb").read()
    dims = struct.unpack_from("<2i", params, 0x18)
    offs = struct.unpack_from("<2i", params, 0x28)
    pre = {off: open("%s_pre_+%02X.bin" % (base, off), "rb").read() for off in (0, 8, 0x10, 0x20, 0x30)}
    gpu_out = np.frombuffer(open(base + "_post_+08.bin", "rb").read(), np.uint8, EXTENT)
    gpu_flags = np.frombuffer(open(base + "_post_+30.bin", "rb").read(), np.int32)

    h = Hip()
    fn = h.module(args.hsaco, KERNEL)
    buf = {off: h.upload(np.frombuffer(data, np.uint8)) for off, data in pre.items()}
    blob = struct.pack("<3Q2iQ2iQ", buf[0], buf[8], buf[0x10], dims[0], dims[1], buf[0x20],
                       offs[0], offs[1], buf[0x30])
    assert len(blob) == 56
    h.launch(fn, (10, 6, 2), (32, 8, 1), blob)
    out = h.download(buf[8], EXTENT)
    flags = h.download(buf[0x30], len(pre[0x30])).view(np.int32)

    x = ffwd_ref.decode_activations(pre[0], 240)
    ref = np.frombuffer(ffwd_ref.encode_activations(ffwd_ref.forward(x, ffwd_ref.decode_weights(pre[0x10]))),
                        np.uint8)

    def report(name, a, b):
        same = a == b
        diff = np.abs(ptxsim.E4M3[a[~same]] - ptxsim.E4M3[b[~same]]) if (~same).any() else np.zeros(1)
        print("native vs %-10s %8.4f%% bytes identical, max |diff| %.4g" % (name, 100 * same.mean(), diff.max()))

    report("translated", out, gpu_out)
    report("reference", out, ref)
    print("translated vs reference %.4f%% (for scale)" % (100 * (gpu_out == ref).mean()))
    print("ready flags: %s" % ("identical to the translated kernel's" if np.array_equal(flags, gpu_flags)
                               else "DIFFERENT: %s vs %s" % (flags[:12], gpu_flags[:12])))
    if args.time:
        ms = h.time(fn, (10, 6, 2), (32, 8, 1), blob, repeat=args.time)
        print("native kernel: %.3f ms per launch (%d launches)" % (ms, args.time))


if __name__ == "__main__":
    main()
