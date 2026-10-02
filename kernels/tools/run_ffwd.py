#!/usr/bin/env python3
"""cc_split_swin_16h_ffwd_512_chained_fp8, emulated on a captured launch.

Runs the kernel's own PTX through ptxsim on the memory the proxy dumped before
launch 8, then compares what it wrote with what the GPU wrote. A match is what
turns the interpreter into a reference a native kernel can be checked against.

Usage: run_ffwd.py [capture dir] [--blocks N]
"""
import argparse
import glob
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ptxsim  # noqa: E402
import workspace  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
GRID, BLOCK = (10, 6, 2), (32, 8, 1)
EXTENT = 120 * 16384      # what one launch writes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", nargs="?", default=os.path.join(HERE, "capture_l8"))
    ap.add_argument("--mma", default="exact", help="exact, f32x2 or f32seq; see ptxsim")
    ap.add_argument("--fma", default="fused", help="fused or unfused; see ptxsim")
    ap.add_argument("--mma-round", default="rn", help="rn or rz: how each mma result becomes f16")
    ap.add_argument("--flush", action="store_true",
                    help="e4m3 subnormal mma inputs read as zero, as ZLUDA's gfx11 path does")
    ap.add_argument("--ptx", default=os.path.join(workspace.captures(), "ffwd_512.ptx"),
                    help="the kernel to run: the original, or a variant from ffwd_variants.py")
    ap.add_argument("--f16", action="store_true",
                    help="compare the output as f16 values, for the variants that write "
                         "raw accumulators")
    args = ap.parse_args()
    ptxsim.Machine.mma_mode = args.mma
    ptxsim.Machine.fma_mode = args.fma
    ptxsim.Machine.mma_round = args.mma_round
    ptxsim.Machine.mma_flush_subnormal = args.flush
    print("mma accumulation: %s, f16 fma: %s" % (args.mma, args.fma))

    base = glob.glob(os.path.join(args.capture, "*_args.bin"))[0][:-9]
    params = open(base + "_args.bin", "rb").read()
    ptrs = struct.unpack_from("<7Q", params)
    names = {0x00: ptrs[0], 0x08: ptrs[1], 0x10: ptrs[2], 0x20: ptrs[4], 0x30: ptrs[6]}

    mem = ptxsim.GlobalMemory()
    for off, addr in names.items():
        mem.add(addr, open("%s_pre_+%02X.bin" % (base, off), "rb").read())
    print("global memory: %d regions, %.1f MB" %
          (len(mem.regions), sum(len(d) for _, d in mem.regions) / 1e6))

    kernel = ptxsim.Kernel(open(args.ptx, encoding="utf-8").read())
    print("kernel: %d instructions, %d branches, shared %d bytes %s" %
          (len(kernel.instrs), len(kernel.reconv), kernel.shared_size,
           [(n[-30:], o, s) for n, (o, s) in kernel.shared.items()]))

    m = ptxsim.Machine(kernel, GRID, BLOCK, params, mem)
    t0 = time.time()
    m.run()
    print("ran %d steps in %.1f s, %d global accesses outside the dumps" %
          (m.steps, time.time() - t0, mem.misses))

    out = mem.read(ptrs[1], EXTENT)
    ref = np.frombuffer(open(base + "_post_+08.bin", "rb").read()[:EXTENT], dtype=np.uint8)
    if args.f16:
        compare_f16(out, ref)
        return
    pre = np.frombuffer(open(base + "_pre_+08.bin", "rb").read()[:EXTENT], dtype=np.uint8)
    same = int((out == ref).sum())
    print("\noutput: %d of %d bytes identical to the GPU's (%.3f%%)" %
          (same, EXTENT, 100.0 * same / EXTENT))
    print("        %d bytes still hold what was there before the launch" % int((out == pre).sum()))
    if same != EXTENT:
        a, b = ptxsim.E4M3[out.astype(np.int64)], ptxsim.E4M3[ref.astype(np.int64)]
        finite = np.isfinite(a) & np.isfinite(b)
        d = np.abs(a[finite] - b[finite])
        codes = np.abs(out.astype(np.int64) - ref.astype(np.int64))
        print("        differing: max |value| diff %.4f, mean %.6f; codes off by 1: %d, by more: %d"
              % (d.max(), d.mean(), int((codes == 1).sum()), int((codes > 1).sum())))
        first = np.nonzero(out != ref)[0][:6]
        for i in first:
            print("          byte %7d: emulated 0x%02X (%g)  gpu 0x%02X (%g)"
                  % (i, out[i], ptxsim.E4M3[out[i]], ref[i], ptxsim.E4M3[ref[i]]))

    flags = mem.read(ptrs[6], 480)
    ref_flags = open(base + "_post_+30.bin", "rb").read()[:480]
    print("ready flags: %s" % ("identical" if bytes(flags) == ref_flags else "DIFFERENT"))


def compare_f16(out, ref):
    """The output read as f16 values, and how far apart they are in ulps.

    ulps are counted on the ordered integer form of the f16 bit pattern, so a
    step across zero or between binades counts the same as any other."""
    a = out.view(np.uint16).astype(np.int64)
    b = ref.view(np.uint16).astype(np.int64)

    def ordered(x):
        return np.where(x & 0x8000, 0x8000 - (x & 0x7FFF), x + 0x8000)

    ulps = np.abs(ordered(a) - ordered(b))
    n = len(a)
    print("\nf16 values: %d of %d identical (%.3f%%)" % (int((ulps == 0).sum()), n,
                                                        100.0 * (ulps == 0).sum() / n))
    for lo, hi in ((1, 1), (2, 2), (3, 4), (5, 16), (17, 1 << 20)):
        print("   %6s ulp: %d" % ("%d-%d" % (lo, hi) if lo != hi else lo,
                                   int(((ulps >= lo) & (ulps <= hi)).sum())))
    va = out.view(np.float16).astype(np.float64)
    vb = ref.view(np.float16).astype(np.float64)
    worst = np.argsort(-ulps)[:5]
    for i in worst:
        print("   value %7d: emulated %-12g gpu %-12g (%d ulp)" % (i, va[i], vb[i], ulps[i]))


if __name__ == "__main__":
    main()
