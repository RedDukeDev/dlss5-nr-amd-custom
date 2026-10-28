#!/usr/bin/env python3
"""Replay one captured launch on the GPU, through ZLUDA, from a kernel's PTX -- and look inside it.

The capture is the gate's (runtime/gate/capture.h: DLSS5NR_GATE_CAPTURE, _DIR, _LAUNCH): args.bin, info.txt and the
memory the arguments point into, before (pre_<base>.bin) and after (post_<base>.bin) the launch. The pointers in the
arguments are rebased onto fresh allocations holding the "pre" bytes, the kernel is built from PTX by ZLUDA and run
on the whole grid, and what it leaves behind is compared with the capture. When that is byte for byte, the PTX
text and ZLUDA are known to be what produced the capture, and the PTX can be changed and run again.

Two uses:

  replay_gpu.py <capture dir> <kernel.ptx>
      the check itself: how many bytes differ from the capture (0 expected).

  replay_gpu.py <capture dir> <kernel.ptx> --mma-log BX,BY out.npz
      every mma of the block (BX, BY) writes its result to a debug buffer: the hardware's own answer for each
      16x8 product tile, in program order, as float16 (n, 16, 8). A tile found in a reference's matrix says that
      the reference and the GPU agree up to that point; the first tile that is not says where they part. The
      interpreter (ptxsim.py) can be wrong about instructions and about how the GPU rounds; this cannot.
      The block is a single warp here: the kernel must be launched with 32 threads per block.

The tools that need the extracted PTX or a capture read them from a directory on the command line; nothing derived
from NVIDIA's network belongs in the repository (see workspace.py).
"""
import argparse
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import workspace  # noqa: E402
import zluda_run  # noqa: E402

HEADER = ".version 9.4\n.target sm_120\n.address_size 64\n"
MMA = re.compile(r"(mma\.sync\.aligned[^;]*?\{\s*(%r\d+)\s*,\s*(%r\d+)\s*\}[^;]*;)", re.S)


def read_info(directory):
    info = {"regions": {}, "pointers": []}
    for line in open(os.path.join(directory, "info.txt")):
        f = line.split()
        if f[0] in ("grid", "block"):
            info[f[0]] = tuple(int(v) for v in f[1:])
        elif f[0] == "region":
            info["regions"][int(f[1], 16)] = int(f[2])
        elif f[0] == "pointer":
            info["pointers"].append((int(f[1]), int(f[2], 16), int(f[3])))
    return info


def instrument_mma(text, bx, by):
    """PTX with one more parameter, a pointer to a debug buffer, and a store after every mma: the two destination
    registers of each lane, at debug[index * 256 + lane * 8]. Only the block (bx, by) stores. Returns the text, the
    number of mma instructions and the size of the buffer."""
    param = re.search(r"\.param \.align 8 \.b8 (\w+)\[(\d+)\]", text)
    name, size = param.group(1), int(param.group(2))
    text = text.replace(param.group(0), ".param .align 8 .b8 %s[%d]" % (name, size + 8))
    declarations = "\n.reg .pred %pdbg;\n.reg .b64 %rdb<3>;\n.reg .b32 %rdt<3>;\n"
    prologue = (
        "ld.param.u64 %rdb0, [{n}+{o}];\n"
        "cvta.to.global.u64 %rdb0, %rdb0;\n"
        "mov.u32 %rdt0, %laneid;\n"
        "mul.wide.u32 %rdb1, %rdt0, 8;\n"
        "add.s64 %rdb0, %rdb0, %rdb1;\n"
        "mov.u32 %rdt1, %ctaid.x;\n"
        "mov.u32 %rdt2, %ctaid.y;\n"
        "xor.b32 %rdt1, %rdt1, {bx};\n"
        "xor.b32 %rdt2, %rdt2, {by};\n"
        "or.b32 %rdt1, %rdt1, %rdt2;\n"
        "setp.eq.u32 %pdbg, %rdt1, 0;\n"
    ).format(n=name, o=size, bx=bx, by=by)
    # The kernel's own declarations are the first run of consecutive .reg lines: inner blocks declare their own.
    top = re.search(r"(?:\.reg [^\n]*\n)+", text)
    text = text[:top.end()] + declarations + prologue + text[top.end():]
    count = 0

    def store_after(m):
        nonlocal count
        k = count
        count += 1
        return m.group(1) + "\n@%%pdbg st.global.v2.u32 [%%rdb0+%d], {%s, %s};\n" % (k * 256, m.group(2), m.group(3))

    return MMA.sub(store_after, text), count


def run(directory, ptx_text, debug_bytes=0):
    """Returns (info, post memory per region, debug buffer or None)."""
    info = read_info(directory)
    z = zluda_run.Zluda()
    args = bytearray(open(os.path.join(directory, "args.bin"), "rb").read())
    new_base = {}
    for base in info["regions"]:
        new_base[base] = z.upload(np.fromfile(os.path.join(directory, "pre_%x.bin" % base), np.uint8))
    for offset, base, delta in info["pointers"]:
        args[offset:offset + 8] = (new_base[base] + delta).to_bytes(8, "little")
    debug = 0
    if debug_bytes:
        debug = z.upload(np.zeros(debug_bytes, np.uint8))
        args += debug.to_bytes(8, "little")
    name = re.search(r"\.entry\s+(\w+)", ptx_text).group(1)
    if ".version" not in ptx_text:
        ptx_text = HEADER + ptx_text
    started = time.time()
    fn = z.module(ptx_text, name)
    print("module built in %.1f s" % (time.time() - started))
    z.launch(fn, info["grid"], info["block"], bytes(args))
    post = {base: z.download(new_base[base], size) for base, size in info["regions"].items()}
    return info, post, (z.download(debug, debug_bytes) if debug_bytes else None)


def compare(directory, info, post):
    total = 0
    for base in info["regions"]:
        captured = np.fromfile(os.path.join(directory, "post_%x.bin" % base), np.uint8)
        before = np.fromfile(os.path.join(directory, "pre_%x.bin" % base), np.uint8)
        diff = int((post[base] != captured).sum())
        total += diff
        print("region %x: the launch changed %d bytes, the replay differs from the capture in %d"
              % (base, int((captured != before).sum()), diff))
    return total


def tiles(debug, count):
    """The debug buffer as float16 (count, 16, 8) tiles: lane l holds rows l // 4 and l // 4 + 8, columns
    2 * (l % 4) and 2 * (l % 4) + 1, as f16x2 pairs (the m16n8 accumulator layout)."""
    words = debug.view(np.uint32).reshape(count, 32, 2)
    low = (words & 0xFFFF).astype(np.uint16).view(np.float16)
    high = (words >> 16).astype(np.uint16).view(np.float16)
    out = np.zeros((count, 16, 8), np.float16)
    for lane in range(32):
        g, t = lane // 4, lane % 4
        for r in range(2):
            out[:, g + 8 * r, 2 * t] = low[:, lane, r]
            out[:, g + 8 * r, 2 * t + 1] = high[:, lane, r]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture")
    ap.add_argument("ptx")
    ap.add_argument("--mma-log", nargs=2, metavar=("BX,BY", "OUT.npz"))
    args = ap.parse_args()
    # ZLUDA's cache for these runs: a scratch directory unless the caller chose one, never the shipped cache.
    os.environ.setdefault("ZLUDA_CACHE_DIR", os.path.join(os.path.abspath(args.capture), "..", "zluda_cache"))
    os.makedirs(os.environ["ZLUDA_CACHE_DIR"], exist_ok=True)
    text = open(args.ptx, encoding="utf-8").read()
    if args.mma_log:
        bx, by = (int(v) for v in args.mma_log[0].split(","))
        text, count = instrument_mma(text, bx, by)
        info, post, debug = run(args.capture, text, debug_bytes=count * 256)
        if compare(args.capture, info, post):
            print("the instrumented kernel no longer reproduces the capture: not saved")
            return 1
        np.savez_compressed(args.mma_log[1], d=tiles(debug, count))
        print("%d mma results of block (%d, %d) in %s" % (count, bx, by, args.mma_log[1]))
        return 0
    info, post, _ = run(args.capture, text)
    return 0 if compare(args.capture, info, post) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
