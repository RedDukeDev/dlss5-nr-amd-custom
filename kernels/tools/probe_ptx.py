#!/usr/bin/env python3
"""Put a probe into a translated kernel: one register, per lane, per block.

A native kernel is checked against the translation by its output, but when the
two disagree the output says only that they do. The way to find where is to
make the translation itself hand over an intermediate value: this writes a
store of one PTX register into the kernel's text, and probe_zluda.py runs that
text through ZLUDA on a captured launch. The value comes back from the same
hardware the native kernel runs on, so it is an exact reference, which ptxsim's
emulation of the WMMA is not.

The probe writes through a pointer in a parameter slot the kernel never reads
(+64 in the 96-byte parameter of the fused Swin kernels), so nothing it does is
visible to the kernel. Each block gets <stride> words; the probe uses the first
32, one per lane, which leaves room for the native kernel to write its own view
of the same value further along and for the two to be compared side by side.

Usage: probe_ptx.py <ptx> <line> <%register> <output ptx> [--stride 640] [--slot 64]

<line> is the line of the PTX after which the store goes, 1-based, as the file
numbers it: the line of the instruction that defines the register, or the
closing brace of the inline-asm block that does.
"""
import argparse
import io
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ptx")
    ap.add_argument("line", type=int)
    ap.add_argument("register")
    ap.add_argument("out")
    ap.add_argument("--stride", type=int, default=640, help="probe words per block")
    ap.add_argument("--slot", type=int, default=64, help="parameter offset holding the probe pointer")
    args = ap.parse_args()

    lines = io.open(args.ptx, encoding="utf-8").read().split("\n")
    param = None
    for text in lines:
        m = re.search(r"\.param\s+\.align\s+\d+\s+\.b8\s+(\S+)\[", text)
        if m:
            param = m.group(1)
            break
    if param is None:
        raise SystemExit("no parameter declaration in %s" % args.ptx)

    store = """
ld.param.b64 %rdprobe1, [{param}+{slot}];
mov.u32 %rprobe1, %ctaid.x;
mov.u32 %rprobe2, %ctaid.y;
mov.u32 %rprobe6, %nctaid.x;
mad.lo.s32 %rprobe3, %rprobe2, %rprobe6, %rprobe1;
mov.u32 %rprobe4, %laneid;
mad.lo.s32 %rprobe5, %rprobe3, {stride}, %rprobe4;
mul.wide.u32 %rdprobe2, %rprobe5, 4;
add.s64 %rdprobe3, %rdprobe1, %rdprobe2;
st.global.b32 [%rdprobe3], {reg};
""".format(param=param, slot=args.slot, stride=args.stride, reg=args.register)

    for i, text in enumerate(lines):
        if text.startswith(".reg .b64 %rd<"):
            lines[i] = text + "\n.reg .b32 %rprobe<8>;\n.reg .b64 %rdprobe<8>;"
            break
    else:
        raise SystemExit("no register block in %s" % args.ptx)
    lines.insert(args.line, store)
    io.open(args.out, "w", encoding="utf-8").write("\n".join(lines))
    print("probe on %s after line %d -> %s (%d words per block)"
          % (args.register, args.line, args.out, args.stride))


if __name__ == "__main__":
    main()
