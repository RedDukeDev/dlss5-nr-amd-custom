#!/usr/bin/env python3
"""A kernel's PTX with a store after every mma of all its operands, for runtime/gate/swap.h.

usage: instrument_ptx.py <kernel.ptx> <out.ptx>

The patched kernel takes one more 8-byte parameter, right after its own, a pointer to a debug buffer whose first
two words are the (x, y) of the block to log (the gate writes them from DLSS5NR_GATE_SWAP_BLOCK). For that block,
mma number k (program order) stores, at 256 + k * 1280 + lane * 40, ten words: its two destination registers, the
four of A, the two of B and the two of C -- what the multiply was handed and what it gave back, as the GPU
computed them. The buffer is DLSS5NR_GATE_SWAP_DEBUG bytes: 256 + 1280 * (number of mma). The tool prints that
number. A kernel with one wave per block is assumed (the lane is %laneid).

replay_gpu.py does the same for captures it can replay; this is for kernels that read textures, whose inputs live
in the running process.
"""
import re
import sys

REGS = r"(%r\d+)"
MMA = re.compile(r"(mma\.sync\.aligned[^;]*?\{\s*" + REGS + r"\s*,\s*" + REGS + r"\s*\}\s*,\s*\{\s*" + REGS + r"\s*,\s*" + REGS +
                 r"\s*,\s*" + REGS + r"\s*,\s*" + REGS + r"\s*\}\s*,\s*\{\s*" + REGS + r"\s*,\s*" + REGS + r"\s*\}\s*,\s*\{\s*" + REGS +
                 r"\s*,\s*" + REGS + r"\s*\}\s*;)", re.S)


def instrument(text):
    param = re.search(r"\.param \.align 8 \.b8 (\w+)\[(\d+)\]", text)
    name, size = param.group(1), int(param.group(2))
    text = text.replace(param.group(0), ".param .align 8 .b8 %s[%d]" % (name, size + 8))
    declarations = "\n.reg .pred %pdbg;\n.reg .b64 %rdb<3>;\n.reg .b32 %rdt<4>;\n"
    prologue = (
        "ld.param.u64 %rdb2, [{n}+{o}];\n"
        "cvta.to.global.u64 %rdb2, %rdb2;\n"
        "ld.global.v2.u32 {{%rdt1, %rdt2}}, [%rdb2];\n"
        "mov.u32 %rdt0, %laneid;\n"
        "mul.wide.u32 %rdb1, %rdt0, 40;\n"
        "add.s64 %rdb0, %rdb2, %rdb1;\n"
        "add.s64 %rdb0, %rdb0, 256;\n"
        "mov.u32 %rdt3, %ctaid.x;\n"
        "xor.b32 %rdt1, %rdt1, %rdt3;\n"
        "mov.u32 %rdt3, %ctaid.y;\n"
        "xor.b32 %rdt2, %rdt2, %rdt3;\n"
        "or.b32 %rdt1, %rdt1, %rdt2;\n"
        "setp.eq.u32 %pdbg, %rdt1, 0;\n"
    ).format(n=name, o=size)
    top = re.search(r"(?:\.reg [^\n]*\n)+", text)
    text = text[:top.end()] + declarations + prologue + text[top.end():]
    count = 0

    def store_after(m):
        nonlocal count
        d0, d1, a0, a1, a2, a3, b0, b1, c0, c1 = m.groups()[1:]
        base = count * 1280
        count += 1
        words = [d0, d1, a0, a1, a2, a3, b0, b1, c0, c1]
        stores = "".join("@%%pdbg st.global.u32 [%%rdb0+%d], %s;\n" % (base + 4 * k, w) for k, w in enumerate(words))
        return m.group(1) + "\n" + stores

    return MMA.sub(store_after, text), count


def main():
    text = open(sys.argv[1], encoding="utf-8").read()
    patched, count = instrument(text)
    if ".version" not in patched:
        patched = ".version 9.4\n.target sm_120\n.address_size 64\n" + patched
    with open(sys.argv[2], "w", encoding="utf-8", newline="\n") as f:
        f.write(patched)
    print("%d mma instrumented; DLSS5NR_GATE_SWAP_DEBUG=%d" % (count, 256 + 1280 * count))


if __name__ == "__main__":
    main()
