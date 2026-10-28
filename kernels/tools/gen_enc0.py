#!/usr/bin/env python3
"""The generated halves of the native rr2_enc0_kernel: its feature gather and its guide tail, from the user's PTX.

usage: gen_enc0.py <rr2_enc0.ptx> <out.inc>

The kernel is three stretches of PTX. Before its first mma it gathers 43 features per pixel from the game's
textures (thousands of instructions, textures, branches, no structure to re-derive by hand): translated here, by
ptx2hip.py. From the first mma to the last it is the embedding and the same window attention and MLP as every other
stage (rr2_stage.hip), written by hand. After the last mma it gates the guide image with the head and writes it,
reading more textures: translated again.

What passes between the stretches is registers, and the analysis here says which. Out of the gather: the A
fragments of the first eight mmas (the 64 feature slots of 64 tokens, as e4m3 codes) and the values the tail goes
on to use. Into the tail: those, and the destination registers of the last eight mmas (the head).

Writes <out.inc>: struct Regs, the two functions, and unpack_features / pack_head, which move the fragments between
those registers and the LDS arrays the stage works on. The output is derived from NVIDIA's network and is never
committed; it is made at build time from the copy of the library the user has.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ptx2hip  # noqa: E402

REG = re.compile(r"%(?:rs|rd|fd|r|f|p)\d+")


def dests(line):
    t = re.sub(r"^@!?%p\d+\s+", "", line.strip())
    m = re.match(r"(?:\{\s*)?([a-z][a-z0-9_.]*)\s+(.*)", t)
    if not m:
        return set()
    op, rest = m.group(1), m.group(2)
    if op.startswith(("st", "bra", "bar", "ret", "sust", "red", "membar", "fence")):
        return set()
    if op.startswith("mma"):
        return set(REG.findall(rest.split("}")[0]))
    f = rest.split(";")[0]
    d = f[1:f.index("}")] if f.startswith("{") else f.split(",")[0]
    return set(REG.findall(d))


def statements(lines, first, last):
    """Instruction texts with their first line number, over lines first..last (0-based, inclusive)."""
    out, i = [], first
    while i <= last:
        text, start = lines[i], i
        while ";" not in lines[i] and i < last:
            i += 1
            text += " " + lines[i]
        out.append((start, text))
        i += 1
    return out


def main():
    src, out = sys.argv[1], sys.argv[2]
    text = open(src, encoding="utf-8").read()
    lines = text.split("\n")
    param = re.search(r"\.param \.align \d+ \.b8 (\w+)\[", text).group(1)
    mma_lines = [i for i, l in enumerate(lines) if "mma.sync" in l]
    first, last_start = mma_lines[0], mma_lines[-1]
    last_end = last_start
    while ";" not in lines[last_end]:
        last_end += 1

    # the mma statements: D, A, B, C registers
    mma_re = re.compile(r"mma\.sync[^{]*\{([^}]*)\}\s*,\s*\{([^}]*)\}\s*,\s*\{([^}]*)\}\s*,\s*\{([^}]*)\}")
    mmas = []
    for start, st in statements(lines, first, last_end):
        m = mma_re.search(st)
        if m:
            mmas.append([[r.strip() for r in g.split(",")] for g in m.groups()])
    assert len(mmas) == len(mma_lines), (len(mmas), len(mma_lines))

    ret_line = next(i for i in range(last_end, len(lines)) if lines[i].strip() == "ret;")
    kernel_end = max(i for i, l in enumerate(lines) if l.strip() == "}")   # 0-based: the kernel's closing brace

    # the A fragments of the first eight distinct operands, in order of first appearance: (token block, k step)
    tiles = []
    for d, a, b, c in mmas:
        if a not in tiles:
            tiles.append(a)
        if len(tiles) == 8:
            break
    heads = mmas[-8:]

    # liveness, by name: defined before the first mma, mentioned after it
    prefix_defs = set()
    ret_cold = next(i for i in range(last_end, len(lines)) if lines[i].strip() == "ret;")
    for i in list(range(0, first)) + list(range(ret_cold + 1, len(lines))):
        prefix_defs |= dests(lines[i])
    prefix_defs = {r for r in prefix_defs if r != ""}
    tail_lines = range(last_end + 1, ret_cold + 1)
    tail_used = set()
    for i in tail_lines:
        tail_used |= set(REG.findall(lines[i]))
    after_used = set()
    for i in range(first, ret_cold + 1):
        after_used |= set(REG.findall(lines[i]))
    gather_out = sorted(prefix_defs & after_used)
    head_regs = [r for h in heads for r in h[0]]
    tail_in = sorted((prefix_defs & tail_used) | set(head_regs))
    # registers the middle defined that the tail reads, other than the head: redone at the tail's start
    middle_defs = {}
    for i in range(first, last_start):
        for r in dests(lines[i]):
            middle_defs[r] = i
    redo = sorted((set(middle_defs) & tail_used) - set(head_regs) - prefix_defs, key=lambda r: middle_defs[r])
    prologue = []
    for r in redo:
        prologue.append(lines[middle_defs[r]].strip().rstrip(";"))

    all_in_out = sorted(set(gather_out) | set(tail_in), key=lambda r: (re.match(r"%([a-z]+)", r).group(1), int(re.search(r"\d+", r).group())))
    cls = lambda r: re.match(r"%([a-z]+)", r).group(1)
    fields = []
    for r in all_in_out:
        c = ptx2hip.CTYPE[cls(r)]
        fields.append("    %s %s;" % (c, ptx2hip.cname(r)))

    # the body starts after the kernel's opening brace and its run of .reg declarations
    brace = next(i for i, l in enumerate(lines) if l.strip() == "{")
    body = brace + 1
    while lines[body].strip().startswith(".reg") or not lines[body].strip():
        body += 1
    gather = ptx2hip.translate(lines, param, body + 1, first, "enc0_gather", ins=[], outs=gather_out, more=[(ret_line + 2, kernel_end)])
    tail = ptx2hip.translate(lines, param, last_end + 2, ret_line + 1, "enc0_tail", ins=tail_in, outs=[], prologue=prologue)

    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("// Generated by gen_enc0.py from %s. Derived from NVIDIA's network: do not commit.\n" % os.path.basename(src))
        f.write("struct Regs {\n" + "\n".join(fields) + "\n};\n\n")
        f.write(gather + "\n" + tail + "\n")
        # features: the A fragments of the eight first tiles -> a [64 tokens][64 slots] code array
        f.write("// tile q = 2 * token block + k step; lane (g, t) holds rows g and g + 8, k 4t .. 4t + 3 and 16 + 4t ..\n")
        f.write("__device__ inline void unpack_features(const Regs &io, uint8_t *features, int lane) {\n")
        f.write("    const int g = lane >> 2, t = lane & 3;\n")
        for q, a in enumerate(tiles):
            tb, s = q // 2, q % 2
            n = [ptx2hip.cname(r) for r in a]
            f.write("    *(uint32_t *)(features + (%d + g) * 64 + %d + 4 * t) = io.%s;\n" % (16 * tb, 32 * s, n[0]))
            f.write("    *(uint32_t *)(features + (%d + g + 8) * 64 + %d + 4 * t) = io.%s;\n" % (16 * tb, 32 * s, n[1]))
            f.write("    *(uint32_t *)(features + (%d + g) * 64 + %d + 16 + 4 * t) = io.%s;\n" % (16 * tb, 32 * s, n[2]))
            f.write("    *(uint32_t *)(features + (%d + g + 8) * 64 + %d + 16 + 4 * t) = io.%s;\n" % (16 * tb, 32 * s, n[3]))
        f.write("}\n\n")
        # head: the destination registers of the last eight mmas, tile (token block, 8 channels) in program order
        f.write("// head tile h = 2 * token block + 8-channel half; lane (g, t) holds rows g and g + 8, columns 2t, 2t + 1\n")
        f.write("__device__ inline void pack_head(Regs &io, const uint16_t *head, int lane) {\n")
        f.write("    const int g = lane >> 2, t = lane & 3;\n")
        for h, (d, a, b, c) in enumerate(heads):
            tb, half = h // 2, h % 2
            f.write("    io.%s = *(const uint32_t *)(head + (%d + g) * 16 + %d + 2 * t);\n" % (ptx2hip.cname(d[0]), 16 * tb, 8 * half))
            f.write("    io.%s = *(const uint32_t *)(head + (%d + g + 8) * 16 + %d + 2 * t);\n" % (ptx2hip.cname(d[1]), 16 * tb, 8 * half))
        f.write("}\n")
    print("gather out %d regs, tail in %d regs (%d from the head), redone in the tail: %s" % (len(gather_out), len(tail_in), len(head_regs), redo))
    print("A tiles:", tiles)
    print("head tiles D:", [h[0] for h in heads])


if __name__ == "__main__":
    main()
