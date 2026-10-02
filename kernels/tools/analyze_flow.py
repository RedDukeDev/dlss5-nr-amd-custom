#!/usr/bin/env python3
"""The layer graph of a kernel, read off its PTX instead of its listing.

These kernels are ten to fifteen thousand lines of straight-line code, and what
matters in them is small: which matrices are multiplied, in what order, with
what between them. This walks the definitions backwards from every mma and says,
for each operand, where it came from -- a load off which parameter, the result
of which earlier multiply, a transpose, a shuffle -- and groups consecutive
multiplies that share a shape into the layer they belong to.

A register can be written on two paths (a window that is inside the image is
loaded, one that is outside is zeroed), so every definition is followed and the
answers are joined with "|".

Usage: analyze_flow.py <kernel.ptx> [--detail]
"""
import argparse
import re
import sys

MAX_DEPTH = 60


def parse(path):
    """Instructions as (op, dests, srcs, text), and every register's definitions."""
    text = open(path, encoding="utf-8", errors="replace").read()
    joined, buf = [], ""
    for raw in text.splitlines():
        piece = raw.split("//")[0].strip()
        if not piece:
            continue
        buf = (buf + " " + piece).strip() if buf else piece
        if piece.endswith((";", "{", "}", ":")):
            joined.append(buf)
            buf = ""
    instrs, defs = [], {}
    for raw in joined:
        line = raw.strip().rstrip(";")
        # The f16 operations come wrapped in braces of their own -- "{max.f16x2
        # %r1,%r2,%r3; }" -- so a brace that merely surrounds an instruction is
        # dropped, while the braces of an operand group are left alone.
        while line.startswith("{") and re.match(r"^\{\s*[a-z]", line):
            line = line[1:].strip()
        # Only an unmatched trailing brace is the wrapper's; an operand group
        # closes its own.
        while line.endswith("}") and line.count("}") > line.count("{"):
            line = line[:-1].strip().rstrip(";")
        if not line or line.startswith(("//", ".", "$", "}", "{")):
            continue
        line = re.sub(r"^@!?%\w+\s+", "", line)
        m = re.match(r"^([a-z][\w.:]*)\s*(.*)$", line)
        if not m:
            continue
        op, rest = m.group(1), m.group(2)
        head = rest.split("},")[0] if rest.startswith("{") else rest.split(",")[0]
        dests = re.findall(r"%\w+", head)
        srcs = [r for r in re.findall(r"%\w+", rest) if r not in dests]
        instrs.append((op, dests, srcs, rest))
        for d in dests:
            defs.setdefault(d, []).append(len(instrs) - 1)
    return instrs, defs


class Flow:
    def __init__(self, instrs, defs):
        self.instrs, self.defs = instrs, defs
        self.mma_index = {}                    # instruction index -> mma number
        n = 0
        for i, (op, *_ ) in enumerate(instrs):
            if op.startswith("mma"):
                self.mma_index[i] = n
                n += 1
        self.params = {}                       # register -> parameter offset
        for i, (op, dests, srcs, rest) in enumerate(instrs):
            if op.startswith("ld.param") and dests:
                off = re.search(r"\+(\d+)\]", rest)
                self.params[dests[0]] = int(off.group(1)) if off else 0

    def base_param(self, reg, depth=0, seen=None):
        """Which kernel parameter an address register descends from."""
        seen = seen or set()
        if reg in self.params:
            return "+0x%02X" % self.params[reg]
        if reg in seen or depth > MAX_DEPTH or reg not in self.defs:
            return "?"
        seen.add(reg)
        for i in self.defs[reg]:
            op, dests, srcs, rest = self.instrs[i]
            for s in srcs:
                got = self.base_param(s, depth + 1, seen)
                if got != "?":
                    return got
        return "?"

    def origin(self, reg, depth=0, seen=None):
        seen = seen or set()
        if reg in seen or depth > MAX_DEPTH:
            return {"?"}
        seen = seen | {reg}
        if reg not in self.defs:
            return {"undef"}
        out = set()
        for i in self.defs[reg]:
            op, dests, srcs, rest = self.instrs[i]
            if op.startswith("ld.param"):
                out.add("param")
            elif op.startswith(("ld.global", "ld.weak.global")):
                addr = re.search(r"\[(%\w+)", rest)
                out.add("load(%s)" % (self.base_param(addr.group(1)) if addr else "?"))
            elif op.startswith("mma"):
                out.add("mma#%d" % self.mma_index[i])
            elif op.startswith("cvt.rn.satfinite"):
                out.add("narrow(%s)" % self.join(srcs, depth, seen))
            elif op.startswith("cvt.rn.f16x2.e4m3x2"):
                out.add("widen(%s)" % self.join(srcs, depth, seen))
            elif op.startswith("shfl"):
                out.add("shfl(%s)" % self.join(srcs[:1], depth, seen))
            elif op.startswith("movmatrix"):
                out.add("transpose(%s)" % self.join(srcs, depth, seen))
            elif op.startswith(("mul.f16", "add.f16", "fma.rn.f16", "max.f16", "min.f16",
                                "abs.f16", "sub.f16")):
                out.add("f16(%s)" % self.join(srcs, depth, seen))
            elif op.startswith(("rcp", "rsqrt", "ex2", "lg2", "cvt.f32", "cvt.rn.f16.f32")):
                out.add("f32math")
            elif op.startswith(("mov", "prmt", "and", "or", "xor", "shl", "shr", "add", "sub",
                                "selp", "cvt")):
                inner = self.join(srcs, depth, seen)
                out.add(inner if inner not in ("", "?") else "const")
            else:
                out.add(op.split(".")[0])
        return out or {"?"}

    def join(self, srcs, depth, seen):
        got = set()
        for s in srcs:
            got |= self.origin(s, depth + 1, seen)
        got.discard("?")
        got.discard("undef")
        got.discard("const")
        return "|".join(sorted(got)) if got else "const"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ptx")
    ap.add_argument("--detail", action="store_true")
    args = ap.parse_args()
    instrs, defs = parse(args.ptx)
    flow = Flow(instrs, defs)

    groups = []
    for i, (op, dests, srcs, rest) in enumerate(instrs):
        if not op.startswith("mma"):
            continue
        fields = [re.findall(r"%\w+", f) for f in re.findall(r"\{([^}]*)\}", rest)]
        d, aop, bop, cop = (fields + [[], [], [], []])[:4]
        kinds = tuple(flow.join(x[:1], 0, set()) for x in (aop, bop, cop))
        if groups and groups[-1][0] == kinds:
            groups[-1][1] += 1
        else:
            groups.append([kinds, 1, flow.mma_index[i]])
        if args.detail:
            print("mma#%-4d A=%-28s B=%-28s C=%s" % (flow.mma_index[i], kinds[0], kinds[1], kinds[2]))

    print("%s: %d instructions, %d mma" % (args.ptx, len(instrs), len(flow.mma_index)))
    for kinds, n, first in groups:
        print("  %4d x mma from #%-4d  A = %-30s B = %-24s C = %s" % (n, first, kinds[0], kinds[1], kinds[2]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
