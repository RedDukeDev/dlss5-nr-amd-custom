#!/usr/bin/env python3
"""Straight PTX to HIP, for the stretches of a kernel that are plain arithmetic, texture fetches and branches.

    ptx2hip.py <kernel.ptx> <first line> <last line> <function name> <out.inc>
              [--in REG,REG,...] [--out REG,REG,...] [--prologue "PTX;PTX"]

The lines (1-based, inclusive) of the kernel's body become a HIP device function

    __device__ void <name>(const uint8_t *P, const Special &S, Regs &io)

that reads the registers named by --in from io before it starts, runs the instructions, and writes the ones named
by --out back. P is the kernel's parameter block, S the thread's special registers (rr_ptx.h). Every PTX register
is kept as its raw bits (uint32_t for %r and %f, uint16_t for %rs, uint64_t for %rd and %fd, bool for %p) and each
instruction reads and writes them through the conversions its type asks for, so the generated code is the PTX,
instruction for instruction, with the semantics ZLUDA gives each instruction (rr_ptx.h has them).

What this is for: the ray reconstruction's encoder 0 begins with a gather of per-pixel features from the game's
textures, thousands of instructions long and with no structure worth re-deriving by hand. The generated code is
derived from NVIDIA's network and is therefore never committed: it is made at build time from the user's own copy.

Not handled, by design: mma, movmatrix, bar. A line range that holds one is refused.
"""
import re
import sys

REG = re.compile(r"%[A-Za-z_][A-Za-z0-9_]*")
CTYPE = {"p": "bool", "rs": "uint16_t", "r": "uint32_t", "f": "uint32_t", "rd": "uint64_t", "fd": "uint64_t"}
SPECIAL = {
    "%tid.x": "S.tid_x", "%tid.y": "S.tid_y", "%tid.z": "S.tid_z", "%ntid.x": "S.ntid_x", "%ntid.y": "S.ntid_y",
    "%ntid.z": "S.ntid_z", "%ctaid.x": "S.ctaid_x", "%ctaid.y": "S.ctaid_y", "%ctaid.z": "S.ctaid_z",
    "%nctaid.x": "S.nctaid_x", "%nctaid.y": "S.nctaid_y", "%nctaid.z": "S.nctaid_z", "%laneid": "S.lane",
    "%warpid": "S.warp", "WARP_SZ": "32u",
}


def reg_class(name):
    m = re.match(r"%([a-z]+)\d+$", name)
    return m.group(1) if m else None


def cname(name):
    return name.lstrip("%").replace(".", "_")


def split_args(text):
    out, depth, cur = [], 0, ""
    for ch in text:
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


class Segment:
    """The statements of a line range: labels, instructions, and the nested blocks PTX uses for scoped registers."""

    def __init__(self, lines, first, last):
        self.items = []
        self.locals = {}          # scoped register name -> (unique C name, C type)
        scope = [{}]
        counter = 0
        text = "\n".join(l.replace("\r", "") for l in lines[first - 1:last])
        pos, line = 0, first
        stmt_start = True
        buf = ""
        buf_line = line

        def flush():
            nonlocal buf
            s = buf.strip()
            buf = ""
            if s:
                self._statement(s, buf_line, scope)

        i = 0
        while i < len(text):
            ch = text[i]
            if ch == "\n":
                line += 1
            if ch == "/" and text[i:i + 2] == "//":
                while i < len(text) and text[i] != "\n":
                    i += 1
                continue
            if stmt_start and not buf.strip():
                buf_line = line
                if ch == "{":
                    self.items.append({"k": "open"})
                    scope.append({})
                    i += 1
                    continue
                if ch == "}":
                    self.items.append({"k": "close"})
                    scope.pop()
                    i += 1
                    continue
            if ch == ";":
                flush()
                stmt_start = True
                i += 1
                continue
            buf += ch
            if buf.strip():
                stmt_start = False
            # a label ends at its colon
            if ch == ":" and re.fullmatch(r"\s*[$A-Za-z_][\w$]*:", buf):
                self.items.append({"k": "label", "name": buf.strip()[:-1], "line": buf_line})
                buf = ""
                stmt_start = True
            i += 1
        flush()

    def _statement(self, s, line, scope):
        if s.startswith(".reg"):
            m = re.match(r"\.reg\s+\.(\w+)\s+(.*)", s)
            ptype, names = m.group(1), m.group(2)
            for n in [x.strip() for x in names.split(",")]:
                if "<" in n:   # a range declaration of the whole kernel: handled by the caller
                    continue
                unique = "L%d_%s" % (len(self.locals), n)
                ctype = {"f16": "uint16_t", "b16": "uint16_t", "u16": "uint16_t", "s16": "uint16_t", "b32": "uint32_t",
                         "u32": "uint32_t", "s32": "uint32_t", "f32": "uint32_t", "b64": "uint64_t", "u64": "uint64_t",
                         "pred": "bool"}[ptype]
                scope[-1][n] = unique
                self.locals[unique] = ctype
            return
        if s.startswith(".pragma") or s.startswith(".loc") or s.startswith(".file"):
            return
        pred = None
        m = re.match(r"@(!?)(%p\d+)\s+(.*)", s, re.S)
        if m:
            pred = (m.group(1) == "!", m.group(2))
            s = m.group(3)
        m = re.match(r"([A-Za-z_][\w.]*)\s*(.*)", s, re.S)
        if m is None:
            raise SystemExit("line %d: cannot read the statement %r" % (line, s[:100]))
        op, rest = m.group(1), m.group(2).strip()
        args = split_args(rest) if rest else []
        names = {}
        for sc in scope:
            names.update(sc)
        self.items.append({"k": "inst", "pred": pred, "op": op, "args": args, "line": line, "names": dict(names), "text": s})


class Emitter:
    def __init__(self, seg, param):
        self.seg, self.param = seg, param
        self.regs = {}   # C name -> class

    # ---------------------------------------------------------------- operands
    def use(self, tok, names):
        """C name of a register token, noting it as used."""
        tok = tok.strip()
        if tok in names:
            return names[tok]
        if tok in SPECIAL:
            return SPECIAL[tok]
        cls = reg_class(tok)
        if cls is None:
            raise ValueError("unknown register %s" % tok)
        c = cname(tok)
        self.regs[c] = cls
        return c

    def is_reg(self, tok, names):
        tok = tok.strip()
        return tok in names or tok in SPECIAL or (tok.startswith("%") and reg_class(tok) is not None)

    def imm(self, tok, kind):
        tok = tok.strip()
        if re.fullmatch(r"0[fF][0-9a-fA-F]{8}", tok):
            bits = int(tok[2:], 16)
            return ("f", bits)
        if re.fullmatch(r"0[dD][0-9a-fA-F]{16}", tok):
            return ("d", int(tok[2:], 16))
        t = tok.rstrip("UuSs") if not tok.lower().startswith("0x") else tok.rstrip("Uu")
        return ("i", int(t, 0))

    def read(self, tok, kind, names):
        """C expression of an operand as `kind`: u32 s32 f32 u64 s64 u16 s16 h (half bits) pred."""
        tok = tok.strip()
        if self.is_reg(tok, names):
            v = self.use(tok, names)
            special = tok in SPECIAL
            if special:
                return {"u32": "(uint32_t)%s" % v, "s32": "(int32_t)%s" % v, "u64": "(uint64_t)%s" % v}[kind]
            return {
                "u32": "(uint32_t)%s" % v, "s32": "(int32_t)%s" % v, "f32": "u2f(%s)" % v,
                "u64": "(uint64_t)%s" % v, "s64": "(int64_t)%s" % v, "f64": "u2d(%s)" % v,
                "u16": "(uint16_t)%s" % v, "s16": "(int16_t)%s" % v, "h": "(uint16_t)%s" % v, "pred": "(bool)%s" % v,
            }[kind]
        tag, val = self.imm(tok, kind)
        if tag == "f":
            return {"f32": "u2f(0x%08Xu)" % val, "u32": "0x%08Xu" % val, "s32": "(int32_t)0x%08Xu" % val}[kind]
        if tag == "d":
            return "u2d(0x%016XULL)" % val
        return {
            "u32": "%du" % (val & 0xFFFFFFFF) if val >= 0 else "(uint32_t)(%d)" % val,
            "s32": "(int32_t)(%d)" % val, "u64": "(uint64_t)(%dLL)" % val, "s64": "(int64_t)(%dLL)" % val,
            "u16": "(uint16_t)(%d)" % val, "s16": "(int16_t)(%d)" % val, "h": "(uint16_t)(%d)" % val,
            "f32": "(float)(%d)" % val, "pred": "(bool)(%d)" % val,
        }[kind]

    def dest(self, tok, names):
        return self.use(tok, names)

    def store(self, tok, names, kind, expr):
        d = self.dest(tok, names)
        conv = {"u32": "(uint32_t)(%s)", "s32": "(uint32_t)(int32_t)(%s)", "f32": "f2u(%s)", "u64": "(uint64_t)(%s)",
                "s64": "(uint64_t)(int64_t)(%s)", "f64": "d2u(%s)", "u16": "(uint16_t)(%s)", "h": "(uint16_t)(%s)",
                "pred": "(bool)(%s)"}[kind]
        return "%s = %s;" % (d, conv % expr)

    def address(self, tok, names):
        """C expression of the pointer of [reg+off] / [symbol+off]."""
        inner = tok.strip()[1:-1].replace(" ", "")
        m = re.fullmatch(r"([^+\-]+)(?:([+\-]-?\d+))?", inner) or re.fullmatch(r"(.+?)(?:\+(-?\d+))?", inner)
        base, off = m.group(1), int(m.group(2) or 0)
        if base == self.param:
            return "(P + %d)" % off
        return "((const uint8_t *)(uintptr_t)(%s) + (%d))" % (self.read(base, "u64", names), off)

    # ---------------------------------------------------------------- instructions
    def emit(self, it):
        names = it["names"]
        op, a = it["op"], it["args"]
        parts = op.split(".")
        base = parts[0]
        mods = parts[1:]
        T = mods[-1] if mods else ""
        out = []

        def guarded(lines):
            if it["pred"] is None:
                return lines
            neg, p = it["pred"]
            cond = ("!" if neg else "") + self.use(p, names)
            return ["if (%s) { %s }" % (cond, " ".join(lines))]

        if base == "bra":
            tgt = a[0].lstrip("$")
            if it["pred"] is None:
                return ["goto %s;" % tgt]
            neg, p = it["pred"]
            return ["if (%s%s) goto %s;" % ("!" if neg else "", self.use(p, names), tgt)]
        if base == "ret":
            return ["goto L_return;" if it["pred"] is None else "if (%s%s) goto L_return;" % ("!" if it["pred"][0] else "", self.use(it["pred"][1], names))]
        body = self.body(base, mods, T, a, names, it)
        return guarded(body)

    def body(self, base, mods, T, a, names, it):
        R, S = self.read, self.store
        m = set(mods)
        ftz = "ftz" in m
        if base == "mov":
            if a[0].startswith("{"):   # unpack
                parts = split_args(a[0][1:-1])
                src = R(a[1], "u64" if T in ("b64", "u64") else "u32", names)
                if T in ("b64", "u64"):
                    return [S(parts[0], names, "u32", "(uint32_t)(%s)" % src), S(parts[1], names, "u32", "(uint32_t)((%s) >> 32)" % src)]
                return [S(parts[0], names, "u16", "(uint16_t)(%s)" % src), S(parts[1], names, "u16", "(uint16_t)((%s) >> 16)" % src)]
            if a[1].startswith("{"):   # pack
                parts = split_args(a[1][1:-1])
                if T in ("b64", "u64"):
                    return [S(a[0], names, "u64", "(uint64_t)%s | ((uint64_t)%s << 32)" % (R(parts[0], "u32", names), R(parts[1], "u32", names)))]
                return [S(a[0], names, "u32", "(uint32_t)%s | ((uint32_t)%s << 16)" % (R(parts[0], "u16", names), R(parts[1], "u16", names)))]
            if a[1] == self.param:
                return [S(a[0], names, "u64", "(uint64_t)(uintptr_t)P")]
            kind = {"b32": "u32", "u32": "u32", "s32": "u32", "f32": "u32", "b16": "u16", "u16": "u16", "s16": "u16",
                    "b64": "u64", "u64": "u64", "s64": "u64", "pred": "pred", "f16": "u16"}[T]
            return [S(a[0], names, kind, R(a[1], kind, names))]
        if base == "selp":
            kind = {"b32": "u32", "f32": "u32", "u32": "u32", "s32": "u32", "b16": "u16", "u16": "u16", "b64": "u64", "u64": "u64"}[T]
            return [S(a[0], names, kind, "%s ? %s : %s" % (R(a[3], "pred", names), R(a[1], kind, names), R(a[2], kind, names)))]
        if base in ("add", "sub", "mul", "mad", "min", "max", "abs", "neg") and T in ("s32", "u32", "s64", "u64"):
            w = 64 if T.endswith("64") else 32
            sg = T.startswith("s")
            kind = ("s" if sg else "u") + str(w)
            ukind = "u" + str(w)
            ctype = "int%d_t" % w if sg else "uint%d_t" % w
            u = "uint%d_t" % w
            x = lambda i: R(a[i], kind, names)
            if base == "mul" and "wide" in m:
                wide = "int64_t" if sg else "uint64_t"
                return [S(a[0], names, "u64", "(uint64_t)((%s)%s * (%s)%s)" % (wide, x(1), wide, x(2)))]
            if base == "mul" and "hi" in m:
                raise NotImplementedError("mul.hi")
            if base == "add":
                return [S(a[0], names, ukind, "(%s)%s + (%s)%s" % (u, x(1), u, x(2)))]
            if base == "sub":
                return [S(a[0], names, ukind, "(%s)%s - (%s)%s" % (u, x(1), u, x(2)))]
            if base == "mul":
                return [S(a[0], names, ukind, "(%s)%s * (%s)%s" % (u, x(1), u, x(2)))]
            if base == "mad":
                return [S(a[0], names, ukind, "(%s)%s * (%s)%s + (%s)%s" % (u, x(1), u, x(2), u, x(3)))]
            if base in ("min", "max"):
                c = "<" if base == "min" else ">"
                return [S(a[0], names, ukind, "(%s)(%s %s %s ? %s : %s)" % (u, x(1), c, x(2), x(1), x(2)))]
            if base == "abs":
                return [S(a[0], names, ukind, "(%s)(%s < 0 ? -(%s)%s : %s)" % (u, x(1), u, x(1), x(1)))]
            if base == "neg":
                return [S(a[0], names, ukind, "(%s)0 - (%s)%s" % (u, u, x(1)))]
        if base in ("and", "or", "xor", "not") and T == "pred":
            if base == "not":
                return [S(a[0], names, "pred", "!%s" % R(a[1], "pred", names))]
            c = {"and": "&&", "or": "||", "xor": "!="}[base]
            return [S(a[0], names, "pred", "%s %s %s" % (R(a[1], "pred", names), c, R(a[2], "pred", names)))]
        if base in ("and", "or", "xor", "not", "shl", "shr") and T in ("b32", "u32", "s32", "b64", "u64", "u16", "s16", "b16"):
            w = 64 if T.endswith("64") else 16 if T.endswith("16") else 32
            kind = "u%d" % w
            u = "uint%d_t" % w
            x = lambda i: R(a[i], kind, names)
            if base == "not":
                return [S(a[0], names, kind, "~%s" % x(1))]
            if base in ("and", "or", "xor"):
                c = {"and": "&", "or": "|", "xor": "^"}[base]
                return [S(a[0], names, kind, "%s %s %s" % (x(1), c, x(2)))]
            n = R(a[2], "u32", names)
            if base == "shl":
                return [S(a[0], names, kind, "(%s)(%s >= %d ? 0 : (%s)(%s << %s))" % (u, n, w, u, x(1), n))]
            if T.startswith("s"):
                sk = "s%d" % w
                return [S(a[0], names, kind, "(%s)((%s)%s >> (%s > %d ? %d : %s))" % (u, "int%d_t" % w, R(a[1], sk, names), n, w - 1, w - 1, n))]
            return [S(a[0], names, kind, "(%s)(%s >= %d ? 0 : (%s)(%s >> %s))" % (u, n, w, u, x(1), n))]
        if base == "bfi":
            return [S(a[0], names, "u32", "bfi32(%s, %s, %s, %s)" % (R(a[1], "u32", names), R(a[2], "u32", names), R(a[3], "u32", names), R(a[4], "u32", names)))]
        if base == "prmt":
            return [S(a[0], names, "u32", "prmt32(%s, %s, %s)" % (R(a[1], "u32", names), R(a[2], "u32", names), R(a[3], "u32", names)))]
        if base == "setp":
            cmp = mods[0]
            ty = T
            kind = {"s32": "s32", "u32": "u32", "s16": "s16", "u16": "u16", "s64": "s64", "u64": "u64", "f32": "f32", "b32": "u32"}[ty]
            x, y = R(a[1], kind, names), R(a[2], kind, names)
            cpp = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">=",
                   "equ": "==", "neu": "!=", "ltu": "<", "leu": "<=", "gtu": ">", "geu": ">="}
            if kind == "f32":
                if cmp in ("num", "nan"):
                    e = "(%s == %s && %s == %s)" % (x, x, y, y) if cmp == "num" else "(%s != %s || %s != %s)" % (x, x, y, y)
                elif cmp.endswith("u") and cmp not in ("eq",):
                    e = "(%s %s %s || %s != %s || %s != %s)" % (x, cpp[cmp], y, x, x, y, y)
                else:
                    e = "(%s %s %s)" % (x, cpp[cmp], y)
            else:
                e = "(%s %s %s)" % (x, cpp[cmp], y)
            return [S(a[0].split("|")[0], names, "pred", e)]
        if base == "cvt":
            return self.cvt(mods, a, names)
        if base == "cvta":
            return [S(a[0], names, "u64", R(a[1], "u64", names))]
        if base in ("add", "sub", "mul", "fma", "min", "max", "abs", "neg", "div", "rcp", "sqrt", "lg2", "ex2") and T == "f32":
            x = lambda i: R(a[i], "f32", names)
            e = {
                "add": lambda: "%s + %s" % (x(1), x(2)), "sub": lambda: "%s - %s" % (x(1), x(2)), "mul": lambda: "%s * %s" % (x(1), x(2)),
                "fma": lambda: "__builtin_fmaf(%s, %s, %s)" % (x(1), x(2), x(3)), "min": lambda: "fmin_ptx(%s, %s)" % (x(1), x(2)),
                "max": lambda: "fmax_ptx(%s, %s)" % (x(1), x(2)), "abs": lambda: "__builtin_fabsf(%s)" % x(1), "neg": lambda: "-%s" % x(1),
                "div": lambda: "div_approx(%s, %s)" % (x(1), x(2)), "rcp": lambda: "rcp_approx(%s)" % x(1), "sqrt": lambda: "sqrt_approx(%s)" % x(1),
                "lg2": lambda: "lg2_approx(%s)" % x(1), "ex2": lambda: "ex2_approx(%s)" % x(1),
            }[base]()
            if ftz and base in ("add", "sub", "mul", "fma", "min", "max", "neg"):
                e = "ftz(%s)" % e
            return [S(a[0], names, "f32", e)]
        if base in ("add", "sub", "mul", "min", "max", "fma") and T in ("f16", "f16x2"):
            if T == "f16":
                x = lambda i: "u2h(%s)" % R(a[i], "h", names)
                e = {"add": "%s + %s", "sub": "%s - %s", "mul": "%s * %s"}[base] % (x(1), x(2))
                return [S(a[0], names, "h", "h2u((_Float16)(%s))" % e)]
            x = lambda i: "u2h2(%s)" % R(a[i], "u32", names)
            if base == "fma":
                e = "__builtin_elementwise_fma(%s, %s, %s)" % (x(1), x(2), x(3))
            elif base in ("min", "max"):
                e = "__builtin_elementwise_%s(%s, %s)" % (base, x(1), x(2))
            else:
                e = "%s %s %s" % (x(1), {"add": "+", "sub": "-", "mul": "*"}[base], x(2))
            return [S(a[0], names, "u32", "h22u(%s)" % e)]
        if base in ("ld", "st"):
            return self.memory(base, mods, a, names)
        if base == "tex":
            return self.tex(mods, a, names)
        if base == "sust":
            return self.sust(mods, a, names)
        if base == "shfl":
            mode = mods[1]
            dst = a[0]
            res = "shfl_%s(%s, %s, %s)" % (mode, R(a[1], "u32", names), R(a[2], "s32", names), R(a[3], "u32", names))
            if "|" in dst:
                d1, d2 = [x.strip() for x in dst.split("|")]
                return ["{ ShflResult sr = %s; %s %s }" % (res, S(d1, names, "u32", "sr.value"), S(d2, names, "pred", "sr.valid"))]
            return [S(dst, names, "u32", "%s.value" % res)]
        if base == "vote":
            return [S(a[0], names, "u32", "__builtin_amdgcn_ballot_w32(%s)" % R(a[1], "pred", names))]
        raise NotImplementedError("%s %s" % (base + "." + ".".join(mods), a))

    def cvt(self, mods, a, names):
        R, S = self.read, self.store
        to, frm = mods[-2], mods[-1]
        sat = "satfinite" in mods
        if (to, frm) == ("rn", "f16"):
            pass
        if to == "e4m3x2" and frm == "f16x2":
            return [S(a[0], names, "u16", "cvt_e4m3x2(%s)" % R(a[1], "u32", names))]
        if to == "f16" and frm == "f32":
            e = "(_Float16)%s" % (("clamp_f16(%s)" if sat else "%s") % R(a[1], "f32", names))
            return [S(a[0], names, "h", "h2u(%s)" % e)]
        if to == "f32" and frm == "f16":
            return [S(a[0], names, "f32", "(float)u2h(%s)" % R(a[1], "h", names))]
        if to == "f32" and frm == "s32":
            return [S(a[0], names, "f32", "(float)%s" % R(a[1], "s32", names))]
        if to == "u16" and frm == "u32":
            return [S(a[0], names, "u16", "(uint16_t)%s" % R(a[1], "u32", names))]
        if to == "u32" and frm == "u16":
            return [S(a[0], names, "u32", "(uint32_t)%s" % R(a[1], "u16", names))]
        if to == "u64" and frm == "u32":
            return [S(a[0], names, "u64", "(uint64_t)%s" % R(a[1], "u32", names))]
        raise NotImplementedError("cvt." + ".".join(mods))

    def memory(self, base, mods, a, names):
        R, S = self.read, self.store
        T = mods[-1]
        vec = next((m for m in mods if m in ("v2", "v4")), None)
        n = 2 if vec == "v2" else 4 if vec == "v4" else 1
        ctype = {"u8": "uint8_t", "u16": "uint16_t", "u32": "uint32_t", "f32": "uint32_t", "u64": "uint64_t", "b32": "uint32_t", "b16": "uint16_t"}[T]
        kind = {"uint8_t": "u16", "uint16_t": "u16", "uint32_t": "u32", "uint64_t": "u64"}[ctype]
        if base == "ld":
            addr = self.address(a[1], names)
            if n == 1:
                dest_cls = reg_class(a[0].strip()) if a[0].strip() not in names else "r"
                widen = {"u16": "u16", "u32": "u32", "u64": "u64"}[kind]
                if dest_cls == "r" and kind == "u16":
                    widen = "u32"
                return [S(a[0], names, widen, "*(const %s *)%s" % (ctype, addr))]
            regs = split_args(a[0].strip()[1:-1])
            out = []
            for i, r in enumerate(regs):
                dest_cls = reg_class(r.strip()) if r.strip() not in names else "r"
                widen = "u32" if dest_cls == "r" and kind == "u16" else kind
                out.append(S(r, names, widen, "((const %s *)%s)[%d]" % (ctype, addr, i)))
            return out
        addr = self.address(a[0], names)
        if n == 1:
            return ["*(%s *)%s = (%s)%s;" % (ctype, addr.replace("const ", ""), ctype, R(a[1], kind, names))]
        regs = split_args(a[1].strip()[1:-1])
        return ["((%s *)%s)[%d] = (%s)%s;" % (ctype, addr.replace("const ", ""), i, ctype, R(r, kind, names)) for i, r in enumerate(regs)]

    def tex(self, mods, a, names):
        R, S = self.read, self.store
        dest = split_args(a[0].strip()[1:-1])
        m = re.match(r"\[(.+?),\s*\{(.+?)\}\]", a[1].strip())
        handle = R(m.group(1), "u64", names)
        co = [c.strip() for c in m.group(2).split(",")]
        mode = "texel" if mods[-1] == "s32" else "lod" if "level" in mods else "plain"
        if mode == "texel":
            call = "tex2d_texel(%s, %s, %s)" % (handle, R(co[0], "s32", names), R(co[1], "s32", names))
        elif mode == "lod":
            call = "tex2d_lod(%s, %s, %s, %s)" % (handle, R(co[0], "f32", names), R(co[1], "f32", names), R(a[2], "f32", names))
        else:
            call = "tex2d(%s, %s, %s)" % (handle, R(co[0], "f32", names), R(co[1], "f32", names))
        out = ["{ float4v t = %s;" % call]
        for i, d in enumerate(dest):
            out.append(S(d, names, "f32", "t[%d]" % i))
        out.append("}")
        return [" ".join(out)]

    def sust(self, mods, a, names):
        R = self.read
        m = re.match(r"\[(.+?),\s*\{(.+?)\}\]", a[0].strip())
        handle = R(m.group(1), "u64", names)
        co = [c.strip() for c in m.group(2).split(",")]
        vals = split_args(a[1].strip()[1:-1])
        n = len(vals)
        v = ", ".join("(uint16_t)%s" % R(x, "u16", names) for x in vals)
        return ["sust_b16(%s, %s, %s, %d, %s);" % (handle, R(co[0], "s32", names), R(co[1], "s32", names), n, v)]


def declare(regs, locals_):
    decl = []
    for c, cls in sorted(regs.items()):
        if cls == "p":
            decl.append("bool %s = false;" % c)
        else:
            decl.append("%s %s = 0;" % (CTYPE[cls], c))
    for c, t in sorted(locals_.items()):
        decl.append("%s %s = 0;" % (t, c))
    return decl


def translate(lines, param, first, last, name, ins=(), outs=(), prologue=(), more=()):
    """The HIP text of lines first..last (1-based) as a device function. `prologue` is PTX statements run first
    (to redo, in a segment that starts late, what the kernel did before it with registers only the thread knows);
    `more` is further (first, last) line ranges that belong to the same function -- out-of-line blocks the
    compiler placed elsewhere -- joined after a return so that nothing falls into them."""
    pieces = [lines[first - 1:last]]
    for a, b in more:
        pieces.append(["ret;"])
        pieces.append(lines[a - 1:b])
    seg_lines = [l for piece in pieces for l in piece]
    extra = [p.strip() + ";" for p in prologue if p.strip()]
    seg_lines = extra + seg_lines
    first, last = 1, len(seg_lines)
    seg = Segment(seg_lines, first, last)
    for it in seg.items:
        if it["k"] == "inst" and it["op"].startswith(("mma", "movmatrix", "bar")):
            raise SystemExit("line %d: %s cannot be translated" % (it["line"], it["op"]))
    em = Emitter(seg, param)
    body = []
    for it in seg.items:
        if it["k"] == "label":
            body.append("%s:;" % it["name"].lstrip("$"))
        elif it["k"] == "open":
            body.append("{")
        elif it["k"] == "close":
            body.append("}")
        else:
            try:
                for l in em.emit(it):
                    body.append(l + "  // %d" % it["line"])
            except Exception as e:
                raise SystemExit("line %d: %s: %s" % (it["line"], it["text"][:80], e))
    for r in list(ins) + list(outs):
        em.regs.setdefault(cname(r), reg_class(r))
    text = ["__device__ void %s(const uint8_t *P, const Special &S, Regs &io) {" % name]
    text += ["    " + d for d in declare(em.regs, seg.locals)]
    text += ["    %s = io.%s;" % (cname(r), cname(r)) for r in ins]
    text += ["    " + l for l in body]
    text += ["L_return:;"]
    text += ["    io.%s = %s;" % (cname(r), cname(r)) for r in outs]
    text += ["}", ""]
    n_inst = sum(1 for i in seg.items if i["k"] == "inst")
    print("%s: %d instructions, %d registers" % (name, n_inst, len(em.regs)))
    return chr(10).join(text)


def main():
    args = sys.argv[1:]
    src, first, last, name, out = args[0], int(args[1]), int(args[2]), args[3], args[4]
    ins = [x for x in args[args.index("--in") + 1].split(",") if x] if "--in" in args else []
    outs = [x for x in args[args.index("--out") + 1].split(",") if x] if "--out" in args else []
    text = open(src, encoding="utf-8").read()
    param = re.search(r"\.param \.align \d+ \.b8 (\w+)\[", text).group(1)
    code = translate(text.split(chr(10)), param, first, last, name, ins, outs)
    with open(out, "w", encoding="utf-8", newline=chr(10)) as f:
        f.write(code)


if __name__ == "__main__":
    main()
