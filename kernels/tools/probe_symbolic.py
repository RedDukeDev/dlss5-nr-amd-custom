"""Symbolic lockstep run of a straight-line stretch of the attention PTX over
32 lanes. Scores from the score mmas become labels ("s", query, key); f16x2
registers are (lo, hi) pairs of expression trees; integer registers are ints.
Prints, for a target register, which tree each lane holds.

Usage: probe_symbolic.py FIRST LAST TARGET [PTX]"""
import io, re, sys
import workspace  # noqa: E402

PTX = sys.argv[4] if len(sys.argv) > 4 else sys.exit("the PTX path is required (fourth argument)")
lines = io.open(PTX, encoding="utf-8").read().splitlines()
first, last, target = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]


class Sym(tuple):
    pass


def S(lo, hi):
    return Sym((lo, hi))


# ---- parse into instructions
ins = []
k = first - 1
while k < last:
    l = lines[k].strip()
    k += 1
    if l.startswith("mma.sync"):
        d = re.findall(r"%r\d+", l)
        ins.append(("mma", d[:2]))
        k += 3
        continue
    l = l.strip("{} ;").strip()
    if l.count("{") > l.count("}"):
        l += "}"
    if not l or l.startswith(".reg") or l.startswith("//") or l.startswith("$L") or l.startswith("@") or l.startswith("bra"):
        continue
    m = re.match(r"(\S+)\s+(.*)", l)
    if m:
        args = [a.strip() for a in re.split(r",(?![^{]*\})", m.group(2))]
        ins.append((m.group(1), args))

R = [{"%laneid": lane, target: S(("acc",), ("acc",))} for lane in range(32)]
mma_n = 0


def val(lane, a):
    a = a.strip()
    if a in R[lane]:
        return R[lane][a]
    if re.fullmatch(r"-?\d+", a):
        return int(a)
    if a.lower().startswith("0x"):
        return int(a.rstrip("Uu"), 16)
    return None


for op, args in ins:
    base = op.split(".")[0]
    if op == "mma":
        tile, nt = divmod(mma_n, 8)
        for lane in range(32):
            g, t = lane >> 2, lane & 3
            R[lane][args[0]] = S(("s", 16 * tile + g, 8 * nt + 2 * t), ("s", 16 * tile + g, 8 * nt + 2 * t + 1))
            R[lane][args[1]] = S(("s", 16 * tile + g + 8, 8 * nt + 2 * t), ("s", 16 * tile + g + 8, 8 * nt + 2 * t + 1))
        mma_n += 1
        continue
    if op.startswith("shfl.sync.idx"):
        dst = args[0].split("|")[0]
        before = [R[lane].get(args[1]) for lane in range(32)]
        for lane in range(32):
            src = val(lane, args[2])
            if isinstance(src, int):
                R[lane][dst] = before[src & 31]
        continue
    for lane in range(32):
        v = lambda a: val(lane, a)
        if base == "mov" and args[0].startswith("{"):
            parts = [p.strip() for p in args[0].strip("{}").split(",")]
            x = v(args[1])
            if isinstance(x, Sym):
                R[lane][parts[0]] = ("half", x[0])
                R[lane][parts[1]] = ("half", x[1])
            elif isinstance(x, int):
                R[lane][parts[0]] = x & 0xFFFF
                R[lane][parts[1]] = x >> 16
            continue
        if base == "mov" and len(args) > 1 and args[1].startswith("{"):
            parts = [p.strip() for p in args[1].strip("{}").split(",")]
            if len(parts) < 2:
                continue
            a, b = v(parts[0]), v(parts[1])
            if isinstance(a, tuple) and a and a[0] == "half" and isinstance(b, tuple) and b and b[0] == "half":
                R[lane][args[0]] = S(a[1], b[1])
            continue
        if base in ("mov", "cvt"):
            x = v(args[1])
            if x is not None:
                R[lane][args[0]] = x
            continue
        if op.startswith("add.f16x2"):
            a, b = v(args[1]), v(args[2])
            if isinstance(a, Sym) and isinstance(b, Sym):
                R[lane][args[0]] = S(("+", a[0], b[0]), ("+", a[1], b[1]))
            continue
        if base in ("fma", "max", "min"):
            a = v(args[1])
            if isinstance(a, Sym):
                R[lane][args[0]] = a
            continue
        vs = [v(a) for a in args[1:]]
        if base in ("shl", "add") and vs and isinstance(vs[0], Sym):
            R[lane][args[0]] = vs[0]
            continue
        if base == "selp":
            R[lane][args[0]] = vs[0] if vs[2] else vs[1]
            continue
        if base == "prmt":
            a, b = vs[0], vs[1]
            if isinstance(a, Sym) and isinstance(b, Sym) and vs[2] == 0x5410:
                R[lane][args[0]] = S(a[0], b[0])
            continue
        if not vs or not all(isinstance(x, (int, bool)) for x in vs):
            continue
        if base == "setp":
            cmp = op.split(".")[1]
            a, b = vs[0], vs[1]
            R[lane][args[0]] = {"eq": a == b, "ne": a != b, "lt": a < b, "le": a <= b,
                                "gt": a > b, "ge": a >= b}[cmp]
            continue
        a = vs[0]; b = vs[1] if len(vs) > 1 else 0
        ops = {"and": lambda: a & b, "or": lambda: a | b, "xor": lambda: a ^ b,
               "shl": lambda: (a << b) & 0xFFFFFFFF, "shr": lambda: a >> b, "sub": lambda: a - b,
               "mul": lambda: a * b, "add": lambda: a + b, "not": lambda: ~a}
        if base in ops:
            R[lane][args[0]] = ops[base]()


def fmt(e):
    if isinstance(e, tuple) and e and e[0] == "+":
        return "(" + fmt(e[1]) + " + " + fmt(e[2]) + ")"
    if isinstance(e, tuple) and e and e[0] == "s":
        return "p%d_%d" % (e[1], e[2])
    return str(e)


for lane in (0, 1, 2, 3, 4, 8, 31):
    x = R[lane].get(target)
    if isinstance(x, Sym):
        print("lane", lane, "lo:", fmt(x[0]))
        print("        hi:", fmt(x[1]))
    else:
        print("lane", lane, x)
