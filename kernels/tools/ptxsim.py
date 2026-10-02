#!/usr/bin/env python3
"""A PTX interpreter for one kernel, run on real captured memory.

The point is not speed but certainty. A native kernel has to reproduce what the
original computes, and the original is 2300 lines of PTX whose layouts are
nowhere written down. Reading it and re-deriving the algorithm by hand gets
every index right only if every index is guessed right; this instead executes
the PTX itself, instruction by instruction, on the exact bytes a real launch was
handed -- and then checks the result against what the GPU actually wrote. Once
it matches, it is a reference that can be instrumented: which input byte lands
in which output, what an intermediate holds, where a stage begins.

How it runs:

  - every lane of every block at once, vectorised with numpy: a lane is a
    thread, and registers are arrays with one entry per lane;
  - divergence the way the hardware does it: a stack of active masks, each
    branch reconverging at its immediate post-dominator, so lanes that take
    different paths meet again where the control flow does;
  - warp-collective instructions (mma.sync, elect.sync) on groups of 32 lanes;
  - asynchronous copies and mbarriers made synchronous: the copy happens when
    it is issued, and a wait always finds it done;
  - global memory rebuilt from the proxy's dumps; shared memory per block.

Numerics: e4m3 and f16 are decoded exactly. mma.sync computes each product and
sum in float64 and rounds once to f16 -- whether NVIDIA's hardware rounds the
same way is exactly what comparing against a real capture answers.
"""
import re
import struct
import numpy as np

# ---------------------------------------------------------------- formats

def _e4m3_table():
    values = np.zeros(256, dtype=np.float64)
    for code in range(256):
        sign = -1.0 if code & 0x80 else 1.0
        e, m = (code >> 3) & 0xF, code & 7
        if e == 15 and m == 7:
            values[code] = np.nan
        elif e == 0:
            values[code] = sign * (m / 8.0) * 2.0 ** -6
        else:
            values[code] = sign * (1 + m / 8.0) * 2.0 ** (e - 7)
    return values


E4M3 = _e4m3_table()
# The non-negative finite codes in increasing order: 0x00..0x7E.
E4M3_POS = E4M3[:0x7F]
E4M3_MIN_NORMAL = 2.0 ** -6


def f32_bits_to_f64(bits):
    """The f32 values of raw bits."""
    return (np.asarray(bits, np.uint64) & MASK32).astype(np.uint32).view(np.float32).astype(np.float64)


def f64_to_f32_bits(x):
    """float64 values to the bits of their f32 rounding (to nearest even)."""
    return np.asarray(x, np.float64).astype(np.float32).view(np.uint32).astype(np.uint64)


def f16_to_e4m3(x):
    """float64 values (already f16-exact) to e4m3 codes, RN-even, satfinite.

    The semantics of ZLUDA's routine, which in turn matches HIP's: NaN and
    infinity alike become 0x7F with the sign, anything beyond 448 saturates to
    448, and a tie goes to the even code.
    """
    x = np.asarray(x, dtype=np.float64)
    sign = np.where(np.signbit(x), 0x80, 0x00).astype(np.uint32)
    a = np.abs(x)
    special = ~np.isfinite(a)
    a = np.where(special, 0.0, a)
    a = np.minimum(a, 448.0)
    hi = np.searchsorted(E4M3_POS, a, side="left")          # first >= a
    hi = np.clip(hi, 0, 0x7E)
    lo = np.clip(hi - 1, 0, 0x7E)
    d_hi = np.abs(E4M3_POS[hi] - a)
    d_lo = np.abs(a - E4M3_POS[lo])
    pick_lo = (d_lo < d_hi) | ((d_lo == d_hi) & (lo % 2 == 0) & (lo != hi))
    code = np.where(pick_lo, lo, hi).astype(np.uint32)
    code = np.where(special, 0x7F, code)
    return code | sign


def f16_bits_to_f64(bits):
    return np.asarray(bits, dtype=np.uint16).view(np.float16).astype(np.float64)


def f64_to_f16_bits(values):
    return np.asarray(values, dtype=np.float64).astype(np.float16).view(np.uint16).astype(np.uint64)


def split_f16x2(v):
    v = np.asarray(v, dtype=np.uint64)
    return f16_bits_to_f64((v & 0xFFFF).astype(np.uint16)), f16_bits_to_f64((v >> 16).astype(np.uint16))


def pack_f16x2(lo, hi):
    return (f64_to_f16_bits(lo) | (f64_to_f16_bits(hi) << np.uint64(16))).astype(np.uint64)

# ---------------------------------------------------------------- parsing

SPECIAL = {"%tid.x", "%tid.y", "%tid.z", "%ctaid.x", "%ctaid.y", "%ctaid.z",
           "%nctaid.x", "%nctaid.y", "%nctaid.z",
           "%ntid.x", "%ntid.y", "%ntid.z", "%laneid"}


def split_top(text, sep=","):
    parts, depth, cur = [], 0, []
    for ch in text:
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        parts.append("".join(cur).strip())
    return parts


def statements(body):
    """Statements and labels, with scope braces dropped and vector braces kept."""
    out, i, n = [], 0, len(body)
    label = re.compile(r"\$[\w]+:")
    while i < n:
        c = body[i]
        if c.isspace() or c in "{}":
            i += 1
            continue
        m = label.match(body, i)
        if m:
            out.append(("label", m.group(0)[:-1]))
            i = m.end()
            continue
        depth, j = 0, i
        while j < n:
            ch = body[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                if depth == 0:
                    break
                depth -= 1
            elif ch == ";" and depth == 0:
                break
            j += 1
        out.append(("stmt", " ".join(body[i:j].split())))
        i = j + 1
    return out


class Instr:
    __slots__ = ("guard", "neg", "op", "parts", "args", "text", "target")

    def __repr__(self):
        return self.text


class Kernel:
    def __init__(self, text):
        text = re.sub(r"//[^\n]*", "", text)
        head_end = text.index("{", text.index(")"))
        header = text[:head_end]
        m = re.search(r"\.param\s+\.align\s+\d+\s+\.b8\s+(\w+)\[(\d+)\]", header)
        self.param_name, self.param_size = m.group(1), int(m.group(2))
        body = text[head_end + 1:text.rindex("}")]
        self.shared = {}          # name -> (offset, size)
        self.shared_size = 0
        self.instrs, self.labels = [], {}
        for kind, s in statements(body):
            if kind == "label":
                self.labels[s] = len(self.instrs)
                continue
            if s.startswith(".reg"):
                continue
            if s.startswith(".shared"):
                sm = re.match(r"\.shared\s+\.align\s+(\d+)\s+\.b8\s+(\S+?)\[(\d+)\]", s)
                align, name, size = int(sm.group(1)), sm.group(2), int(sm.group(3))
                off = (self.shared_size + align - 1) // align * align
                self.shared[name] = (off, size)
                self.shared_size = off + size
                continue
            self.instrs.append(self._parse(s))
        for ins in self.instrs:
            if ins.op in ("bra", "bra.uni"):
                ins.target = self.labels[ins.args[0][1]]
        self._post_dominators()

    def _operand(self, t):
        t = t.strip()
        if t.startswith("{"):
            return ("vec", [self._operand(x) for x in split_top(t[1:-1])])
        if t.startswith("["):
            inner = t[1:-1].replace(" ", "")
            m = re.match(r"^([^+]+)(?:\+(-?\w+))?$", inner)
            base, off = m.group(1), int(m.group(2), 0) if m.group(2) else 0
            return ("mem", self._operand(base), off)
        if "|" in t:
            a, b = t.split("|")
            return ("pair", self._operand(a), self._operand(b))
        if t == "_":
            return ("sink",)
        if t in SPECIAL:
            return ("sreg", t)
        if t.startswith("$"):
            return ("label", t)
        # Integer literals may carry a U or S suffix, as prmt's selectors do.
        lit = re.match(r"^-?(0x[0-9a-fA-F]+|\d+)[UuSs]?$", t)
        if lit:
            return ("imm", int(t.rstrip("UuSs"), 0))
        if re.match(r"^0f[0-9a-fA-F]{8}$", t):
            return ("imm", int(t[2:], 16))
        if t == self.param_name or t in self.shared:
            return ("sym", t)
        return ("reg", t)

    def _parse(self, s):
        ins = Instr()
        ins.text = s
        ins.guard, ins.neg, ins.target = None, False, None
        m = re.match(r"^@(!?)(%?\w+)\s+(.*)$", s)
        if m:
            ins.neg, ins.guard, s = m.group(1) == "!", m.group(2), m.group(3)
        op, _, rest = s.partition(" ")
        ins.op = op
        ins.parts = op.split(".")
        ins.args = [self._operand(a) for a in split_top(rest)] if rest.strip() else []
        return ins

    def _post_dominators(self):
        """Immediate post-dominator of every branch, as an instruction index.

        Basic blocks are cut at labels and after branches; the virtual exit is
        len(instrs). Classic iterative dataflow, small enough not to matter.
        """
        n = len(self.instrs)
        starts = sorted({0} | set(self.labels.values()) |
                        {i + 1 for i, x in enumerate(self.instrs)
                         if x.op in ("bra", "bra.uni", "ret") and i + 1 < n})
        starts = [s for s in starts if s < n]
        block_of = {}
        blocks = []
        for bi, s in enumerate(starts):
            e = starts[bi + 1] if bi + 1 < len(starts) else n
            blocks.append((s, e))
            for k in range(s, e):
                block_of[k] = bi
        EXIT = len(blocks)
        succ = []
        for bi, (s, e) in enumerate(blocks):
            last = self.instrs[e - 1]
            nxt = block_of.get(e, EXIT)
            if last.op == "ret":
                succ.append({EXIT})
            elif last.op in ("bra", "bra.uni"):
                t = block_of[last.target]
                succ.append({t} if (last.guard is None) else {t, nxt})
            else:
                succ.append({nxt})
        allb = set(range(EXIT + 1))
        pdom = {b: set(allb) for b in range(EXIT)}
        pdom[EXIT] = {EXIT}
        changed = True
        while changed:
            changed = False
            for b in range(EXIT - 1, -1, -1):
                new = set.intersection(*(pdom[s] for s in succ[b])) | {b}
                if new != pdom[b]:
                    pdom[b], changed = new, True
        ipdom = {}
        for b in range(EXIT):
            cands = pdom[b] - {b}
            # The immediate one is the candidate every other candidate post-dominates.
            for c in cands:
                if all(o in pdom[c] for o in cands):
                    ipdom[b] = c
                    break
        self.reconv = {}
        for bi, (s, e) in enumerate(blocks):
            last = e - 1
            if self.instrs[last].op in ("bra", "bra.uni"):
                r = ipdom.get(bi, EXIT)
                self.reconv[last] = blocks[r][0] if r < EXIT else n

# ---------------------------------------------------------------- memory

class GlobalMemory:
    """Disjoint regions rebuilt from dumps, overlapping windows merged."""

    def __init__(self):
        self.regions = []   # (base, np.uint8 array)
        self.misses = 0

    def add(self, base, data):
        """One more dump. Windows that overlap become one region: the dump of
        the input pointer, for one, runs on into the output, and the same
        address must not live in two places. Where two dumps cover the same
        byte, the one added later wins; dumps taken at the same moment agree
        there anyway."""
        pieces = self.regions + [(base, np.frombuffer(data, dtype=np.uint8).copy())]
        order = sorted(range(len(pieces)), key=lambda i: pieces[i][0])
        merged = []
        for i in order:
            b, d = pieces[i]
            if merged and b <= merged[-1][0] + len(merged[-1][1]):
                mb, md = merged[-1]
                hi = max(mb + len(md), b + len(d))
                out = np.zeros(hi - mb, dtype=np.uint8)
                out[:len(md)] = md
                # Later additions overwrite: the index says which came last.
                later = i == len(pieces) - 1
                if later:
                    out[b - mb:b - mb + len(d)] = d
                else:
                    tail = out[b - mb:b - mb + len(d)]
                    fill = np.arange(len(d)) + b >= mb + len(md)
                    tail[fill] = d[fill]
                merged[-1] = (mb, out)
            else:
                merged.append((b, d))
        self.regions = merged

    def _each(self, addr, nbytes):
        for base, data in self.regions:
            sel = (addr >= base) & (addr + nbytes <= base + len(data))
            if sel.any():
                yield sel, base, data

    def load(self, addr, nbytes):
        addr = addr.astype(np.int64)
        out = np.zeros((len(addr), nbytes), dtype=np.uint8)
        hit = np.zeros(len(addr), dtype=bool)
        for sel, base, data in self._each(addr, nbytes):
            idx = (addr[sel] - base)[:, None] + np.arange(nbytes)
            out[sel] = data[idx]
            hit |= sel
        self.misses += int((~hit).sum())
        return out

    def store(self, addr, raw):
        addr = addr.astype(np.int64)
        nbytes = raw.shape[1]
        hit = np.zeros(len(addr), dtype=bool)
        for sel, base, data in self._each(addr, nbytes):
            idx = (addr[sel] - base)[:, None] + np.arange(nbytes)
            data[idx] = raw[sel]
            hit |= sel
        self.misses += int((~hit).sum())

    def read(self, base, nbytes):
        return self.load(np.array([base], dtype=np.int64), nbytes)[0]


def bytes_to_words(raw, width):
    """(n, k*width) bytes -> list of k arrays of little-endian words."""
    n, total = raw.shape
    words = raw.reshape(n, total // width, width).astype(np.uint64)
    shifts = (np.arange(width, dtype=np.uint64) * np.uint64(8))
    vals = (words << shifts).sum(axis=2).astype(np.uint64)
    return [vals[:, i] for i in range(vals.shape[1])]


def words_to_bytes(words, width):
    arr = np.stack([np.asarray(w, dtype=np.uint64) for w in words], axis=1)
    shifts = (np.arange(width, dtype=np.uint64) * np.uint64(8))
    b = ((arr[:, :, None] >> shifts) & np.uint64(0xFF)).astype(np.uint8)
    return b.reshape(arr.shape[0], -1)

# ---------------------------------------------------------------- machine

MASK32 = np.uint64(0xFFFFFFFF)

# Fragment layouts of mma.m16n8k32 with 8-bit A/B and f16 C/D (PTX ISA). With
# g = lane >> 2 and t = lane & 3:
#   A, four .b32 of four bytes, element i = 4*reg + byte:
#       row = g + 8 for regs 1 and 3, g otherwise
#       col = 4t + byte, + 16 for regs 2 and 3
#   B, two .b32, element i = 4*reg + byte: k = 4t + byte (+16 for reg 1), n = g
#   C/D, two .f16x2, element j = 2*reg + half: row = g (+8 for reg 1),
#       col = 2t + half
LANE = np.arange(32)
G, T = LANE >> 2, LANE & 3


class Machine:
    def __init__(self, kernel, grid, block, params, memory, z_slice=None, blocks=None):
        """z_slice: run only the blocks with that ctaid.z (the grid still
        reports its full size). Kernels whose z slices wait for each other --
        a split K handed from one slice to the next through a flag -- cannot run
        in the lockstep this machine uses, so their slices are run in order.

        blocks: (first, last) of the flattened grid, for kernels with more
        blocks than fit in memory at once. Every block is independent of the
        others except through the flags, which a capture has already raised."""
        self.k = kernel
        self.grid, self.block = grid, block
        self.threads = block[0] * block[1] * block[2]
        planes = 1 if z_slice is not None else grid[2]
        total = grid[0] * grid[1] * planes
        first, last = blocks if blocks else (0, total)
        last = min(last, total)
        self.nblocks = max(0, last - first)
        self.nl = self.threads * self.nblocks
        lane = np.arange(self.nl)
        b, t = lane // self.threads + first, lane % self.threads
        self.block_of = lane // self.threads
        z = np.full(self.nl, z_slice) if z_slice is not None else b // (grid[0] * grid[1])
        self.sreg = {
            "%tid.x": t % block[0], "%tid.y": (t // block[0]) % block[1],
            "%tid.z": t // (block[0] * block[1]),
            "%ctaid.x": b % grid[0], "%ctaid.y": (b // grid[0]) % grid[1],
            "%ctaid.z": z,
            "%ntid.x": np.full(self.nl, block[0]), "%ntid.y": np.full(self.nl, block[1]),
            "%ntid.z": np.full(self.nl, block[2]), "%laneid": t % 32,
            "%nctaid.x": np.full(self.nl, grid[0]), "%nctaid.y": np.full(self.nl, grid[1]),
            "%nctaid.z": np.full(self.nl, grid[2]),
        }
        self.sreg = {k: v.astype(np.uint64) for k, v in self.sreg.items()}
        self.regs = {}
        self.vec128 = {}
        self.params = params
        self.mem = memory
        self.shared = np.zeros((self.nblocks, max(kernel.shared_size, 16)), dtype=np.uint8)
        self.steps = 0
        self.spin_guard = {}

    # ----- operands
    def reg(self, name):
        r = self.regs.get(name)
        if r is None:
            r = np.zeros(self.nl, dtype=np.uint64)
            self.regs[name] = r
        return r

    def val(self, o):
        kind = o[0]
        if kind == "reg":
            return self.reg(o[1])
        if kind == "imm":
            return np.full(self.nl, np.uint64(o[1] & 0xFFFFFFFFFFFFFFFF), dtype=np.uint64)
        if kind == "sreg":
            return self.sreg[o[1]]
        if kind == "sym":
            off = self.k.shared[o[1]][0] if o[1] in self.k.shared else 0
            return np.full(self.nl, np.uint64(off), dtype=np.uint64)
        raise ValueError("cannot read %r" % (o,))

    def put(self, o, value, mask, width=32):
        if o[0] == "sink":
            return
        v = np.asarray(value, dtype=np.uint64)
        if width < 64:
            v = v & np.uint64((1 << width) - 1)
        r = self.reg(o[1])
        r[mask] = v[mask] if v.ndim else v

    # ----- helpers
    @staticmethod
    def width(ty):
        return {"b16": 16, "u16": 16, "s16": 16, "f16": 16, "b32": 32, "u32": 32, "s32": 32,
                "f32": 32, "f16x2": 32, "b64": 64, "u64": 64, "s64": 64, "pred": 1}[ty]

    @staticmethod
    def signed(v, width):
        v = np.asarray(v, dtype=np.uint64)
        if width == 64:
            return v.view(np.int64)
        v = v & np.uint64((1 << width) - 1)
        s = v.astype(np.int64)
        return np.where(s >= (1 << (width - 1)), s - (1 << width), s)

    def addr(self, mem_operand):
        _, base, off = mem_operand
        if base[0] == "sym":
            if base[1] == self.k.param_name:
                return ("param", off)
            return self.val(base) + np.uint64(off & 0xFFFFFFFFFFFFFFFF)
        return (self.val(base).astype(np.int64) + off).astype(np.uint64)

    def shared_load(self, addr, nbytes):
        idx = addr.astype(np.int64)[:, None] + np.arange(nbytes)
        return self.shared[self.block_of[:, None], idx]

    def shared_store(self, addr, raw, mask):
        lanes = np.nonzero(mask)[0]
        idx = addr[lanes].astype(np.int64)[:, None] + np.arange(raw.shape[1])
        self.shared[self.block_of[lanes][:, None], idx] = raw[lanes]

    # ----- the run
    def run(self, max_steps=5_000_000):
        n = len(self.k.instrs)
        stack = [[0, np.ones(self.nl, dtype=bool), n]]
        while stack:
            top = stack[-1]
            pc, mask, rpc = top
            if pc == rpc or not mask.any():
                stack.pop()
                continue
            ins = self.k.instrs[pc]
            self._pc = pc
            self.steps += 1
            if self.steps > max_steps:
                raise RuntimeError("step limit at %d: %s" % (pc, ins.text))
            active = mask
            if ins.guard is not None:
                p = self.reg(ins.guard).astype(bool)
                active = mask & (~p if ins.neg else p)
            if ins.op in ("bra", "bra.uni"):
                taken, fall = active, mask & ~active
                if not fall.any():
                    self._spin_check(pc, ins)
                    top[0] = ins.target
                elif not taken.any():
                    top[0] = pc + 1
                else:
                    r = self.k.reconv[pc]
                    top[0] = r
                    stack.append([pc + 1, fall, r])
                    stack.append([ins.target, taken, r])
                continue
            if ins.op == "ret":
                for entry in stack:
                    entry[1] = entry[1] & ~active
                top[0] = pc + 1
                continue
            if active.any():
                self.execute(ins, active)
            top[0] = pc + 1

    def _spin_check(self, pc, ins):
        # A loop that never lets go is a flag nobody in this capture raises.
        count = self.spin_guard.get(pc, 0) + 1
        self.spin_guard[pc] = count
        if count > 200000:
            raise RuntimeError("stuck in a loop at %s" % ins.text)

    def execute(self, ins, m):
        op, parts, a = ins.op, ins.parts, ins.args
        head = parts[0]
        ty = parts[-1]

        if head == "mov":
            if ty == "b128":
                self.vec128[a[0][1]] = [self.val(x) for x in a[1][1]]
                return
            src = a[1]
            if a[0][0] == "vec":
                # Unpacking: the pieces of one register into several.
                n = len(a[0][1])
                w = self.width(ty) // n
                v = self.val(src)
                for i, d in enumerate(a[0][1]):
                    self.put(d, (v >> np.uint64(i * w)) & np.uint64((1 << w) - 1), m, w)
                return
            if src[0] == "vec":
                w = 32 // len(src[1]) if ty == "b32" else 64 // len(src[1])
                v = np.zeros(self.nl, dtype=np.uint64)
                for i, e in enumerate(src[1]):
                    v |= (self.val(e) & np.uint64((1 << w) - 1)) << np.uint64(i * w)
                self.put(a[0], v, m, self.width(ty))
            else:
                self.put(a[0], self.val(src), m, 1 if ty == "pred" else self.width(ty))
            return

        if head == "ld" and parts[1] == "param":
            _, off = self.addr(a[1])
            if a[0][0] == "vec":
                w = self.width(ty)
                for i, d in enumerate(a[0][1]):
                    raw = self.params[off + i * w // 8: off + (i + 1) * w // 8]
                    self.put(d, np.uint64(int.from_bytes(raw, "little")), m, w)
            else:
                w = self.width(ty)
                raw = self.params[off: off + w // 8]
                self.put(a[0], np.uint64(int.from_bytes(raw, "little")), m, w)
            return

        if head in ("ld", "st"):
            space = "shared" if any(p.startswith("shared") for p in parts) else "global"
            if head == "st" and ty == "b128":
                vals = self.vec128[a[1][1]]
                addr = self.addr(a[0])
                raw = words_to_bytes(vals, 4)
                self._store(space, addr, raw, m)
                return
            w = self.width(ty) // 8
            count = 4 if "v4" in parts else 2 if "v2" in parts else 1
            if head == "ld":
                addr = self.addr(a[1])
                raw = self._load(space, addr, w * count, m)
                words = bytes_to_words(raw, w)
                dests = a[0][1] if a[0][0] == "vec" else [a[0]]
                for d, v in zip(dests, words):
                    self.put(d, v, m, w * 8)
            else:
                addr = self.addr(a[0])
                srcs = a[1][1] if a[1][0] == "vec" else [a[1]]
                raw = words_to_bytes([self.val(s) for s in srcs], w)
                self._store(space, addr, raw, m)
            return

        if head == "cvta":
            self.put(a[0], self.val(a[1]), m, 64)
            return

        if head in ("add", "sub", "mul", "mad", "and", "or", "xor", "shl", "shr", "min", "max", "not",
                    "div", "rem"):
            if ty == "pred":
                x = self.val(a[1]).astype(bool)
                if head == "not":
                    r = ~x
                else:
                    y = self.val(a[2]).astype(bool)
                    r = (x | y) if head == "or" else (x & y) if head == "and" else (x ^ y)
                self.put(a[0], r.astype(np.uint64), m, 1)
                return
            if ty == "f16x2":
                self._f16x2(head, a, m)
                return
            w = self.width(ty)
            sgn = ty.startswith("s")
            x = self.val(a[1])
            if head == "not":
                self.put(a[0], ~x, m, w)
                return
            y = self.val(a[2])
            if head == "mul" and "wide" in parts:
                xs = self.signed(x, w) if sgn else (x & np.uint64((1 << w) - 1)).astype(np.int64)
                ys = self.signed(y, w) if sgn else (y & np.uint64((1 << w) - 1)).astype(np.int64)
                self.put(a[0], (xs * ys).astype(np.int64).view(np.uint64), m, 2 * w)
                return
            xs, ys = self.signed(x, w), self.signed(y, w)
            if head == "add":
                r = xs + ys
            elif head == "sub":
                r = xs - ys
            elif head == "mul":
                r = xs * ys
            elif head == "mad":
                r = xs * ys + self.signed(self.val(a[3]), w)
            elif head in ("div", "rem"):
                # C semantics: the quotient truncates toward zero. A zero
                # divisor is undefined in PTX; it yields 0 here.
                if not sgn:
                    xs = (x & np.uint64((1 << w) - 1)).astype(np.int64)
                    ys = (y & np.uint64((1 << w) - 1)).astype(np.int64)
                safe = np.where(ys == 0, 1, ys)
                q = np.sign(xs) * np.sign(safe) * (np.abs(xs) // np.abs(safe))
                q = np.where(ys == 0, 0, q)
                r = q if head == "div" else np.where(ys == 0, 0, xs - q * safe)
            elif head in ("min", "max"):
                if sgn:
                    r = np.minimum(xs, ys) if head == "min" else np.maximum(xs, ys)
                else:
                    ux, uy = x & np.uint64((1 << w) - 1), y & np.uint64((1 << w) - 1)
                    self.put(a[0], np.minimum(ux, uy) if head == "min" else np.maximum(ux, uy), m, w)
                    return
            elif head in ("and", "or", "xor"):
                r = (x & y) if head == "and" else (x | y) if head == "or" else (x ^ y)
                self.put(a[0], r, m, w)
                return
            elif head == "shl":
                self.put(a[0], x << (y & np.uint64(63)), m, w)
                return
            elif head == "shr":
                s = (y & np.uint64(63)).astype(np.int64)
                if sgn:
                    r = xs >> s
                else:
                    self.put(a[0], (x & np.uint64((1 << w) - 1)) >> s.astype(np.uint64), m, w)
                    return
            self.put(a[0], r.astype(np.int64).view(np.uint64), m, w)
            return

        if head in ("abs", "fma") and ty == "f16x2":
            self._f16x2(head, a, m)
            return

        if head == "setp" and parts[-1] in ("f32", "f16"):
            # Float comparisons, including the unordered forms; NaN is false
            # for the ordered ones and true for "u" ones.
            cmp = parts[1]
            conv = f32_bits_to_f64 if parts[-1] == "f32" else f16_bits_to_f64
            x, y = conv(self.val(a[1])), conv(self.val(a[2]))
            nan = np.isnan(x) | np.isnan(y)
            base = {"eq": x == y, "ne": x != y, "lt": x < y, "le": x <= y, "gt": x > y, "ge": x >= y,
                    "equ": x == y, "neu": x != y, "ltu": x < y, "leu": x <= y, "gtu": x > y,
                    "geu": x >= y, "num": ~nan, "nan": nan}[cmp]
            if cmp.endswith("u") and cmp != "num":
                base = base | nan
            self.put(a[0], base.astype(np.uint64), m, 1)
            return

        if head in ("add", "sub", "mul", "fma", "min", "max", "neg", "abs", "rcp", "rsqrt", "sqrt",
                    "ex2", "lg2", "div", "mad") and parts[-1] in ("f32", "f16"):
            # One f32 (or f16) value per lane. The approximate instructions are
            # computed exactly here: what the hardware's approximation does is a
            # property of the card, checked against captures, not of this model.
            wide = parts[-1] == "f32"
            conv = f32_bits_to_f64 if wide else f16_bits_to_f64
            back = f64_to_f32_bits if wide else f64_to_f16_bits
            x = conv(self.val(a[1]))
            if head in ("neg", "abs", "rcp", "rsqrt", "sqrt", "ex2", "lg2"):
                r = {"neg": lambda v: -v, "abs": np.abs, "rcp": lambda v: 1.0 / v,
                     "rsqrt": lambda v: 1.0 / np.sqrt(v), "sqrt": np.sqrt,
                     "ex2": np.exp2, "lg2": np.log2}[head](x)
            else:
                y = conv(self.val(a[2]))
                if head in ("fma", "mad"):
                    r = x * y + conv(self.val(a[3]))
                else:
                    r = {"add": lambda: x + y, "sub": lambda: x - y, "mul": lambda: x * y,
                         "div": lambda: x / y, "min": lambda: np.fmin(x, y),
                         "max": lambda: np.fmax(x, y)}[head]()
            if "ftz" in parts:
                tiny = np.abs(r) < (2.0 ** -126 if wide else 2.0 ** -14)
                r = np.where(tiny & np.isfinite(r), np.copysign(0.0, r), r)
            self.put(a[0], back(r), m, 32 if wide else 16)
            return

        if head == "setp":
            cmp, t = parts[1], parts[2]
            w = self.width(t)
            if t.startswith("s"):
                x, y = self.signed(self.val(a[1]), w), self.signed(self.val(a[2]), w)
            else:
                mk = np.uint64((1 << w) - 1) if w < 64 else np.uint64(0xFFFFFFFFFFFFFFFF)
                x, y = self.val(a[1]) & mk, self.val(a[2]) & mk
            r = {"eq": x == y, "ne": x != y, "lt": x < y, "le": x <= y,
                 "gt": x > y, "ge": x >= y}[cmp]
            self.put(a[0], r.astype(np.uint64), m, 1)
            return

        if head == "selp":
            p = self.val(a[3]).astype(bool)
            self.put(a[0], np.where(p, self.val(a[1]), self.val(a[2])), m, self.width(ty))
            return

        if head == "cvt":
            self._cvt(ins, a, m)
            return

        if head == "mma":
            self._mma(a, m)
            return

        if head == "prmt":
            # Byte permute: four selectors of four bits each pick bytes of the
            # concatenation {b, a}; bit 3 of a selector replicates the byte's
            # sign instead (the "msb" mode of the same instruction).
            x, y, sel = self.val(a[1]), self.val(a[2]), self.val(a[3])
            src = x | (y << np.uint64(32))
            out = np.zeros(self.nl, dtype=np.uint64)
            msb = "msb" in parts
            for i in range(4):
                s4 = (sel >> np.uint64(4 * i)) & np.uint64(0xF)
                idx = s4 & np.uint64(7)
                byte = (src >> (idx * np.uint64(8))) & np.uint64(0xFF)
                if msb:
                    byte = np.where((s4 & np.uint64(8)) != 0, np.where(byte >= 0x80, 0xFF, 0x00), byte)
                else:
                    byte = np.where((s4 & np.uint64(8)) != 0, np.where(byte >= 0x80, 0xFF, 0x00), byte)
                out |= byte.astype(np.uint64) << np.uint64(8 * i)
            self.put(a[0], out, m, 32)
            return

        if head == "movmatrix":
            # movmatrix.sync.trans.aligned.m8n8.b16: the warp holds an 8x8
            # matrix of 16-bit values, two consecutive columns of one row per
            # lane (lane l: row l/4, columns 2(l%4) and +1), and gets the
            # transpose back in the same layout.
            v = self.val(a[1])
            lanes = np.arange(self.nl)
            warp, l = lanes // 32, lanes % 32
            lo = v & np.uint64(0xFFFF)
            hi = (v >> np.uint64(16)) & np.uint64(0xFFFF)
            mat = np.zeros((self.nl // 32, 8, 8), dtype=np.uint64)
            mat[warp, l // 4, 2 * (l % 4)] = lo
            mat[warp, l // 4, 2 * (l % 4) + 1] = hi
            t = np.transpose(mat, (0, 2, 1))
            out = t[warp, l // 4, 2 * (l % 4)] | (t[warp, l // 4, 2 * (l % 4) + 1] << np.uint64(16))
            self.put(a[0], out, m, 32)
            return

        if head == "shfl":
            # shfl.sync.<mode>.b32 d[|p], a, b, c, membermask. The lane a value
            # comes from, and whether it is in range, follow the PTX definition;
            # out-of-range lanes keep their own value.
            mode = parts[2]
            dest = [a[0][1], a[0][2]] if a[0][0] == "pair" else [a[0]]
            src = self.val(a[1])
            bval = (self.val(a[2]) & np.uint64(0x1F)).astype(np.int64)
            c = self.val(a[3]).astype(np.int64)
            cval, segmask = c & 0x1F, (c >> 8) & 0x1F
            lane = (np.arange(self.nl) % 32).astype(np.int64)
            maxlane = (lane & segmask) | (cval & ~segmask & 0x1F)
            minlane = lane & segmask
            if mode == "up":
                j = lane - bval
                ok = j >= maxlane
            elif mode == "down":
                j = lane + bval
                ok = j <= maxlane
            elif mode == "bfly":
                j = lane ^ bval
                ok = j <= maxlane
            else:                                     # idx
                j = minlane | (bval & ~segmask & 0x1F)
                ok = j <= maxlane
            j = np.where(ok, j, lane)
            base = (np.arange(self.nl) // 32) * 32
            picked = src[base + j]
            self.put(dest[0], picked, m, 32)
            if len(dest) > 1:
                self.put(dest[1], ok.astype(np.uint64), m, 1)
            return

        if head == "elect":
            # One lane per warp, the lowest active one.
            lanes = np.nonzero(m)[0]
            leader = np.zeros(self.nl, dtype=np.uint64)
            warps = lanes // 32
            first = np.unique(warps, return_index=True)[1]
            leader[lanes[first]] = 1
            dest = a[0][2] if a[0][0] == "pair" else a[0]
            self.put(dest, leader, m, 1)
            return

        if head == "cp":
            # cp.async.bulk shared <- global, done on the spot. It is traced
            # like a load: a kernel that stages its inputs this way reads
            # nothing else from global, and the trace would say it reads no
            # activations at all.
            dst = self.addr(a[0])
            src = self.addr(a[1])
            size = int(self.val(a[2])[np.nonzero(m)[0][0]])
            self._note("bulk", src, size, m)
            raw = self.mem.load(src[m], size)
            full = np.zeros((self.nl, size), dtype=np.uint8)
            full[m] = raw
            self.shared_store(dst, full, m)
            return

        if head == "mbarrier":
            kind = parts[1]
            if kind == "try_wait":
                self.put(a[0], np.ones(self.nl, dtype=np.uint64), m, 1)
            elif kind == "arrive":
                self.put(a[0], np.zeros(self.nl, dtype=np.uint64), m, 64)
            return

        if head == "red" and "f16x2" in parts and "add" in parts:
            # Atomic f16x2 add into global memory: each lane's words added to
            # what is there, rounded to f16. Lanes of one instruction hitting the
            # same word would have to be summed in hardware order; that never
            # happens in the kernels this runs, and it is checked.
            count = 4 if "v4" in parts else 2 if "v2" in parts else 1
            addr = self.addr(a[0])
            srcs = a[1][1] if a[1][0] == "vec" else [a[1]]
            lanes = np.nonzero(m)[0]
            if len(np.unique(addr[lanes])) != len(lanes):
                raise NotImplementedError("red with lanes sharing an address: " + ins.text)
            old_raw = self.mem.load(addr[lanes], 4 * count)
            old_words = bytes_to_words(old_raw, 4)
            new = []
            for i, src in enumerate(srcs):
                xl, xh = split_f16x2(old_words[i])
                yl, yh = split_f16x2(self.val(src)[lanes])
                new.append(pack_f16x2(xl + yl, xh + yh))
            self.mem.store(addr[lanes], words_to_bytes(new, 4))
            return

        # Directives (".pragma") split to an empty head; they are hints, not work.
        if head in ("bar", "nanosleep", "membar", "fence", "prefetch", "discard", ""):
            return

        raise NotImplementedError(ins.text)

    # Set to a list to record every global access as
    # (kind, first address, bytes per lane, lanes): what a kernel reads and
    # writes, which is how a layer's layout is read off rather than guessed.
    trace = None

    def _note(self, kind, addr, nbytes, m):
        if self.trace is None:
            return
        lanes = np.nonzero(m)[0]
        if len(lanes):
            self.trace.append((kind, int(addr[lanes[0]]), nbytes, len(lanes),
                               int(addr[lanes].min()), int(addr[lanes].max())))

    def _load(self, space, addr, nbytes, m):
        if space == "shared":
            return self.shared_load(addr, nbytes)
        self._note("read", addr, nbytes, m)
        out = np.zeros((self.nl, nbytes), dtype=np.uint8)
        out[m] = self.mem.load(addr[m], nbytes)
        return out

    def _store(self, space, addr, raw, m):
        if space == "shared":
            self.shared_store(addr, raw, m)
        else:
            self._note("write", addr, raw.shape[1], m)
            self.mem.store(addr[m], raw[m])

    def _f16x2(self, head, a, m):
        xl, xh = split_f16x2(self.val(a[1]))
        if head == "abs":
            v = self.val(a[1]) & np.uint64(0x7FFF7FFF)
            self.put(a[0], v, m, 32)
            return
        yl, yh = split_f16x2(self.val(a[2]))
        if head == "min":
            rl, rh = np.fmin(xl, yl), np.fmin(xh, yh)
        elif head == "max":
            rl, rh = np.fmax(xl, yl), np.fmax(xh, yh)
        elif head == "mul":
            rl, rh = xl * yl, xh * yh
        elif head == "add":
            rl, rh = xl + yl, xh + yh
        elif head == "sub":
            rl, rh = xl - yl, xh - yh
        elif head == "fma":
            zl, zh = split_f16x2(self.val(a[3]))
            if self.fma_mode == "unfused":
                # The product rounded to f16 before the add: what a lowering
                # that splits the instruction in two computes.
                pl = xl * yl
                ph = xh * yh
                pl = pl.astype(np.float16).astype(np.float64)
                ph = ph.astype(np.float16).astype(np.float64)
                rl, rh = pl + zl, ph + zh
            else:
                rl, rh = xl * yl + zl, xh * yh + zh
        else:
            raise NotImplementedError(head)
        self.put(a[0], pack_f16x2(rl, rh), m, 32)

    def _cvt(self, ins, a, m):
        parts = ins.parts
        to, frm = parts[-2], parts[-1]
        src = self.val(a[1])
        if to == "e4m3x2" and frm == "f16x2":
            lo, hi = split_f16x2(src)
            r = f16_to_e4m3(lo) | (f16_to_e4m3(hi) << np.uint32(8))
            self.put(a[0], r.astype(np.uint64), m, 16)
        elif to == "f16x2" and frm == "e4m3x2":
            lo = E4M3[(src & np.uint64(0xFF)).astype(np.int64)]
            hi = E4M3[((src >> np.uint64(8)) & np.uint64(0xFF)).astype(np.int64)]
            self.put(a[0], pack_f16x2(lo, hi), m, 32)
        elif to == "f32" and frm == "f16":
            self.put(a[0], f64_to_f32_bits(f16_bits_to_f64(src)), m, 32)
        elif to == "f16" and frm == "f32":
            f = (src & MASK32).astype(np.uint32).view(np.float32).astype(np.float64)
            self.put(a[0], f64_to_f16_bits(f), m, 16)
        elif to == "f32" and frm in ("s32", "u32"):
            v = self.signed(src, 32) if frm == "s32" else (src & MASK32).astype(np.int64)
            self.put(a[0], f64_to_f32_bits(v.astype(np.float64)), m, 32)
        elif to[0] in "usb" and frm[0] in "usb" and to[1:].isdigit() and frm[1:].isdigit():
            # Integer to integer: the source read at its width, signed or not,
            # then truncated to the destination's.
            fw, tw = int(frm[1:]), int(to[1:])
            v = self.signed(src, fw).view(np.uint64) if frm[0] == "s" else src & np.uint64((1 << fw) - 1)
            self.put(a[0], v & np.uint64((1 << tw) - 1 if tw < 64 else 0xFFFFFFFFFFFFFFFF), m, tw)
        else:
            raise NotImplementedError(ins.text)

    # How the products of one mma are summed. Every product is exact in any of
    # these (e4m3 x e4m3 fits an f16's mantissa twice over); what differs is
    # where the sum is rounded, which is the one thing a capture from real
    # hardware can decide.
    #
    #   exact   one rounding, to f16, of the true sum -- the PTX semantics taken
    #           literally
    #   f32x2   two k=16 halves, each summed exactly and rounded to f32, the
    #           second starting from the first: what ZLUDA's gfx11 path asks of
    #           two f32-accumulating WMMAs, if the hardware rounds ideally
    #   f32seq  one f32 rounding after every product, in k order
    mma_mode = "exact"

    # fma.rn.f16x2 is one rounding by definition ("fused"); "unfused" rounds
    # the product first, to test whether the hardware path does.
    fma_mode = "fused"

    # How each mma's result is rounded to f16: "rn" to nearest even, as the
    # PTX says; "rz" toward zero, as a device left in that rounding mode would.
    mma_round = "rn"

    # For mma_mode "align": window width in bits, and how the excess is dropped.
    mma_align_bits = 24
    mma_align_trunc = "rz"

    # ZLUDA's gfx11 path widens e4m3 through an f16 multiply that flushes
    # denormals, so e4m3 subnormal inputs reach its WMMA as zero. True follows
    # ZLUDA there; False follows the PTX semantics.
    mma_flush_subnormal = False

    def _accumulate(self, A, B, C):
        P = A[:, :, :, None] * B[:, None, :, :]            # (W, 16, 32, 8)
        if self.mma_mode == "exact":
            return C + P.sum(axis=2)
        if self.mma_mode == "f32x2":
            d = (C + P[:, :, :16].sum(axis=2)).astype(np.float32).astype(np.float64)
            return (d + P[:, :, 16:].sum(axis=2)).astype(np.float32).astype(np.float64)
        if self.mma_mode == "f32seq":
            d = C.astype(np.float32)
            for k in range(32):
                d = (d + P[:, :, k].astype(np.float32)).astype(np.float32)
            return d.astype(np.float64)
        if self.mma_mode == "align":
            # Matrix units commonly sum by aligning every term to the largest
            # exponent in the group and dropping the bits that fall off a
            # fixed-width window, rather than rounding each addition. Two k=16
            # groups, as ZLUDA issues two WMMAs; the accumulator is one of the
            # terms of each; the window is mma_align_bits wide below the
            # largest term's leading bit, and what falls off is truncated
            # toward zero ("rz") or toward minus infinity ("rd").
            d = C
            for half in (slice(0, 16), slice(16, 32)):
                terms = np.concatenate([d[:, :, None, :], P[:, :, half, :]], axis=2)
                biggest = np.max(np.abs(terms), axis=2, keepdims=True)
                _, e = np.frexp(np.where(biggest > 0, biggest, 1.0))
                quantum = np.ldexp(1.0, e - self.mma_align_bits)
                q = terms / quantum
                q = np.trunc(q) if self.mma_align_trunc == "rz" else np.floor(q)
                d = (q * quantum).sum(axis=2).astype(np.float32).astype(np.float64)
            return d
        if self.mma_mode == "f16x2":
            # Two k=16 halves, the running sum rounded to f16 after the first.
            d = (C + P[:, :, :16].sum(axis=2)).astype(np.float16).astype(np.float64)
            return d + P[:, :, 16:].sum(axis=2)
        if self.mma_mode == "f16x4":
            d = C
            for q in range(4):
                d = (d + P[:, :, 8 * q:8 * q + 8].sum(axis=2)).astype(np.float16).astype(np.float64)
            return d
        if self.mma_mode == "f16seq":
            d = C
            for k in range(32):
                d = (d + P[:, :, k]).astype(np.float16).astype(np.float64)
            return d
        raise ValueError(self.mma_mode)

    # Set to a list to keep every mma's operands and result as matrices. The
    # registers cannot be read afterwards -- a kernel reuses them, so what is
    # left at the end belongs to some later multiply -- and this is what makes
    # a stage's inputs checkable against a reference.
    mma_log = None

    def _mma(self, a, m):
        if self.trace is not None:
            self.trace.append(("mma", 0, 0, int(m.sum()), 0, 0))
        dregs, aregs, bregs, cregs = (x[1] for x in a)
        lanes = np.nonzero(m)[0]
        if len(lanes) % 32:
            raise RuntimeError("mma.sync on a partial warp")
        base = lanes.reshape(-1, 32)[:, 0]                 # first lane of each warp
        idx = base[:, None] + LANE[None, :]                # (W, 32)
        W = len(base)
        A = np.zeros((W, 16, 32))
        for r, o in enumerate(aregs):
            v = self.val(o)[idx]                           # (W, 32)
            for byte in range(4):
                code = ((v >> np.uint64(8 * byte)) & np.uint64(0xFF)).astype(np.int64)
                row = G + (8 if r in (1, 3) else 0)
                col = 4 * T + byte + (16 if r >= 2 else 0)
                A[:, row, col] = E4M3[code]
        B = np.zeros((W, 32, 8))
        for r, o in enumerate(bregs):
            v = self.val(o)[idx]
            for byte in range(4):
                code = ((v >> np.uint64(8 * byte)) & np.uint64(0xFF)).astype(np.int64)
                k = 4 * T + byte + (16 if r == 1 else 0)
                B[:, k, G] = E4M3[code]
        C = np.zeros((W, 16, 8))
        for r, o in enumerate(cregs):
            lo, hi = split_f16x2(self.val(o)[idx])
            row = G + (8 if r == 1 else 0)
            C[:, row, 2 * T] = lo
            C[:, row, 2 * T + 1] = hi
        if self.mma_flush_subnormal:
            A[np.abs(A) < E4M3_MIN_NORMAL] = 0.0
            B[np.abs(B) < E4M3_MIN_NORMAL] = 0.0
        D = self._accumulate(A, B, C)
        if self.mma_log is not None:
            self.mma_log.append({"pc": getattr(self, "_pc", -1), "lanes": lanes.copy(),
                                 "a": A.copy(), "b": B.copy(), "c": C.copy(), "d": D.copy()})
        if self.mma_round in ("rna", "tie_stats"):
            # Ties: the exact result sitting exactly halfway between two f16.
            # Products of e4m3 carry few bits, so this is not rare.
            h = D.astype(np.float16)
            hv = h.astype(np.float64)
            other = np.nextafter(h, np.where(D > hv, np.float16(np.inf), np.float16(-np.inf))).astype(np.float64)
            tie = (D != hv) & (np.abs(D - hv) == np.abs(other - D))
            self.ties = getattr(self, "ties", 0) + int(tie.sum())
            self.rounded = getattr(self, "rounded", 0) + D.size
            if self.mma_round == "rna":
                away = np.where(np.abs(other) > np.abs(hv), other, hv)
                D = np.where(tie, away, hv)
        if self.mma_round == "rz":
            # Toward zero instead of to nearest: where rounding to nearest went
            # up in magnitude, step one f16 back toward zero.
            h = D.astype(np.float16)
            over = np.abs(h.astype(np.float64)) > np.abs(D)
            h = np.where(over, np.nextafter(h, np.float16(0)), h)
            D = h.astype(np.float64)
        for r, o in enumerate(dregs):
            row = G + (8 if r == 1 else 0)
            packed = pack_f16x2(D[:, row, 2 * T], D[:, row, 2 * T + 1])   # (W, 32)
            full = self.reg(o[1])
            full[idx] = packed
