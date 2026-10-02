#!/usr/bin/env python3
"""Variants of the ffwd kernel that write an intermediate instead of the result.

The only thing a capture shows is the final output: 32 f16 accumulators per
lane, each narrowed to e4m3. That is too compressed to say where the emulator
and the hardware part ways. These variants write the accumulators themselves,
raw f16, into the same slots -- the slots hold 16 registers a lane, so two
variants cover the 32:

  acc_a   the first register of every packed pair
  acc_b   the second

The rewrite is textual and small: every `cvt.rn.satfinite.e4m3x2.f16x2 %rsX, %rY`
names which accumulator %rsX came from, and the sixteen `mov.b32 %rZ, {%rsX,
%rsW}` that pack the output are replaced by `mov.b32 %rZ, %rY`.

Usage: ffwd_variants.py            writes ffwd_acc_a.ptx and ffwd_acc_b.ptx
"""
import os
import re
import workspace  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def make(module_text, which):
    source = dict(re.findall(r"cvt\.rn\.satfinite\.e4m3x2\.f16x2\s+(%rs\d+),\s*(%r\d+);",
                             module_text))
    # The packing movs are the ones whose two halves were both narrowed from
    # accumulators; the output stores read them.
    pack = re.compile(r"mov\.b32\s+(%r\d+),\s*\{(%rs\d+),\s*(%rs\d+)\};")
    replaced = 0

    def swap(m):
        nonlocal replaced
        dst, lo, hi = m.group(1), m.group(2), m.group(3)
        if lo not in source or hi not in source:
            return m.group(0)
        replaced += 1
        return "mov.b32 %s, %s;" % (dst, source[lo] if which == "a" else source[hi])

    # Only the packing just before the stores: after the last narrowing.
    last_cvt = max(m.end() for m in re.finditer(r"cvt\.rn\.satfinite\.e4m3x2\.f16x2", module_text))
    head, tail = module_text[:last_cvt], module_text[last_cvt:]
    tail = pack.sub(swap, tail)
    return head + tail + "\n// variant acc_%s\n" % which, replaced


def main():
    text = open(os.path.join(workspace.captures(), "ffwd_module.ptx"), encoding="utf-8").read()
    for which in ("a", "b"):
        out, n = make(text, which)
        path = os.path.join(workspace.captures(), "ffwd_acc_%s.ptx" % which)
        open(path, "w", encoding="utf-8", newline="\n").write(out)
        print("%s: %d packing movs replaced" % (os.path.basename(path), n))


if __name__ == "__main__":
    main()
