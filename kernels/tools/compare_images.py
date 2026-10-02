#!/usr/bin/env python3
"""How far apart two frames are, when "identical" is the wrong question.

Bit-for-bit equality is the right test for the gate -- a refused native kernel
must leave the translation's output untouched, to the byte -- but it is the
wrong test for a native kernel that actually ran. A reimplementation does not
reproduce NVIDIA's accumulation order, and f16 rounding is not associative, so
a correct kernel still lands a few least-significant bits away. Demanding the
same sha256 would reject every working kernel, and accepting any difference at
all would hide a broken one.

So the question becomes how far, and the answer is a number: maximum deviation
on any channel, mean deviation, how many pixels moved at all, and PSNR.

Rules of thumb for 8-bit channels, from what the difference means rather than
from convention:

    max 0            the same bytes; nothing to discuss
    max 1-2          rounding, invisible: a native kernel is behaving
    max 3-8          suspicious; correct in structure, wrong in detail
                     (a saturation, a bias, one tile edge)
    max > 8          a different computation, whatever the mean says

The maximum matters more than the mean: a single wrong tile in a corner is a
bug, and a mean taken over two million pixels will not show it. That is why the
worst pixel is located and printed, not just counted.

Usage: compare_images.py <reference.png> <candidate.png> [--max-allowed 2]
"""
import argparse
import sys

try:
    from PIL import Image
except ImportError:
    Image = None
try:
    import numpy as np
except ImportError:
    np = None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--max-allowed", type=int, default=2,
                    help="largest per-channel deviation still called a pass")
    args = ap.parse_args()

    if Image is None or np is None:
        print("needs Pillow and numpy: pip install pillow numpy")
        return 3

    a = np.asarray(Image.open(args.reference).convert("RGB"), dtype=np.int16)
    b = np.asarray(Image.open(args.candidate).convert("RGB"), dtype=np.int16)
    if a.shape != b.shape:
        print("different sizes: %s against %s" % (a.shape, b.shape))
        return 2

    diff = np.abs(a - b)
    worst = int(diff.max())
    mean = float(diff.mean())
    moved = int((diff.any(axis=2)).sum())
    total = a.shape[0] * a.shape[1]

    print("%s  against  %s" % (args.reference, args.candidate))
    print("   size             %dx%d" % (a.shape[1], a.shape[0]))
    print("   max deviation    %d" % worst)
    print("   mean deviation   %.4f" % mean)
    print("   pixels differing %d of %d (%.3f%%)" % (moved, total, 100.0 * moved / total))

    if worst == 0:
        print("\nimages identical byte for byte.")
        return 0

    # PSNR says little on its own but is the number other projects quote, so it
    # is here to be comparable with them.
    mse = float((diff.astype(np.float64) ** 2).mean())
    psnr = 10.0 * np.log10(255.0 * 255.0 / mse) if mse > 0 else float("inf")
    print("   PSNR             %.2f dB" % psnr)

    # The worst pixel, located: a single bad tile hides inside any average.
    flat = int(diff.max(axis=2).argmax())
    y, x = divmod(flat, a.shape[1])
    print("   worst pixel      (%d, %d): reference %s, candidate %s"
          % (x, y, tuple(int(v) for v in a[y, x]), tuple(int(v) for v in b[y, x])))

    if worst <= args.max_allowed:
        print("\nPASSED: deviation within %d, consistent with a different accumulation\n"
              "        order and not with a different computation." % args.max_allowed)
        return 0
    print("\nFAILED: deviation %d beyond the limit of %d." % (worst, args.max_allowed))
    return 1


if __name__ == "__main__":
    sys.exit(main())
