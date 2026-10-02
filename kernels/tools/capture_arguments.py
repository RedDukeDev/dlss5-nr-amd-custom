#!/usr/bin/env python3
"""The ABI one kernel is called with, captured from a real run.

A native kernel has to accept exactly what the translated one accepts. These
kernels take a single packed argument buffer, and nothing in the PTX says what
is inside it -- the only reliable source is a real launch. The proxy already
prints that buffer (NVCUDA_PROXY_ARGS) and can read back the buffers it points
at (NVCUDA_PROXY_ARGHASH); this drives it and tidies the result.

Usage:
  capture_arguments.py <kernel name> [--grid1] [--repeat 1]

Prints, for each launch of that kernel: the grid and block geometry, the
argument buffer word by word, and which words look like device pointers.
"""
import argparse
import os
import re
import subprocess
import sys
import workspace  # noqa: E402

HARNESS = workspace.harness()
SNIPPET = workspace.snippet()
# The cache must answer for the build being run; its key carries ZLUDA's git SHA.
CACHE = os.path.join(HARNESS, "cache_per_matrix")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kernel")
    ap.add_argument("--grid1", action="store_true",
                    help="run the kernel as a single block (the output is meaningless)")
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()

    log = os.path.join(HARNESS, "abi_%s.log" % args.kernel[:40])
    if os.path.exists(log):
        os.remove(log)
    env = dict(os.environ)
    # An inherited variable from the previous measurement would silently change
    # what is being captured, so the experiment's own knobs are cleared first.
    for stale in ("ZLUDA_SELECTIVE_FUSE", "ZLUDA_MMA_OPEN", "ZLUDA_MMA_CHAIN",
                  "ZLUDA_MMA_CHAIN_N", "ZLUDA_PROBE_NO_REAL_WMMA",
                  "ZLUDA_PROBE_NO_CROSS_LANE", "ZLUDA_PROBE_NO_CROSS_LANE_KIND",
                  "NVCUDA_PROXY_SWAP_KERNEL", "NVCUDA_PROXY_SWAP_PTX",
                  "NVCUDA_PROXY_GRID1", "NVCUDA_PROXY_RESTORE"):
        env.pop(stale, None)
    env.update({
        "ZLUDA_CACHE_DIR": CACHE,
        "ZLUDA_TARGET_ARCH": "gfx11-generic",
        "ZLUDA_CODEGEN_PARTS": "auto",
        "NVCUDA_PROXY_LOG": log,
        "NVCUDA_PROXY_ARGS": args.kernel,
        "NVCUDA_PROXY_ARGHASH": "1",
        "NVCUDA_PROXY_SYNC": "1",
        "DLSS_REPEAT": str(args.repeat),
    })
    if args.grid1:
        env["NVCUDA_PROXY_GRID1"] = "1"

    subprocess.run(
        [os.path.join(HARNESS, "dlss_image.exe"), "in.png", "abi.png", SNIPPET,
         os.path.join(HARNESS, "nvcuda.dll")],
        cwd=HARNESS, capture_output=True, text=True, errors="replace", env=env,
        timeout=7200)

    if not os.path.exists(log):
        print("the proxy wrote no log; was NVCUDA_PROXY_LOG honoured?")
        return 2

    # The proxy prints the buffer as "[+0xNN] 0x...." lines under a header
    # naming the kernel, and follows a pointer-looking word with what it found.
    launches, current = [], None
    for line in open(log, encoding="utf-8", errors="replace"):
        if f"arguments of {args.kernel}" in line:
            current = {"words": [], "size": re.search(r"(\d+) byte", line).group(1)}
            launches.append(current)
        elif current is not None:
            m = re.match(r"\s+\[\+0x([0-9A-F]+)\] 0x([0-9A-F]+)", line)
            if m:
                current["words"].append((int(m.group(1), 16), int(m.group(2), 16), ""))
            elif current["words"] and line.strip().startswith(("inside", "outside", "read")):
                off, val, _ = current["words"][-1]
                current["words"][-1] = (off, val, line.strip()[:70])
            elif line.startswith("["):
                current = None

    if not launches:
        print(f"{args.kernel} was never launched, or the name does not match")
        return 1

    first = launches[0]
    print(f"{args.kernel}: {len(launches)} launches captured, "
          f"argument buffer {first['size']} bytes\n")
    for off, val, note in first["words"]:
        looks_like_pointer = 0x100000000 <= val < 0x1000000000
        kind = "pointer" if looks_like_pointer else ("zero" if val == 0 else "value")
        print(f"  +0x{off:02X}  0x{val:016X}  {kind:8} {note}")
    if len(launches) > 1:
        moving = {off for off, val, _ in first["words"]
                  if any(l["words"][i][1] != val for l in launches[1:]
                         for i, (o, _, _) in enumerate(first["words"]) if o == off
                         and i < len(l["words"]))}
        print(f"\n  words that differ between launches: "
              f"{', '.join('+0x%02X' % o for o in sorted(moving)) or 'none'}")
        print("  (those are the per-tile or per-stage arguments; the rest is the "
              "same buffer every time)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
