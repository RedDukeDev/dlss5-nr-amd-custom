#!/usr/bin/env python3
"""A native kernel against the translated one, on the real network.

Two things make a measurement here worthless, and both are easy to miss, so
neither is left to the reader:

  - if the image is not the reference one, the native kernel is wrong and its
    time is not a speedup, it is a different computation;
  - if no launch was actually taken natively, the "native" run measured the
    translation twice. The gate refuses silently by design, so a run with zero
    native launches looks exactly like a run with no gain.

Both are checked before any percentage is printed.

Usage: ab_native.py <kernel name> [--manifest kernels.txt] [--rounds 3]
"""
import argparse
import glob
import hashlib
import os
import re
import subprocess
import sys
import workspace  # noqa: E402

HARNESS = workspace.harness()
NATIVE = workspace.kernels()
SNIPPET = workspace.snippet()
CACHE = os.path.join(HARNESS, "cache_per_matrix")
REFERENCE = "a48928166420fc45"


def run_one(kernel, tag, manifest):
    log = os.path.join(HARNESS, "abn_%s.log" % tag)
    png = os.path.join(HARNESS, "abn_%s.png" % tag)
    for stale in glob.glob(png + ".r*.png") + [png, log]:
        if os.path.exists(stale):
            os.remove(stale)

    env = dict(os.environ)
    for name in ("ZLUDA_SELECTIVE_FUSE", "ZLUDA_MMA_OPEN", "ZLUDA_MMA_CHAIN",
                 "ZLUDA_MMA_CHAIN_N", "ZLUDA_PROBE_NO_REAL_WMMA",
                 "ZLUDA_PROBE_NO_CROSS_LANE", "ZLUDA_PROBE_NO_CROSS_LANE_KIND",
                 "NVCUDA_PROXY_SWAP_KERNEL", "NVCUDA_PROXY_SWAP_PTX",
                 "NVCUDA_PROXY_GRID1", "NVCUDA_PROXY_RESTORE", "NVCUDA_PROXY_ARGS",
                 "NVCUDA_PROXY_NATIVE"):
        env.pop(name, None)
    env["ZLUDA_CACHE_DIR"] = CACHE
    env["ZLUDA_TARGET_ARCH"] = "gfx11-generic"
    env["ZLUDA_CODEGEN_PARTS"] = "auto"
    env["NVCUDA_PROXY_LOG"] = log
    # The per-launch timings come from the synchronise, so it stays on for both
    # sides: it costs the same on each and is the only per-kernel number there is.
    env["NVCUDA_PROXY_SYNC"] = "1"
    env["DLSS_REPEAT"] = "3"
    if manifest:
        env["NVCUDA_PROXY_NATIVE"] = manifest

    subprocess.run(
        [os.path.join(HARNESS, "dlss_image.exe"), "in.png", png, SNIPPET,
         os.path.join(HARNESS, "nvcuda.dll")],
        cwd=HARNESS, capture_output=True, text=True, errors="replace", env=env, timeout=7200)

    own, launches, native_taken, frame = 0.0, 0, 0, 0.0
    if os.path.exists(log):
        for line in open(log, encoding="utf-8", errors="replace"):
            if "<-- NATIVE" in line:
                native_taken += 1
            m = re.match(r"\s+([\d.]+) ms\s+(\S+)", line)
            if m:
                frame += float(m.group(1))
                if m.group(2) == kernel:
                    own += float(m.group(1))
                    launches += 1
    produced = sorted(glob.glob(png + ".r*.png")) or ([png] if os.path.exists(png) else [])
    digest = hashlib.sha256(open(produced[-1], "rb").read()).hexdigest()[:16] if produced else "-"
    return {"own": own, "launches": launches, "native": native_taken, "frame": frame,
            "digest": digest}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kernel")
    ap.add_argument("--manifest", default=os.path.join(NATIVE, "kernels.txt"))
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()

    print("%s\nmanifest: %s\n" % (args.kernel, args.manifest))
    got = {"translated": [], "native": []}
    seen, natives = {}, []
    for i in range(args.rounds):
        for tag, manifest in (("translated", None), ("native", args.manifest)):
            r = run_one(args.kernel, "%s%d" % (tag, i), manifest)
            got[tag].append((r["own"], r["frame"]))
            seen.setdefault(tag, set()).add(r["digest"])
            if tag == "native":
                natives.append(r["native"])
            print("round %d %-11s kernel %7.2f ms in %2d launches   frame %7.2f ms   "
                  "img=%s  native=%d" % (i, tag, r["own"], r["launches"], r["frame"],
                                         r["digest"], r["native"]), flush=True)

    print()
    if not any(natives):
        print("NOT MEASURABLE: no launch took the native path.")
        print("The gate refused: the [native] lines in the log say why.")
        return 2
    wrong = {d for d in seen.get("native", set()) if d != REFERENCE}
    if wrong:
        print("WRONG IMAGE from the native path: %s, expected %s."
              % (", ".join(sorted(wrong)), REFERENCE))
        print("The time means nothing: it is a different computation, not a gain.")
        return 1

    def median(values):
        s = sorted(values)
        return s[len(s) // 2]

    k_old = median([o for o, _ in got["translated"]])
    k_new = median([o for o, _ in got["native"]])
    f_old = median([f for _, f in got["translated"]])
    f_new = median([f for _, f in got["native"]])
    print("kernel  %7.2f -> %7.2f ms   %+.1f%%" % (k_old, k_new, 100.0 * (k_new - k_old) / k_old))
    print("network %7.2f -> %7.2f ms   %+.1f%%" % (f_old, f_new, 100.0 * (f_new - f_old) / f_old))
    print("\nimage identical to the reference, %d launches taken natively per run"
          % (sum(natives) // max(1, len(natives))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
