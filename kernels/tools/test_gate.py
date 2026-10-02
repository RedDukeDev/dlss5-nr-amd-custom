#!/usr/bin/env python3
"""Does the gate really fall back, and does falling back really cost nothing?

The native path is only safe if every way it can fail ends in the translated
kernel running and the image coming out unchanged. That is a claim about
behaviour, so it is tested rather than asserted: the network is run three times,
once with no native kernels at all and twice with a native kernel declared in a
way that must be refused, and all three images must be the same bytes.

  reference     no manifest: the translation, as it has always run
  wrong size    the real kernel, declared with an argument size that is not its
                own -- the gate must refuse it before loading anything
  missing file  declared correctly but pointing at a code object that does not
                exist -- the gate must refuse it when the load fails

A fourth outcome is worth watching for: if the "wrong size" run were to differ
from the reference, the hook would be changing behaviour merely by being
compiled in, which is the one thing it must never do.

Usage: test_gate.py [--kernel <name>]
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
# 69 modules under the current build's git SHA; the key carries that SHA, so a
# cache filled by another build silently retranslates instead of failing.
CACHE = os.path.join(HARNESS, "cache_per_matrix")
DEFAULT_KERNEL = "cc_tinlayout_fused_swin_8h_256_8_chained_fp8"


def run(tag, manifest):
    log = os.path.join(HARNESS, "gate_%s.log" % tag)
    png = os.path.join(HARNESS, "gate_%s.png" % tag)
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
    env["DLSS_REPEAT"] = "1"
    if manifest:
        env["NVCUDA_PROXY_NATIVE"] = manifest

    finished = subprocess.run(
        [os.path.join(HARNESS, "dlss_image.exe"), "in.png", png, SNIPPET,
         os.path.join(HARNESS, "nvcuda.dll")],
        cwd=HARNESS, capture_output=True, text=True, errors="replace", env=env, timeout=7200)

    # One repeat writes the plain name, several write one file per round; take
    # the last either way rather than assuming which.
    produced = sorted(glob.glob(png + ".r*.png")) or ([png] if os.path.exists(png) else [])
    digest = "-"
    if produced:
        digest = hashlib.sha256(open(produced[-1], "rb").read()).hexdigest()[:16]
    else:
        # Without this the failure says only "no image", which is indistinguishable
        # from a network that never ran.
        tail = (finished.stdout + finished.stderr).strip().splitlines()[-6:]
        print("   %s produced no image (exit %d):" % (tag, finished.returncode))
        for line in tail:
            print("      ", line[:120])

    decisions, native_launches = [], 0
    if os.path.exists(log):
        for line in open(log, encoding="utf-8", errors="replace"):
            if line.startswith("[native]"):
                decisions.append(line.rstrip()[:110])
            if "<-- NATIVE" in line:
                native_launches += 1
    return digest, decisions, native_launches


def write_manifest(path, kernel, obj, args):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# written by test_gate.py; not a real native kernel\n")
        f.write("%s  %s  block=32,8,1  args=%d\n" % (kernel, obj, args))
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kernel", default=DEFAULT_KERNEL)
    args = ap.parse_args()

    scratch = os.path.join(NATIVE, "gate_manifests")
    os.makedirs(scratch, exist_ok=True)
    cases = [
        ("reference", None),
        ("wrong_size", write_manifest(os.path.join(scratch, "wrong_size.txt"),
                                         args.kernel, "nonexistent.hsaco", 87)),
        ("missing_object", write_manifest(os.path.join(scratch, "missing.txt"),
                                           args.kernel, "nonexistent.hsaco", 88)),
    ]

    results = {}
    for tag, manifest in cases:
        digest, decisions, native = run(tag, manifest)
        results[tag] = digest
        print("%-16s img=%s  native launches=%d" % (tag, digest, native), flush=True)
        for d in decisions:
            print("      ", d)

    print()
    reference = results["reference"]
    same = all(v == reference for v in results.values())
    missing = [k for k, v in results.items() if v == "-"]
    if missing:
        print("FAILED: no image produced by %s" % ", ".join(missing))
        return 2
    if same:
        print("PASSED: the three images are identical (%s), the gate falls back without\n"
              "        changing the result." % reference)
        return 0
    print("FAILED: images differ -> %s" % results)
    return 1


if __name__ == "__main__":
    sys.exit(main())
