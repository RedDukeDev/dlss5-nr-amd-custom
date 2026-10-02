#!/usr/bin/env python3
"""Capture one launch of any kernel: its argument buffer and its buffers before
and after it runs, as the translated kernel computes them.

The data a native replacement is checked against. Every 8-byte word of the
argument buffer that points into device memory gets a pre and a post dump of
up to --bytes bytes (the proxy clamps to the allocation).

Usage: capture_launch.py <kernel> <dump dir> [--launch N] [--bytes N]
"""
import argparse
import os
import subprocess
import workspace  # noqa: E402

HARNESS = workspace.harness()
SNIPPET = workspace.snippet()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel")
    ap.add_argument("out_dir")
    ap.add_argument("--launch", default="1")
    ap.add_argument("--bytes", default=str(8 << 20))
    args = ap.parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    env = dict(os.environ)
    for stale in ("ZLUDA_SELECTIVE_FUSE", "ZLUDA_MMA_OPEN", "ZLUDA_MMA_PAIR", "ZLUDA_MMA_CHAIN",
                  "ZLUDA_PROBE_NO_CROSS_LANE", "ZLUDA_PROBE_NO_CROSS_LANE_KIND",
                  "NVCUDA_PROXY_NATIVE", "NVCUDA_PROXY_ARGS", "NVCUDA_PROXY_SYNC",
                  "NVCUDA_PROXY_SWAP_KERNEL", "NVCUDA_PROXY_SWAP_PTX", "NVCUDA_PROXY_SWAP_LAUNCH"):
        env.pop(stale, None)
    env.update({
        "ZLUDA_CACHE_DIR": os.path.join(HARNESS, "cache_per_matrix"),
        "ZLUDA_TARGET_ARCH": "gfx11-generic",
        "ZLUDA_CODEGEN_PARTS": "auto",
        "NVCUDA_PROXY_LOG": os.path.join(out_dir, "proxy.log"),
        "NVCUDA_PROXY_DUMP": args.kernel,
        "NVCUDA_PROXY_DUMP_DIR": out_dir,
        "NVCUDA_PROXY_DUMP_LAUNCH": args.launch,
        "NVCUDA_PROXY_DUMP_BYTES": args.bytes,
        "DLSS_REPEAT": "1",
    })
    subprocess.run([os.path.join(HARNESS, "dlss_image.exe"), "in.png", os.path.join(out_dir, "image.png"),
                    SNIPPET, os.path.join(HARNESS, "nvcuda.dll")],
                   cwd=HARNESS, capture_output=True, env=env, timeout=3600)
    files = sorted(f for f in os.listdir(out_dir) if f.endswith(".bin"))
    print("%s launch %s: %d dump files in %s" % (args.kernel, args.launch, len(files), out_dir))
    for f in files:
        print("   %-70s %9d bytes" % (f, os.path.getsize(os.path.join(out_dir, f))))


if __name__ == "__main__":
    main()
