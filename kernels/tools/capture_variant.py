#!/usr/bin/env python3
"""Run the network with one kernel swapped for a PTX variant, and dump a launch.

Usage: capture_variant.py <variant.ptx> <dump dir> [launch]
"""
import os
import subprocess
import sys
import workspace  # noqa: E402

HARNESS = workspace.harness()
KERNEL = "cc_split_swin_16h_ffwd_512_chained_fp8"


def main():
    ptx, out_dir = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    launch = sys.argv[3] if len(sys.argv) > 3 else "8"
    os.makedirs(out_dir, exist_ok=True)
    env = dict(os.environ)
    for stale in ("ZLUDA_SELECTIVE_FUSE", "ZLUDA_MMA_OPEN", "ZLUDA_MMA_PAIR", "ZLUDA_MMA_CHAIN",
                  "ZLUDA_PROBE_NO_CROSS_LANE", "ZLUDA_PROBE_NO_CROSS_LANE_KIND",
                  "NVCUDA_PROXY_NATIVE", "NVCUDA_PROXY_ARGS", "NVCUDA_PROXY_SYNC"):
        env.pop(stale, None)
    env.update({
        "ZLUDA_CACHE_DIR": os.path.join(HARNESS, "cache_per_matrix"),
        "ZLUDA_TARGET_ARCH": "gfx11-generic",
        "ZLUDA_CODEGEN_PARTS": "auto",
        "NVCUDA_PROXY_SWAP_KERNEL": KERNEL,
        "NVCUDA_PROXY_SWAP_PTX": ptx,
        # The variant only for the launch being dumped: every other launch
        # runs the original, so the one under study gets its real inputs.
        "NVCUDA_PROXY_SWAP_LAUNCH": launch,
        "NVCUDA_PROXY_LOG": os.path.join(out_dir, "proxy.log"),
        "NVCUDA_PROXY_DUMP": KERNEL,
        "NVCUDA_PROXY_DUMP_DIR": out_dir,
        "NVCUDA_PROXY_DUMP_LAUNCH": launch,
        "NVCUDA_PROXY_DUMP_BYTES": "4194304",
        "DLSS_REPEAT": "1",
    })
    subprocess.run([os.path.join(HARNESS, "dlss_image.exe"), "in.png",
                    os.path.join(out_dir, "image.png"),
                    workspace.snippet(),
                    os.path.join(HARNESS, "nvcuda.dll")],
                   cwd=HARNESS, capture_output=True, env=env, timeout=3600)
    log = open(os.path.join(out_dir, "proxy.log"), encoding="utf-8", errors="replace").read()
    swapped = "taken from the replacement module -> 0" in log
    dumped = len([f for f in os.listdir(out_dir) if f.endswith(".bin")])
    print("%s: swapped=%s, %d dump files in %s" % (os.path.basename(ptx), swapped, dumped, out_dir))


if __name__ == "__main__":
    main()
