#!/usr/bin/env python3
"""Runs the harness through the whole HIP chain on Windows, and compares the image.

The chain is the one a game under Proton uses, with Windows standing in for
Wine's door to native code:

  ZLUDA -> trampoline (amdhip64_7.dll) -> test bridge -> AMD's own HIP runtime

AMD's runtime is copied under another name so that the trampoline can take the
place of the real one, and ZLUDA is pointed away from the HIP SDK's folder (it
would load the SDK's copy first). The image the network makes through the chain
has to be the one it makes without it, byte for byte: every one of the forwarded
functions the network uses is on the way.

  python linux/tests/test_chain_windows.py --snippet nvngx_dlssnr.dll [--image in.png]
                                           [--runtime-dir dist/OptiScaler/dlss5nr]
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
LINUX = os.path.dirname(HERE)
ROOT = os.path.dirname(LINUX)
HARNESS = os.path.join(ROOT, "runtime", "build", "bin", "dlss5nr_harness.exe")


def digest(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def run_harness(arguments, environment, out):
    command = [HARNESS, "--image", arguments.image, "--out", out, "--snippet", arguments.snippet,
               "--runtime-dir", arguments.runtime_dir, "--frames", "40", "--evaluations", "3"]
    result = subprocess.run(command, env=environment, capture_output=True, text=True)
    if result.returncode or not os.path.isfile(out):
        sys.exit("the harness failed:\n" + result.stdout[-2000:] + result.stderr[-2000:])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snippet", required=True, help="the original, signed nvngx_dlssnr.dll")
    ap.add_argument("--image", default=os.path.join(ROOT, "..", "proxy_harness", "in.png"))
    ap.add_argument("--runtime-dir", default=os.path.join(ROOT, "dist", "OptiScaler", "dlss5nr"))
    ap.add_argument("--hip", default=os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "amdhip64_7.dll"),
                    help="AMD's HIP runtime to forward to")
    arguments = ap.parse_args()
    for name in ("snippet", "image", "runtime_dir", "hip"):
        setattr(arguments, name, os.path.abspath(getattr(arguments, name)))

    for target in ("trampoline", "test-bridge"):
        if subprocess.run([sys.executable, os.path.join(LINUX, "build.py"), target]).returncode:
            sys.exit("could not build the %s" % target)

    work = tempfile.mkdtemp(prefix="dlss5nr_chain_")
    try:
        real = os.path.join(work, "amdhip64_real.dll")
        shutil.copy(arguments.hip, real)
        # The folder the harness runs from gets the trampoline for the length of the run;
        # the SDK's folder is replaced by an empty one.
        harness_dir = os.path.dirname(HARNESS)
        trampoline = os.path.join(harness_dir, "amdhip64_7.dll")
        base_out = os.path.join(work, "without.png")
        chain_out = os.path.join(work, "through.png")

        environment = dict(os.environ)
        for name in ("DLSSNR_BRIDGE_DLL", "DLSSNR_HIP_LIB", "HIP_PATH"):
            environment.pop(name, None)
        # Without the trampoline: what the image is.
        run_harness(arguments, dict(environment, HIP_PATH=os.path.join(work, "empty")), base_out)

        shutil.copy(os.path.join(LINUX, "build", "amdhip64_7.dll"), trampoline)
        try:
            chain = dict(environment, HIP_PATH=os.path.join(work, "empty"),
                         DLSSNR_BRIDGE_DLL=os.path.join(LINUX, "build", "dlss5nr_hip_bridge_win.dll"),
                         DLSSNR_HIP_LIB=real)
            run_harness(arguments, chain, chain_out)
        finally:
            os.remove(trampoline)

        same = digest(base_out) == digest(chain_out)
        print("image without the chain:", digest(base_out)[:16])
        print("image through the chain:", digest(chain_out)[:16])
        print("IDENTICAL" if same else "DIFFERENT")
        return 0 if same else 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
