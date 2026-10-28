#!/usr/bin/env python3
"""ab_stage.py over every native ray reconstruction stage: one capture per stage in <captures>/<stage>/, named
enc1 ... enc5, dec4_0, dec4_1, ... dec0_0 (runtime/gate/capture.h, one DLSS5NR_GATE_CAPTURE run per kernel).

Usage: ab_rr_all.py <captures dir> [code object, default kernels/rr2_stage.hsaco] [--time N]
Prints one line per stage: IDENTICAL or the number of differing bytes, and the time per launch.
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = ["enc1", "enc2", "enc3", "enc4", "enc5", "dec4_0", "dec4_1", "dec3_0", "dec3_1", "dec2_0", "dec2_1",
          "dec1_0", "dec1_1", "dec0_0"]


# stages launched with more waves than the PTX uses (the manifest's waves=)
WAVES = {"enc1": 8, "dec0_0": 4, "dec1_0": 8, "dec1_1": 8, "enc2": 8}


def entry(stage):
    return "rr2_%s_kernel_native" % stage if stage.startswith("enc") else "rr2_%s_kernel_%s_native" % (stage[:4], stage[5])


def main():
    captures = sys.argv[1]
    obj = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else os.path.join(HERE, "..", "rr2_stage.hsaco")
    repeat = sys.argv[sys.argv.index("--time") + 1] if "--time" in sys.argv else "5"
    total = 0.0
    bad = 0
    for stage in STAGES:
        out = subprocess.run([sys.executable, os.path.join(HERE, "ab_stage.py"), os.path.join(captures, stage), obj, entry(stage),
                              "--time", repeat] + (["--waves", str(WAVES[stage])] if stage in WAVES else []), capture_output=True, text=True).stdout
        ms = re.search(r"([\d.]+) ms per launch", out)
        verdict = out.strip().splitlines()[-1] if out.strip() else "no output"
        if ms:
            total += float(ms.group(1))
        bad += not verdict.startswith("IDENTICAL")
        print("%-7s %-10s %s ms" % (stage, verdict if not verdict.startswith("IDENTICAL") else "IDENTICAL", ms.group(1) if ms else "?"))
    print("total %.2f ms, %s" % (total, "all identical" if not bad else "%d DIFFERENT" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
