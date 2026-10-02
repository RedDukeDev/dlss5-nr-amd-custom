#!/usr/bin/env python3
"""One kernel's PTX, lifted out of the extracted corpus.

The corpus is what tools/extract_dlssnr_ptx.py writes: whole modules, each
holding many entries. run_ptx.py wants one entry -- its header, its body and
the shared arrays it names -- with the .version line prepended at load time,
which is the shape the swin_*.ptx files in this directory already have.

Usage: extract_kernel.py <kernel name> <corpus dir> <out.ptx>
"""
import glob
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from kernel_census import kernel_bodies  # noqa: E402

SHARED = re.compile(r"^\.shared[^;]*;", re.M)


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    name, corpus, out = sys.argv[1:]
    for path in sorted(glob.glob(os.path.join(corpus, "*.ptx"))):
        text = io.open(path, "rb").read().decode("utf-8", "replace")
        for got, body in kernel_bodies(text):
            if got != name:
                continue
            # Only the shared arrays this kernel names: a module declares every
            # kernel's, and ptxsim would otherwise reserve all of them.
            decls = [d for d in SHARED.findall(text) if d.split()[-1].split("[")[0] in body]
            io.open(out, "w", encoding="utf-8", newline="\n").write(
                "\n".join(decls + [body]) + "\n")
            print("%s: %d bytes from %s, %d shared arrays"
                  % (out, len(body), os.path.basename(path), len(decls)))
            return 0
    print("not found: " + name)
    return 1


if __name__ == "__main__":
    sys.exit(main())
