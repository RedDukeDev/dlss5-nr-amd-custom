#!/usr/bin/env python3
"""What a new DLSS build changed, and which native kernels it invalidates.

This is the tool that answers "do we have to rewrite the kernels?" when NVIDIA
ships a new nvngx_dlssnr.dll. A native kernel is gated on the name and the hash
of the PTX body it replaces: unchanged hash, still valid; changed hash, it falls
back to the translated path until someone looks at it.

Usage:
  diff_versions.py old.json new.json [--native k1 k2 ...]

`--native` is the list of kernels we have native implementations for; without it
every changed kernel is reported, which is the same question asked more widely.
"""
import argparse
import json
import sys


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def family(name):
    """Kernel names carry their shape -- swin_8h_256_8 is eight heads, 256
    channels. Collapsing the numbers groups a family so a reshaped network is
    visible as a family that moved, not as forty unrelated kernels."""
    import re
    return re.sub(r"\d+", "N", name)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--native", nargs="*", default=[],
                    help="kernels we replace natively")
    args = ap.parse_args()

    old, new = load(args.old), load(args.new)
    ko, kn = old["kernels"], new["kernels"]

    same = [k for k in ko if k in kn and ko[k]["ptx_sha256"] == kn[k]["ptx_sha256"]]
    changed = [k for k in ko if k in kn and ko[k]["ptx_sha256"] != kn[k]["ptx_sha256"]]
    removed = [k for k in ko if k not in kn]
    added = [k for k in kn if k not in ko]

    print(f"old: {old.get('source','?')}")
    print(f"new: {new.get('source','?')}")
    print(f"\n{len(same)} identical, {len(changed)} changed, {len(removed)} removed, "
          f"{len(added)} added")

    if changed:
        print("\nchanged kernels, by family:")
        fams = {}
        for k in changed:
            fams.setdefault(family(k), []).append(k)
        for fam, ks in sorted(fams.items(), key=lambda kv: -len(kv[1]))[:12]:
            print(f"  {fam:52} x{len(ks)}")
    if removed:
        print(f"\nremoved: {', '.join(sorted(removed)[:8])}"
              f"{' ...' if len(removed) > 8 else ''}")
    if added:
        print(f"added:   {', '.join(sorted(added)[:8])}{' ...' if len(added) > 8 else ''}")

    if args.native:
        print("\n--- the native kernels")
        broken = []
        for k in args.native:
            if k not in kn:
                state, note = "GONE", "the kernel no longer exists"
            elif k not in ko:
                state, note = "NEW", "was not in the old build either"
            elif ko[k]["ptx_sha256"] == kn[k]["ptx_sha256"]:
                state, note = "valid", "body identical, native path keeps working"
            else:
                state, note = "STALE", "body changed, falls back to translation"
            if state in ("GONE", "STALE"):
                broken.append(k)
            print(f"  {state:6} {k[:56]:56} {note}")
        print(f"\n{len(args.native) - len(broken)} of {len(args.native)} native kernels "
              f"survive this update")
        return 1 if broken else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
