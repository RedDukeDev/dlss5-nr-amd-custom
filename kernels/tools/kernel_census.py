#!/usr/bin/env python3
"""Every kernel in the network, with the identity a native replacement is gated on.

A native kernel may only stand in for a PTX kernel it actually matches. The
identity used here is the kernel's name plus a hash of its PTX body: a retrained
network keeps both, a reshaped one does not, and that is exactly the line along
which a native kernel stays valid or stops being valid.

Usage:
  kernel_census.py <corpus dir with *.ptx> [-o manifest.json]
  kernel_census.py <nvngx_dlssnr.dll> [-o manifest.json]   (extracts first)

The DLL form writes the fatbins next to the manifest, so a new DLSS build can be
censused without any other tool.
"""
import argparse
import hashlib
import json
import os
import re
import struct
import sys

FATBIN_MAGIC = struct.pack("<I", 0xBA55ED50)
ENTRY = re.compile(r"\.visible \.entry\s+([A-Za-z0-9_$]+)")


def modules_from_dll(path, out_dir):
    """The PTX modules, decompressed out of the fatbins.

    The neural rendering snippet compresses its PTX with zstd, so reading the
    fatbin bytes finds no kernels at all. The entry walk is the one proved in
    tools/extract_dlssnr_ptx.py: within each fatbin, an entry of kind 1 carries
    the compressed size at +16 and the decompressed size at +56, the payload
    starting after that entry's own header.
    """
    import zstandard

    data = open(path, "rb").read()
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for m in re.finditer(re.escape(FATBIN_MAGIC), data):
        offset = m.start()
        header_size, = struct.unpack_from("<H", data, offset + 6)
        body_size, = struct.unpack_from("<Q", data, offset + 8)
        if header_size != 16 or not (0 < body_size < 200_000_000):
            continue
        base = "fb_%03d_%08x" % (len(written), offset)
        entry = offset + header_size
        while entry < offset + header_size + body_size:
            kind, _, entry_header = struct.unpack_from("<HHI", data, entry)
            entry_size, = struct.unpack_from("<Q", data, entry + 8)
            if kind == 1:
                packed_size, = struct.unpack_from("<I", data, entry + 16)
                plain_size, = struct.unpack_from("<Q", data, entry + 56)
                payload = data[entry + entry_header:entry + entry_header + packed_size]
                try:
                    ptx = zstandard.ZstdDecompressor().decompress(
                        payload, max_output_size=plain_size)
                except Exception:
                    # Some entries are stored plain; the fallback keeps them.
                    ptx = data[entry + entry_header:entry + entry_header + entry_size]
                if ptx[:8].startswith(b"//") or b".version" in ptx[:200]:
                    name = os.path.join(out_dir, base + ".ptx")
                    with open(name, "wb") as f:
                        f.write(ptx)
                    written.append(name)
            entry += entry_header + entry_size
    return written


def kernel_bodies(text):
    """Yield (name, body) for each entry, the body being brace-balanced."""
    for m in ENTRY.finditer(text):
        name = m.group(1)
        brace = text.find("{", m.end())
        if brace < 0:
            continue
        depth, i = 0, brace
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        yield name, text[m.start():i + 1]


def census_file(path):
    # PTX carries NUL bytes when it is lifted straight out of a fatbin, so read
    # bytes and decode loosely rather than trusting the file to be text.
    text = open(path, "rb").read().decode("utf-8", "replace")
    out = {}
    for name, body in kernel_bodies(text):
        params = len(re.findall(r"\.param\s+\.\w+\s+[A-Za-z0-9_$]+", body.split("{", 1)[0]))
        out[name] = {
            "module": os.path.basename(path),
            "ptx_sha256": hashlib.sha256(body.encode("utf-8", "replace")).hexdigest(),
            "params": params,
            "lines": body.count("\n") + 1,
            "bytes": len(body),
            "fp8": name.endswith("_fp8"),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="directory of .ptx/.fatbin files, or nvngx_dlssnr.dll")
    ap.add_argument("-o", "--out", default="manifest.json")
    ap.add_argument("--extract-to", default=None,
                    help="where to put the fatbins when the source is a DLL")
    args = ap.parse_args()

    if os.path.isdir(args.source):
        files = [os.path.join(args.source, f) for f in sorted(os.listdir(args.source))
                 if f.endswith((".ptx", ".fatbin"))]
        origin = os.path.abspath(args.source)
        origin_sha = ""
    else:
        out_dir = args.extract_to or os.path.join(os.path.dirname(os.path.abspath(args.out)),
                                                  "moduli")
        files = modules_from_dll(args.source, out_dir)
        origin = os.path.abspath(args.source)
        origin_sha = hashlib.sha256(open(args.source, "rb").read()).hexdigest()
        print(f"{len(files)} modules extracted to {out_dir}")

    kernels = {}
    for path in files:
        found = census_file(path)
        for name, info in found.items():
            if name in kernels:
                info["also_in"] = kernels[name]["module"]
            kernels[name] = info
        print(f"  {os.path.basename(path):28} {len(found):4} kernels")

    manifest = {"source": origin, "source_sha256": origin_sha,
                "kernels": dict(sorted(kernels.items()))}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)

    fp8 = sum(1 for k in kernels.values() if k["fp8"])
    print(f"\n{len(kernels)} kernels ({fp8} fp8, {len(kernels)-fp8} plain) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
