#!/usr/bin/env python3
"""The weight tables, found by the shape of the records rather than by address.

The published AMD ports pin the archive at a fixed offset and refuse any DLL
whose SHA-256 they do not know, which is why a new DLSS build stops them dead.
The archive is self-describing enough not to need either: every record is

    8 bytes  name length
    n bytes  ASCII name, e.g. block17.layer0.layer
    8 bytes  payload length
    m bytes  payload

so anchoring on a name finds the first record, and the 8 bytes before it are the
archive header. Verified against 310.8.0.0: the scan lands on 0x114A160, the
address the other projects hardcode, and walks all 153 records.

Usage:
  extract_weights.py <nvngx_dlssnr.dll>                  list the records
  extract_weights.py <nvngx_dlssnr.dll> -o <directory>   write them out
"""
import argparse
import hashlib
import json
import os
import re
import struct
import sys

# Record names seen so far. A build that renames these needs a new pattern here,
# which is a one-line change and an honest place to notice the format moved.
NAME_PATTERN = re.compile(rb"block\d+\.layer\d+\.[a-z_]+")


def find_archive(data):
    """Returns (header offset, first record offset, declared size) or None."""
    for m in NAME_PATTERN.finditer(data):
        record = m.start() - 8
        if record < 0:
            continue
        name_len, = struct.unpack_from("<Q", data, record)
        if name_len != len(m.group(0)):
            continue
        header = record - 8
        if header < 0:
            continue
        declared, = struct.unpack_from("<Q", data, header)
        if 1_000_000 < declared <= len(data) - header:
            return header, record, declared
    return None


def walk(data, first, limit):
    pos = first
    while pos < limit:
        name_len, = struct.unpack_from("<Q", data, pos)
        if not (1 <= name_len <= 200):
            break
        name = data[pos + 8:pos + 8 + name_len].decode("ascii", "replace")
        span, = struct.unpack_from("<Q", data, pos + 8 + name_len)
        if not (0 < span <= limit - pos):
            break
        payload = pos + 8 + name_len + 8
        yield name, payload, span
        pos = payload + span


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dll")
    ap.add_argument("-o", "--out", default=None, help="write each record here")
    args = ap.parse_args()

    data = open(args.dll, "rb").read()
    found = find_archive(data)
    if not found:
        print("no weight archive found: the record format is not the one this knows")
        return 2
    header, first, declared = found
    print(f"archive header 0x{header:x}, first record 0x{first:x}, "
          f"declared {declared/1024/1024:.1f} MB")

    records = list(walk(data, first, min(header + 8 + declared, len(data))))
    total = sum(span for _, _, span in records)
    print(f"{len(records)} records, {total/1024/1024:.1f} MB of payload")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
    manifest = []
    for name, payload, span in records:
        blob = data[payload:payload + span]
        entry = {"name": name, "offset": payload, "size": span,
                 "sha256": hashlib.sha256(blob).hexdigest()}
        manifest.append(entry)
        if args.out:
            with open(os.path.join(args.out, name.replace(".", "_") + ".bin"), "wb") as f:
                f.write(blob)
    if args.out:
        with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump({"dll": os.path.abspath(args.dll),
                       "dll_sha256": hashlib.sha256(data).hexdigest(),
                       "records": manifest}, f, indent=1)
        print(f"written to {args.out}")
    else:
        for e in manifest[:10]:
            print(f"  {e['name']:34} {e['size']:>10} bytes  {e['sha256'][:12]}")
        if len(manifest) > 10:
            print(f"  ... {len(manifest)-10} more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
