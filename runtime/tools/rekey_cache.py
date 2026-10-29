#!/usr/bin/env python3
"""Carries translations over to a newer ZLUDA build, for the modules the newer build would translate the same way.

    python rekey_cache.py <old ComputeCache dir> <new ComputeCache dir> [--devices gfx1100,gfx11-generic]
                          [--modules-of gfx1100]

The cache is keyed by ZLUDA's version, so rows translated by one commit are never read by another. Translating
the neural rendering network again takes hours; but when the commits between the two did not touch how the
instructions of a module are lowered, the old translation is as good as a new one, and only its key has to change.
That holds for the neural rendering's modules, which have no texture or surface instructions (the part of
ZLUDA that changed). It does not hold for a module that has them: those are translated again, by prepare_cache.py.

Rows of <old> are copied into <new>'s zluda2.db with their version replaced by the one <new>'s rows carry (so <new>
has to hold at least one row of the build being shipped), for the devices named, and, with --modules-of, only
for the modules that <old> also holds for that device (gfx1100 holds the neural rendering's modules alone). Rows
<new> has are kept.
"""
import argparse
import os
import shutil
import sqlite3
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--devices", default="gfx1100,gfx11-generic")
    ap.add_argument("--modules-of", help="only the modules the old cache holds for this device")
    args = ap.parse_args()

    old_db, new_db = os.path.join(args.old, "zluda2.db"), os.path.join(args.new, "zluda2.db")
    for path in (old_db, new_db):
        if not os.path.isfile(path):
            raise SystemExit("not found: " + path)
    new = sqlite3.connect(new_db)
    versions = [r[0] for r in new.execute("SELECT DISTINCT zluda_version FROM modules")]
    if len(versions) != 1:
        raise SystemExit("the new cache must hold rows of exactly one ZLUDA build, it has %s" % versions)
    version = versions[0]

    shutil.copyfile(new_db, new_db + ".before-rekey")
    new.execute("ATTACH DATABASE ? AS old", (old_db,))
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    marks = ",".join("?" for _ in devices)
    where = "device IN (%s)" % marks
    params = list(devices)
    if args.modules_of:
        where += " AND hash IN (SELECT hash FROM old.modules WHERE device = ?)"
        params.append(args.modules_of)
    before = new.execute("SELECT COUNT(*) FROM main.modules").fetchone()[0]
    new.execute(
        "INSERT OR IGNORE INTO main.modules (hash, compiler_version, zluda_version, device, backend_key, binary, last_access) "
        "SELECT hash, compiler_version, ?, device, backend_key, binary, last_access FROM old.modules WHERE " + where,
        [version] + params)
    new.commit()
    after = new.execute("SELECT COUNT(*) FROM main.modules").fetchone()[0]
    new.execute("DETACH DATABASE old")
    new.execute("VACUUM")
    new.commit()
    for device, count in new.execute("SELECT device, COUNT(*) FROM main.modules GROUP BY device"):
        print("  %-18s %d modules" % (device, count))
    new.close()
    print("%d rows carried over to %s (%d before, %d after); the cache as it was is zluda2.db.before-rekey"
          % (after - before, version[:12], before, after))
    return 0


if __name__ == "__main__":
    sys.exit(main())
