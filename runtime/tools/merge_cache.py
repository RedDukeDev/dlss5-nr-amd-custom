#!/usr/bin/env python3
"""Adds the translations of one cache to another.

    python merge_cache.py <from ComputeCache dir> [<into ComputeCache dir>]

prepare_cache.py translates every module of every library it is given. The frame generation library
(nvngx_dlssg.dll) is translated apart from the others, which take hours, and its rows are added to the shipped cache
here: both are keyed by ZLUDA's version, so they have to come from the same ZLUDA, and a row already there is kept.

<into> defaults to runtime\\build\\shipped\\ComputeCache, where package.py takes the cache from.
"""
import os
import shutil
import sqlite3
import sys

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    source = os.path.join(sys.argv[1], "zluda2.db")
    into = sys.argv[2] if len(sys.argv) > 2 else os.path.join(PROJECT, "build", "shipped", "ComputeCache")
    destination = os.path.join(into, "zluda2.db")
    for path in (source, destination):
        if not os.path.isfile(path):
            raise SystemExit("not found: " + path)

    # What the two caches were translated by: rows of another ZLUDA would never be read.
    def versions(path):
        connection = sqlite3.connect(path)
        rows = {row[0] for row in connection.execute("SELECT DISTINCT zluda_version FROM modules")}
        connection.close()
        return rows

    mine, theirs = versions(source), versions(destination)
    if mine != theirs:
        raise SystemExit("the caches come from different ZLUDA builds (%s and %s)" % (sorted(mine), sorted(theirs)))

    backup = os.path.join(os.path.dirname(into.rstrip("\/")), "zluda2.db.before-merge")
    shutil.copyfile(destination, backup)
    connection = sqlite3.connect(destination)
    before = connection.execute("SELECT COUNT(*) FROM modules").fetchone()[0]
    connection.execute("ATTACH DATABASE ? AS added", (source,))
    connection.execute(
        "INSERT OR IGNORE INTO modules (hash, compiler_version, zluda_version, device, backend_key, binary, last_access) "
        "SELECT hash, compiler_version, zluda_version, device, backend_key, binary, last_access FROM added.modules")
    connection.commit()
    connection.execute("DETACH DATABASE added")
    after = connection.execute("SELECT COUNT(*) FROM modules").fetchone()[0]
    connection.execute("VACUUM")
    connection.commit()
    connection.close()
    print("%d rows before, %d after (%d added); the previous cache is kept beside the cache directory as %s" % (before, after, after - before, os.path.basename(backup)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
