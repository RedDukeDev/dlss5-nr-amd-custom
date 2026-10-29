#!/usr/bin/env python3
"""Finds the modules a translation cache is missing and translates them into it.

    python fill_cache.py <cache dir> <work dir> [--libs nvngx_dlss,nvngx_dlssd,nvngx_dlssg]
                         [--target gfx11-generic] [--driver zluda_real.dll]

<cache dir> holds the zluda2.db to check; it is changed in place (keep a copy). Every module of the libraries in
third_party\\ is loaded through ZLUDA, one process each, for one GPU target. A module the cache answers loads in about
half a second; one it does not answer is translated and filed, which takes seconds to minutes and is printed with the
number of rows it added. Run it again and every line should say "+0 rows" and take under a second.

Why this and not the harnesses: the cache key depends on the module's PTX and on how ZLUDA lowers it (the
ptx_impl_revision in the key, which differs from module to module), and a library holds several networks. A harness
loads the modules of the network it runs. A package checked only with the presets one harness uses can miss the
modules of another, and then ZLUDA translates them in the game, on the render thread, for the card's own target:
the game freezes for minutes. This looks at all of them.

After it, delete the rows an older revision of the same module left behind (they are never read):

    DELETE FROM modules WHERE EXISTS (SELECT 1 FROM modules n WHERE n.hash = modules.hash
      AND n.device = modules.device AND n.id <> modules.id
      AND CAST(SUBSTR(n.backend_key, INSTR(n.backend_key, 'revision":') + 10) AS INTEGER)
        > CAST(SUBSTR(modules.backend_key, INSTR(modules.backend_key, 'revision":') + 10) AS INTEGER));

The neural rendering network (nvngx_dlssnr.dll) is not in the default list: it has no texture or surface
instructions, and its rows are made for gfx1100 as well as the generic target by prepare_cache.py.
"""
import argparse
import importlib.util
import os
import sqlite3
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.abspath(os.path.join(HERE, "..", ".."))

spec = importlib.util.spec_from_file_location("prepare_cache", os.path.join(HERE, "prepare_cache.py"))
prepare_cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_cache)


def rows_in(database):
    connection = sqlite3.connect(database)
    try:
        return connection.execute("SELECT COUNT(*) FROM modules").fetchone()[0]
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cache")
    parser.add_argument("work")
    parser.add_argument("--libs", default="nvngx_dlss,nvngx_dlssd,nvngx_dlssg")
    parser.add_argument("--target", default="gfx11-generic")
    parser.add_argument("--driver", default=os.path.join(PROJECT, "runtime", "build", "bin", "zluda", "zluda_real.dll"))
    parser.add_argument("--one", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.one:
        # The child process: one module, one target.
        return prepare_cache.translate_one(args.driver, args.one)

    database = os.path.join(args.cache, "zluda2.db")
    if not os.path.isfile(database):
        raise SystemExit("not found: " + database)
    # ZLUDA reads its settings once per process, so every switch of the parent's is dropped but these two.
    environment = {k: v for k, v in os.environ.items() if not k.upper().startswith("ZLUDA_")}
    environment["ZLUDA_TARGET_ARCH"] = args.target
    environment["ZLUDA_CACHE_DIR"] = os.path.abspath(args.cache)

    translated = 0
    for library in args.libs.split(","):
        directory = os.path.join(args.work, library)
        os.makedirs(directory, exist_ok=True)
        for module in prepare_cache.extract_modules(os.path.join(PROJECT, "third_party", library + ".dll"), directory):
            before = rows_in(database)
            started = time.time()
            result = subprocess.run([sys.executable, os.path.abspath(__file__), "--one", module, "--driver", args.driver,
                                     args.cache, args.work], env=environment, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)
            added = rows_in(database) - before
            translated += added > 0
            print("%-14s %-34s exit %d  %6.1f s  +%d rows" % (library, os.path.basename(module), result.returncode,
                                                             time.time() - started, added), flush=True)
            if result.returncode not in (0, 1) or (result.returncode == 1 and added == 0):
                print("   " + result.stdout.strip().replace("\n", "\n   ")[-400:], flush=True)
    print("%d modules had to be translated" % translated)
    return 0


if __name__ == "__main__":
    sys.exit(main())
