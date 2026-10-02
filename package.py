#!/usr/bin/env python3
"""Assembles what goes into a game's folder.

The runtime goes in OptiScaler\\dlss5nr\\: OptiScaler takes its OptiScaler\\
subfolder, beside the game's executable, as its own directory, and looks for
the runtime there.

    python package.py --zluda <dir> [--optiscaler <dir>] [--runtime <dir>]
                      [--cache <dir> | --no-cache] [--no-network] [--out dist]

--zluda      a ZLUDA build: its nvcuda.dll becomes zluda\\zluda_real.dll, and
             its nvapi64.dll comes along
--optiscaler the OptiScaler fork's build output (OptiScaler.dll and its files),
             copied to the top of the package. Its setup_linux.sh is then replaced
             by the one in optiscaler\\, the version this repository points to: the
             build output keeps the copy it was last built with.
--runtime    where dlss5nr_runtime.dll, zluda\\nvcuda.dll (the gate) and zluda\\nvngx.dll
             come from: runtime\\build\\bin, the Visual Studio build, unless it is
             runtime\\build-mingw\\bin, the mingw-w64 one (runtime/build.sh). The cache
             is still checked against runtime\\build\\bin\\zluda\\zluda_real.dll.
--cache      the translation cache built by runtime\\tools\\prepare_cache.py
             (default runtime\\build\\shipped\\ComputeCache), shipped as
             OptiScaler\\dlss5nr\\cache\\ComputeCache so that no player waits for the
             network's code to be translated. It only answers the ZLUDA build
             that produced it: rebuild it whenever a different ZLUDA is shipped.
--no-network leaves NVIDIA's network out, to be added to OptiScaler\\dlss5nr\\
             by hand.

The Linux side (amdhip64_7.dll and libdlss5nr_hip_bridge.so from linux\\build, built
with linux/build.sh, plus launch.sh and README.txt) goes into linux\\ beside
OptiScaler.dll, where setup_linux.sh finds it. It is left out, with a warning, when
it has not been built or its sources are newer than what was built.

The runtime (runtime\\build\\bin) and the native kernels (kernels\\, built with
build_kernels.py) must be built first. Only the kernels the manifest names are
packaged. NVIDIA's network, third_party\\nvngx_dlssnr.dll, goes into
OptiScaler\\dlss5nr\\ as it is.
"""
import argparse
import filecmp
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def copy(source, target):
    if not os.path.exists(source):
        sys.exit("missing: %s" % source)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    shutil.copy2(source, target)


def newest(folder, skip=("build", "__pycache__")):
    """The latest modification time of any file under folder."""
    latest = 0.0
    for where, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if d not in skip]
        for name in files:
            latest = max(latest, os.path.getmtime(os.path.join(where, name)))
    return latest


def package_linux(out):
    """dist\\linux: what a Linux install adds. None of it belongs beside a Windows
    install (a stray amdhip64_7.dll beside the executable would be loaded by ZLUDA
    instead of AMD's), which is why it has a folder of its own."""
    linux = os.path.join(ROOT, "linux")
    built = [os.path.join(linux, "build", "amdhip64_7.dll"), os.path.join(linux, "build", "libdlss5nr_hip_bridge.so")]
    missing = [b for b in built if not os.path.isfile(b)]
    if missing:
        print("warning: the Linux side is not built (sh linux/build.sh), so dist/linux is left out: missing %s"
              % ", ".join(os.path.basename(m) for m in missing))
        return
    sources = max(newest(os.path.join(linux, "bridge")), newest(os.path.join(linux, "trampoline")))
    if any(os.path.getmtime(b) < sources for b in built):
        print("warning: linux/bridge or linux/trampoline changed after the Linux side was built: "
              "run sh linux/build.sh again")
    target = os.path.join(out, "linux")
    for b in built:
        copy(b, os.path.join(target, os.path.basename(b)))
    copy(os.path.join(linux, "launch.sh"), os.path.join(target, "launch.sh"))
    copy(os.path.join(linux, "PACKAGE_README.txt"), os.path.join(target, "README.txt"))
    os.chmod(os.path.join(target, "launch.sh"), 0o755)
    os.chmod(os.path.join(target, "libdlss5nr_hip_bridge.so"), 0o755)
    print("packaged the Linux side into %s" % target)


def make_scripts_executable(out):
    """Every .sh in the package runs as it is on Linux. Whatever the package was
    assembled on, the copies come out without the bit: the OptiScaler build's
    setup_linux.sh is made on Windows, and a checkout on one does not keep it."""
    for where, _, files in os.walk(out):
        for name in files:
            if name.endswith(".sh"):
                path = os.path.join(where, name)
                os.chmod(path, os.stat(path).st_mode | 0o111)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zluda", required=True)
    ap.add_argument("--optiscaler")
    ap.add_argument("--runtime", default=os.path.join(ROOT, "runtime", "build", "bin"))
    ap.add_argument("--cache", default=os.path.join(ROOT, "runtime", "build", "shipped", "ComputeCache"))
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-network", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "dist"))
    args = ap.parse_args()

    runtime = args.runtime
    kernels = os.path.join(ROOT, "kernels")
    target = os.path.join(args.out, "OptiScaler", "dlss5nr")
    # Checked before the old package is removed, which a missing network
    # would otherwise leave half rebuilt.
    network = os.path.join(ROOT, "third_party", "nvngx_dlssnr.dll")
    if not args.no_network and not os.path.isfile(network):
        sys.exit("missing: %s (NVIDIA's network, version 310.8.0.0, is not in the repository: "
                 "put it there before packaging, or pass --no-network)" % network)
    if os.path.exists(args.out):
        shutil.rmtree(args.out)

    copy(os.path.join(runtime, "dlss5nr_runtime.dll"), os.path.join(target, "dlss5nr_runtime.dll"))
    copy(os.path.join(runtime, "zluda", "nvcuda.dll"), os.path.join(target, "zluda", "nvcuda.dll"))
    copy(os.path.join(runtime, "zluda", "nvngx.dll"), os.path.join(target, "zluda", "nvngx.dll"))
    copy(os.path.join(args.zluda, "nvcuda.dll"), os.path.join(target, "zluda", "zluda_real.dll"))
    copy(os.path.join(args.zluda, "nvapi64.dll"), os.path.join(target, "zluda", "nvapi64.dll"))
    # Where the runtime looks for it when the host names no other path. It is
    # not in the repository: whoever packages keeps a copy in third_party\.
    if not args.no_network:
        copy(network, os.path.join(target, "nvngx_dlssnr.dll"))

    if not args.no_cache:
        database = os.path.join(args.cache, "zluda2.db")
        if not os.path.isfile(database):
            sys.exit("missing: %s (build it with runtime\\tools\\prepare_cache.py, or pass --no-cache)"
                     % database)
        # The cache is keyed by ZLUDA's version: one translated by another
        # build is never read, and every player translates from scratch.
        built_with = os.path.join(ROOT, "runtime", "build", "bin", "zluda", "zluda_real.dll")
        if not os.path.isfile(built_with) or not filecmp.cmp(built_with, os.path.join(args.zluda, "nvcuda.dll"),
                                                           shallow=False):
            sys.exit("the cache was built with %s, which is not the ZLUDA being packaged: rebuild the "
                     "cache with this ZLUDA, or pass --no-cache" % built_with)
        copy(database, os.path.join(target, "cache", "ComputeCache", "zluda2.db"))
        # What the runtime checks the GPU against before it starts the network.
        coverage = os.path.join(args.cache, "coverage.txt")
        if not os.path.isfile(coverage):
            sys.exit("missing: %s (rebuild the cache with runtime\\tools\\prepare_cache.py)" % coverage)
        copy(coverage, os.path.join(target, "cache", "ComputeCache", "coverage.txt"))

    manifest = os.path.join(kernels, "kernels.txt")
    copy(manifest, os.path.join(target, "kernels", "kernels.txt"))
    objects = set()
    for line in open(manifest, encoding="utf-8"):
        fields = line.split("#", 1)[0].split()
        if len(fields) >= 2:
            objects.add(fields[1])
    for name in sorted(objects):
        copy(os.path.join(kernels, name), os.path.join(target, "kernels", name))

    if args.optiscaler:
        for name in os.listdir(args.optiscaler):
            source = os.path.join(args.optiscaler, name)
            if name.lower().endswith((".pdb", ".lib", ".exp", ".ilk")):
                continue
            if os.path.isdir(source):
                shutil.copytree(source, os.path.join(args.out, name), dirs_exist_ok=True)
            else:
                shutil.copy2(source, os.path.join(args.out, name))
        setup = os.path.join(ROOT, "optiscaler", "setup_linux.sh")
        if os.path.isfile(setup):
            shutil.copy2(setup, os.path.join(args.out, "setup_linux.sh"))

    for name in ("LICENSE", "THIRD_PARTY.md", "README.md"):
        copy(os.path.join(ROOT, name), os.path.join(target, name))
    package_linux(args.out)
    make_scripts_executable(args.out)
    print("packaged %d native kernels%s into %s"
          % (len(objects), "" if args.no_cache else " and the translation cache", args.out))


if __name__ == "__main__":
    main()
