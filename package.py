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
             copied to the top of the package. Its setup_linux.sh and setup_windows.bat
             are then replaced by the ones in optiscaler\\, the version this repository
             points to: the build output keeps the copies it was last built with.
--runtime    where dlss5nr_runtime.dll, zluda\\nvcuda.dll (the gate) and zluda\\nvngx.dll
             come from: runtime\\build\\bin, the Visual Studio build, unless it is
             runtime\\build-mingw\\bin, the mingw-w64 one (runtime/build.sh). The cache
             is still checked against runtime\\build\\bin\\zluda\\zluda_real.dll.
--cache      the translation cache built by runtime\\tools\\prepare_cache.py
             (default runtime\\build\\shipped\\ComputeCache), shipped as
             OptiScaler\\dlss5nr\\cache\\ComputeCache so that no player waits for the
             network's code to be translated. It only answers the ZLUDA build
             that produced it: rebuild it whenever a different ZLUDA is shipped.
--no-network leaves NVIDIA's libraries out (nvngx_dlssnr.dll, the neural rendering;
             nvngx_dlss.dll, super resolution; nvngx_dlssd.dll, ray reconstruction; nvngx_dlssg.dll, frame
             generation), to be
             added to OptiScaler\\dlss5nr\\ by hand.

The Linux side (amdhip64_7.dll and libdlss5nr_hip_bridge.so from linux\\build, built
with linux/build.sh, plus launch.sh and README.txt) goes into linux\\ beside
OptiScaler.dll, where setup_linux.sh finds it. It is left out, with a warning, when
it has not been built or its sources are newer than what was built.

The 32-bit bridge (bridge32\\, built with bridge32\\build.py): its client goes into bridge32\\ and its host
beside the runtime in OptiScaler\\dlss5nr\\.

DXVK (dxvk\\, 64-bit) goes into dxvk\\ beside OptiScaler.dll, for setup_windows.bat to copy
beside a Direct3D 9 or 10 game.

The runtime (runtime\\build\\bin) and the native kernels (kernels\\, built with
build_kernels.py, build_fg_kernels.py and build_rr_kernels.py) must be built first. Only the kernels the manifest names are
packaged. The native super resolution layers (kernels\\sr, built with
build_sr_kernels.py) and their manifest, kernels_sr.txt, go with them. NVIDIA's libraries, in
third_party\\, go into OptiScaler\\dlss5nr\\ as they are. The upscalers' NGX hosts (zluda\\ngx_sr,
zluda\\ngx_rr) are copies of zluda\\nvngx.dll: each network has a folder of its own so that
they can run beside the neural rendering.
"""
import argparse
import filecmp
import hashlib
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

# NVIDIA's libraries, in third_party\\: the neural rendering network, super resolution, ray reconstruction.
NETWORKS = ("nvngx_dlssnr.dll", "nvngx_dlss.dll", "nvngx_dlssd.dll", "nvngx_dlssg.dll")


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


def package_dxvk(out):
    """dist\\dxvk: DXVK, which lets Direct3D 9 and 10 games use the neural rendering (OptiScaler
    has no support for those of its own). setup_windows.bat copies what a game needs from here
    beside its executable, and Remove_OptiScaler.bat takes it away again. 64-bit only: so is
    OptiScaler."""
    dxvk = os.path.join(ROOT, "dxvk")
    target = os.path.join(out, "dxvk")
    for name in ("d3d9.dll", "d3d10core.dll", "d3d11.dll", "dxgi.dll"):
        copy(os.path.join(dxvk, "x64", name), os.path.join(target, "x64", name))
    for name in ("LICENSE", "README.txt"):
        copy(os.path.join(dxvk, name), os.path.join(target, name))
    print("packaged DXVK into %s" % target)


def package_bridge(out):
    """The 32-bit bridge: its two clients go to bridge32\\ (setup_windows.bat copies them beside a 32-bit
    game as d3d9.dll, for Direct3D 9, and dxgi.dll, for Direct3D 10 and 11), its host beside the runtime,
    where the clients look for it. Built with bridge32\\build.py."""
    bridge = os.path.join(ROOT, "bridge32")
    client = os.path.join(bridge, "build", "x86", "d3d9.dll")
    client_dxgi = os.path.join(bridge, "build", "x86_dxgi", "dxgi.dll")
    host = os.path.join(bridge, "build", "x64", "dlss5nr_host.exe")
    missing = [b for b in (client, client_dxgi, host) if not os.path.isfile(b)]
    if missing:
        print("warning: the 32-bit bridge is not built (python bridge32/build.py), so it is left out: missing %s"
              % ", ".join(os.path.basename(m) for m in missing))
        return
    sources = max(newest(os.path.join(bridge, "client")), newest(os.path.join(bridge, "host")), newest(os.path.join(bridge, "common")))
    if any(os.path.getmtime(b) < sources for b in (client, client_dxgi, host)):
        print("warning: the bridge's sources changed after it was built: run python bridge32/build.py again")
    copy(client, os.path.join(out, "bridge32", "d3d9.dll"))
    copy(client_dxgi, os.path.join(out, "bridge32", "dxgi.dll"))
    copy(os.path.join(bridge, "README.txt"), os.path.join(out, "bridge32", "README.txt"))
    copy(host, os.path.join(out, "OptiScaler", "dlss5nr", "dlss5nr_host.exe"))
    print("packaged the 32-bit bridge into %s" % os.path.join(out, "bridge32"))


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
    networks = [os.path.join(ROOT, "third_party", name) for name in NETWORKS]
    missing = [n for n in networks if not os.path.isfile(n)]
    if not args.no_network and missing:
        sys.exit("missing: %s (NVIDIA's libraries are not in the repository: the neural rendering "
                 "network, version 310.9.16.0, and the DLSS ones, 310.9.1, go in third_party\\ "
                 "before packaging; or pass --no-network)" % ", ".join(missing))
    if os.path.exists(args.out):
        shutil.rmtree(args.out)

    copy(os.path.join(runtime, "dlss5nr_runtime.dll"), os.path.join(target, "dlss5nr_runtime.dll"))
    copy(os.path.join(runtime, "zluda", "nvcuda.dll"), os.path.join(target, "zluda", "nvcuda.dll"))
    copy(os.path.join(runtime, "zluda", "nvngx.dll"), os.path.join(target, "zluda", "nvngx.dll"))
    for host in ("ngx_sr", "ngx_rr", "ngx_fg"):
        copy(os.path.join(runtime, "zluda", "nvngx.dll"), os.path.join(target, "zluda", host, "nvngx.dll"))
    copy(os.path.join(args.zluda, "nvcuda.dll"), os.path.join(target, "zluda", "zluda_real.dll"))
    # NVAPI: ZLUDA's stays beside the stand-in for frame generation, which is what the processes ask for by name. It
    # answers what is not frame generation's with ZLUDA's, and brings along the pairing of the library's cubins with
    # their PTX and the native kernels it launches in their place.
    copy(os.path.join(args.zluda, "nvapi64.dll"), os.path.join(target, "zluda", "zluda_nvapi.dll"))
    copy(os.path.join(runtime, "zluda", "nvapi64.dll"), os.path.join(target, "zluda", "nvapi64.dll"))
    copy(os.path.join(ROOT, "runtime", "fg", "data", "cubin_fatbin_map.txt"),
         os.path.join(target, "zluda", "cubin_fatbin_map.txt"))
    for kernel in ("fg_conv.hsaco", "fg_custom.hsaco"):
        built = os.path.join(kernels, kernel)
        if os.path.isfile(built):
            copy(built, os.path.join(target, "zluda", kernel))
        else:
            print("warning: %s is not built (kernels\build_fg_kernels.py): frame generation runs on the translation alone" % kernel)
    # Where the runtime looks for it when the host names no other path. It is
    # not in the repository: whoever packages keeps a copy in third_party\.
    if not args.no_network:
        for network in networks:
            copy(network, os.path.join(target, os.path.basename(network)))

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

    # The super resolution layers: the manifest and the code objects build_sr_kernels.py wrote.
    upscaler_kernels = os.path.join(kernels, "sr")
    upscaler_manifest = os.path.join(upscaler_kernels, "kernels_sr.txt")
    copy(upscaler_manifest, os.path.join(target, "kernels", "kernels_sr.txt"))
    for line in open(upscaler_manifest, encoding="utf-8"):
        fields = line.split("#", 1)[0].split()
        if len(fields) >= 2:
            objects.add(fields[1])
            copy(os.path.join(upscaler_kernels, fields[1]), os.path.join(target, "kernels", fields[1]))

    # The ray reconstruction stages: the manifest (kernels/rr) and the code objects build_kernel.py wrote in
    # kernels/ itself.
    # kernels_rr_generated.txt is there only where tools/build_rr_generated.py has made encoder 0 and the last decoder from the library.
    rr_lines = []
    for name in ("kernels_rr.txt", "kernels_rr_generated.txt"):
        path = os.path.join(kernels, "rr", name)
        if os.path.isfile(path):
            rr_lines += open(path, encoding="utf-8").read().splitlines()
    generated = os.path.join(kernels, "rr", "kernels_rr_generated.txt")
    if not os.path.isfile(generated):
        print("warning: encoder 0 and the last decoder of the ray reconstruction are not built "
              "(python kernels\\build_rr_kernels.py): they run translated, which is a good deal slower")
    else:
        # The generated kernels are made from one build of the library: shipping another one beside them would
        # run code written for kernels that are no longer there.
        recorded = next((l.split()[-1] for l in rr_lines if l.startswith("# library sha256")), None)
        library = os.path.join(ROOT, "third_party", "nvngx_dlssd.dll")
        if recorded and os.path.isfile(library):
            with open(library, "rb") as f:
                actual = hashlib.sha256(f.read()).hexdigest()
            if actual != recorded:
                sys.exit("kernels/rr was generated from another nvngx_dlssd.dll than third_party's: run "
                         "python kernels\\build_rr_kernels.py again")
    with open(os.path.join(target, "kernels", "kernels_rr.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(rr_lines) + "\n")
    for line in rr_lines:
        fields = line.split("#", 1)[0].split()
        if len(fields) >= 2:
            objects.add(fields[1])
            copy(os.path.join(kernels, fields[1]), os.path.join(target, "kernels", fields[1]))

    if args.optiscaler:
        for name in os.listdir(args.optiscaler):
            source = os.path.join(args.optiscaler, name)
            if name.lower().endswith((".pdb", ".lib", ".exp", ".ilk")):
                continue
            if os.path.isdir(source):
                shutil.copytree(source, os.path.join(args.out, name), dirs_exist_ok=True)
            else:
                shutil.copy2(source, os.path.join(args.out, name))
        # The setup scripts come from optiscaler\\, the version this repository points to: the
        # build output keeps the copies it was last linked with.
        for script in ("setup_linux.sh", "setup_windows.bat"):
            setup = os.path.join(ROOT, "optiscaler", script)
            if os.path.isfile(setup):
                shutil.copy2(setup, os.path.join(args.out, script))

    for name in ("LICENSE", "THIRD_PARTY.md", "README.md"):
        copy(os.path.join(ROOT, name), os.path.join(target, name))
    package_linux(args.out)
    package_dxvk(args.out)
    package_bridge(args.out)
    make_scripts_executable(args.out)
    print("packaged %d native kernels%s into %s"
          % (len(objects), "" if args.no_cache else " and the translation cache", args.out))


if __name__ == "__main__":
    main()
