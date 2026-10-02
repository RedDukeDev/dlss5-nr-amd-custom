#!/bin/sh
# Starts a game under Proton with the HIP bridge loaded.
#
# In Steam, as the game's launch option:
#
#     /path/to/launch.sh %command%
#
# It preloads the bridge into every process of the session (the bridge does
# nothing until the trampoline asks it for something), and tells Steam's
# container about the folders it has to be able to see.
#
# Needs: libamdhip64.so.7 from ROCm 7 on this system. If it is not on the
# library path, say where with DLSSNR_HIP_LIB=/opt/rocm/lib/libamdhip64.so.7.
# DLSSNR_BRIDGE_LOG=1 makes the bridge say what it does, on the game's stderr.

here=$(dirname "$(readlink -f "$0")")
bridge="$here/libdlss5nr_hip_bridge.so"

if [ ! -f "$bridge" ]; then
    echo "launch.sh: $bridge is missing: build it with  python linux/build.py bridge" >&2
    exit 1
fi

# Steam's container only shows what it is told to: the bridge's folder, and
# the folder ROCm is in.
# Inside Steam's container the loader knows nothing of ROCm's folder, so the
# library is named by its path, not by its soname.
if [ -z "$DLSSNR_HIP_LIB" ]; then
    for dir in /opt/rocm/core/lib /opt/rocm/lib /usr/lib /usr/lib64 /usr/lib/x86_64-linux-gnu; do
        if [ -e "$dir/libamdhip64.so.7" ]; then
            DLSSNR_HIP_LIB="$dir/libamdhip64.so.7"
            break
        fi
    done
fi
[ -n "$DLSSNR_HIP_LIB" ] && export DLSSNR_HIP_LIB

# ROCm needs more than its library (device bitcode in amdgcn/, LLVM beside it),
# so the whole of /opt/rocm is shown when that is where it is.
mounts="$here"
case "$DLSSNR_HIP_LIB" in
    /opt/rocm/*) mounts="$mounts:/opt/rocm" ;;
    "") [ -d /opt/rocm ] && mounts="$mounts:/opt/rocm" ;;
    *) mounts="$mounts:$(dirname "$(readlink -f "$DLSSNR_HIP_LIB")")" ;;
esac
export STEAM_COMPAT_MOUNTS="${STEAM_COMPAT_MOUNTS:+$STEAM_COMPAT_MOUNTS:}$mounts"

export LD_PRELOAD="$bridge${LD_PRELOAD:+:$LD_PRELOAD}"

# The CUDA driver here is ours (zluda\nvcuda.dll), not Wine's. Proton sets
# nvcuda=b whenever NVAPI is on and puts it after anything set here, so NVAPI
# has to be off for nvcuda=n to hold. (OptiScaler brings its own NVAPI.)
export PROTON_DISABLE_NVAPI=1
export WINEDLLOVERRIDES="nvcuda=n${WINEDLLOVERRIDES:+;$WINEDLLOVERRIDES}"
exec "$@"
