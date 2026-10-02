#!/bin/sh
# Builds the Linux side of neural rendering into linux/build/:
#
#   libdlss5nr_hip_bridge.so   the native half of the HIP bridge (gcc)
#   amdhip64_7.dll             its Windows half, the trampoline (mingw-w64)
#
# package.py picks both up from there and puts them in dist/linux/ with
# launch.sh and README.txt.
#
#   sh linux/build.sh            the two files above
#   sh linux/build.sh runtime    also builds dlss5nr_runtime.dll with mingw into
#                                linux/build/mingw/, to try the runtime on a
#                                machine without Visual Studio. It is not packaged:
#                                the package has the Visual Studio build.
#
# Needs gcc, mingw-w64 (x86_64-w64-mingw32-gcc) and python 3.
set -e
here=$(cd "$(dirname "$0")" && pwd)
root=$(dirname "$here")
out="$here/build"
mkdir -p "$out"

python3 "$here/build.py" bridge

# -static: nothing of mingw's has to be in a prefix. The trampoline has to be a
# PE DLL named like AMD's.
x86_64-w64-mingw32-gcc -O2 -shared -static -D_CRT_SECURE_NO_WARNINGS \
    -o "$out/amdhip64_7.dll" "$here/trampoline/core.c" "$here/trampoline/generated/exports.c"

if [ "$1" = "runtime" ]; then
    mkdir -p "$out/mingw"
    x86_64-w64-mingw32-g++ -std=c++17 -O2 -shared -static -DUNICODE -D_UNICODE -DNOMINMAX \
        -D_CRT_SECURE_NO_WARNINGS -DDLSS5NR_BUILD \
        -o "$out/mingw/dlss5nr_runtime.dll" "$root/runtime/core/runtime.cpp" "$root/runtime/core/network.cpp" \
        "$root/runtime/core/passes.cpp" -ld3d12 -ld3dcompiler_47 -ldxgi -lole32
fi

echo "built into $out"
