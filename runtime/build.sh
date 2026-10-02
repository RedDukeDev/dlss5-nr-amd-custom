#!/bin/sh
# Builds the runtime on Linux with mingw-w64, into build-mingw/bin laid out as
# build.bat's build/bin is: dlss5nr_runtime.dll, zluda/nvcuda.dll (the gate),
# zluda/nvngx.dll, the harness and the interop probe.
#
#   sh runtime/build.sh
#
# Needs mingw-w64 (x86_64-w64-mingw32-g++), cmake, ninja and python 3. The sources
# are the Visual Studio build's; what differs is the compiler (see ngx_cuda.h for
# the one place that has to know). Programs are not run here: they run under Wine.
set -e
here=$(cd "$(dirname "$0")" && pwd)
cd "$here"
cmake -G Ninja -B build-mingw -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=mingw.cmake
cmake --build build-mingw
