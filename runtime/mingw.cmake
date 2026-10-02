# Toolchain file for building the runtime on Linux with mingw-w64 (see build.sh).
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)
set(CMAKE_C_COMPILER x86_64-w64-mingw32-gcc)
set(CMAKE_CXX_COMPILER x86_64-w64-mingw32-g++)
# Python runs on the build machine, whatever it is that is being built.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
