@echo off
REM Builds the runtime into build\bin. Needs Visual Studio (C++ and CMake) and
REM Python 3 on the path.
setlocal
set VS=C:\Program Files\Microsoft Visual Studio\18\Community
if not exist "%VS%\VC\Auxiliary\Build\vcvars64.bat" (
    echo Visual Studio not found at "%VS%".
    exit /b 1
)
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d %~dp0
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 exit /b 1
cmake --build build
exit /b %errorlevel%
