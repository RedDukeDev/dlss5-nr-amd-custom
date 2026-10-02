@echo off
REM Builds the research proxy into build\nvcuda_proxy.dll.
REM
REM This is the instrument the native kernels were developed with: it logs the
REM network's calls, dumps launches (NVCUDA_PROXY_DUMP) and swaps kernels. The
REM product gate, which does only the swapping, is runtime\gate.
REM
REM Note the name it has to be given where it is used: the DLSS snippet loads
REM "nvcuda.dll" by bare name, so the proxy must be copied to that name in a
REM directory of its own, next to the real driver renamed zluda_real.dll.
setlocal
set VS=C:\Program Files\Microsoft Visual Studio\18\Community
if not exist "%VS%\VC\Auxiliary\Build\vcvars64.bat" (
    echo Visual Studio not found at "%VS%".
    exit /b 1
)
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul

set ROOT=%~dp0
set OUT=%ROOT%build
if not exist "%OUT%" mkdir "%OUT%"

cl /nologo /std:c++17 /EHsc /O2 /W3 /D_CRT_SECURE_NO_WARNINGS /LD ^
   "%ROOT%nvcuda_proxy.cpp" ^
   /Fo:"%OUT%\\" /Fe:"%OUT%\nvcuda_proxy.dll" /link /IMPLIB:"%OUT%\nvcuda_proxy.lib"
if errorlevel 1 exit /b 1

del "%OUT%\*.obj" >nul 2>&1
echo Built %OUT%\nvcuda_proxy.dll
exit /b 0
