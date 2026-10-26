@echo off
rem Builds the frame generation NVAPI stand-in on its own, for the harness: nvapi64.dll in out\.
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d %~dp0
if not exist out mkdir out
cl /nologo /std:c++17 /O2 /EHsc /MT /W3 /utf-8 /D_CRT_SECURE_NO_WARNINGS /LD nvapi_fg.cpp ..\core\convert.cpp /Fo:out\ /Fe:out\nvapi64.dll /link d3d12.lib dxgi.lib d3dcompiler.lib
