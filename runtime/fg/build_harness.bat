@echo off
rem Builds the frame generation harness with d4r's command hooks and MinHook compiled in.
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d %~dp0
rem The OptiScaler fork to take d4r's hooks from: the submodule, or a working copy (set OPTI first).
if not defined OPTI set OPTI=%~dp0..\..\optiscaler
if not exist out mkdir out
if not exist out\hook mkdir out\hook
cl /nologo /O2 /MT /W3 /c /Fo:out\hook\ "%OPTI%\external\MinHook\src\hook.c" "%OPTI%\external\MinHook\src\buffer.c" "%OPTI%\external\MinHook\src\trampoline.c" "%OPTI%\external\MinHook\src\hde\hde64.c" /I"%OPTI%\external\MinHook\include" /I"%OPTI%\external\MinHook\src" || exit /b 1
cl /nologo /std:c++17 /O2 /EHsc /MT /W3 /bigobj /utf-8 /D_CRT_SECURE_NO_WARNINGS /DUNICODE /D_UNICODE ^
   ..\tests\fg_harness.cpp "%OPTI%\OptiScaler\zluda\d4r_commands\d3d12_command_hooks.cpp" out\hook\*.obj ^
   /I"%OPTI%\OptiScaler\zluda\d4r_commands" /I"%OPTI%\external\MinHook\include" ^
   /Fo:out\ /Fe:out\dlss5nr_fg_harness.exe /link d3d12.lib dxgi.lib dxguid.lib
