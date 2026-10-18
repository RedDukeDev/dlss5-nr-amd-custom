#!/usr/bin/env python3
r"""Builds the bridge for 32-bit games with MSVC.

    python build.py

Output:
    build\x86\d3d9.dll            the client, for a 32-bit game's folder
    build\x64\dlss5nr_host.exe    the host, which goes beside dlss5nr_runtime.dll

The client is 32-bit and the host 64-bit, so each is compiled in its own environment. The
compiler is cl.exe straight from Visual Studio's, as for the other small tools in this repository.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
VCVARS = r"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat"


def run(arch, out_dir, command):
    out = os.path.join(ROOT, "build", out_dir)
    os.makedirs(out, exist_ok=True)
    script = 'call "%s" %s >nul && cd /d "%s" && %s' % (VCVARS, arch, out, command)
    print("== %s" % out_dir)
    # One string, so that cmd sees the quotes as they are: with a list Python escapes them.
    result = subprocess.run('cmd /c "%s"' % script, capture_output=True, text=True)
    print(result.stdout[-3000:])
    if result.returncode != 0:
        print(result.stderr[-3000:])
        sys.exit("build failed: %s" % out_dir)


common = "/nologo /O2 /MT /EHsc /utf-8 /W3 /DUNICODE /D_UNICODE /DNOMINMAX /D_CRT_SECURE_NO_WARNINGS"
run("x86", "x86", 'cl %s /LD "%s" /Fe:d3d9.dll /link /OPT:REF kernel32.lib user32.lib' %
    (common, os.path.join(ROOT, "client", "d3d9_client.cpp")))
run("x64", "x64", 'cl %s "%s" /Fe:dlss5nr_host.exe /link /OPT:REF d3d12.lib dxgi.lib' %
    (common, os.path.join(ROOT, "host", "host.cpp")))
print("built")
