#!/usr/bin/env python3
"""Reads the Linux branch of the bridge with the Windows compiler, against stand-in headers.

The bridge is built with gcc on Linux, which not every machine has. This only
catches what a compiler catches without running the code -- a wrong type, a
missing declaration, a misspelt name -- in the part of core.c that a Windows
build never compiles.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LINUX = os.path.dirname(HERE)
VS = os.environ.get("VS_DIR", r"C:\Program Files\Microsoft Visual Studio\18\Community")
VCVARS = os.path.join(VS, "VC", "Auxiliary", "Build", "vcvars64.bat")
STUBS = os.path.join(HERE, "stubs")

for source in (os.path.join("bridge", "core.c"), os.path.join("bridge", "generated", "thunks.c")):
    command = ('call "%s" >nul && cl /nologo /Zs /W3 /D_CRT_SECURE_NO_WARNINGS /I"%s" /FI"%s" "%s"'
               % (VCVARS, STUBS, os.path.join(STUBS, "prelude.h"), os.path.join(LINUX, source)))
    if subprocess.run(command, shell=True, cwd=LINUX).returncode:
        sys.exit("%s does not read cleanly" % source)
print("the Linux branch reads cleanly")
