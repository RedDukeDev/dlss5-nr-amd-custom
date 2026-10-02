#!/usr/bin/env python3
"""Builds the pieces that let the network run under Wine and Proton.

  python linux/build.py trampoline    build\\amdhip64_7.dll, the Windows side (needs MSVC; any OS
                                      that has it -- it is an ordinary Windows DLL)
  python linux/build.py bridge        build/libdlss5nr_hip_bridge.so, the native side (needs gcc, on Linux)
  python linux/build.py test-bridge   build\\dlss5nr_hip_bridge_win.dll, the bridge as a Windows DLL that
                                      calls AMD's own HIP runtime: for testing the chain on Windows

Everything goes to linux/build. The forwarders in the generated folders come from
gen_forwarders.py and are kept in the repository, so none of this needs ZLUDA's
sources.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "build")
VS = os.environ.get("VS_DIR", r"C:\Program Files\Microsoft Visual Studio\18\Community")


def msvc(arguments, workdir):
    vcvars = os.path.join(VS, "VC", "Auxiliary", "Build", "vcvars64.bat")
    command = 'call "%s" >nul && cl /nologo %s' % (vcvars, arguments)
    result = subprocess.run(command, cwd=workdir, shell=True)
    if result.returncode:
        sys.exit("the compiler failed")


def trampoline():
    # The static runtime: this DLL is loaded into a Wine prefix, where nothing
    # should have to supply the compiler's.
    msvc('/O2 /MT /LD /W3 /D_CRT_SECURE_NO_WARNINGS "%s" "%s" /Fo:"%s\\\\" /Fe:"%s"' %
         (os.path.join(HERE, "trampoline", "core.c"),
          os.path.join(HERE, "trampoline", "generated", "exports.c"),
          OUT, os.path.join(OUT, "amdhip64_7.dll")), HERE)


def test_bridge():
    msvc('/O2 /MT /LD /W3 /D_CRT_SECURE_NO_WARNINGS /DDLSSNR_WIN_TEST "%s" "%s" /Fo:"%s\\\\" /Fe:"%s"' %
         (os.path.join(HERE, "bridge", "core.c"),
          os.path.join(HERE, "bridge", "generated", "thunks.c"),
          OUT, os.path.join(OUT, "dlss5nr_hip_bridge_win.dll")), HERE)


def bridge():
    # -D_GNU_SOURCE: dlfcn's RTLD_DEFAULT and friends. The library is loaded by
    # LD_PRELOAD into every process of the Wine session, so it asks nothing of
    # them: libc and libdl only.
    command = ["gcc", "-O2", "-fPIC", "-shared", "-Wall", "-D_GNU_SOURCE", "-o",
               os.path.join(OUT, "libdlss5nr_hip_bridge.so"),
               os.path.join(HERE, "bridge", "core.c"),
               os.path.join(HERE, "bridge", "generated", "thunks.c"), "-ldl"]
    if subprocess.run(command).returncode:
        sys.exit("the compiler failed")


def main():
    targets = {"trampoline": trampoline, "bridge": bridge, "test-bridge": test_bridge}
    if len(sys.argv) != 2 or sys.argv[1] not in targets:
        sys.exit(__doc__)
    os.makedirs(OUT, exist_ok=True)
    targets[sys.argv[1]]()
    print("built into", OUT)


if __name__ == "__main__":
    main()
