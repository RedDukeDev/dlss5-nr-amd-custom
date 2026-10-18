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


common = "/nologo /O2 /MT /EHsc /std:c++17 /utf-8 /W3 /DUNICODE /D_UNICODE /DNOMINMAX /D_CRT_SECURE_NO_WARNINGS"
def compile_shaders():
    """The shaders are compiled to headers with fxc, from the Windows SDK, and the headers are kept in
    the repository, so that a machine without fxc builds with them as they are."""
    import glob
    found = sorted(glob.glob(os.path.join("C:"+os.sep, "Program Files (x86)", "Windows Kits", "10", "bin", "*", "x64", "fxc.exe")))
    if not found:
        print("fxc not found: the shader headers are used as they are")
        return
    folder = os.path.join(ROOT, "client", "shaders")
    for name, profile, variable in (("gain_plain", "ps_2_0", "kGainPlain"), ("gain_mc", "ps_2_0", "kGainMc"),
                                    ("down", "ps_2_0", "kDown"), ("me_coarse", "ps_3_0", "kMeCoarse"),
                                    ("me_refine", "ps_3_0", "kMeRefine"), ("me_median", "ps_3_0", "kMeMedian"),
                                    ("quad_vs", "vs_3_0", "kQuadVs")):
        result = subprocess.run([found[-1], "-nologo", "-T", profile, "-E", "main", "-Fh", name + ".h", "-Vn", variable, name + ".hlsl"],
                                cwd=folder, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout + result.stderr)
            sys.exit("shader failed: " + name)


compile_shaders()
imgui = os.path.join(ROOT, "third_party", "imgui")
client = [os.path.join(ROOT, "client", name) for name in ("d3d9_client.cpp", "menu.cpp", "motion.cpp", "input.cpp")]
client += [os.path.join(imgui, name) for name in (
    "imgui.cpp", "imgui_draw.cpp", "imgui_tables.cpp", "imgui_widgets.cpp", "imgui_impl_dx9.cpp")]
run("x86", "x86", 'cl %s /I"%s" /LD %s /Fe:d3d9.dll /link /OPT:REF kernel32.lib user32.lib gdi32.lib imm32.lib dwmapi.lib' %
    (common, imgui, " ".join('"%s"' % f for f in client)))
run("x86", "x86_test", 'cl %s /Fe:me_test.exe "%s" "%s" /link /OPT:REF d3d9.lib user32.lib' %
    (common, os.path.join(ROOT, "tests", "me_test.cpp"), os.path.join(ROOT, "client", "motion.cpp")))
run("x64", "x64", 'cl %s "%s" /Fe:dlss5nr_host.exe /link /OPT:REF d3d12.lib dxgi.lib' %
    (common, os.path.join(ROOT, "host", "host.cpp")))
print("built")
