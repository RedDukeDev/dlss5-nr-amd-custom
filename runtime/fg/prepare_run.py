"""Assembles the folder the harness runs in: the stand-in as nvapi64.dll, ZLUDA behind it, the gate, the NGX host."""
import os, shutil, sys

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))        # dlss5-nr-amd-custom
work = os.path.abspath(os.path.join(root, ".."))                                     # D:\reshade_dlss5_amd
run = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "out", "run")
zluda = sys.argv[2] if len(sys.argv) > 2 else os.path.join(work, "ZLUDA", "target", "release")
os.makedirs(run, exist_ok=True)
here = os.path.dirname(__file__)
files = {
    os.path.join(here, "out", "nvapi64.dll"): "nvapi64.dll",
    os.path.join(zluda, "nvapi64.dll"): "zluda_nvapi.dll",
    os.path.join(zluda, "nvcuda.dll"): "zluda_real.dll",
    os.path.join(root, "runtime", "build", "bin", "zluda", "nvcuda.dll"): "nvcuda.dll",
    os.path.join(root, "runtime", "build", "bin", "zluda", "nvngx.dll"): "nvngx.dll",
    os.path.join(work, "fg_probe", "cubin_fatbin_map.txt"): "cubin_fatbin_map.txt",
}
for src, name in files.items():
    if not os.path.isfile(src):
        sys.exit("missing " + src)
    shutil.copy2(src, os.path.join(run, name))
for native in ("fg_conv.hsaco", "fg_custom.hsaco"):
    src = os.path.join(work, "fg_probe", "native", native)
    if os.path.isfile(src):
        shutil.copy2(src, os.path.join(run, native))
shutil.copy2(os.path.join(here, "out", "dlss5nr_fg_harness.exe"), os.path.join(run, "dlss5nr_fg_harness.exe"))
print("ready:", run)
