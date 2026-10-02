"""Where the tools find what is not part of this repository.

Nothing derived from NVIDIA's network is committed here: captured launches,
extracted PTX, translated caches. The tools that need them read their
locations from the environment:

  DLSS5NR_SNIPPET   nvngx_dlssnr.dll (310.8.0.0), the network itself; by
                    default the one in third_party\\
  DLSS5NR_HARNESS   a directory with the headless runner (dlss_image.exe), the
                    research proxy as nvcuda.dll (kernels/harness), ZLUDA as
                    zluda_real.dll and its translation cache (cache_per_matrix)
  DLSS5NR_ZLUDA     ZLUDA's nvcuda.dll, when not the harness's zluda_real.dll
  DLSS5NR_CAPTURES  the data directory some checks read by name: captures
                    (capture_pre16, capture_post16) and kernels' extracted PTX
                    (pre_block_32.ptx, post_block_32.ptx, ffwd_*.ptx)

Captures are passed to each tool as a directory on the command line.
"""
import os
import sys


def _required(name, default=None):
    value = os.environ.get(name, default)
    if not value:
        sys.exit("%s is not set: see kernels/tools/workspace.py" % name)
    return value


def harness():
    return _required("DLSS5NR_HARNESS")


def snippet():
    shipped = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "third_party",
                           "nvngx_dlssnr.dll")
    return _required("DLSS5NR_SNIPPET", os.path.normpath(shipped) if os.path.isfile(shipped) else None)


def zluda():
    return os.environ.get("DLSS5NR_ZLUDA") or os.path.join(harness(), "zluda_real.dll")


def captures():
    return _required("DLSS5NR_CAPTURES")


def kernels():
    """The kernels directory of this repository."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
