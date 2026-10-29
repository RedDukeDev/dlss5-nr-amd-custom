# Third-party work

## RenoDX DLSS 5 add-on

The runtime's colour handling (`runtime/core/shaders.h`) follows the design of
the DLSS 5 add-on in RenoDX by clshortfuse
(https://github.com/clshortfuse/renodx): the network is shown a display encoded
proxy of the frame, and its answer is composed back as a two-branch luminance
ratio against the frame, with the network's hue restored in OkLab, a blend
between a luminance-only result and the network's colour, and out-of-gamut
colours moved towards neutral. We learned of it through the OptiScaler-DLSSNR
fork that adopted it. It is their design, reimplemented here; no code was
copied.

## d4r

The upscaler side of this project started from d4r by countervolts (https://github.com/countervolts/d4r, GPL-3):
it showed that NVIDIA's DLSS upscalers can run on AMD through ZLUDA with a command-list split around the
upscale, and its hand-written kernels are what the native super resolution layers are. What we use of it:

- **Super resolution kernels:** the eleven `dltss_pwin_*_layer` kernels (preset K) and the five Swin blocks
  (presets L and M) are d4r's WMMA code. `kernels/build_sr_kernels.py` builds them from the `third_party/d4r` submodule
  (the `windows` branch); the compiled kernels in the release package are built from
  GPL-3 code and stay under that licence.
- **Command-list module:** `zluda/d4r_commands` in the OptiScaler fork (the split of the game's command list
  at a boundary, `record_boundary`, resource states, retirement) is d4r's, used by the DLSS upscaler feature
  and the frame generation stand-in. The fork is GPL-3 too.

The shared pieces around them (ZLUDA fork, gate, NGX host, NVAPI stand-in, runtime) are ours.

## OkLab

The OkLab conversion uses the matrices published by Bjorn Ottosson
(https://bottosson.github.io/posts/oklab/), public domain or MIT.

## ZLUDA

`third_party/ZLUDA` is a submodule and keeps its own MIT / Apache-2.0 licence.

## DXVK

`dxvk/x64` holds the 64-bit `d3d9.dll`, `d3d10core.dll`, `d3d11.dll` and `dxgi.dll` of DXVK 3.1.1
(https://github.com/doitsujin/dxvk, zlib licence, in `dxvk/LICENSE`), unmodified, as released.
They are shipped so that Direct3D 9 and 10 games can run on Vulkan and use the neural rendering;
`dxvk/README.txt` says where they come from and how to check them.

## Dear ImGui

`bridge32/third_party/imgui` is Dear ImGui 1.92.0 by Omar Cornut (https://github.com/ocornut/imgui,
tag v1.92.0, MIT licence in `LICENSE.txt` there), with its DirectX 9 and Win32 backends, unmodified.
It draws the menu of the 32-bit bridge, the same library OptiScaler's menu uses.
