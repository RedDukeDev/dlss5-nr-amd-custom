# Release notes: since 1.2.0

## NVIDIA's DLSS on AMD

- **Super resolution and ray reconstruction** (`nvngx_dlss.dll`, `nvngx_dlssd.dll`) run on the same runtime as the neural
  rendering, and OptiScaler offers them as the DLSS upscaler on a GPU that is not NVIDIA's.
- **Ray reconstruction is about 2.4 times faster** (1280x720 to 1920x1080 on an RX 7900 XT: 80 ms a frame down to 35).
  Sixteen of the network's kernels now run native: encoders 1 to 5 and decoders 4 to 0 from one hand-written stage, and
  the first encoder and the last decoder through code translated at build time from the library's own. Every one of them
  gives, byte for byte, what the translated kernel gave, checked on captured launches and, over nine frames, on the
  final image.
- **NVIDIA's frame generation** runs through a stand-in NVAPI with 27 native kernels.
- Ray reconstruction in games: optional camera matrices, and presets under their own names.

## Linux

- Textures are shared with ROCm where Wine allows it (a dma-buf from the D3D12 resource), with a copy as the fallback;
  the upscalers and frame generation run under Proton.
- Under Wine a game with no driver version in DXGI is given one when an NVIDIA GPU is spoofed.

## OptiScaler

- Merged with upstream's latest. The DXGI hooks stay in place when the neural rendering is enabled or its runtime is
  installed, so it can be switched on from the menu with `OverlayMenu=false`.
- The DLSS parameter table in use is no longer freed after switching to it.
- The log says what the game asks about ray reconstruction and whether it is offered.

## ZLUDA

- Merged with upstream's latest (surfaces, GameWorks support). Raw surface stores and texel fetches for the DLSS
  upscalers are kept on top. The translation cache is made again for this build.
- A surface object now carries the log2 of its pixel size in its three bottom bits (upstream's surfaces). The frame
  generation stand-in, which reads the object's descriptor page to make 8 and 16 bit surfaces normalised, masks them
  off; without that the generated frame came out empty.

## For people who build or debug it

- `kernels/build_rr_kernels.py` builds the ray reconstruction kernels. The two generated ones are made from the user's
  own `nvngx_dlssd.dll` and are never in the repository.
- The gate can capture a launch (`DLSS5NR_GATE_CAPTURE`), compare a native kernel with the translated one in the
  running process (`DLSS5NR_GATE_AB`), and run an instrumented copy of a kernel's PTX (`DLSS5NR_GATE_SWAP_*`).
  `kernels/tools/replay_gpu.py` replays a capture on the GPU from the PTX and logs every matrix multiply.
- The manifest takes `waves=N` to launch a native kernel with more waves per block than the PTX used. Ship the gate and
  the manifests together: an older gate ignores the line and leaves the kernel translated.
- `runtime/tools/fill_cache.py` loads every module of the upscaler and frame generation libraries through ZLUDA and
  translates the ones a cache lacks. It found the modules of ray reconstruction's older presets (D and E), which the
  cache of this release first missed: the game translated them at start-up and froze for minutes.
- `runtime/tools/rekey_cache.py` carries translations over to a newer ZLUDA for modules that have no texture or surface
  instructions.
