# DLSS 5 Neural Rendering on AMD - Featuring asyncronous mode (almost no performance loss!) and Linux support

DLSS 5 Neural Rendering, NVIDIA's own network, running in Direct3D 12, Direct3D 11 and Vulkan games on
AMD Radeon GPUs through OptiScaler.

The network is NVIDIA's `nvngx_dlssnr.dll`, included in the release package. Its CUDA code
runs on AMD through [a fork of ZLUDA](https://github.com/RedDukeDev/ZLUDA), and the hottest of its kernels are
replaced by hand-written HIP kernels that produce the same bytes. A runtime
drives it beside the game, and a fork of OptiScaler hands the runtime the
game's frame.

Also, instead of waiting for the dlss network to finish, this implementation is able to run it in an asyncrous way, resulting in a huge perfomance gain over existing implementations.
More infos in [DESIGN.md](docs/DESIGN.md)

## Layout

| folder | what | licence |
|---|---|---|
| [`kernels/`](kernels/) | the native HIP kernels, their verification tools and the research proxy they were built with | MIT |
| [`runtime/`](runtime/) | `dlss5nr_runtime.dll` and its C interface, the gate in front of ZLUDA, our NGX host, the cache preparation tool, a test harness | MIT |
| [`linux/`](linux/) | extra code for Linux support, read the "Linux" section in this redme | MIT |
| [`optiscaler/`](optiscaler/) | submodule: the fork of OptiScaler that calls the runtime | GPL-3.0 |
| [`third_party/ZLUDA`](third_party/) | submodule: the fork of ZLUDA the network runs on | MIT / Apache-2.0 |
| [`docs/`](docs/) | [design](docs/DESIGN.md) | MIT |

How the pieces fit, and why the network runs beside the game rather than
inside its frame, is in [docs/DESIGN.md](docs/DESIGN.md).

## Status / Known Issues

There can be some artifacts in some scenarios, specially ghosting/trail effects.

The raw performance should be comparable to other DLSS5 on AMD implementations, 
for example on RDNA3 it runs at around 20 fps at 1440p with "Before the upsclaer" on and 
async mode off ("Wait for the network" on).
With async mode (enabled by default) this huge performance loss is almost gone

Hardware-specific status:
- **RDNA4** This implementration has been tested on RDNA3, it already should be able of take advantage of RDNA4 hardware FP8 features, but there bay still be some issues. Please report any problem you encounter.
- **RDNA3** Should be fine, but current code has some custom optimizations for gfx1100 and gfx1101 targets, different RDNA3 gpus can behave slightly differently.
- **RDNA2** RDNA2 gpus should be in theory able to run dlss5 too, but the performance impact is huge. Async mode could make dlss5 osable there, but there also can be too much artifacts to be enjoyable, 

## Requirements

- Windows 10/11 or Linux, an AMD Radeon RX 9000 (RDNA4), RX 7000 (RDNA3) or RX 6000 (RDNA2). 
- A game using DirectX 12, DirectX 11 or Vulkan (see *Direct3D 11* and *Vulkan* below), or DirectX 9 or 10 through
  DXVK, which is included (see *Older Direct3D* below)
- A 64-bit game, or a 32-bit Direct3D 9 one through the bridge (see *32-bit games* below)
- The AMD HIP SDK for Windows, if using Windows.
- ROCm if using Linux.

## How to use it

1. **Get the package.** It holds OptiScaler's files, and inside its
   `OptiScaler` folder a `dlss5nr` folder with everything else, NVIDIA's
   network included.
2. **Copy it into the game's folder.** Everything goes next to the game's
   executable. In Unreal Engine games that is usually
   `<game>\<name>\Binaries\Win64`, not the folder with the launcher.
3. **Install OptiScaler.** Run `setup_windows.bat` (windows) or `setup_linux.sh` (linux) from the game's folder. It
   renames `OptiScaler.dll` to a name the game loads, such as `dxgi.dll`. See
   OptiScaler's own documentation if the game needs a different name.
4. **Start the game and, if it has one, turn on an upscaler** in its graphics
   settings: DLSS, FSR or XeSS. That is where neural rendering works best: the
   upscaler hands OptiScaler the game's depth and motion vectors, which let
   the network's result follow the scene. Without an upscaler it still runs,
   on the finished image (see *Without an upscaler* below).
5. **Open OptiScaler's menu** with the Insert key. In the *Neural Rendering*
   tab (the same box is in the *Custom* tab too), tick *Enable*. Use *Save Settings* to keep it on
   for the next time. Alternatively, set `Enabled=true` in the
   `[NeuralRendering]` section of `OptiScaler.ini` before starting the game.

The menu shows what the network is doing. Right after the start it says
*starting* while the network is loaded, a few seconds with the cache that
comes with the package. The game runs normally in the meantime, without the
effect. If something the package should hold is missing, or the cache does
not cover your GPU, the menu says *failed* and what is missing: the plugin
never translates the network's code on its own (see *Note on the
translation cache* below).

The network is slower than the game, so it does not enhance every frame. It
enhances a frame every so often, and between two of them its result follows
the scene along the game's motion vectors. A slower network lowers how often
the image is refreshed, not the game's frame rate.

What this costs shows behind things that move: the patch of background an
object uncovers was hidden when the network looked, so for a moment it gets
the network's light but not its detail. The faster the network, the smaller
the patch, which is why it runs before the upscaler by default. *Wait for the
network* removes it altogether, at the price of the frame rate.

### Without an upscaler

If the game uses no upscaler, the network runs on the finished image instead,
just before it is shown. Two things are different then:

- There are no motion vectors, so the result cannot follow the scene: it
  trails behind whatever moves. Turn on *Wait for the network* to avoid that (note: it introduces a huge performance loss).
- The game's own interface is part of the image, and is enhanced with it.

An HDR10 image is not supported in this mode; SDR and scRGB are.

## Direct3D 11

A Direct3D 11 game goes through OptiScaler's Direct3D 11 to 12 bridge: the colour, depth and motion
vectors are copied to textures shared with a Direct3D 12 device, the runtime works there, and the
result is copied back. It costs the game about 0.2 to 0.3 ms a frame at 1080p. It works with the
game's upscaler, whichever of OptiScaler's Direct3D 11 backends is selected (the native FSR 2.2 and
FSR 3.1 ones, or those running on Direct3D 12), before or after it as for Direct3D 12, and without one,
on the finished image, as above. Tested on Windows with FSR 2.2, native and on Direct3D 12; not yet
with OptiScaler's frame generation.

## Vulkan

Two ways, by the upscaler OptiScaler uses in the game (`VulkanUpscaler` in OptiScaler.ini). While
neural rendering is enabled and `VulkanUpscaler` is left on `auto`, OptiScaler picks `ffx_12`, the first
way; any value set there is respected.

- **A Vulkan upscaler running on Direct3D 12** (`ffx_12`, `fsr21_12`): the upscaler's inputs already
  go to Direct3D 12 for it, so the neural rendering works there as in a Direct3D 12 game, before
  or after the upscaler, with depth and motion vectors.
- **Any other** (the default, native `fsr22`, `ffx`, `xess`... or no upscaler at all): the finished
  image, at `vkQueuePresentKHR`. The swapchain image is copied into an image shared with a
  Direct3D 12 device, the runtime works there, and the result is copied back over the swapchain image,
  with the Direct3D 12 fences imported as timeline semaphores for the order between the two APIs.
  The menu is drawn after it, so it is not enhanced. As without an upscaler in Direct3D 12, there are no
  depth or motion vectors to follow, and the game's interface is part of the image: the result trails
  behind what moves, which *Wait for the network* avoids at a cost in frame rate.

The first gives the better picture, at the cost of OptiScaler's copies of the upscaler's inputs and
output, and is what `auto` chooses. Set a native one, `fsr22` for instance, to get the second.

Tested on Windows with two small Vulkan programs (`runtime/tests/dx11/vkapp.cpp` and `vkngx.cpp`):
BGRA8 and RGBA8 swapchains come back byte for byte at zero strength, and `ffx_12` and the default
upscaler both hold 60 fps. Not tested yet: HDR swapchains and window resizing.

## Older Direct3D (9 and 10)

OptiScaler has no Direct3D 9 or 10 support of its own, and those games give no motion vectors (nor a
depth that can be read) in any case: all there is is the finished image, so the result trails behind what
moves, as without an upscaler in Direct3D 12 (*Wait for the network* avoids that at a cost in frame rate).

- **Direct3D 9 with DXVK** works: with DXVK's `d3d9.dll` beside the game, the game is a Vulkan one, and
  the Vulkan way above enhances its image. OptiScaler goes in under a name the game loads that DXVK does
  not use (`version.dll`). Tested on Windows with `runtime/tests/dx11/dx9app.cpp`: at zero
  strength the picture comes back byte for byte.
- **Direct3D 10 with DXVK** (the files DXVK needs beside the game are `d3d10core.dll`, `d3d11.dll` and
  `dxgi.dll`) works too, on Windows, the same way: DXVK runs it as Direct3D 11, whose bridge cannot be used
  with it (the textures DXVK shares cannot be opened by the native Direct3D 12, and the driver crashes in
  it), so OptiScaler enhances its image at the Vulkan present instead. The same goes for a Direct3D 11 game
  running on DXVK on Windows. Tested with `runtime/tests/dx11/dx10app.cpp`. Under Wine (Proton) DXVK and
  vkd3d-proton do share, so nothing is changed there.
- Direct3D 9 and 10 **without DXVK** are not supported.
- **Only 64-bit games through DXVK**: OptiScaler is 64-bit, and so are the DXVK files in the package.
  32-bit games have the bridge, below.

### DXVK comes with the package

DXVK 3.1.1 (zlib licence; the exact files and their hashes are listed in `dxvk/README.txt`) is in the
`dxvk` folder of the package, so there is nothing to download. `setup_windows.bat` takes care of it:

1. Run it in the game's folder as usual. It asks whether the game is an old one, Direct3D 9 or 10 (answer 1,
   the default, for Direct3D 11, 12 and Vulkan games, which do not need it).
2. For 2 it copies `d3d9.dll`, `dxgi.dll`, `d3d10core.dll` and `d3d11.dll` beside the executable: the game
   loads the ones it uses, so there is no need to know which of the two it is. A different file of the same
   name that is already there is left alone, with a warning.
3. It installs OptiScaler as `version.dll` unless you choose another name; `dxgi.dll` is DXVK's, so that
   one is refused.
4. Turn on *Neural Rendering* in OptiScaler's menu (Insert) or in `OptiScaler.ini` as for any other game.

`Remove_OptiScaler.bat` removes DXVK again, the files identical to those in the package only. On Linux
(Proton) none of this applies: Proton already runs Direct3D 9 to 11 on DXVK.

Tested on Windows end to end with `runtime/tests/dx11/dx9app.cpp` and `dx10app.cpp`: package, setup,
a run with the neural rendering off and on (the image changes, the runtime is only loaded when it is on)
and the removal, which leaves the folder as it was.

## 32-bit games

OptiScaler and the network are 64-bit, and a 32-bit game cannot load them. For 32-bit Direct3D 9 games (most
of the older ones) the package carries a bridge: a 32-bit `d3d9.dll` that hooks the game's `Present` and
sends the finished frame to a 64-bit helper, `dlss5nr_host.exe`, which runs the network through the same
runtime and returns what it did to each pixel as a gain, which is applied to the frame the game is
showing. The helper starts with the game and ends with it.

`setup_windows.bat` notices that the game is 32-bit and offers to install it: `d3d9.dll` goes beside the
executable (a `d3d9.dll` that was there is kept as `d3d9_chain.dll`, and the bridge loads it), and the
neural rendering is switched on. `Remove_OptiScaler.bat` takes it away and puts the old one back.

There is no OptiScaler menu in such a game, so the bridge has one of its own: Insert opens it, with the
neural rendering part of OptiScaler's menu; the settings are kept in the same `OptiScaler.ini`. A game that
reads its mouse through DirectInput gets that input hidden while the menu is open.

There are no motion vectors, so what the network adds is a few frames old when it arrives and trails behind
what moves. What helps, and is the default: *Wait for the network* (the answer is applied to the very frame it
was made from, and the frame rate is the network's), with the network at half resolution (the result is
stretched over the whole frame, so the detail stays) and *Temporal smoothing* against the flicker the
network's own decisions show from picture to picture. At 1440p that is roughly 35-40 frames a second with
vsync on, on a Radeon RX 7900 XT. *Reduce trailing* (fade the result where the picture has changed) and
*Follow motion* (motion estimated from the pictures, experimental) are for using it without waiting.

Only Direct3D 9, and only 32 bits a pixel in the backbuffer. Its sources are in `bridge32/`: `python
bridge32/build.py` builds the client and the helper; `bridge32/README.txt` is the user's version of this.

## Linux 

The project also supports DX12 games running on Linux using Proton. 
The setup_linux.sh script should altready copy the required files, 
but you also have to change the launch options to include the launch.sh script included into the "linux" folder.
On steam, add %command% after the linux.sh path

Example:
***/path_to_dlss5-nr-amd-custom/linux/launch.sh %command%***

Make sure that the launch.sh script is executable.

Note: some games, like for example Hogwarts Legacy, may need OverlayMenu=false in OptiScaler.ini to work on Linux

## Settings

All in the *Neural Rendering* tab of the menu, also shown in the *Custom* tab:

- **Before the upscaler** (on by default): runs the network on the lower
  resolution image the game renders, before upscaling. Much faster, slightly
  lower quality. Off, it runs on the upscaled image.
- **Without an upscaler** (on by default): lets the network run on the
  finished image when the game uses no upscaler.
- **Wait for the network:** the game waits for the network on every frame and
  the frame rate becomes the network's: use it with a low network resolution,
  or before the upscaler. Each frame gets the answer for the frame before it,
  carried along the motion vectors. **Apply on the same frame** (experimental,
  off) makes the GPU wait inside the frame instead and applies the answer to
  that very frame: nothing trails and the network keeps its history, which
  steadies the detail, but every frame takes about twice as long.
- **Network resolution:** runs the network on a smaller copy of the image. Its
  cost follows its pixels. Not shown when running before the upscaler, where
  the image is the game's own render resolution.
- **Network:** which network the library runs. *Library default* leaves the choice
  to NVIDIA's library, which takes the smaller one since version 310.9.16.0;
  *Preset 1* is the larger network the versions before carried (it is slower),
  *Preset 2* the smaller one. A change restarts the network.
- **Style, Intensity and the tone and structure strengths:** the network's own
  controls. Changing them restarts the network, which takes a moment.
  *Skin structure* is the detail it adds to faces and skin. It is automatic by
  default and follows the structure strength; untick Auto to set it yourself,
  where 0 leaves faces as the game drew them.
- **Passes:** how many times the effect is applied, one on top of the other: each pass works on the result of the one before. More passes give a stronger effect, but too many look artificial and shift the colours (a blue scarf turns grey), and each takes as long as the first, so the network gets that much slower. **Keep local tone** says whether the later passes keep applying the local tone, which would push the lighting further each time; it only matters with more than one pass.
- **Detail:** how much of the network's light and detail is applied. 0 is the
  game's own image, 1 is the network's.
- **Colour:** 0 keeps the game's colours, 1 takes the network's, up to 4 pushes
  them further.
- **Max ratio:** the most the network may brighten or darken a pixel. Lower it
  if the image flickers.
- **Blend frames:** how many frames a new result takes to replace the one
  before (0 to 8, 4 by default). The network decides some of its detail anew
  with every result, which shows as a flicker each time it switches; blending
  the two over a few frames smooths that out. 0 switches at once.
- **Follow motion:** moves the network's result along the motion vectors to
  where the scene is now. Leave it on.
- **Network history:** lets the network blend each result with its previous
  one, as it does on NVIDIA cards. Experimental, off by default.
- **Show tracking:** tints the pixels the network never saw because something
  was in front of them: cyan and blue where its answer was found elsewhere,
  green where only its light was filled in, red where nothing was found;
  yellow where a halo the network put around an edge was taken off, because the
  edge has moved away since; orange where the network's answer belonged to
  something else, and only its light is kept.
- **Input and Exposure:** how the game's colours are handed to the network.
  Exposure is measured from the image by default (Auto). Taking the game's own
  exposure (Game) showed visible artifacts around moving things in some
  games, so it is not the default; change these only if the image looks washed
  out or too dark.

### If something goes wrong

The runtime writes its log to `OptiScaler\dlss5nr\cache\dlss5nr.log`, next to
OptiScaler's own log. Check it first, and attach both logs when you report a
problem.

## Building

- **Runtime:** `runtime\build.bat` (Visual Studio with C++ and CMake, Python 3).
- **Linux bridge:** `linux\build.sh` (Needs a Linux hosts / WSL and gcc, mingw-w64 (x86_64-w64-mingw32-gcc), Python 3).
- **Kernels:** `python kernels\build_kernels.py`, which builds every kernel
  `kernels\kernels.txt` names, in parallel (the AMD HIP SDK and Visual Studio).
- **ZLUDA:** see `third_party/ZLUDA`.
- **OptiScaler:** `optiscaler\OptiScaler.sln`.
- **NVIDIA's network:** `nvngx_dlssnr.dll` is too large for the repository and
  is only shipped in the release package. To build from source, take version
  310.9.16.0 from the release package, or from a game that ships DLSS 5, and put
  it in `third_party\`. (310.8.0.0, the version before, still runs, but the
  translation cache of the package is made for 310.9.16.0.) It has to be the original signed file: builds patched
  to run on older NVIDIA cards cannot be translated.
- **Package:** `python package.py --zluda <ZLUDA build> --optiscaler <OptiScaler
  build>` puts everything, the network and the translation cache included,
  into `dist\`.

`runtime/README.md` describes how the built pieces are laid out in a game's
folder.

### Note on the translation cache

The network's CUDA modules have to be translated into code an AMD GPU can
run. ZLUDA does exactly that, but some modules take a long time, up to tens of
minutes each. That's why the package comes with a pre-built cache, shipped in
`OptiScaler\dlss5nr\cache\ComputeCache`.

The runtime only ever reads this cache. If it is missing, or does not hold
a translation for your GPU, or a native kernel the package should hold is
missing, neural rendering fails to start with a message saying which, rather
than translating the network inside the game. The cache covers the GPU
targets listed in `coverage.txt` beside it.

To run from a cache of your own, or to let ZLUDA translate as the network
loads (slow, and only for development), set `DLSS5NR_ALLOW_TRANSLATION=1`.

#### Building the cache manually (for releases)

The cache only works with the exact ZLUDA build that produced it. If you
compiled ZLUDA from source, or changed its code, the cache has to be rebuilt.
To release a cache anyone can use, run:

    python runtime\tools\prepare_cache.py

It reads the code modules out of `third_party\nvngx_dlssnr.dll`, translates
them once per GPU target, several at a time, and merges everything into
`runtime\build\shipped\ComputeCache\zluda2.db` (with the `coverage.txt` the
runtime checks the GPU against), where `package.py` takes it from. It needs ZLUDA's driver in `runtime\build\bin\zluda\zluda_real.dll` (the
same one you will package), an AMD GPU with the HIP SDK, and Python. Expect
it to take hours.

The script builds the cache for the generic targets of RDNA2, RDNA3 and
RDNA4, and also for gfx1100 and gfx1101, since those cards get some custom
optimizations. You can choose which targets to build by editing
`NATIVE_TARGETS` and `GENERIC_TARGETS` in the script, or with `--targets`
(which builds every target it names without those optimizations).

## Contributing

This project have been made with the help of AI tools and in general i'm not against AI-generated code, but you must be able to explain what the code does and why you want it merged.
Any PR posted without a proper explanation will be rejected

## Frequently Asked Questions (FAQ)

### Do you know DLSS-NR-on-AMD by danielblnc?

Yes, i'm aware of that project, but that's totally unrelated to mine. 
Since there's no code available, i really can't tell how the two projects differ.

### Can the asyncronous approach be used on Nvidia too?

Likely, but i'm not currently able to test that.
Feel free to re-implement the idea on nvidia-related projects