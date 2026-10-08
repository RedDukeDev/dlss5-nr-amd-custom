# DLSS 5 Neural Rendering on AMD - Featuring asyncronous mode (almost no performance loss!) and Linux support

DLSS 5 Neural Rendering, NVIDIA's own network, running in Direct3D 12 games on
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
- A game using DirectX 12
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
5. **Open OptiScaler's menu** with the Insert key. Under
   *Neural Rendering (DLSS 5)*, tick *Enable*. Use *Save Settings* to keep it on
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

All in the same section of the menu:

- **Before the upscaler** (on by default): runs the network on the lower
  resolution image the game renders, before upscaling. Much faster, slightly
  lower quality. Off, it runs on the upscaled image.
- **Without an upscaler** (on by default): lets the network run on the
  finished image when the game uses no upscaler.
- **Wait for the network:** the game waits for the network on every frame.
  Nothing trails behind moving objects, but the frame rate becomes the
  network's: use it with a low network resolution, or before the upscaler.
- **Network resolution:** runs the network on a smaller copy of the image. Its
  cost follows its pixels. Not shown when running before the upscaler, where
  the image is the game's own render resolution.
- **Style, Intensity and the tone and structure strengths:** the network's own
  controls. Changing them restarts the network, which takes a moment.
  *Skin structure* is the detail it adds to faces and skin. It is automatic by
  default and follows the structure strength; untick Auto to set it yourself,
  where 0 leaves faces as the game drew them.
- **Detail:** how much of the network's light and detail is applied. 0 is the
  game's own image, 1 is the network's.
- **Colour:** 0 keeps the game's colours, 1 takes the network's, up to 4 pushes
  them further.
- **Max ratio:** the most the network may brighten or darken a pixel. Lower it
  if the image flickers.
- **Follow motion:** moves the network's result along the motion vectors to
  where the scene is now. Leave it on.
- **Network history:** lets the network blend each result with its previous
  one, as it does on NVIDIA cards. Experimental, off by default.
- **Show tracking:** tints the pixels the network never saw because something
  was in front of them: cyan and blue where its answer was found elsewhere,
  green where only its light was filled in, red where nothing was found.
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
  310.8.0.0 from the release package, or from a game that ships DLSS 5, and put
  it in `third_party\`. It has to be the original signed file: builds patched
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