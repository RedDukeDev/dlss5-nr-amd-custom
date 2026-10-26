# Runtime

`dlss5nr_runtime.dll` puts NVIDIA's neural rendering network into a Direct3D 12
frame on an AMD GPU. The host is OptiScaler's fork, a ReShade add-on or the
test harness here. It hands the runtime the game's image, depth and motion
vectors inside the game's own command list. The runtime records a few compute
passes there, and runs the network on a thread of its own.
[`include/dlss5nr.h`](include/dlss5nr.h) is the whole interface, and
[`../docs/DESIGN.md`](../docs/DESIGN.md) explains its shape.

## Pieces

| folder | builds | what it is |
|---|---|---|
| `core/` | `dlss5nr_runtime.dll` | the runtime: slots, the network's worker thread, the Direct3D 12 passes (exposure, capture, residual, compose) |
| `gate/` | `zluda\nvcuda.dll` | a thin CUDA driver in front of ZLUDA that launches the native kernels in place of the translated ones they reproduce; everything else is forwarded to `zluda_real.dll` |
| `ngx/` | `zluda\nvngx.dll` | our NGX host. The network only accepts calls that come from a module named `nvngx.dll`, so every NGX call is made from inside it |
| `tests/` | `dlss5nr_harness.exe` | the runtime driven as a host drives it, on a still image |
| `common/` | | the slices of the CUDA and NGX interfaces used here |

## Layout in a game

With OptiScaler the runtime goes in a `dlss5nr` folder inside OptiScaler's own
`OptiScaler` folder, which OptiScaler takes as its directory:

```
<game>\
    OptiScaler.dll (renamed dxgi.dll or similar), OptiScaler.ini, ...
    OptiScaler\
      dlss5nr\
        dlss5nr_runtime.dll
        nvngx_dlssnr.dll          NVIDIA's network, from third_party\
        zluda\
            zluda_real.dll        ZLUDA's nvcuda.dll, renamed
            nvcuda.dll            the gate
            nvapi64.dll           ZLUDA's NVAPI stand-in
            nvngx.dll             our NGX host
        kernels\
            kernels.txt           which native kernel stands in for which
            *.hsaco               the native kernels
        cache\                    created: the translation cache and the logs
```

## The first start

The network's code has to be translated for the GPU, and that takes tens of
minutes, so the runtime never does it: it reads the cache that ships with it
(`cache\ComputeCache`, built by `tools\prepare_cache.py`) and the native
kernels in `kernels\`. If the cache is missing, does not hold a translation
for the GPU (`coverage.txt` lists what it covers), or a kernel is missing,
the runtime goes to the *failed* state with a message saying which. The game
runs normally, without the effect.

For development, `DLSS5NR_ALLOW_TRANSLATION=1` lets ZLUDA translate as the
network loads, and skips these checks. The harness needs it unless it is
pointed at a complete package (`--runtime-dir`).

The compute passes (core/shaders.h) are not compiled at the first frame: their bytecode
ships in core/shaders_dxbc.h, made by `tools/gen_dxbc.cpp`. Run that again whenever
shaders.h changes; a stale header is ignored and the shaders are compiled at the first
frame instead, which takes 15 to 20 seconds of the game's render thread.

## The harness

    dlss5nr_harness.exe --image in.png --out out.png --snippet nvngx_dlssnr.dll

`--image` must be an ordinary PNG. The harness runs frames through the
runtime until the network has produced three results (`--evaluations`), then
writes the composed image. It also prints:

- the network's time per evaluation;
- how many evaluations per second arrived;
- how many native kernels the gate runs.

At full resolution, with the residual unclamped, the output is the network's
own image. It can be compared with what other tools produce for the same
picture.

## Upscalers: super resolution and ray reconstruction

The same runtime runs NVIDIA's DLSS upscalers (`nvngx_dlss.dll`, `nvngx_dlssd.dll`), through
the C interface in the "Upscalers" section of [`include/dlss5nr.h`](include/dlss5nr.h). An
upscaler has to answer in the frame it is asked for, so the host cannot call it from inside a
command list it is still recording: it splits the list at the upscale and calls
`dlss5nr_upscaler_evaluate` in between, with everything it recorded before submitted to the queue.
OptiScaler's fork does that (`upscalers/dlssz`, with the command-list module taken from d4r) and
offers the result as the DLSS upscaler on a GPU that is not NVIDIA's; the runtime converts the game's
textures (any format, any state, a sub-rect of a larger texture, dynamic resolution, exposure, the
reactive and transparency masks) into the network's own, runs it, and converts the answer into the
game's output.

Beside the neural rendering's files the `dlss5nr` folder then holds `nvngx_dlss.dll` and
`nvngx_dlssd.dll`, and `zluda\ngx_sr`, `zluda\ngx_rr` (an NGX host per network, so that they run
beside the neural rendering). `kernels\kernels_sr.txt` lists the native kernels
(`kernels\build_sr_kernels.py` in the repository, from d4r's sources): the eleven layers of preset K and the five Swin
blocks of presets L and M. One cache holds all the networks' translations
(`tools\prepare_cache.py` for each, merged).

What was measured on an RX 7900 XT, 960x540 to 1920x1080:

| network | time per frame | note |
|---|---|---|
| super resolution, preset M (the default) | 5.7 ms | its heavy layers native; translated they take 20 to 30 ms |
| super resolution, preset K and J | 4 ms | its heavy layers native, the rest translated |
| ray reconstruction | 72 ms | translated, no native kernels yet |

The upscaler harness (`dlss5up_harness.exe --kind sr|rr --runtime-dir DIR --data-dir DIR --snippet DLL`)
feeds synthetic frames of ordinary game textures and has what is needed to tell a wrong picture from a
right one: `--scene gradient|const|corner|topleft|ramp`, `--means` (the output's mean colour after each
frame), `--preset`, `--flags`, `--subrect`, `--dynamic WxH`, `--exposure`, `--masks`, `--reset-every`,
`--reset-at`, `--reset-parity`. Run it from a folder with no `dxgi.dll` of OptiScaler's in it.
`tests/dx12/dx12ngx.cpp` is a Direct3D 12 "game" that upscales through the NGX API, to try the
OptiScaler integration.

## Frame generation

NVIDIA's frame generation (`nvngx_dlssg.dll`, 310.9.1) runs here too, for a game that has DLSS-G of its own.
Unlike the upscalers it has no CUDA path: its kernels are launched through NVAPI inside a Direct3D 12 command
list, on buffers and textures the library creates on the device. `fg/nvapi_fg.cpp` is the `nvapi64.dll` that
answers that (ZLUDA's NVAPI is `zluda_nvapi.dll` beside it, and gets everything that is not frame generation's):

- the buffers the library creates are found by hooking the device, only while the library runs, and their memory
  is imported into HIP, so that a GPU virtual address in a kernel's arguments becomes the HIP pointer;
- a texture or surface the library asks for through a descriptor is the resource behind it, imported the same way;
  a normalised 8 or 16 bit array gets a normalised surface, as Direct3D's would have (`sust.p` converts floats);
- a launch waits for the command list: the host splits the game's list (d4r's `record_boundary`) and the launches
  of one split run together, between what was recorded before and after, with HIP and the queue waiting for each
  other on a shared fence;
- the game's textures are not shared with HIP, and may be in formats the library does not take, so `fgnv_evaluate`
  copies them (with the converter the upscalers use, `core/convert.cpp`) into shared textures of its own, hands the
  library those, and copies what it generates back into the game's texture. 10 bit and HDR colour is brought down
  to 8 bit RGBA for the library.

The kernels `fg_conv.hip` and `fg_custom.hip` (`kernels/build_fg_kernels.py`) stand in for 27 of the network's
translated kernels, byte for byte what they compute. The NGX host (`zluda\ngx_fg\nvngx.dll`) is what the library
insists on being called from; `fgnv_*` are the stand-in's entry points, which OptiScaler's `Nvngx_Nvidia` provider
uses. `cubin_fatbin_map.txt` pairs the machine code the library hands over with the PTX that is next to it in its
file; it is for 310.9.1 only.

`dlss5nr_fg_harness.exe` (`fg/build_harness.bat`, then `fg/prepare_run.py`) does what the provider does, with
Direct3D 12 textures of its own: a bright disc moves across a background and the frame generated between two is
written as a PNG, where the disc should be half way. `FG_HARNESS_SESSION=1` gives the library the game's textures
as a game would (not shared), `FG_HARNESS_FORMAT=bgra8|rgb10a2` changes the backbuffer's format,
`FG_HARNESS_DIST=1` runs as the package does. `FG_NVAPI_PERKERNEL=1` synchronises after every kernel and reports
what it left in the textures it touched.

In OptiScaler the library can be driven by the game's own DLSS-G (`FGInput=NvngxFG`), or, with
`FGOutput=DLSSG` and `FGNvngxReplacement=Nvidia`, by any other input that gives OptiScaler the frame: FSR 3 frame
generation, XeFG (the game's depth and motion vectors come from its tags) or the upscaler. The FSR-FG swapchain
paces the frames and the library makes them, with no Streamline of OptiScaler's.

Every launch of the library goes on one stream of the stand-in's own, native or translated: the upscalers of the
same process run with a default stream per thread (`ZLUDA_PER_THREAD_STREAM`), and launches split between that and
HIP's null stream do not wait for each other (the generated frame comes out black, with no error).

## Logs

The logs are in `cache\`:

- `dlss5nr.log`: the runtime;
- `gate.log`: the native kernels, which ones load and which fall back;
- NVIDIA's own NGX log;
- `zluda\fg_nvapi.log`: the frame generation stand-in (what it attached to, mirrors made, launches that failed).

The network reports its fatal errors to the NGX log and then ends the
process. When a game closes on its own, look there.
