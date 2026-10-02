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

## Logs

The logs are in `cache\`:

- `dlss5nr.log`: the runtime;
- `gate.log`: the native kernels, which ones load and which fall back;
- NVIDIA's own NGX log.

The network reports its fatal errors to the NGX log and then ends the
process. When a game closes on its own, look there.
