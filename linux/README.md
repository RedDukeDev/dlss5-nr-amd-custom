# Linux and Proton

Neural rendering runs under Wine and Proton with the whole Windows stack in the
game's prefix -- the runtime, ZLUDA, the network -- and only one piece
replaced: ZLUDA's calls to the HIP runtime go to ROCm on the Linux side instead
of AMD's Windows DLL. This folder is that piece. What a player needs to know is in
`PACKAGE_README.txt`, which ships as `linux/README.txt` in the package; this file is
for whoever works on it.

**State:** verified with Hogwarts Legacy under GE-Proton11-7 (Steam Linux Runtime
4.0) and with the test harness under Proton Experimental, on an RX 7900 XT with ROCm
7.15: the network loads from the shipped cache, evaluates, and the image it
produces is the right one. The runtime used was the mingw build (see "Building");
the Visual Studio build of the same sources has not been run under Wine yet.

## How a call goes

```
ZLUDA (zluda_real.dll, in the prefix)
  -> amdhip64_7.dll          the trampoline: exports the 108 HIP functions ZLUDA uses, and
                             packs each call into a structure         (trampoline/)
  -> ntdll!__wine_unix_call_dispatcher
                             Wine's own door from Windows code to native code in the
                             same process: it takes the address of a table of native
                             functions, a number in it and a pointer, and calls
  -> libdlss5nr_hip_bridge.so
                             preloaded into the process; unpacks the structure and calls
                             the real function                         (bridge/)
  -> libamdhip64.so.7        ROCm
```

Both sides are one process with one address space, so a pointer is a pointer on
either side. Every function is a call with integers and pointers only: each
argument goes in a 64-bit slot of one structure, the result in a 32-bit one.

How the trampoline finds the table: every process of the session that has the
bridge preloaded writes `/tmp/dlss5nr-bridge-<pid>.handle` with the address of a tag
(a magic number and the table's address) and removes it on exit. The trampoline reads
them all, through Wine's `Z:` drive, and takes the one whose address holds the tag in
its own memory: only this process's bridge can be there. (The pid in the name cannot
be used: a file read from Windows code is opened by the wineserver, so `/proc/self` is
the wineserver's. A variable in the environment would not do either: a game started
by another Windows program gets that program's environment.)

The images are not shared with ROCm. A shared Direct3D 12 handle under Wine is a
`DxgkSharedResource`, not a file descriptor (`runtime/tests/interop_probe.cpp` asks
the question: `wine_server_handle_to_fd` answers with an object type mismatch), so
the runtime copies instead (`Network::set_copy_mode`): the inputs to host memory and
into CUDA arrays before each evaluation, the output back after it. It is switched on
when the process is under Wine, or with `DLSS5NR_COPY=1`, and nothing changes
otherwise. `hipImportExternalMemory`, the one function the bridge writes by hand, is
therefore not called any more.

## Files

| | |
|---|---|
| `hip_functions.txt` | the HIP functions to forward: everything `zluda_real.dll` imports from `amdhip64_7.dll` |
| `gen_forwarders.py` | writes the three generated files from that list and ZLUDA's bindings of the HIP headers; stops at a function it cannot reduce to integers and pointers |
| `generated/`, `trampoline/generated/`, `bridge/generated/` | what it writes: the enumeration and structures, the trampoline's exports, the bridge's thunks and table |
| `trampoline/` | the Windows DLL (`amdhip64_7.dll`) |
| `bridge/` | the native library, built with gcc on Linux; built as a Windows DLL it forwards to AMD's runtime, for testing |
| `build.sh` | builds the bridge and the trampoline into `build/` (gcc, mingw-w64); `package.py` takes them from there |
| `build.py` | the pieces one by one: `trampoline` (MSVC), `bridge`, `test-bridge` |
| `launch.sh` | the Steam launch option wrapper |
| `PACKAGE_README.txt` | the player's README |
| `tests/test_chain_windows.py` | runs the harness through trampoline -> bridge -> AMD's runtime, and compares the image with the one without |
| `tests/check_linux_syntax.py` | reads the Linux branch of the bridge with the Windows compiler |

## Where the trampoline goes

Beside the game's executable. ZLUDA loads `amdhip64_7.dll` by name, and Wine looks
in the executable's folder, then in the system's; the folder of `zluda_real.dll` is
not searched. `setup_linux.sh` copies it from `linux/` there, `remove_optiscaler.sh`
removes it. It must not be in a Windows package: with it beside the executable ZLUDA
would load it instead of AMD's.

## Building and packaging

```
sh linux/build.sh            # bridge.so (gcc), amdhip64_7.dll (mingw-w64) -> linux/build/
python package.py --zluda <dir> ...   # copies them into dist/linux/ with launch.sh and README.txt
sh linux/build.sh runtime    # also dlss5nr_runtime.dll with mingw, into linux/build/mingw/
```

The runtime that ships is the Visual Studio one (`runtime/build.bat`): the sources are
the same, and the Wine-specific behaviour is chosen when it runs. The mingw build
exists for trying the runtime on a machine without Visual Studio; it is not packaged.
One difference between the compilers is handled in `runtime/common/ngx_cuda.h`:
MSVC lays out the overloads of one virtual function in the reverse of the order they
are declared in, GCC does not, so for mingw the overloads of `NVSDK_NGX_Parameter`
are declared reversed. Without it every `Set` goes to the wrong overload (intensity
arrived as 0).

## What was found, and why the code is the way it is

- **Handle files** are tried one by one (above), not named by pid.
- **Two LLVMs.** ROCm brings its own, through comgr, and so does Mesa's RADV. In one
  process the one that loads second finds the first one's state, and
  `hipStreamCreate` fails with "out of memory" (ROCm cannot build its blit kernels).
  Reproduced outside Wine. The bridge therefore loads `libamdhip64` in its
  constructor, before anything else in the process, with nothing initialised.
- **`nvcuda`.** Proton sets `nvcuda=b` whenever NVAPI is on and puts it after any
  override given to it, so the plugin's own `nvcuda.dll` is replaced by Wine's, which
  has no `cuInit`. `launch.sh` turns NVAPI off (`PROTON_DISABLE_NVAPI=1`) and
  sets `nvcuda=n`.
- **Spaces in the path.** `ld.so` splits `LD_PRELOAD` at spaces and colons and cannot
  quote them, so a game in `World of Warcraft/` had the bridge taken for three
  libraries. `launch.sh` puts the library's name in `LD_PRELOAD` and its folder in
  `LD_LIBRARY_PATH`, which splits at colons only (a colon in the path still
  breaks it).
- **Steam's container** does not know `/opt/rocm`: `dlopen("libamdhip64.so.7")` by
  soname fails there (`cuInit` answers 801, not supported). `launch.sh` finds the
  library, names it by path (`DLSSNR_HIP_LIB`) and shows `/opt/rocm` to the container.
- **Shaders.** Wine's `d3dcompiler_47` is vkd3d's, and cannot build the compute
  shaders (`RWByteAddressBuffer.Load`). `runtime/core/shaders_dxbc.h` holds them
  compiled (`runtime/tools/gen_dxbc.cpp`, run again when `shaders.h` changes), and
  `passes.cpp` uses that when compiling fails and the hash of the source matches.
- **Copies.** ROCm copies wrongly, and says nothing, to and from host memory that a
  Direct3D driver has mapped: an array filled from vkd3d's readback buffer held almost
  none of it. Each image has an ordinary buffer of its own (`SharedImage::host`) that
  the copies go through.
- **Arrays.** The network reads its inputs through texture objects with normalized
  coordinates (`CUDA_TEXTURE_DESC::flags = 2`) and writes through `sust.p`; both work
  on arrays made with `cuArray3DCreate`, checked with small kernels of our own. Only
  some `sust.b` forms (16-bit and 8-bit elements) store nothing on such an array,
  because ZLUDA writes their raw bits as floats; the network does not use them.

## Finding what a program does with HIP

`DLSSNR_BRIDGE_TRACE=1` (with `DLSSNR_BRIDGE_LOG=1`) makes the bridge name every call
in order, with the grid and block of each kernel launch, the name of the kernel and the
size of each array. The parameters of the plugin's own 16 x 16 kernels are printed in
full. `ZLUDA_DUMP_IR=<dir>` writes the LLVM IR of every module ZLUDA translates, which
shows what a kernel does with its textures and surfaces; with a cache that already
holds the big modules, delete only the small ones from its `zluda2.db` to have just
those translated again.
