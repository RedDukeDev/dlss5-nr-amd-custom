DLSS 5 neural rendering on Linux (Wine / Proton)
================================================

This folder is the Linux side of the plugin. Everything else in the package is
shared with Windows. Nothing here is needed on Windows.

  amdhip64_7.dll              the Windows half of the HIP bridge. Wine looks for
                              it beside the game's executable; setup_linux.sh
                              copies it there.
  libdlss5nr_hip_bridge.so    the native half. launch.sh preloads it into the
                              game's processes.
  launch.sh                   the Steam launch option wrapper.

What it is for: the network runs on ZLUDA, which calls AMD's HIP runtime. On
Linux that runtime is ROCm's libamdhip64.so, native, and the bridge connects the
two: ZLUDA's calls go through amdhip64_7.dll into the .so and on to ROCm.


Requirements
------------

- An AMD GPU that the shipped translation cache covers (see
  OptiScaler\dlss5nr\cache\ComputeCache\coverage.txt), with ROCm 7 installed:
  libamdhip64.so.7 is needed, from /opt/rocm or the system's libraries.
- Wine or Proton with vkd3d-proton. Tested with GE-Proton11-7 (Steam Linux
  Runtime 4.0) and Proton Experimental, Wine 11.
- NVIDIA's network, nvngx_dlssnr.dll 310.8.0.0, unmodified and signed, in
  OptiScaler\dlss5nr\ (the package does not include it).

Nothing else is needed in the Wine prefix: the compute shaders come already
compiled, so Microsoft's d3dcompiler_47.dll is not required.


Installing
----------

1. Extract the whole package to the game's folder (the one with the executable;
   for Unreal Engine games, <Game>/Binaries/Win64).
2. Run ./setup_linux.sh there. Besides setting up OptiScaler it copies
   linux/amdhip64_7.dll beside the executable. ./remove_optiscaler.sh takes it
   away again.
3. In Steam, set the game's launch option to

       WINEDLLOVERRIDES="dxgi=n,b" "/path/to/the/game/folder/linux/launch.sh" %command%

   with the file name you chose for OptiScaler in setup_linux.sh instead of
   dxgi if it is another.
4. Start the game, open OptiScaler's menu (Insert) and turn on Neural Rendering.

launch.sh does the following, so none of it has to be set by hand:

- preloads libdlss5nr_hip_bridge.so;
- sets PROTON_DISABLE_NVAPI=1 and WINEDLLOVERRIDES=nvcuda=n (Proton puts
  nvcuda=b after anything set here when NVAPI is on, and the CUDA driver must be
  the plugin's own, not Wine's);
- finds libamdhip64.so.7 and tells the bridge where it is (DLSSNR_HIP_LIB), and
  shows /opt/rocm to Steam's container, which cannot see it otherwise.


Known limits
------------

- The game's images cannot be shared with ROCm under Wine (a shared Direct3D 12
  handle is not a file descriptor ROCm can import), so every evaluation copies
  its inputs to host memory and its output back. That costs time per frame, and
  the cost grows with the network's resolution: lower "Resolution scale" if the
  frame rate suffers.
- With the default OptiScaler menu, an overlay drawn through Vulkan, some Proton
  versions show a black screen (seen with GE-Proton11-7). Setting
  OverlayMenu=false in OptiScaler.ini avoids it; the menu is then drawn before the
  game's own interface and can end up under parts of it.
- ROCm 7's runtime is loaded by the bridge in every process of the session
  before the Vulkan driver, on purpose: both bring their own LLVM, and the
  second one to load breaks the first.


When something goes wrong
-------------------------

Logs:
  OptiScaler\dlss5nr\cache\dlss5nr.log   the plugin's own log
  OptiScaler.log                          OptiScaler's (LogToFile=true in OptiScaler.ini)

Environment variables, for a terminal launch:
  DLSSNR_BRIDGE_LOG=1     the bridge says what it loads and what is missing
  DLSSNR_BRIDGE_TRACE=1   every HIP call, with kernel launches and array sizes
                          (together with DLSSNR_BRIDGE_LOG)
  DLSSNR_HIP_LIB=<path>   the ROCm library to use instead of the one found
  DLSSNR_COPY=0|1         0 forbids and 1 forces the copy through host memory

Messages:
  "cuInit failed: 801"    the bridge could not load libamdhip64.so.7 (not
                          installed, or not visible in Steam's container: check
                          the paths launch.sh finds, or set DLSSNR_HIP_LIB)
  "no handle file holds this process's tag"
                          the bridge .so is not preloaded: the launch option
                          does not go through launch.sh
  "could not load ... nvcuda.dll" / "does not export cuInit"
                          Wine's own nvcuda is being used: launch.sh was not
                          used, or NVAPI is enabled for the game
