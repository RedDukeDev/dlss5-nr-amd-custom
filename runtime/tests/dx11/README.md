# Direct3D 11 and Vulkan test games

Small programs to try the neural rendering in a Direct3D 11 or Vulkan game without one at hand.
Each draws an animated scene in a window, presents it, and writes what the desktop shows of the
window to a BMP at a chosen time, so what OptiScaler did to the picture is in it.

    dx11app.cpp   no upscaler: a flip-model swap chain and a procedural scene (the finished-image path)
    vkapp.cpp     a Vulkan game: a swapchain, the scene drawn on the CPU and copied in (no shaders), so
                  OptiScaler enhances the finished image at vkQueuePresentKHR
    dx9app.cpp    a Direct3D 9 game (run it with DXVK's d3d9.dll, and OptiScaler as version.dll, which it imports)
    dx10app.cpp   a Direct3D 10 game (with DXVK's d3d10core.dll, d3d11.dll and dxgi.dll, OptiScaler as version.dll)
    vkngx.cpp     the same through the NGX Vulkan API, with depth and motion vectors (OptiScaler answers
                  as FSR 2.2; `VulkanUpscaler=ffx_12` takes the upscaler to Direct3D 12)
    dx11ngx.cpp   the scene at render resolution with depth and motion vectors, upscaled through the
                  NGX Direct3D 11 API as a DLSS game does (OptiScaler answers, as FSR 2.2 unless
                  Dx11Upscaler says otherwise)

Build with the Visual Studio developer prompt (dx11ngx needs OptiScaler's NGX headers):

    cl /O2 /EHsc /std:c++17 dx11app.cpp
    cl /O2 /EHsc /std:c++17 dx9app.cpp
    cl /O2 /EHsc /std:c++17 dx10app.cpp
    cl /O2 /EHsc /std:c++17 /I <OptiScaler>\external\vulkan\include vkapp.cpp
    cl /O2 /EHsc /std:c++17 /I <OptiScaler>\external\vulkan\include /I <OptiScaler>\external\nvngx_dlss_sdk vkngx.cpp
    cl /O2 /EHsc /std:c++17 /I <OptiScaler>\external\nvngx_dlss_sdk dx11ngx.cpp

Run them in a folder holding the package with OptiScaler.dll copied to `dxgi.dll` beside the exe
(both load it by that name) and `Enabled=true` in the `[NeuralRendering]` section of OptiScaler.ini:

    dx11app.exe  [seconds] [capture_at] [out.bmp] [width] [height] [rgba|bgra|srgb|fp16] [freeze_time]
    vkapp.exe    [seconds] [capture_at] [out.bmp] [width] [height] [freeze_time] [bgra|rgba]
    dx9app.exe   [seconds] [capture_at] [out.bmp] [width] [height] [freeze_time]   (dx10app.exe the same)
    vkngx.exe    [seconds] [capture_at] [out.bmp] [out_w] [out_h] [render_percent] [freeze_time]
    dx11ngx.exe  [seconds] [capture_at] [out.bmp] [out_w] [out_h] [render_percent] [freeze_time]

`freeze_time` stops the animation, which makes two runs comparable. `NOVSYNC=1` in the environment
presents without waiting for the display, to measure what the bridge costs a frame.

Checked this way: with `DetailStrength=0` and `ColourStrength=0` the picture comes back byte for byte
as it went in, for RGBA8, BGRA8 and FP16 swap chains in Direct3D 11 and RGBA8 and BGRA8 ones in Vulkan: the round trip through the bridge is lossless.
