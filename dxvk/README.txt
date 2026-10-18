DXVK 3.1.1, 64-bit
==================

d3d9.dll, d3d10core.dll, d3d11.dll and dxgi.dll in x64\ are DXVK as released, unmodified:

    https://github.com/doitsujin/dxvk/releases/tag/v3.1.1   (dxvk-3.1.1.tar.gz)
    archive sha256  40565b4a724aadc4433fa4e010b4b23916d9b1f1baeee64e17186db94f54e608

    sha256 of the files
    c126316f7478eaff1ad8087702eda5930f2400187b9c8b3592219c34f6632780  x64\d3d9.dll
    beffdd081dce91c2bccdc87845b88cbbb7848d39d4642143462d2556d69bd5d9  x64\d3d10core.dll
    0a203b6255c893430f0a8461c618962df04e62aa41538d5a875b21868317cf9f  x64\d3d11.dll
    e3c1178eb7f0dd59f91bcfcc257549e5b9cfdb5e8508719293de7f8704f53160  x64\dxgi.dll

DXVK is under the zlib licence (LICENSE).

What it is here for
-------------------
OptiScaler has no Direct3D 9 or Direct3D 10 support, so the neural rendering has nothing to hook in
those games. DXVK turns them into Vulkan (Direct3D 9) and Direct3D 11 (Direct3D 10) games, and OptiScaler
then enhances their image at the Vulkan present.

You do not copy these by hand: setup_windows.bat asks whether the game is Direct3D 9 or 10, copies
the files that kind needs beside the game's executable (d3d9.dll for Direct3D 9; dxgi.dll,
d3d10core.dll and d3d11.dll for Direct3D 10) and installs OptiScaler as version.dll, since dxgi.dll is
DXVK's. Remove_OptiScaler.bat takes them away again, the ones identical to these only.

On Linux (Proton) none of this is needed: Proton already runs Direct3D 9 to 11 on DXVK.
Only 64-bit games work: OptiScaler is 64-bit, and these are the 64-bit files.
