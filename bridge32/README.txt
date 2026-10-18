DLSS 5 neural rendering in 32-bit Direct3D 9, 10 and 11 games
=============================================================

OptiScaler and the network are 64-bit and cannot be loaded into a 32-bit game. This is what stands in
for them there:

  bridge32\d3d9.dll                  a 32-bit d3d9.dll, for Direct3D 9 games. It loads the real one (or
                                     the one renamed d3d9_chain.dll: another wrapper's, DXVK's) and hooks
                                     the game's Present. It sends the finished frame to the helper and
                                     applies what comes back.
  bridge32\dxgi.dll                  the same for Direct3D 10 and 11 games, as a 32-bit dxgi.dll (the
                                     real one, or dxgi_chain.dll). Direct3D 11 for now: Direct3D 10 is
                                     recognised and left alone, with a line in the log.
  OptiScaler\dlss5nr\dlss5nr_host.exe  the 64-bit helper. The client starts it with the game, and it
                                     ends with the game. It runs the network through the same runtime
                                     the 64-bit games use, and returns what the network did to each pixel
                                     as a gain, which the client applies to the frame the game is showing.

You do not copy these by hand: setup_windows.bat notices that the game is 32-bit and installs both clients
beside the executable, d3d9.dll and dxgi.dll (a DLL of the same name that was there is kept as d3d9_chain.dll
or dxgi_chain.dll). The game loads whichever it uses, so there is no need to know which. Remove_OptiScaler.bat
takes them away and puts the old ones back.

Using it
--------
Insert opens a menu in the game, the neural rendering part of OptiScaler's own. Its settings are kept in
OptiScaler.ini, in the [NeuralRendering] section, and the file can be edited while the game runs.

The network has no motion vectors here, so what it adds to the picture is a few frames old when it
arrives, and trails behind what moves. What the bridge does about it:

  Wait for the network   (BridgeWaitForNetwork, on)   The game waits for the network on every frame and
                         the answer is applied to that very frame: no trail, but the frame rate is the
                         network's. Pair it with a low network resolution.
  Network resolution     (ResolutionScale, 0.5)       The network works at this fraction of the frame, and
                         what the bridge moves between the game and the network shrinks with it. The result
                         is stretched over the whole frame, so the detail stays. At 1440p, 0.5 gives
                         roughly 35-40 frames a second on a Radeon RX 7900 XT with vsync on.
  Temporal smoothing     (BridgeSmoothing, 0.75)      The network decides some of its detail anew with every
                         picture, which shows as a flicker. Where the game's picture has not changed, the
                         change of the result from the one before is held back.
  Reduce trailing        (BridgeReduceTrailing)       Without waiting: fades the result where the picture
                         has changed since the network saw it. (Off while waiting: nothing to fade.)
  Follow motion          (BridgeFollowMotion, off)    Experimental: estimates the motion from the pictures
                         and carries the result along it.

Limits
------
Direct3D 9 and 11, 32-bit only (64-bit games use OptiScaler itself). The backbuffer has to be 32 bits a pixel
(eight bits a channel, sRGB or not): HDR and ten-bit ones are left alone. In Direct3D 11 there is no Reduce
trailing and no Follow motion yet: use it waiting for the network. The interface of the game is part of the frame, so it is enhanced with the rest. Games that
read the mouse through DirectInput get their input hidden while the menu is open (the menu takes it).

If something does not work, dlss5nr_bridge.log (Direct3D 9) or dlss5nr_bridge_dxgi.log (Direct3D 11), beside
the game's executable, and OptiScaler\dlss5nr\dlss5nr_host.log say what the client and the helper did.
