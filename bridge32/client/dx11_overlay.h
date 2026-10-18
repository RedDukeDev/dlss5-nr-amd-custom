// What the DXGI client does to a Direct3D 11 game's frame, once per Present: take a copy of the
// finished frame, send it to the host when the host wants one, apply what came back, draw the menu.
#pragma once
#include <d3d11.h>
#include <dxgi.h>

namespace dx11 {

// Called from the hook on IDXGISwapChain::Present, before the real one. Returns false when the swap chain
// is not one this can work with (the reason is in the log, once).
void on_present(IDXGISwapChain *swap_chain);

} // namespace dx11
