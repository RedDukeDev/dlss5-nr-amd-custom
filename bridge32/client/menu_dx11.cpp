// The Direct3D 11 side of the menu: ImGui's renderer for it.
#include "menu.h"
#include <d3d11.h>
#include "imgui.h"
#include "imgui_impl_dx11.h"

namespace menu {
namespace {
ID3D11Device *g_device = nullptr;
}

void init(ID3D11Device *device, ID3D11DeviceContext *context, HWND window) {
    attach(window);
    if (g_device != device) {
        if (g_device) ImGui_ImplDX11_Shutdown();
        ImGui_ImplDX11_Init(device, context);
        g_device = device;
    }
}

void frame(ID3D11DeviceContext *context, ID3D11RenderTargetView *target, unsigned width, unsigned height, const Status &status) {
    if (!g_device || !begin(width, height, status, &ImGui_ImplDX11_NewFrame)) return;
    ImDrawData *data = (ImDrawData *)end();
    ID3D11RenderTargetView *previous[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT] = {};
    ID3D11DepthStencilView *previous_depth = nullptr;
    context->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, previous, &previous_depth);
    context->OMSetRenderTargets(1, &target, nullptr);
    ImGui_ImplDX11_RenderDrawData(data);
    context->OMSetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, previous, previous_depth);
    for (ID3D11RenderTargetView *t : previous)
        if (t) t->Release();
    if (previous_depth) previous_depth->Release();
}

} // namespace menu
