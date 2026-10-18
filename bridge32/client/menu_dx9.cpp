// The Direct3D 9 side of the menu: ImGui's renderer for it, and the state the menu is drawn in.
#include "menu.h"
#include <d3d9.h>
#include "imgui.h"
#include "imgui_impl_dx9.h"

namespace menu {
namespace {
IDirect3DDevice9 *g_device = nullptr;
}

void init(IDirect3DDevice9 *device, HWND window) {
    attach(window);
    if (g_device != device) {
        if (g_device) ImGui_ImplDX9_Shutdown();
        ImGui_ImplDX9_Init(device);
        g_device = device;
    }
}

void device_lost() {
    if (g_device) ImGui_ImplDX9_InvalidateDeviceObjects();
}

void device_restored() {
    if (g_device) ImGui_ImplDX9_CreateDeviceObjects();
}

void frame(IDirect3DDevice9 *device, unsigned width, unsigned height, const Status &status) {
    if (!g_device || !begin(width, height, status, &ImGui_ImplDX9_NewFrame)) return;
    ImDrawData *data = (ImDrawData *)end();

    IDirect3DSurface9 *back = nullptr, *target = nullptr;
    if (FAILED(device->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO, &back))) return;
    device->GetRenderTarget(0, &target);
    if (target != back) device->SetRenderTarget(0, back);
    device->BeginScene();
    ImGui_ImplDX9_RenderDrawData(data);
    device->EndScene();
    if (target != back) device->SetRenderTarget(0, target);
    if (target) target->Release();
    back->Release();
}

} // namespace menu
