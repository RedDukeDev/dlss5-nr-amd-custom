// A game component that loads d3d9.dll, makes a Direct3D 9 object, unloads the DLL, and later does it again.
#include <windows.h>
#include <d3d9.h>
#include <stdio.h>
typedef IDirect3D9 *(WINAPI *Create_t)(UINT);
int main() {
    WNDCLASSA wc = {};
    wc.lpfnWndProc = DefWindowProcA;
    wc.hInstance = GetModuleHandleA(nullptr);
    wc.lpszClassName = "unload_test";
    RegisterClassA(&wc);
    HWND w = CreateWindowA("unload_test", "t", WS_OVERLAPPEDWINDOW, 0, 0, 200, 200, nullptr, nullptr, wc.hInstance, nullptr);
    for (int round = 0; round < 2; ++round) {
        HMODULE dll = LoadLibraryW(L"d3d9.dll");
        Create_t create = (Create_t)GetProcAddress(dll, "Direct3DCreate9");
        IDirect3D9 *d3d = create(D3D_SDK_VERSION);
        printf("round %d: module %p, object %p\n", round, (void *)dll, (void *)d3d);
        if (round == 1) {
            D3DPRESENT_PARAMETERS pp = {};
            pp.Windowed = TRUE;
            pp.SwapEffect = D3DSWAPEFFECT_DISCARD;
            pp.BackBufferFormat = D3DFMT_UNKNOWN;
            pp.hDeviceWindow = w;
            IDirect3DDevice9 *dev = nullptr;
            HRESULT hr = d3d->CreateDevice(0, D3DDEVTYPE_HAL, w, D3DCREATE_SOFTWARE_VERTEXPROCESSING, &pp, &dev);
            printf("CreateDevice: %08lx\n", hr);
            if (dev) dev->Release();
        }
        d3d->Release();
        FreeLibrary(dll);
        printf("freed\n");
    }
    return 0;
}
