// A dxgi.dll for 32-bit Direct3D 10 and 11 games: it loads the real one (or the one renamed
// dxgi_chain.dll), hooks the creation of swap chains and their Present, and hands each finished frame to
// the part that does what the Direct3D 9 client does (dx11_overlay.cpp).
//
// Everything the real dxgi.dll exports is exported here, so that the game's own d3d11.dll, which binds
// to whichever dxgi.dll is already loaded, finds all it needs.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <vector>
#include "core.h"
#include "dx11_overlay.h"
#include "input.h"

// ------------------------------------------------------------------ what is passed through

// Every export the system's dxgi.dll has, but the three that create a factory.
#define DXGI_PASSTHROUGH(X)                                                                                              \
    X(ApplyCompatResolutionQuirking, 1) X(CompatString, 2) X(CompatValue, 3) X(DXGIDumpJournal, 4) X(PIXBeginCapture, 5) \
    X(PIXEndCapture, 6) X(PIXGetCaptureState, 7) X(SetAppCompatStringPointer, 8) X(UpdateHMDEmulationStatus, 9)        \
    X(DXGID3D10CreateDevice, 13) X(DXGID3D10CreateLayeredDevice, 14) X(DXGID3D10GetLayeredDeviceSize, 15)             \
    X(DXGID3D10RegisterLayers, 16) X(DXGIDeclareAdapterRemovalSupport, 17) X(DXGIDisableVBlankVirtualization, 18)     \
    X(DXGIGetDebugInterface1, 19) X(DXGIReportAdapterConfiguration, 20)

#define SLOT(name, ordinal) extern "C" FARPROC p_##name = nullptr;
DXGI_PASSTHROUGH(SLOT)
#undef SLOT

#define THUNK(name, ordinal)                                                                                 \
    extern "C" __declspec(naked) void thunk_##name() { __asm { jmp dword ptr [p_##name] } }                  \
    __pragma(comment(linker, "/EXPORT:" #name "=_thunk_" #name ",@" #ordinal))
DXGI_PASSTHROUGH(THUNK)
#undef THUNK

namespace {

HMODULE g_real = nullptr;

typedef HRESULT(WINAPI *CreateFactory_t)(REFIID, void **);
typedef HRESULT(WINAPI *CreateFactory2_t)(UINT, REFIID, void **);
CreateFactory_t g_create = nullptr, g_create1 = nullptr;
CreateFactory2_t g_create2 = nullptr;

// ------------------------------------------------------------------ the hooks

typedef HRESULT(STDMETHODCALLTYPE *CreateSwapChain_t)(IDXGIFactory *, IUnknown *, DXGI_SWAP_CHAIN_DESC *, IDXGISwapChain **);
typedef HRESULT(STDMETHODCALLTYPE *CreateForHwnd_t)(IDXGIFactory2 *, IUnknown *, HWND, const DXGI_SWAP_CHAIN_DESC1 *,
                                                    const DXGI_SWAP_CHAIN_FULLSCREEN_DESC *, IDXGIOutput *, IDXGISwapChain1 **);
typedef HRESULT(STDMETHODCALLTYPE *Present_t)(IDXGISwapChain *, UINT, UINT);
typedef HRESULT(STDMETHODCALLTYPE *Present1_t)(IDXGISwapChain1 *, UINT, UINT, const DXGI_PRESENT_PARAMETERS *);

// The vtable indices: IDXGIFactory CreateSwapChain 10; IDXGIFactory2 CreateSwapChainForHwnd 15;
// IDXGISwapChain Present 8; IDXGISwapChain1 Present1 22.
constexpr int kCreateSwapChain = 10, kCreateForHwnd = 15, kPresent = 8, kPresent1 = 22;

struct FactoryHook {
    void **vtable;
    CreateSwapChain_t create;
    CreateForHwnd_t create_for_hwnd;
};
struct SwapHook {
    void **vtable;
    Present_t present;
    Present1_t present1;
};
std::vector<FactoryHook> g_factories;
std::vector<SwapHook> g_swaps;

bool patch(void **vtable, int index, void *hook, void **original) {
    DWORD old;
    if (!VirtualProtect(&vtable[index], sizeof(void *), PAGE_EXECUTE_READWRITE, &old)) return false;
    if (vtable[index] != hook) {
        *original = vtable[index];
        vtable[index] = hook;
    }
    VirtualProtect(&vtable[index], sizeof(void *), old, &old);
    return true;
}

SwapHook *swap_hook(void *object) {
    void **vtable = *(void ***)object;
    for (SwapHook &h : g_swaps)
        if (h.vtable == vtable) return &h;
    return nullptr;
}

HRESULT STDMETHODCALLTYPE hk_present(IDXGISwapChain *swap_chain, UINT sync, UINT flags) {
    SwapHook *h = swap_hook(swap_chain);
    if (!(flags & DXGI_PRESENT_TEST)) dx11::on_present(swap_chain);
    return h->present(swap_chain, sync, flags);
}

HRESULT STDMETHODCALLTYPE hk_present1(IDXGISwapChain1 *swap_chain, UINT sync, UINT flags, const DXGI_PRESENT_PARAMETERS *parameters) {
    SwapHook *h = swap_hook(swap_chain);
    if (!(flags & DXGI_PRESENT_TEST)) dx11::on_present(swap_chain);
    return h->present1(swap_chain, sync, flags, parameters);
}

void hook_swap_chain(IUnknown *object) {
    IDXGISwapChain *sc = nullptr;
    if (SUCCEEDED(object->QueryInterface(__uuidof(IDXGISwapChain), (void **)&sc))) {
        void **vtable = *(void ***)sc;
        if (!swap_hook(sc)) {
            SwapHook h = {};
            h.vtable = vtable;
            patch(vtable, kPresent, (void *)hk_present, (void **)&h.present);
            // Present1 is on the IDXGISwapChain1 interface, which may have a vtable of its own.
            IDXGISwapChain1 *sc1 = nullptr;
            if (SUCCEEDED(object->QueryInterface(__uuidof(IDXGISwapChain1), (void **)&sc1))) {
                void **vtable1 = *(void ***)sc1;
                if (vtable1 == vtable) {
                    patch(vtable1, kPresent1, (void *)hk_present1, (void **)&h.present1);
                } else {
                    SwapHook h1 = {};
                    h1.vtable = vtable1;
                    patch(vtable1, kPresent, (void *)hk_present, (void **)&h1.present);
                    patch(vtable1, kPresent1, (void *)hk_present1, (void **)&h1.present1);
                    g_swaps.push_back(h1);
                }
                sc1->Release();
            }
            g_swaps.push_back(h);
            log("hooked a swap chain (vtable %p)", vtable);
        }
        sc->Release();
    }
}

HRESULT STDMETHODCALLTYPE hk_create_swap_chain(IDXGIFactory *factory, IUnknown *device, DXGI_SWAP_CHAIN_DESC *desc, IDXGISwapChain **out) {
    CreateSwapChain_t original = nullptr;
    for (FactoryHook &f : g_factories)
        if (f.vtable == *(void ***)factory) original = f.create;
    const HRESULT hr = original(factory, device, desc, out);
    if (SUCCEEDED(hr) && out && *out) hook_swap_chain(*out);
    return hr;
}

HRESULT STDMETHODCALLTYPE hk_create_for_hwnd(IDXGIFactory2 *factory, IUnknown *device, HWND window, const DXGI_SWAP_CHAIN_DESC1 *desc,
                                             const DXGI_SWAP_CHAIN_FULLSCREEN_DESC *fullscreen, IDXGIOutput *output, IDXGISwapChain1 **out) {
    CreateForHwnd_t original = nullptr;
    for (FactoryHook &f : g_factories)
        if (f.vtable == *(void ***)factory) original = f.create_for_hwnd;
    const HRESULT hr = original(factory, device, window, desc, fullscreen, output, out);
    if (SUCCEEDED(hr) && out && *out) hook_swap_chain(*out);
    return hr;
}

void hook_factory(IUnknown *object) {
    IDXGIFactory *f = nullptr;
    if (SUCCEEDED(object->QueryInterface(__uuidof(IDXGIFactory), (void **)&f))) {
        void **vtable = *(void ***)f;
        bool known = false;
        for (FactoryHook &h : g_factories) known |= h.vtable == vtable;
        if (!known) {
            FactoryHook h = {};
            h.vtable = vtable;
            patch(vtable, kCreateSwapChain, (void *)hk_create_swap_chain, (void **)&h.create);
            IDXGIFactory2 *f2 = nullptr;
            if (SUCCEEDED(object->QueryInterface(__uuidof(IDXGIFactory2), (void **)&f2))) {
                void **vtable2 = *(void ***)f2;
                if (vtable2 == vtable) {
                    patch(vtable2, kCreateForHwnd, (void *)hk_create_for_hwnd, (void **)&h.create_for_hwnd);
                } else {
                    FactoryHook h2 = {};
                    h2.vtable = vtable2;
                    patch(vtable2, kCreateSwapChain, (void *)hk_create_swap_chain, (void **)&h2.create);
                    patch(vtable2, kCreateForHwnd, (void *)hk_create_for_hwnd, (void **)&h2.create_for_hwnd);
                    g_factories.push_back(h2);
                }
                f2->Release();
            }
            g_factories.push_back(h);
            log("hooked a DXGI factory (vtable %p)", vtable);
        }
        f->Release();
    }
}

// ------------------------------------------------------------------ the real dxgi.dll

bool load_real() {
    if (g_real) return true;
    std::wstring chain = g_dir + L"dxgi_chain.dll";
    if (GetFileAttributesW(chain.c_str()) != INVALID_FILE_ATTRIBUTES) {
        g_real = LoadLibraryW(chain.c_str());
        log("chained to %ls: %s", chain.c_str(), g_real ? "ok" : "failed");
    }
    if (!g_real) {
        wchar_t system[MAX_PATH];
        GetSystemWow64DirectoryW(system, MAX_PATH);
        g_real = LoadLibraryW((std::wstring(system) + L"\\dxgi.dll").c_str());
        log("the system dxgi.dll: %s", g_real ? "loaded" : "failed");
    }
    if (!g_real) return false;
#define BIND(name, ordinal) p_##name = GetProcAddress(g_real, #name);
    DXGI_PASSTHROUGH(BIND)
#undef BIND
    g_create = (CreateFactory_t)GetProcAddress(g_real, "CreateDXGIFactory");
    g_create1 = (CreateFactory_t)GetProcAddress(g_real, "CreateDXGIFactory1");
    g_create2 = (CreateFactory2_t)GetProcAddress(g_real, "CreateDXGIFactory2");
    return true;
}

} // namespace

// ------------------------------------------------------------------ the exports

extern "C" {

HRESULT WINAPI my_CreateDXGIFactory(REFIID riid, void **out) {
    if (!load_real() || !g_create) return DXGI_ERROR_UNSUPPORTED;
    const HRESULT hr = g_create(riid, out);
    if (SUCCEEDED(hr) && out && *out) hook_factory((IUnknown *)*out);
    return hr;
}

HRESULT WINAPI my_CreateDXGIFactory1(REFIID riid, void **out) {
    if (!load_real() || !g_create1) return DXGI_ERROR_UNSUPPORTED;
    const HRESULT hr = g_create1(riid, out);
    if (SUCCEEDED(hr) && out && *out) hook_factory((IUnknown *)*out);
    return hr;
}

HRESULT WINAPI my_CreateDXGIFactory2(UINT flags, REFIID riid, void **out) {
    if (!load_real() || !g_create2) return DXGI_ERROR_UNSUPPORTED;
    const HRESULT hr = g_create2(flags, riid, out);
    if (SUCCEEDED(hr) && out && *out) hook_factory((IUnknown *)*out);
    return hr;
}

} // extern "C"

#pragma comment(linker, "/EXPORT:CreateDXGIFactory=_my_CreateDXGIFactory@8,@12")
#pragma comment(linker, "/EXPORT:CreateDXGIFactory1=_my_CreateDXGIFactory1@8,@10")
#pragma comment(linker, "/EXPORT:CreateDXGIFactory2=_my_CreateDXGIFactory2@12,@11")

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        DisableThreadLibraryCalls(instance);
        core_init(instance, L"dlss5nr_bridge_dxgi.log");
        load_real();
        input::install();
    } else if (reason == DLL_PROCESS_DETACH) {
        g_bridge.stop();
    }
    return TRUE;
}
