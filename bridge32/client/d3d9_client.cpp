// A d3d9.dll for 32-bit Direct3D 9 games: it loads the real one (or the one renamed
// d3d9_chain.dll, DXVK's for instance), hooks Present, and sends the finished frame to
// dlss5nr_host.exe, a 64-bit process that runs the network. What comes back is a gain per
// channel, which is applied to whatever frame the game is showing by then.
//
// OptiScaler and the network are 64-bit; this is the only part that has to be 32-bit.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d9.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>
#include <algorithm>
#include <math.h>
#include "../common/protocol.h"
#include "core.h"
#include "input.h"
#include "menu.h"
#include "motion.h"
#include "shaders/gain_mc.h"
#include "shaders/gain_plain.h"

namespace {

// ------------------------------------------------------------------ the real d3d9.dll

typedef IDirect3D9 *(WINAPI *Create9_t)(UINT);
typedef HRESULT(WINAPI *Create9Ex_t)(UINT, IDirect3D9Ex **);
typedef int(WINAPI *BeginEvent_t)(D3DCOLOR, LPCWSTR);
typedef int(WINAPI *EndEvent_t)();
typedef void(WINAPI *Marker_t)(D3DCOLOR, LPCWSTR);
typedef BOOL(WINAPI *QueryRepeat_t)();
typedef void(WINAPI *SetOptions_t)(DWORD);
typedef DWORD(WINAPI *GetStatus_t)();
typedef void(WINAPI *Shim_t)(BOOL);

HMODULE g_real = nullptr;
Create9_t g_create9 = nullptr;
Create9Ex_t g_create9ex = nullptr;
BeginEvent_t g_begin_event = nullptr;
EndEvent_t g_end_event = nullptr;
Marker_t g_set_marker = nullptr, g_set_region = nullptr;
QueryRepeat_t g_query_repeat = nullptr;
SetOptions_t g_set_options = nullptr;
GetStatus_t g_get_status = nullptr;
Shim_t g_shim = nullptr;

bool load_real() {
    if (g_real) return true;
    std::wstring chain = g_dir + L"d3d9_chain.dll";
    if (GetFileAttributesW(chain.c_str()) != INVALID_FILE_ATTRIBUTES) {
        g_real = LoadLibraryW(chain.c_str());
        log("chained to %ls: %s", chain.c_str(), g_real ? "ok" : "failed");
    }
    if (!g_real) {
        wchar_t system[MAX_PATH];
        GetSystemWow64DirectoryW(system, MAX_PATH);
        std::wstring path = std::wstring(system) + L"\\d3d9.dll";
        g_real = LoadLibraryW(path.c_str());
        log("the system d3d9.dll: %s", g_real ? "loaded" : "failed");
    }
    if (!g_real) return false;
    g_create9 = (Create9_t)GetProcAddress(g_real, "Direct3DCreate9");
    g_create9ex = (Create9Ex_t)GetProcAddress(g_real, "Direct3DCreate9Ex");
    g_begin_event = (BeginEvent_t)GetProcAddress(g_real, "D3DPERF_BeginEvent");
    g_end_event = (EndEvent_t)GetProcAddress(g_real, "D3DPERF_EndEvent");
    g_set_marker = (Marker_t)GetProcAddress(g_real, "D3DPERF_SetMarker");
    g_set_region = (Marker_t)GetProcAddress(g_real, "D3DPERF_SetRegion");
    g_query_repeat = (QueryRepeat_t)GetProcAddress(g_real, "D3DPERF_QueryRepeatFrame");
    g_set_options = (SetOptions_t)GetProcAddress(g_real, "D3DPERF_SetOptions");
    g_get_status = (GetStatus_t)GetProcAddress(g_real, "D3DPERF_GetStatus");
    g_shim = (Shim_t)GetProcAddress(g_real, "Direct3D9EnableMaximizedWindowedModeShim");
    return g_create9 != nullptr;
}

// ------------------------------------------------------------------ the hooks

typedef HRESULT(STDMETHODCALLTYPE *CreateDevice_t)(IDirect3D9 *, UINT, D3DDEVTYPE, HWND, DWORD, D3DPRESENT_PARAMETERS *,
                                                    IDirect3DDevice9 **);
typedef HRESULT(STDMETHODCALLTYPE *CreateDeviceEx_t)(IDirect3D9Ex *, UINT, D3DDEVTYPE, HWND, DWORD, D3DPRESENT_PARAMETERS *,
                                                      D3DDISPLAYMODEEX *, IDirect3DDevice9Ex **);
typedef HRESULT(STDMETHODCALLTYPE *Present_t)(IDirect3DDevice9 *, const RECT *, const RECT *, HWND, const RGNDATA *);
typedef HRESULT(STDMETHODCALLTYPE *PresentEx_t)(IDirect3DDevice9Ex *, const RECT *, const RECT *, HWND, const RGNDATA *, DWORD);
typedef HRESULT(STDMETHODCALLTYPE *Reset_t)(IDirect3DDevice9 *, D3DPRESENT_PARAMETERS *);
typedef HRESULT(STDMETHODCALLTYPE *ResetEx_t)(IDirect3DDevice9Ex *, D3DPRESENT_PARAMETERS *, D3DDISPLAYMODEEX *);

// The vtable indices: IDirect3D9 CreateDevice 16; IDirect3D9Ex CreateDeviceEx 20;
// IDirect3DDevice9 Reset 16, Present 17; IDirect3DDevice9Ex PresentEx 121, ResetEx 132.
constexpr int kCreateDevice = 16, kCreateDeviceEx = 20, kReset = 16, kPresent = 17, kPresentEx = 121, kResetEx = 132;

struct Originals {
    void **vtable;
    Present_t present;
    PresentEx_t present_ex;
    Reset_t reset;
    ResetEx_t reset_ex;
};
std::vector<Originals> g_devices;
struct FactoryOriginal {
    void **vtable;
    CreateDevice_t create;
    CreateDeviceEx_t create_ex;
};
std::vector<FactoryOriginal> g_factories;

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

Originals *find_device(void *object) {
    void **vtable = *(void ***)object;
    for (Originals &o : g_devices)
        if (o.vtable == vtable) return &o;
    return nullptr;
}

// ------------------------------------------------------------------ the overlay

// What is kept for the device whose frames are enhanced.
struct Overlay {
    IDirect3DDevice9 *device = nullptr;
    UINT w = 0, h = 0;          // the backbuffer
    UINT cw = 0, ch = 0;        // the capture, the gain and the frame the network saw: the network's resolution
    D3DFORMAT format = D3DFMT_UNKNOWN;
    bool multisampled = false;
    bool gain16 = true;     // the gain texture is A16B16G16R16, else A8R8G8B8
    IDirect3DSurface9 *system = nullptr;    // the capture lands here
    IDirect3DSurface9 *resolved = nullptr;  // the backbuffer resolved, when it is multisampled
    IDirect3DTexture9 *gain = nullptr;
    IDirect3DTexture9 *copy_tex = nullptr;      // the frame now, for the masked pass to read
    IDirect3DSurface9 *copy_surf = nullptr;
    IDirect3DTexture9 *small_tex = nullptr;     // the frame now at the network's resolution, when that is not the frame's
    IDirect3DSurface9 *small_surf = nullptr;
    IDirect3DTexture9 *seen_tex = nullptr;      // the frame the network saw
    IDirect3DPixelShader9 *mask_ps = nullptr;
    bool mask_ok = false;
    // Motion estimation, and the shader that applies the gain along it.
    Motion motion;
    IDirect3DPixelShader9 *mc_ps = nullptr;
    bool me_ok = false;
    bool scaled() const { return cw != w || ch != h; }
    IDirect3DTexture9 *now_small() const { return small_tex ? small_tex : copy_tex; }
    bool seen_dirty = false;    // the frame the network saw was replaced since the pyramid was built
    IDirect3DStateBlock9 *state = nullptr;
    bool have_gain = false;
    bool broken = false;
    HWND window = nullptr;
    bool menu_ready = false;
    unsigned frames = 0;
    unsigned sent = 0, applied = 0;
    DWORD last_report = 0;

    void release() {
        if (system) system->Release();
        if (resolved) resolved->Release();
        if (gain) gain->Release();
        if (copy_surf) copy_surf->Release();
        if (copy_tex) copy_tex->Release();
        if (small_surf) small_surf->Release();
        if (small_tex) small_tex->Release();
        small_surf = nullptr;
        small_tex = nullptr;
        if (seen_tex) seen_tex->Release();
        if (mask_ps) mask_ps->Release();
        motion.release();
        if (mc_ps) mc_ps->Release();
        mc_ps = nullptr;
        me_ok = false;
        if (state) state->Release();
        system = resolved = nullptr;
        copy_surf = nullptr;
        copy_tex = seen_tex = nullptr;
        mask_ps = nullptr;
        mask_ok = false;
        gain = nullptr;
        state = nullptr;
        have_gain = false;
        w = h = cw = ch = 0;
    }
};
Overlay g_overlay;

bool setup(IDirect3DDevice9 *device) {
    Overlay &o = g_overlay;
    IDirect3DSurface9 *back = nullptr;
    if (FAILED(device->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO, &back))) return false;
    D3DSURFACE_DESC desc;
    back->GetDesc(&desc);
    back->Release();
    if (desc.Format != D3DFMT_X8R8G8B8 && desc.Format != D3DFMT_A8R8G8B8) {
        log("overlay: the backbuffer is format %d, which is not handled", (int)desc.Format);
        o.broken = true;
        return false;
    }
    o.w = desc.Width;
    o.h = desc.Height;
    o.format = desc.Format;
    o.multisampled = desc.MultiSampleType != D3DMULTISAMPLE_NONE;
    network_size(o.w, o.h, o.cw, o.ch);
    if (FAILED(device->CreateOffscreenPlainSurface(o.cw, o.ch, o.format, D3DPOOL_SYSTEMMEM, &o.system, nullptr))) {
        log("overlay: no system memory surface");
        o.broken = true;
        return false;
    }
    if (o.multisampled &&
        FAILED(device->CreateRenderTarget(o.w, o.h, o.format, D3DMULTISAMPLE_NONE, 0, FALSE, &o.resolved, nullptr))) {
        log("overlay: no surface to resolve into");
        o.broken = true;
        return false;
    }
    // The gain wants 16 bits a channel where the card has them: at 8 bits the steps are 1.5 %.
    IDirect3D9 *d3d = nullptr;
    device->GetDirect3D(&d3d);
    D3DDEVICE_CREATION_PARAMETERS cp;
    device->GetCreationParameters(&cp);
    D3DDISPLAYMODE mode = {};
    d3d->GetAdapterDisplayMode(cp.AdapterOrdinal, &mode);
    // When the network works below the frame's size the gain is stretched over it, which wants a
    // texture that can be filtered.
    o.gain16 = SUCCEEDED(d3d->CheckDeviceFormat(cp.AdapterOrdinal, cp.DeviceType, mode.Format,
                                                D3DUSAGE_DYNAMIC | (o.scaled() ? D3DUSAGE_QUERY_FILTER : 0), D3DRTYPE_TEXTURE, D3DFMT_A16B16G16R16));
    d3d->Release();
    if (FAILED(device->CreateTexture(o.cw, o.ch, 1, D3DUSAGE_DYNAMIC, o.gain16 ? D3DFMT_A16B16G16R16 : D3DFMT_A8R8G8B8,
                                     D3DPOOL_DEFAULT, &o.gain, nullptr))) {
        log("overlay: no gain texture");
        o.broken = true;
        return false;
    }
    // The masked pass: the frame now (a copy it can read), the frame the network saw, the shader.
    o.mask_ok = SUCCEEDED(device->CreateTexture(o.w, o.h, 1, D3DUSAGE_RENDERTARGET, o.format, D3DPOOL_DEFAULT, &o.copy_tex, nullptr)) &&
                SUCCEEDED(o.copy_tex->GetSurfaceLevel(0, &o.copy_surf)) &&
                SUCCEEDED(device->CreateTexture(o.cw, o.ch, 1, D3DUSAGE_DYNAMIC, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT, &o.seen_tex, nullptr)) &&
                (!o.scaled() || (SUCCEEDED(device->CreateTexture(o.cw, o.ch, 1, D3DUSAGE_RENDERTARGET, o.format, D3DPOOL_DEFAULT, &o.small_tex, nullptr)) &&
                                 SUCCEEDED(o.small_tex->GetSurfaceLevel(0, &o.small_surf)))) &&
                SUCCEEDED(device->CreatePixelShader((const DWORD *)kGainPlain, &o.mask_ps));
    if (!o.mask_ok) log("overlay: the masked pass is not available here, trailing cannot be reduced");

    // The motion search wants Shader Model 3 and half-float render targets.
    o.me_ok = o.mask_ok && o.motion.create(device, o.cw, o.ch) && SUCCEEDED(device->CreatePixelShader((const DWORD *)kGainMc, &o.mc_ps));
    log("overlay: motion search %s", o.me_ok ? "available" : "NOT available");
    if (FAILED(device->CreateStateBlock(D3DSBT_ALL, &o.state))) {
        log("overlay: no state block");
        o.broken = true;
        return false;
    }
    log("overlay: %ux%u, the network at %ux%u, format %d, %s, gain %s", o.w, o.h, o.cw, o.ch, (int)o.format,
        o.multisampled ? "multisampled" : "plain", o.gain16 ? "16-bit" : "8-bit");
    return true;
}

// A surface of the network's size into the shared memory.
bool capture(IDirect3DSurface9 *source, uint8_t *to) {
    Overlay &o = g_overlay;
    if (FAILED(o.device->GetRenderTargetData(source, o.system))) return false;
    D3DLOCKED_RECT rect;
    if (FAILED(o.system->LockRect(&rect, nullptr, D3DLOCK_READONLY))) return false;
    for (UINT y = 0; y < o.ch; ++y) memcpy(to + (size_t)y * o.cw * 4, (const uint8_t *)rect.pBits + (size_t)y * rect.Pitch, (size_t)o.cw * 4);
    o.system->UnlockRect();
    return true;
}

// The result into the gain texture.
bool upload_gain(const uint8_t *from) {
    Overlay &o = g_overlay;
    D3DLOCKED_RECT rect;
    if (FAILED(o.gain->LockRect(0, &rect, nullptr, D3DLOCK_DISCARD))) return false;
    if (o.gain16) {
        for (UINT y = 0; y < o.ch; ++y)
            memcpy((uint8_t *)rect.pBits + (size_t)y * rect.Pitch, from + (size_t)y * o.cw * 8, (size_t)o.cw * 8);
    } else {
        for (UINT y = 0; y < o.ch; ++y) {
            const uint16_t *in = (const uint16_t *)(from + (size_t)y * o.cw * 8);
            uint8_t *out = (uint8_t *)rect.pBits + (size_t)y * rect.Pitch;
            for (UINT x = 0; x < o.cw; ++x) {
                out[4 * x + 0] = (uint8_t)(in[4 * x + 2] >> 8);   // B
                out[4 * x + 1] = (uint8_t)(in[4 * x + 1] >> 8);   // G
                out[4 * x + 2] = (uint8_t)(in[4 * x + 0] >> 8);   // R
                out[4 * x + 3] = 255;
            }
        }
    }
    o.gain->UnlockRect(0);
    return true;
}

// The frame the network saw, from the capture buffer, as the gain's partner.
bool upload_seen(const uint8_t *from) {
    Overlay &o = g_overlay;
    D3DLOCKED_RECT rect;
    if (FAILED(o.seen_tex->LockRect(0, &rect, nullptr, D3DLOCK_DISCARD))) return false;
    for (UINT y = 0; y < o.ch; ++y) memcpy((uint8_t *)rect.pBits + (size_t)y * rect.Pitch, from + (size_t)y * o.cw * 4, (size_t)o.cw * 4);
    o.seen_tex->UnlockRect(0);
    return true;
}

// The frame now times the gain, but only where it still looks as it did when the network saw it
// (gain_plain.hlsl), or, with the motion search, as it did there once the motion is allowed for
// (gain_mc.hlsl).
void draw_masked(IDirect3DDevice9 *device) {
    Overlay &o = g_overlay;
    IDirect3DSurface9 *back = nullptr, *target = nullptr, *depth = nullptr;
    if (FAILED(device->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO, &back))) return;
    device->GetRenderTarget(0, &target);
    device->GetDepthStencilSurface(&depth);
    o.state->Capture();
    device->SetDepthStencilSurface(nullptr);

    device->SetRenderState(D3DRS_ZENABLE, D3DZB_FALSE);
    device->SetRenderState(D3DRS_ZWRITEENABLE, FALSE);
    device->SetRenderState(D3DRS_STENCILENABLE, FALSE);
    device->SetRenderState(D3DRS_CULLMODE, D3DCULL_NONE);
    device->SetRenderState(D3DRS_LIGHTING, FALSE);
    device->SetRenderState(D3DRS_FOGENABLE, FALSE);
    device->SetRenderState(D3DRS_SCISSORTESTENABLE, FALSE);
    device->SetRenderState(D3DRS_ALPHATESTENABLE, FALSE);
    device->SetRenderState(D3DRS_ALPHABLENDENABLE, FALSE);
    device->SetRenderState(D3DRS_SRGBWRITEENABLE, FALSE);
    device->SetRenderState(D3DRS_COLORWRITEENABLE, D3DCOLORWRITEENABLE_RED | D3DCOLORWRITEENABLE_GREEN | D3DCOLORWRITEENABLE_BLUE | D3DCOLORWRITEENABLE_ALPHA);
    for (DWORD i = 0; i < 5; ++i) {
        // The network's pictures (the frame it saw, the gain, the frame now at its size) are stretched
        // over the frame when it is smaller.
        const D3DTEXTUREFILTERTYPE filter = o.scaled() && (i == 1 || i == 2 || i == 4) ? D3DTEXF_LINEAR : D3DTEXF_POINT;
        device->SetSamplerState(i, D3DSAMP_MINFILTER, filter);
        device->SetSamplerState(i, D3DSAMP_MAGFILTER, filter);
        device->SetSamplerState(i, D3DSAMP_MIPFILTER, D3DTEXF_NONE);
        device->SetSamplerState(i, D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP);
        device->SetSamplerState(i, D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP);
        device->SetSamplerState(i, D3DSAMP_SRGBTEXTURE, FALSE);
    }

    const bool follow = g_config.options.follow && o.me_ok;
    if (follow) {
        o.motion.coarse_penalty = 0.001f;
        o.motion.fine_penalty = 0.0005f;
        o.motion.update(device, o.now_small(), o.seen_tex, o.seen_dirty);
        o.seen_dirty = false;
    }

    // The result: into the backbuffer.
    device->SetRenderTarget(0, back);
    D3DVIEWPORT9 vp = {0, 0, o.w, o.h, 0.0f, 1.0f};
    device->SetViewport(&vp);
    device->SetVertexShader(nullptr);
    device->SetPixelShader(follow ? o.mc_ps : o.mask_ps);
    device->SetTexture(0, o.copy_tex);
    device->SetTexture(1, o.seen_tex);
    device->SetTexture(2, o.gain);
    device->SetTexture(3, follow ? o.motion.field() : nullptr);
    device->SetTexture(4, o.now_small());
    if (follow && o.motion.filterable()) {
        device->SetSamplerState(3, D3DSAMP_MINFILTER, D3DTEXF_LINEAR);
        device->SetSamplerState(3, D3DSAMP_MAGFILTER, D3DTEXF_LINEAR);
    }
    // The fade runs from the tolerance to a few times it.
    const float low = g_config.options.tolerance, high = low * 4.0f + 0.04f;
    const float k[4] = {2.0f / (float)o.cw, 2.0f / (float)o.ch, low, 1.0f / (high - low)};
    const float trust_low = 0.2f, trust_high = 0.7f;
    const float pixel[4] = {1.0f / (float)o.cw, 1.0f / (float)o.ch, trust_low, 1.0f / (trust_high - trust_low)};
    device->SetPixelShaderConstantF(0, k, 1);
    device->SetPixelShaderConstantF(1, pixel, 1);
    device->BeginScene();
    draw_fullscreen_quad(device, o.w, o.h, false);
    device->EndScene();

    o.state->Apply();
    if (target) device->SetRenderTarget(0, target);   // the passes above changed it, even when it was the backbuffer
    device->SetDepthStencilSurface(depth);
    if (depth) depth->Release();
    if (target) target->Release();
    back->Release();
}

// dest = dest * gain: the blend factors give src * dest + dest * src, which is twice the
// product, and the texture holds the gain over kGainRange (2).
void draw_gain(IDirect3DDevice9 *device) {
    Overlay &o = g_overlay;
    IDirect3DSurface9 *back = nullptr, *target = nullptr, *depth = nullptr;
    if (FAILED(device->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO, &back))) return;
    device->GetRenderTarget(0, &target);
    device->GetDepthStencilSurface(&depth);
    o.state->Capture();
    if (target != back) device->SetRenderTarget(0, back);
    device->SetDepthStencilSurface(nullptr);

    D3DVIEWPORT9 vp = {0, 0, o.w, o.h, 0.0f, 1.0f};
    device->SetViewport(&vp);
    device->SetPixelShader(nullptr);
    device->SetVertexShader(nullptr);
    device->SetRenderState(D3DRS_ZENABLE, D3DZB_FALSE);
    device->SetRenderState(D3DRS_ZWRITEENABLE, FALSE);
    device->SetRenderState(D3DRS_STENCILENABLE, FALSE);
    device->SetRenderState(D3DRS_CULLMODE, D3DCULL_NONE);
    device->SetRenderState(D3DRS_LIGHTING, FALSE);
    device->SetRenderState(D3DRS_FOGENABLE, FALSE);
    device->SetRenderState(D3DRS_SCISSORTESTENABLE, FALSE);
    device->SetRenderState(D3DRS_ALPHATESTENABLE, FALSE);
    device->SetRenderState(D3DRS_SRGBWRITEENABLE, FALSE);
    device->SetRenderState(D3DRS_COLORWRITEENABLE, D3DCOLORWRITEENABLE_RED | D3DCOLORWRITEENABLE_GREEN | D3DCOLORWRITEENABLE_BLUE);
    device->SetRenderState(D3DRS_ALPHABLENDENABLE, TRUE);
    device->SetRenderState(D3DRS_SEPARATEALPHABLENDENABLE, FALSE);
    device->SetRenderState(D3DRS_BLENDOP, D3DBLENDOP_ADD);
    device->SetRenderState(D3DRS_SRCBLEND, D3DBLEND_DESTCOLOR);
    device->SetRenderState(D3DRS_DESTBLEND, D3DBLEND_SRCCOLOR);
    device->SetTexture(0, o.gain);
    device->SetSamplerState(0, D3DSAMP_MINFILTER, o.scaled() ? D3DTEXF_LINEAR : D3DTEXF_POINT);
    device->SetSamplerState(0, D3DSAMP_MAGFILTER, o.scaled() ? D3DTEXF_LINEAR : D3DTEXF_POINT);
    device->SetTextureStageState(0, D3DTSS_COLOROP, D3DTOP_SELECTARG1);
    device->SetTextureStageState(0, D3DTSS_COLORARG1, D3DTA_TEXTURE);
    device->SetTextureStageState(0, D3DTSS_ALPHAOP, D3DTOP_SELECTARG1);
    device->SetTextureStageState(0, D3DTSS_ALPHAARG1, D3DTA_TEXTURE);
    device->SetTextureStageState(0, D3DTSS_TEXCOORDINDEX, 0);
    device->SetTextureStageState(0, D3DTSS_TEXTURETRANSFORMFLAGS, D3DTTFF_DISABLE);
    device->SetTextureStageState(1, D3DTSS_COLOROP, D3DTOP_DISABLE);
    device->SetTextureStageState(1, D3DTSS_ALPHAOP, D3DTOP_DISABLE);
    device->SetSamplerState(0, D3DSAMP_MIPFILTER, D3DTEXF_NONE);
    device->SetSamplerState(0, D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP);
    device->SetSamplerState(0, D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP);
    device->SetSamplerState(0, D3DSAMP_SRGBTEXTURE, FALSE);

    struct Vertex {
        float x, y, z, rhw, u, v;
    };
    const float l = -0.5f, t = -0.5f, r = (float)o.w - 0.5f, b = (float)o.h - 0.5f;
    const Vertex quad[4] = {{l, t, 0.0f, 1.0f, 0.0f, 0.0f},
                            {r, t, 0.0f, 1.0f, 1.0f, 0.0f},
                            {l, b, 0.0f, 1.0f, 0.0f, 1.0f},
                            {r, b, 0.0f, 1.0f, 1.0f, 1.0f}};
    device->SetFVF(D3DFVF_XYZRHW | D3DFVF_TEX1);
    device->BeginScene();
    device->DrawPrimitiveUP(D3DPT_TRIANGLESTRIP, 2, quad, sizeof(Vertex));
    device->EndScene();

    o.state->Apply();
    if (target != back) device->SetRenderTarget(0, target);
    device->SetDepthStencilSurface(depth);
    if (depth) depth->Release();
    if (target) target->Release();
    back->Release();
}

float half_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    const int e = (h >> 10) & 31;
    const uint32_t m = h & 0x3FF;
    if (e == 0) {
        const float v = m / 16777216.0f;
        return (h & 0x8000) ? -v : v;
    }
    uint32_t x = e == 31 ? (sign | 0x7F800000 | (m << 13)) : (sign | ((uint32_t)(e - 15 + 127) << 23) | (m << 13));
    float f;
    memcpy(&f, &x, 4);
    return f;
}

// For testing: what the motion search found, summed up in the log.
void log_motion(IDirect3DDevice9 *device) {
    Overlay &o = g_overlay;
    if (!o.me_ok) return;
    IDirect3DSurface9 *system = nullptr;
    if (FAILED(device->CreateOffscreenPlainSurface(o.motion.grid_width(), o.motion.grid_height(), D3DFMT_A16B16G16R16F, D3DPOOL_SYSTEMMEM, &system, nullptr))) return;
    if (SUCCEEDED(device->GetRenderTargetData(o.motion.field_surface(), system))) {
        D3DLOCKED_RECT rect;
        if (SUCCEEDED(system->LockRect(&rect, nullptr, D3DLOCK_READONLY))) {
            std::vector<float> xs, ys;
            unsigned moving = 0;
            for (UINT y = 0; y < o.motion.grid_height(); ++y) {
                const uint16_t *row = (const uint16_t *)((const uint8_t *)rect.pBits + (size_t)y * rect.Pitch);
                for (UINT x = 0; x < o.motion.grid_width(); ++x) {
                    const float mx = half_to_float(row[4 * x]), my = half_to_float(row[4 * x + 1]);
                    xs.push_back(mx);
                    ys.push_back(my);
                    if (fabsf(mx) > 2.0f || fabsf(my) > 2.0f) ++moving;
                }
            }
            // The field itself, for looking at: width, height, then x, y and cost of every block.
            if (FILE *f = _wfsopen((g_dir + L"dump_motion.bin").c_str(), L"wb", _SH_DENYNO)) {
                const uint32_t dims[2] = {o.motion.grid_width(), o.motion.grid_height()};
                fwrite(dims, 4, 2, f);
                for (UINT y = 0; y < o.motion.grid_height(); ++y) {
                    const uint16_t *row = (const uint16_t *)((const uint8_t *)rect.pBits + (size_t)y * rect.Pitch);
                    for (UINT x = 0; x < o.motion.grid_width(); ++x) {
                        const float v[3] = {half_to_float(row[4 * x]), half_to_float(row[4 * x + 1]), half_to_float(row[4 * x + 2])};
                        fwrite(v, 4, 3, f);
                    }
                }
                fclose(f);
            }
            std::vector<float> sx = xs, sy = ys;
            std::sort(sx.begin(), sx.end());
            std::sort(sy.begin(), sy.end());
            const size_t n = sx.size();
            log("motion: median %.1f,%.1f px; 10%% %.1f,%.1f; 90%% %.1f,%.1f; %u of %u blocks move by more than 2 px", sx[n / 2], sy[n / 2],
                sx[n / 10], sy[n / 10], sx[n * 9 / 10], sy[n * 9 / 10], moving, (unsigned)n);
            system->UnlockRect();
        }
    }
    system->Release();
}

// For testing: the backbuffer as a 32-bit BMP, read from the device itself rather than from the screen.
void dump_backbuffer(IDirect3DDevice9 *device, const wchar_t *name) {
    Overlay &o = g_overlay;
    IDirect3DSurface9 *back = nullptr;
    if (FAILED(device->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO, &back))) return;
    IDirect3DSurface9 *source = back;
    if (o.multisampled) {
        device->StretchRect(back, nullptr, o.resolved, nullptr, D3DTEXF_NONE);
        source = o.resolved;
    }
    IDirect3DSurface9 *system = nullptr;
    if (SUCCEEDED(device->CreateOffscreenPlainSurface(o.w, o.h, o.format, D3DPOOL_SYSTEMMEM, &system, nullptr)) &&
        SUCCEEDED(device->GetRenderTargetData(source, system))) {
        D3DLOCKED_RECT rect;
        if (SUCCEEDED(system->LockRect(&rect, nullptr, D3DLOCK_READONLY))) {
            FILE *f = _wfsopen((g_dir + name).c_str(), L"wb", _SH_DENYNO);
            if (f) {
                BITMAPFILEHEADER fh = {};
                BITMAPINFOHEADER ih = {};
                fh.bfType = 0x4D42;
                fh.bfOffBits = sizeof fh + sizeof ih;
                fh.bfSize = fh.bfOffBits + o.w * o.h * 4;
                ih.biSize = sizeof ih;
                ih.biWidth = (LONG)o.w;
                ih.biHeight = -(LONG)o.h;
                ih.biPlanes = 1;
                ih.biBitCount = 32;
                fwrite(&fh, sizeof fh, 1, f);
                fwrite(&ih, sizeof ih, 1, f);
                for (UINT y = 0; y < o.h; ++y) fwrite((const uint8_t *)rect.pBits + (size_t)y * rect.Pitch, 4, o.w, f);
                fclose(f);
            }
            system->UnlockRect();
        }
    }
    if (system) system->Release();
    back->Release();
}

void run_bridge(IDirect3DDevice9 *device) {
    Overlay &o = g_overlay;
    if (o.broken) return;
    if (!o.system) {
        if (!setup(device)) return;
    }
    if (!g_bridge.start()) {
        o.broken = true;
        return;
    }
    bridge::Header *h = g_bridge.header;
    if (h->hostState == bridge::kFailed) {
        if (!o.broken) log("bridge: the host failed: %s", h->message);
        o.broken = true;
        return;
    }
    ++o.frames;
    if (h->hostState != bridge::kRunning) return;
    if (!g_bridge.map(o.cw, o.ch)) return;

    // The network's answer, if there is a new one, with the frame it was made from: the capture
    // buffer still holds it, nothing is written there until the result has been taken.
    auto consume = [&]() {
        if (h->resultSeq == g_bridge.seen_result || h->resultWidth != o.cw || h->resultHeight != o.ch) return;
        if (upload_gain(g_bridge.view + bridge::result_offset(o.cw, o.ch))) {
            if (o.mask_ok && upload_seen(g_bridge.view + bridge::capture_offset())) o.seen_dirty = true;
            o.have_gain = true;
            ++o.applied;
        }
        g_bridge.seen_result = h->resultSeq;
    };
    consume();

    // The masked pass reads a copy of the frame, and, when the network works below the frame's size, a
    // small copy too, every frame. Waiting needs none of it: the result is for this very frame.
    const bool wait = g_config.options.wait;
    bool masked = g_config.options.reduceTrailing && o.mask_ok && !wait;
    IDirect3DSurface9 *back = nullptr;
    device->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO, &back);
    if (!back) return;
    if (masked) {
        masked = SUCCEEDED(device->StretchRect(back, nullptr, o.copy_surf, nullptr, D3DTEXF_NONE));
        if (masked && o.scaled()) masked = SUCCEEDED(device->StretchRect(o.copy_surf, nullptr, o.small_surf, nullptr, D3DTEXF_LINEAR));
    }

    // A new capture, when the host wants one.
    bool sent = false;
    if (InterlockedCompareExchange(&h->hostReady, 0, 1) == 1) {
        h->width = o.cw;
        h->height = o.ch;
        ResetEvent(g_bridge.result_event);   // whatever it was signalled for is old
        IDirect3DSurface9 *source = nullptr;
        if (masked) {
            source = o.scaled() ? o.small_surf : o.copy_surf;
        } else {
            source = back;
            if (o.multisampled && SUCCEEDED(device->StretchRect(back, nullptr, o.resolved, nullptr, D3DTEXF_NONE))) source = o.resolved;
            if (o.scaled() && o.small_surf && SUCCEEDED(device->StretchRect(source, nullptr, o.small_surf, nullptr, D3DTEXF_LINEAR)))
                source = o.small_surf;
            else if (o.scaled())
                source = nullptr;
        }
        if (source && capture(source, g_bridge.view + bridge::capture_offset())) {
            InterlockedIncrement(&h->captureSeq);
            SetEvent(g_bridge.capture_event);
            ++o.sent;
            sent = true;
        } else {
            InterlockedExchange(&h->hostReady, 1);
        }
    }
    back->Release();
    // Waiting: the answer to this very frame is applied to it.
    if (sent && wait) {
        if (WaitForSingleObject(g_bridge.result_event, 500) == WAIT_OBJECT_0) consume();
        else log("bridge: the network did not answer within 500 ms");
    }

    const bool dump = g_config.dump_frame > 0 && o.frames == (unsigned)g_config.dump_frame;
    const bool in_series = g_config.dump_series > 0 && g_config.dump_frame > 0 && o.frames >= (unsigned)g_config.dump_frame &&
                           o.frames < (unsigned)(g_config.dump_frame + g_config.dump_series);
    if (in_series) {
        wchar_t name[64];
        swprintf(name, 64, L"dump_p_%03u.bmp", o.frames - (unsigned)g_config.dump_frame);
        dump_backbuffer(device, name);
    }
    if (dump) dump_backbuffer(device, L"dump_plain.bmp");
    if (o.have_gain) {
        if (masked) draw_masked(device);
        else draw_gain(device);
    }
    if (in_series) {
        wchar_t name[64];
        swprintf(name, 64, L"dump_s_%03u.bmp", o.frames - (unsigned)g_config.dump_frame);
        dump_backbuffer(device, name);
    }
    if (dump) {
        log_motion(device);
        dump_backbuffer(device, L"dump_final.bmp");
        log("dumped frame %u (gain applied: %d)", o.frames, (int)o.have_gain);
    }

    const DWORD now = GetTickCount();
    if (now - o.last_report > 10000) {
        log("bridge: %u frames, %u captures sent, %u results applied, network %.0f ms", o.frames, o.sent, o.applied, h->evaluationMs);
        o.last_report = now;
    }
}

void on_present(IDirect3DDevice9 *device) {
    Overlay &o = g_overlay;
    // While the menu is open its edits are the truth, not the file.
    if (!menu::visible() && read_config() && g_bridge.header) {
        g_bridge.header->settings = g_config.settings;
        InterlockedIncrement(&g_bridge.header->settingsSeq);
    }
    if (o.device && o.device != device) return;   // one device is enhanced: the first
    if (!o.device) {
        o.device = device;
        D3DDEVICE_CREATION_PARAMETERS cp;
        if (SUCCEEDED(device->GetCreationParameters(&cp))) o.window = cp.hFocusWindow;
    }
    // The menu is there whether or not the effect is on: it is how it gets turned on.
    if (!o.menu_ready && o.window) {
        menu::init(device, o.window);
        o.menu_ready = true;
        if (g_config.open_menu) menu::open();
    }
    if (g_config.enabled && o.system) {
        UINT cw, ch;
        network_size(o.w, o.h, cw, ch);
        if (cw != o.cw || ch != o.ch) o.release();   // the network's resolution was changed
    }
    if (g_config.enabled) run_bridge(device);

    if (menu::visible()) {
        IDirect3DSurface9 *back = nullptr;
        D3DSURFACE_DESC desc = {};
        if (SUCCEEDED(device->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO, &back))) {
            back->GetDesc(&desc);
            back->Release();
            menu::Status status = {};
            status.enabled = g_config.enabled;
            status.header = g_bridge.started && !g_bridge.failed ? g_bridge.header : nullptr;
            status.frames = o.frames;
            status.sent = o.sent;
            status.applied = o.applied;
            status.problem = o.broken && !status.header ? "The bridge cannot run here: see dlss5nr_bridge.log" : nullptr;
            status.mask_available = true;
            menu::frame(device, desc.Width, desc.Height, status);
            if (g_config.dump_frame > 0 && o.frames == (unsigned)g_config.dump_frame && o.system) dump_backbuffer(device, L"dump_menu.bmp");
        }
    }
}

HRESULT STDMETHODCALLTYPE hk_present(IDirect3DDevice9 *device, const RECT *a, const RECT *b, HWND w, const RGNDATA *c) {
    Originals *o = find_device(device);
    on_present(device);
    return o->present(device, a, b, w, c);
}

HRESULT STDMETHODCALLTYPE hk_present_ex(IDirect3DDevice9Ex *device, const RECT *a, const RECT *b, HWND w, const RGNDATA *c, DWORD f) {
    Originals *o = find_device(device);
    on_present(device);
    return o->present_ex(device, a, b, w, c, f);
}

HRESULT STDMETHODCALLTYPE hk_reset(IDirect3DDevice9 *device, D3DPRESENT_PARAMETERS *p) {
    Originals *o = find_device(device);
    if (g_overlay.device == device) {
        g_overlay.release();
        menu::device_lost();
    }
    const HRESULT hr = o->reset(device, p);
    if (SUCCEEDED(hr) && g_overlay.device == device) menu::device_restored();
    return hr;
}

HRESULT STDMETHODCALLTYPE hk_reset_ex(IDirect3DDevice9Ex *device, D3DPRESENT_PARAMETERS *p, D3DDISPLAYMODEEX *m) {
    Originals *o = find_device(device);
    if (g_overlay.device == device) {
        g_overlay.release();
        menu::device_lost();
    }
    const HRESULT hr = o->reset_ex(device, p, m);
    if (SUCCEEDED(hr) && g_overlay.device == device) menu::device_restored();
    return hr;
}

void hook_device(IDirect3DDevice9 *device, bool ex) {
    void **vtable = *(void ***)device;
    if (find_device(device)) return;
    Originals o = {};
    o.vtable = vtable;
    patch(vtable, kPresent, (void *)hk_present, (void **)&o.present);
    patch(vtable, kReset, (void *)hk_reset, (void **)&o.reset);
    if (ex) {
        patch(vtable, kPresentEx, (void *)hk_present_ex, (void **)&o.present_ex);
        patch(vtable, kResetEx, (void *)hk_reset_ex, (void **)&o.reset_ex);
    }
    g_devices.push_back(o);
    log("hooked a device (vtable %p)", vtable);
}

HRESULT STDMETHODCALLTYPE hk_create_device(IDirect3D9 *d3d, UINT adapter, D3DDEVTYPE type, HWND window, DWORD flags,
                                           D3DPRESENT_PARAMETERS *params, IDirect3DDevice9 **device) {
    FactoryOriginal *f = nullptr;
    for (FactoryOriginal &x : g_factories)
        if (x.vtable == *(void ***)d3d) f = &x;
    const HRESULT hr = f->create(d3d, adapter, type, window, flags, params, device);
    if (SUCCEEDED(hr) && device && *device) {
        log("CreateDevice: %ux%u, windowed %d, multisample %d", params->BackBufferWidth, params->BackBufferHeight,
            (int)params->Windowed, (int)params->MultiSampleType);
        hook_device(*device, false);
    }
    return hr;
}

HRESULT STDMETHODCALLTYPE hk_create_device_ex(IDirect3D9Ex *d3d, UINT adapter, D3DDEVTYPE type, HWND window, DWORD flags,
                                              D3DPRESENT_PARAMETERS *params, D3DDISPLAYMODEEX *mode, IDirect3DDevice9Ex **device) {
    FactoryOriginal *f = nullptr;
    for (FactoryOriginal &x : g_factories)
        if (x.vtable == *(void ***)d3d) f = &x;
    const HRESULT hr = f->create_ex(d3d, adapter, type, window, flags, params, mode, device);
    if (SUCCEEDED(hr) && device && *device) hook_device(*device, true);
    return hr;
}

void hook_factory(IDirect3D9 *d3d, bool ex) {
    void **vtable = *(void ***)d3d;
    for (FactoryOriginal &x : g_factories)
        if (x.vtable == vtable) return;
    FactoryOriginal f = {};
    f.vtable = vtable;
    patch(vtable, kCreateDevice, (void *)hk_create_device, (void **)&f.create);
    if (ex) patch(vtable, kCreateDeviceEx, (void *)hk_create_device_ex, (void **)&f.create_ex);
    g_factories.push_back(f);
}

} // namespace

// ------------------------------------------------------------------ the exports

extern "C" {

IDirect3D9 *WINAPI Direct3DCreate9(UINT sdk) {
    if (!load_real()) return nullptr;
    IDirect3D9 *d3d = g_create9(sdk);
    if (d3d) hook_factory(d3d, false);
    return d3d;
}

HRESULT WINAPI Direct3DCreate9Ex(UINT sdk, IDirect3D9Ex **out) {
    if (!load_real() || !g_create9ex) return D3DERR_NOTAVAILABLE;
    const HRESULT hr = g_create9ex(sdk, out);
    if (SUCCEEDED(hr) && out && *out) hook_factory(*out, true);
    return hr;
}

int WINAPI D3DPERF_BeginEvent(D3DCOLOR c, LPCWSTR n) { return load_real() && g_begin_event ? g_begin_event(c, n) : 0; }
int WINAPI D3DPERF_EndEvent() { return load_real() && g_end_event ? g_end_event() : 0; }
void WINAPI D3DPERF_SetMarker(D3DCOLOR c, LPCWSTR n) { if (load_real() && g_set_marker) g_set_marker(c, n); }
void WINAPI D3DPERF_SetRegion(D3DCOLOR c, LPCWSTR n) { if (load_real() && g_set_region) g_set_region(c, n); }
BOOL WINAPI D3DPERF_QueryRepeatFrame() { return load_real() && g_query_repeat ? g_query_repeat() : FALSE; }
void WINAPI D3DPERF_SetOptions(DWORD o) { if (load_real() && g_set_options) g_set_options(o); }
DWORD WINAPI D3DPERF_GetStatus() { return load_real() && g_get_status ? g_get_status() : 0; }
void WINAPI Direct3D9EnableMaximizedWindowedModeShim(BOOL b) { if (load_real() && g_shim) g_shim(b); }

} // extern "C"

#pragma comment(linker, "/EXPORT:Direct3DCreate9=_Direct3DCreate9@4,@37")
#pragma comment(linker, "/EXPORT:Direct3DCreate9Ex=_Direct3DCreate9Ex@8,@38")
#pragma comment(linker, "/EXPORT:D3DPERF_BeginEvent=_D3DPERF_BeginEvent@8,@27")
#pragma comment(linker, "/EXPORT:D3DPERF_EndEvent=_D3DPERF_EndEvent@0,@28")
#pragma comment(linker, "/EXPORT:D3DPERF_GetStatus=_D3DPERF_GetStatus@0,@29")
#pragma comment(linker, "/EXPORT:D3DPERF_QueryRepeatFrame=_D3DPERF_QueryRepeatFrame@0,@30")
#pragma comment(linker, "/EXPORT:D3DPERF_SetMarker=_D3DPERF_SetMarker@8,@31")
#pragma comment(linker, "/EXPORT:D3DPERF_SetOptions=_D3DPERF_SetOptions@4,@32")
#pragma comment(linker, "/EXPORT:D3DPERF_SetRegion=_D3DPERF_SetRegion@8,@33")
#pragma comment(linker, "/EXPORT:Direct3D9EnableMaximizedWindowedModeShim=_Direct3D9EnableMaximizedWindowedModeShim@4,@36")

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        g_self = instance;
        DisableThreadLibraryCalls(instance);
        core_init(instance, L"dlss5nr_bridge.log");
        input::install();
    } else if (reason == DLL_PROCESS_DETACH) {
        g_bridge.stop();
    }
    return TRUE;
}
