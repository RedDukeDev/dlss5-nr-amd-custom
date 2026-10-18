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
#include "../common/protocol.h"
#include "menu.h"

namespace {

// ------------------------------------------------------------------ logging

HMODULE g_self = nullptr;
std::wstring g_dir;
FILE *g_log = nullptr;
CRITICAL_SECTION g_log_lock;

void log(const char *format, ...) {
    if (!g_log) return;
    EnterCriticalSection(&g_log_lock);
    SYSTEMTIME t;
    GetLocalTime(&t);
    fprintf(g_log, "[%02d:%02d:%02d.%03d] ", t.wHour, t.wMinute, t.wSecond, t.wMilliseconds);
    va_list args;
    va_start(args, format);
    vfprintf(g_log, format, args);
    va_end(args);
    fputc('\n', g_log);
    fflush(g_log);
    LeaveCriticalSection(&g_log_lock);
}

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

// ------------------------------------------------------------------ settings

struct Config {
    bool enabled = false;
    std::wstring host;       // the host's path
    bool open_menu = false;  // BridgeDebugOpenMenu: the menu starts open
    int dump_frame = 0;      // DebugDumpFrame: write the frame before and after the gain to bmp files
    bridge::Settings settings{};
    FILETIME stamp{};
    std::wstring ini;
};
Config g_config;

float read_float(const wchar_t *key, float otherwise) {
    wchar_t text[64] = {};
    GetPrivateProfileStringW(L"NeuralRendering", key, L"auto", text, 64, g_config.ini.c_str());
    if (!_wcsicmp(text, L"auto") || !text[0]) return otherwise;
    return (float)_wtof(text);
}

int32_t read_int(const wchar_t *key, int32_t otherwise) {
    wchar_t text[64] = {};
    GetPrivateProfileStringW(L"NeuralRendering", key, L"auto", text, 64, g_config.ini.c_str());
    if (!_wcsicmp(text, L"auto") || !text[0]) return otherwise;
    if (!_wcsicmp(text, L"true")) return 1;
    if (!_wcsicmp(text, L"false")) return 0;
    return _wtoi(text);
}

// Reads the [NeuralRendering] keys OptiScaler.ini has, the same ones the 64-bit games use.
// Returns true when something changed.
bool read_config() {
    WIN32_FILE_ATTRIBUTE_DATA data;
    if (!GetFileAttributesExW(g_config.ini.c_str(), GetFileExInfoStandard, &data)) return false;
    if (CompareFileTime(&data.ftLastWriteTime, &g_config.stamp) == 0) return false;
    g_config.stamp = data.ftLastWriteTime;
    bridge::Settings &s = g_config.settings;
    s.enabled = read_int(L"Enabled", 0);
    s.resolutionScale = read_float(L"ResolutionScale", bridge::kAutoFloat);
    s.style = read_int(L"Style", bridge::kAutoInt);
    s.intensity = read_float(L"Intensity", bridge::kAutoFloat);
    s.localTone = read_float(L"LocalTone", bridge::kAutoFloat);
    s.localStructure = read_float(L"LocalStructure", bridge::kAutoFloat);
    s.skinStructure = read_float(L"SkinStructure", bridge::kAutoFloat);
    s.detailStrength = read_float(L"DetailStrength", bridge::kAutoFloat);
    s.colourStrength = read_float(L"ColourStrength", bridge::kAutoFloat);
    s.maxRatio = read_float(L"MaxRatio", bridge::kAutoFloat);
    s.passes = read_int(L"Passes", bridge::kAutoInt);
    s.keepLocalTone = read_int(L"KeepLocalTone", bridge::kAutoInt);
    s.preset = read_int(L"Preset", bridge::kAutoInt);
    g_config.enabled = s.enabled != 0;
    wchar_t host[MAX_PATH] = {};
    GetPrivateProfileStringW(L"NeuralRendering", L"BridgeHost", L"", host, MAX_PATH, g_config.ini.c_str());
    g_config.host = host;
    g_config.dump_frame = read_int(L"BridgeDebugDumpFrame", 0);
    g_config.open_menu = read_int(L"BridgeDebugOpenMenu", 0) != 0;
    return true;
}

// ------------------------------------------------------------------ the bridge

struct Bridge {
    HANDLE mapping = nullptr;
    HANDLE capture_event = nullptr;
    HANDLE job = nullptr;
    HANDLE process = nullptr;
    bridge::Header *header = nullptr;
    uint8_t *view = nullptr;
    uint32_t view_w = 0, view_h = 0;
    bool failed = false;
    bool started = false;
    int32_t seen_settings = 0;
    int32_t seen_result = 0;

    bool start() {
        if (started) return !failed;
        started = true;
        const DWORD pid = GetCurrentProcessId();
        wchar_t name[96];
        swprintf(name, 96, L"Local\\dlss5nr_bridge_%lu", pid);
        const uint64_t bytes = bridge::max_total_bytes();
        mapping = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, (DWORD)(bytes >> 32), (DWORD)bytes, name);
        swprintf(name, 96, L"Local\\dlss5nr_bridge_capture_%lu", pid);
        capture_event = CreateEventW(nullptr, FALSE, FALSE, name);
        if (!mapping || !capture_event) return fail("could not create the shared memory");
        header = (bridge::Header *)MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, bridge::kHeaderBytes);
        if (!header) return fail("could not map the header");
        memset(header, 0, sizeof *header);
        header->magic = bridge::kMagic;
        header->version = bridge::kVersion;
        header->clientPid = pid;
        header->settings = g_config.settings;
        header->settingsSeq = 1;
        return spawn();
    }

    bool fail(const char *what) {
        failed = true;
        log("bridge: %s (error %lu)", what, GetLastError());
        return false;
    }

    bool spawn() {
        std::wstring host = g_config.host;
        if (host.empty()) host = g_dir + L"OptiScaler\\dlss5nr\\dlss5nr_host.exe";
        if (GetFileAttributesW(host.c_str()) == INVALID_FILE_ATTRIBUTES) {
            log("bridge: the host is not there: %ls", host.c_str());
            return fail("no host");
        }
        // The host goes where the game goes: when the game ends, the job takes it with it.
        job = CreateJobObjectW(nullptr, nullptr);
        if (job) {
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits = {};
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof limits);
        }
        wchar_t command[2 * MAX_PATH];
        swprintf(command, 2 * MAX_PATH, L"\"%ls\" --client %lu", host.c_str(), GetCurrentProcessId());
        std::wstring work = host.substr(0, host.find_last_of(L'\\'));
        STARTUPINFOW si = {sizeof si};
        PROCESS_INFORMATION pi = {};
        if (!CreateProcessW(nullptr, command, nullptr, nullptr, FALSE, CREATE_NO_WINDOW | CREATE_SUSPENDED, nullptr,
                            work.c_str(), &si, &pi))
            return fail("could not start the host");
        if (job) AssignProcessToJobObject(job, pi.hProcess);
        ResumeThread(pi.hThread);
        CloseHandle(pi.hThread);
        process = pi.hProcess;
        log("bridge: host started (%ls)", host.c_str());
        return true;
    }

    // The view has to cover the buffers of the current size, and no more: this is a 32-bit
    // process, and a game with a large address space has little to spare.
    bool map(uint32_t w, uint32_t h) {
        if (view && view_w == w && view_h == h) return true;
        if (view) UnmapViewOfFile(view);
        view = nullptr;
        if (w > bridge::kMaxWidth || h > bridge::kMaxHeight) return false;
        view = (uint8_t *)MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, (SIZE_T)bridge::total_bytes(w, h));
        view_w = w;
        view_h = h;
        if (!view) log("bridge: could not map %ux%u (error %lu)", w, h, GetLastError());
        return view != nullptr;
    }

    void stop() {
        if (header) InterlockedExchange(&header->shutdown, 1);
        if (capture_event) SetEvent(capture_event);
    }
};
Bridge g_bridge;

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
    UINT w = 0, h = 0;
    D3DFORMAT format = D3DFMT_UNKNOWN;
    bool multisampled = false;
    bool gain16 = true;     // the gain texture is A16B16G16R16, else A8R8G8B8
    IDirect3DSurface9 *system = nullptr;    // the capture lands here
    IDirect3DSurface9 *resolved = nullptr;  // the backbuffer resolved, when it is multisampled
    IDirect3DTexture9 *gain = nullptr;
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
        if (state) state->Release();
        system = resolved = nullptr;
        gain = nullptr;
        state = nullptr;
        have_gain = false;
        w = h = 0;
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
    if (FAILED(device->CreateOffscreenPlainSurface(o.w, o.h, o.format, D3DPOOL_SYSTEMMEM, &o.system, nullptr))) {
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
    o.gain16 = SUCCEEDED(d3d->CheckDeviceFormat(cp.AdapterOrdinal, cp.DeviceType, mode.Format, D3DUSAGE_DYNAMIC,
                                                D3DRTYPE_TEXTURE, D3DFMT_A16B16G16R16));
    d3d->Release();
    if (FAILED(device->CreateTexture(o.w, o.h, 1, D3DUSAGE_DYNAMIC, o.gain16 ? D3DFMT_A16B16G16R16 : D3DFMT_A8R8G8B8,
                                     D3DPOOL_DEFAULT, &o.gain, nullptr))) {
        log("overlay: no gain texture");
        o.broken = true;
        return false;
    }
    if (FAILED(device->CreateStateBlock(D3DSBT_ALL, &o.state))) {
        log("overlay: no state block");
        o.broken = true;
        return false;
    }
    log("overlay: %ux%u, format %d, %s, gain %s", o.w, o.h, (int)o.format, o.multisampled ? "multisampled" : "plain",
        o.gain16 ? "16-bit" : "8-bit");
    return true;
}

// The backbuffer into the shared memory.
bool capture(IDirect3DDevice9 *device, uint8_t *to) {
    Overlay &o = g_overlay;
    IDirect3DSurface9 *back = nullptr;
    if (FAILED(device->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO, &back))) return false;
    IDirect3DSurface9 *source = back;
    if (o.multisampled) {
        if (FAILED(device->StretchRect(back, nullptr, o.resolved, nullptr, D3DTEXF_NONE))) {
            back->Release();
            return false;
        }
        source = o.resolved;
    }
    HRESULT hr = device->GetRenderTargetData(source, o.system);
    back->Release();
    if (FAILED(hr)) return false;
    D3DLOCKED_RECT rect;
    if (FAILED(o.system->LockRect(&rect, nullptr, D3DLOCK_READONLY))) return false;
    for (UINT y = 0; y < o.h; ++y) memcpy(to + (size_t)y * o.w * 4, (const uint8_t *)rect.pBits + (size_t)y * rect.Pitch, (size_t)o.w * 4);
    o.system->UnlockRect();
    return true;
}

// The result into the gain texture.
bool upload_gain(const uint8_t *from) {
    Overlay &o = g_overlay;
    D3DLOCKED_RECT rect;
    if (FAILED(o.gain->LockRect(0, &rect, nullptr, D3DLOCK_DISCARD))) return false;
    if (o.gain16) {
        for (UINT y = 0; y < o.h; ++y)
            memcpy((uint8_t *)rect.pBits + (size_t)y * rect.Pitch, from + (size_t)y * o.w * 8, (size_t)o.w * 8);
    } else {
        for (UINT y = 0; y < o.h; ++y) {
            const uint16_t *in = (const uint16_t *)(from + (size_t)y * o.w * 8);
            uint8_t *out = (uint8_t *)rect.pBits + (size_t)y * rect.Pitch;
            for (UINT x = 0; x < o.w; ++x) {
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
    device->SetTextureStageState(0, D3DTSS_COLOROP, D3DTOP_SELECTARG1);
    device->SetTextureStageState(0, D3DTSS_COLORARG1, D3DTA_TEXTURE);
    device->SetTextureStageState(0, D3DTSS_ALPHAOP, D3DTOP_SELECTARG1);
    device->SetTextureStageState(0, D3DTSS_ALPHAARG1, D3DTA_TEXTURE);
    device->SetTextureStageState(0, D3DTSS_TEXCOORDINDEX, 0);
    device->SetTextureStageState(0, D3DTSS_TEXTURETRANSFORMFLAGS, D3DTTFF_DISABLE);
    device->SetTextureStageState(1, D3DTSS_COLOROP, D3DTOP_DISABLE);
    device->SetTextureStageState(1, D3DTSS_ALPHAOP, D3DTOP_DISABLE);
    device->SetSamplerState(0, D3DSAMP_MINFILTER, D3DTEXF_POINT);
    device->SetSamplerState(0, D3DSAMP_MAGFILTER, D3DTEXF_POINT);
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
    if (SUCCEEDED(device->GetRenderTargetData(source, o.system))) {
        D3DLOCKED_RECT rect;
        if (SUCCEEDED(o.system->LockRect(&rect, nullptr, D3DLOCK_READONLY))) {
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
            o.system->UnlockRect();
        }
    }
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
    if (!g_bridge.map(o.w, o.h)) return;

    // The network's answer, if there is a new one.
    if (h->resultSeq != g_bridge.seen_result && h->resultWidth == o.w && h->resultHeight == o.h) {
        const uint8_t *from = g_bridge.view + bridge::result_offset(o.w, o.h);
        if (upload_gain(from)) {
            o.have_gain = true;
            ++o.applied;
        }
        g_bridge.seen_result = h->resultSeq;
    }
    // A new capture, when the host wants one.
    if (InterlockedCompareExchange(&h->hostReady, 0, 1) == 1) {
        h->width = o.w;
        h->height = o.h;
        if (capture(device, g_bridge.view + bridge::capture_offset())) {
            InterlockedIncrement(&h->captureSeq);
            SetEvent(g_bridge.capture_event);
            ++o.sent;
        } else {
            InterlockedExchange(&h->hostReady, 1);
        }
    }
    const bool dump = g_config.dump_frame > 0 && o.frames == (unsigned)g_config.dump_frame;
    if (dump) dump_backbuffer(device, L"dump_plain.bmp");
    if (o.have_gain) draw_gain(device);
    if (dump) {
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

void start_logging() {
    wchar_t path[MAX_PATH];
    GetModuleFileNameW(g_self, path, MAX_PATH);
    g_dir = path;
    g_dir = g_dir.substr(0, g_dir.find_last_of(L'\\') + 1);
    InitializeCriticalSection(&g_log_lock);
    g_log = _wfsopen((g_dir + L"dlss5nr_bridge.log").c_str(), L"w", _SH_DENYNO);
    g_config.ini = g_dir + L"OptiScaler.ini";
    read_config();
    log("client loaded in %lu, enabled=%d", GetCurrentProcessId(), (int)g_config.enabled);
}

} // namespace

// What the menu changes goes to the host at once and, when it is final, to OptiScaler.ini.
bridge::Settings &bridge_user_settings() { return g_config.settings; }

void bridge_setting_changed(const wchar_t *key, const wchar_t *text, bool save) {
    g_config.enabled = g_config.settings.enabled != 0;
    if (g_bridge.header) {
        g_bridge.header->settings = g_config.settings;
        InterlockedIncrement(&g_bridge.header->settingsSeq);
    }
    if (save) WritePrivateProfileStringW(L"NeuralRendering", key, text, g_config.ini.c_str());
}

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
        start_logging();
    } else if (reason == DLL_PROCESS_DETACH) {
        g_bridge.stop();
    }
    return TRUE;
}
