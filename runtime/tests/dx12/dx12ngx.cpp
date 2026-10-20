// A Direct3D 12 test "game" that upscales through the NGX API, as a DLSS game does, with OptiScaler
// answering it (loaded as dxgi.dll beside this exe): the scene is made on the CPU at render resolution
// with colour, depth and motion vectors, uploaded and EvaluateFeature'd inside one command list (the
// way a game records its frame), and the output is read back to a BMP at a chosen time.
//
// usage: dx12ngx [seconds=14] [capture_at=10] [out.bmp] [out_w=1920] [out_h=1080] [render_percent=50]
//                [freeze_time] [options]
// options: --subrect     textures larger than the rendered region, which sits in a corner of its own
//          --exposure    hands an exposure texture and turns auto exposure off
//          --masks       hands a reactive mask (empty) and a transparency mask
//          --dynamic N   renders N percent of the render size (dynamic resolution)
//          --rr          asks for ray reconstruction (a flat G-buffer)
//
// Build with the Visual Studio developer prompt (needs OptiScaler's NGX headers):
//   cl /O2 /EHsc /std:c++17 /I <OptiScaler>\external\nvngx_dlss_sdk dx12ngx.cpp
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>

#include "nvsdk_ngx.h"
#include "nvsdk_ngx_params.h"

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "user32.lib")

static LRESULT CALLBACK proc(HWND h, UINT m, WPARAM w, LPARAM l) {
    if (m == WM_DESTROY) PostQuitMessage(0);
    return DefWindowProcW(h, m, w, l);
}

static float halton(int i, int b) { float f = 1, r = 0; while (i > 0) { f /= b; r += f * (i % b); i /= b; } return r; }

static uint16_t to_half(float v) {
    uint32_t x; memcpy(&x, &v, 4);
    const uint32_t sign = (x >> 16) & 0x8000;
    int exp = (int)((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffff;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00);
    return (uint16_t)(sign | (exp << 10) | (mant >> 13));
}

static float hash2(float x, float y) { float s = sinf(x * 127.1f + y * 311.7f) * 43758.5453f; return s - floorf(s); }
static float noise2(float x, float y) {
    static const bool smooth = getenv("SMOOTH") != nullptr;   // a scene without the pixel-scale texture
    if (smooth) return 0.5f + 0.5f * sinf(x * 0.7f) * cosf(y * 0.9f);
    float ix = floorf(x), iy = floorf(y), fx = x - ix, fy = y - iy;
    fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy);
    float a = hash2(ix, iy), b = hash2(ix + 1, iy), c = hash2(ix, iy + 1), d = hash2(ix + 1, iy + 1);
    return (a + (b - a) * fx) * (1 - fy) + (c + (d - c) * fx) * fy;
}
static float mixf(float a, float b, float t) { return a + (b - a) * t; }

struct Pixel { float r, g, b, depth, mvx, mvy; };

// The same scene the Direct3D 11 test draws on the GPU.
static Pixel scene(float px, float py, float w, float h, float t, float moving) {
    float u = px / w, v = py / h, qx = u * (w / h), qy = v;
    Pixel o{};
    float skyt = powf(v, 1.5f);
    float r = mixf(0.25f, 0.9f, skyt), g = mixf(0.45f, 0.75f, skyt), b = mixf(0.85f, 0.6f, skyt);
    float horizon = 0.55f + 0.08f * noise2(qx * 6, qy * 6) + 0.05f * sinf(qx * 5);
    float depth = 1.0f, mvx = 0;
    if (v > horizon) {
        float n = noise2(qx * 40, qy * 40);
        float gr = mixf(0.15f, 0.35f, n), gg = mixf(0.3f, 0.28f, n), gb = mixf(0.1f, 0.15f, n);
        float k = 0.6f + 0.4f * noise2(qx * 160, qy * 160);
        r = gr * k; g = gg * k; b = gb * k;
        depth = 0.2f + 0.7f * (1 - v);
    }
    float spx = 0.3f + 0.4f * (t * 0.1f - floorf(t * 0.1f)), spy = 0.62f;
    float dx = (u - spx) * (w / h), dy = v - spy, rr = sqrtf(dx * dx + dy * dy);
    if (rr < 0.1f) {
        float z = sqrtf(1 - (rr / 0.1f) * (rr / 0.1f)), nx = dx / 0.1f, ny = dy / 0.1f;
        float lx = -0.4f, ly = -0.6f, lz = 0.7f, ln = sqrtf(lx * lx + ly * ly + lz * lz);
        float l = nx * lx / ln + ny * ly / ln + z * lz / ln; l = l < 0 ? 0 : l > 1 ? 1 : l;
        float spec = powf(l, 40) * 0.5f;
        r = 0.8f * (0.15f + 0.85f * l) + spec; g = 0.25f * (0.15f + 0.85f * l) + spec; b = 0.2f * (0.15f + 0.85f * l) + spec;
        depth = 0.15f;
        mvx = -0.04f * w / 60.0f * moving;
    }
    o.r = r; o.g = g; o.b = b; o.depth = depth; o.mvx = mvx; o.mvy = 0;
    return o;
}

#define CHECK(x) do { HRESULT _hr = (x); if (FAILED(_hr)) { printf("%s failed: %08x\n", #x, (unsigned)_hr); return 10; } } while (0)

typedef NVSDK_NGX_Result (*PFN_Init)(unsigned long long, const wchar_t *, ID3D12Device *, const NVSDK_NGX_FeatureCommonInfo *, NVSDK_NGX_Version);
typedef NVSDK_NGX_Result (*PFN_Caps)(NVSDK_NGX_Parameter **);
typedef NVSDK_NGX_Result (*PFN_Create)(ID3D12GraphicsCommandList *, NVSDK_NGX_Feature, NVSDK_NGX_Parameter *, NVSDK_NGX_Handle **);
typedef NVSDK_NGX_Result (*PFN_Eval)(ID3D12GraphicsCommandList *, const NVSDK_NGX_Handle *, NVSDK_NGX_Parameter *, PFN_NVSDK_NGX_ProgressCallback);

static D3D12_HEAP_PROPERTIES heap(D3D12_HEAP_TYPE type) { D3D12_HEAP_PROPERTIES p = {}; p.Type = type; return p; }

static ID3D12Resource *make_texture(ID3D12Device *dev, UINT w, UINT h, DXGI_FORMAT f, bool uav, D3D12_RESOURCE_STATES state) {
    D3D12_RESOURCE_DESC d = {};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; d.Width = w; d.Height = h; d.DepthOrArraySize = 1; d.MipLevels = 1;
    d.Format = f; d.SampleDesc.Count = 1; d.Flags = uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
    auto hp = heap(D3D12_HEAP_TYPE_DEFAULT);
    ID3D12Resource *r = nullptr;
    if (FAILED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, state, nullptr, IID_PPV_ARGS(&r)))) return nullptr;
    return r;
}

static ID3D12Resource *make_buffer(ID3D12Device *dev, UINT64 size, D3D12_HEAP_TYPE type, D3D12_RESOURCE_STATES state) {
    D3D12_RESOURCE_DESC d = {};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; d.Width = size; d.Height = 1; d.DepthOrArraySize = 1; d.MipLevels = 1;
    d.SampleDesc.Count = 1; d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    auto hp = heap(type);
    ID3D12Resource *r = nullptr;
    if (FAILED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, state, nullptr, IID_PPV_ARGS(&r)))) return nullptr;
    return r;
}

static void barrier(ID3D12GraphicsCommandList *l, ID3D12Resource *r, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b) {
    if (a == b) return;
    D3D12_RESOURCE_BARRIER x = {};
    x.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; x.Transition.pResource = r;
    x.Transition.StateBefore = a; x.Transition.StateAfter = b; x.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    l->ResourceBarrier(1, &x);
}

// A texture and where its upload goes.
struct Upload {
    ID3D12Resource *texture = nullptr;
    ID3D12Resource *buffer = nullptr;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    uint8_t *mapped = nullptr;
    UINT width = 0, height = 0, bytes = 0;
};

static bool make_upload(ID3D12Device *dev, Upload &u, UINT w, UINT h, DXGI_FORMAT f, UINT bytes) {
    u.width = w; u.height = h; u.bytes = bytes;
    u.texture = make_texture(dev, w, h, f, false, D3D12_RESOURCE_STATE_COMMON);
    if (!u.texture) return false;
    UINT64 total = 0;
    auto desc = u.texture->GetDesc();
    dev->GetCopyableFootprints(&desc, 0, 1, 0, &u.footprint, nullptr, nullptr, &total);
    u.buffer = make_buffer(dev, total, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    if (!u.buffer) return false;
    D3D12_RANGE none = {0, 0};
    return SUCCEEDED(u.buffer->Map(0, &none, (void **)&u.mapped));
}

static void record_upload(ID3D12GraphicsCommandList *l, Upload &u) {
    barrier(l, u.texture, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_TEXTURE_COPY_LOCATION dst = {}, src = {};
    dst.pResource = u.texture; dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.pResource = u.buffer; src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; src.PlacedFootprint = u.footprint;
    l->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    barrier(l, u.texture, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
}

static float from_half(uint16_t h) {
    const uint32_t sign = (h & 0x8000u) << 16; int exp = (h >> 10) & 31; uint32_t mant = h & 1023;
    uint32_t x;
    if (exp == 0) { if (!mant) x = sign; else { exp = 1; while (!(mant & 1024)) { mant <<= 1; --exp; } mant &= 1023; x = sign | ((uint32_t)(exp + 112) << 23) | (mant << 13); } }
    else if (exp == 31) x = sign | 0x7f800000u | (mant << 13);
    else x = sign | ((uint32_t)(exp + 112) << 23) | (mant << 13);
    float f; memcpy(&f, &x, 4); return f;
}

static void write_bmp(const char *path, const uint8_t *rgba, UINT pitch, int w, int h, bool half = false) {
    BITMAPFILEHEADER fh = {}; BITMAPINFOHEADER ih = {};
    fh.bfType = 0x4D42; fh.bfOffBits = sizeof fh + sizeof ih; fh.bfSize = fh.bfOffBits + w * h * 4;
    ih.biSize = sizeof ih; ih.biWidth = w; ih.biHeight = -h; ih.biPlanes = 1; ih.biBitCount = 32; ih.biCompression = BI_RGB;
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fwrite(&fh, sizeof fh, 1, f); fwrite(&ih, sizeof ih, 1, f);
    std::vector<uint8_t> row(w * 4);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (half) {
                const uint16_t *p = (const uint16_t *)(rgba + (size_t)y * pitch) + x * 4;
                auto q = [](float v) { v = v < 0 ? 0 : v > 1 ? 1 : v; return (uint8_t)(v * 255.f + 0.5f); };
                row[x * 4 + 0] = q(from_half(p[2])); row[x * 4 + 1] = q(from_half(p[1])); row[x * 4 + 2] = q(from_half(p[0])); row[x * 4 + 3] = 255;
                continue;
            }
            const uint8_t *p = rgba + (size_t)y * pitch + x * 4;
            row[x * 4 + 0] = p[2]; row[x * 4 + 1] = p[1]; row[x * 4 + 2] = p[0]; row[x * 4 + 3] = 255;
        }
        fwrite(row.data(), 1, row.size(), f);
    }
    fclose(f);
}

int main(int argc, char **argv) {
    std::vector<std::string> positional;
    bool subrect = false, with_exposure = false, with_masks = false, rr = false, fp16 = false;
    int preset_hint = 0;       // --preset N: the render preset the game asks for (11 K, 13 M), as a hint at every quality
    int dynamic_percent = 0;   // --dynamic N: the region rendered is N percent of the output (a game scales between 50 and 100)
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--subrect") subrect = true;
        else if (a == "--exposure") with_exposure = true;
        else if (a == "--masks") with_masks = true;
        else if (a == "--rr") rr = true;
        else if (a == "--fp16") fp16 = true;   // colour and output as half floats, not 8 bits
        else if (a == "--preset" && i + 1 < argc) preset_hint = atoi(argv[++i]);
        else if (a == "--dynamic" && i + 1 < argc) dynamic_percent = atoi(argv[++i]);
        else positional.push_back(a);
    }
    auto arg = [&](size_t i, const char *fallback) { return i < positional.size() ? positional[i].c_str() : fallback; };
    const double seconds = atof(arg(0, "14")), capture_at = atof(arg(1, "10"));
    const char *out = arg(2, "shot12.bmp");
    const int W = atoi(arg(3, "1920")), H = atoi(arg(4, "1080")), percent = atoi(arg(5, "50"));
    const double freeze = atof(arg(6, "-1"));
    const int RW = W * percent / 100, RH = H * percent / 100;
    const int DW = dynamic_percent ? W * dynamic_percent / 100 : RW, DH = dynamic_percent ? H * dynamic_percent / 100 : RH;   // the region rendered
    const UINT pad_w = subrect ? 64 : 0, pad_h = subrect ? 48 : 0;
    const UINT TW = (UINT)(DW > RW ? DW : RW) + pad_w, TH = (UINT)(DH > RH ? DH : RH) + pad_h;
    const UINT ox = subrect ? 37 : 0, oy = subrect ? 23 : 0;     // where the region starts in the game's textures
    const UINT GW = W + (subrect ? 40 : 0), GH = H + (subrect ? 30 : 0);
    const UINT out_x = subrect ? 17 : 0, out_y = subrect ? 9 : 0;

    WNDCLASSW wc = {}; wc.lpfnWndProc = proc; wc.hInstance = GetModuleHandleW(nullptr); wc.lpszClassName = L"dx12ngx"; wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    RegisterClassW(&wc);
    RECT rc = {0, 0, W, H}; AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);
    HWND hwnd = CreateWindowW(L"dx12ngx", L"dx12ngx", WS_OVERLAPPEDWINDOW | WS_VISIBLE, 20, 20, rc.right - rc.left, rc.bottom - rc.top, nullptr, nullptr, wc.hInstance, nullptr);

    IDXGIFactory4 *factory; CHECK(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
    IDXGIAdapter1 *adapter = nullptr, *pick = nullptr; SIZE_T best = 0;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) == S_OK; ++i) {
        DXGI_ADAPTER_DESC1 d; adapter->GetDesc1(&d);
        if (!(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) && d.DedicatedVideoMemory > best) { best = d.DedicatedVideoMemory; pick = adapter; }
    }
    ID3D12Device *dev; CHECK(D3D12CreateDevice(pick, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev)));
    D3D12_COMMAND_QUEUE_DESC qd = {}; qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ID3D12CommandQueue *queue; CHECK(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)));
    DXGI_SWAP_CHAIN_DESC1 sd = {}; sd.Width = W; sd.Height = H; sd.Format = DXGI_FORMAT_R8G8B8A8_UNORM; sd.SampleDesc.Count = 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT; sd.BufferCount = 2; sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    IDXGISwapChain1 *sc1; CHECK(factory->CreateSwapChainForHwnd(queue, hwnd, &sd, nullptr, nullptr, &sc1));
    IDXGISwapChain3 *sc; CHECK(sc1->QueryInterface(IID_PPV_ARGS(&sc)));

    ID3D12CommandAllocator *alloc; CHECK(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc)));
    ID3D12GraphicsCommandList *list; CHECK(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, nullptr, IID_PPV_ARGS(&list)));
    ID3D12Fence *fence; CHECK(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    UINT64 fence_value = 0;
    auto wait_idle = [&] {
        queue->Signal(fence, ++fence_value);
        if (fence->GetCompletedValue() < fence_value) { fence->SetEventOnCompletion(fence_value, event); WaitForSingleObject(event, 60000); }
    };

    Upload color, depth, motion, exposure_u, reactive_u, transparency_u, diffuse_u, specular_u, normals_u;
    if (!make_upload(dev, color, TW, TH, fp16 ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_R8G8B8A8_UNORM, fp16 ? 8 : 4) || !make_upload(dev, depth, TW, TH, DXGI_FORMAT_R32_FLOAT, 4) ||
        !make_upload(dev, motion, TW, TH, DXGI_FORMAT_R16G16_FLOAT, 4)) { puts("textures"); return 11; }
    if (with_exposure && !make_upload(dev, exposure_u, 1, 1, DXGI_FORMAT_R32_FLOAT, 4)) return 11;
    if (with_masks && (!make_upload(dev, reactive_u, TW, TH, DXGI_FORMAT_R8_UNORM, 1) || !make_upload(dev, transparency_u, TW, TH, DXGI_FORMAT_R8_UNORM, 1))) return 11;
    if (rr && (!make_upload(dev, diffuse_u, TW, TH, DXGI_FORMAT_R8G8B8A8_UNORM, 4) || !make_upload(dev, specular_u, TW, TH, DXGI_FORMAT_R8G8B8A8_UNORM, 4) ||
               !make_upload(dev, normals_u, TW, TH, DXGI_FORMAT_R16G16B16A16_FLOAT, 8))) return 11;
    ID3D12Resource *output = make_texture(dev, GW, GH, fp16 ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_R8G8B8A8_UNORM, true, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (!output) { puts("output"); return 11; }
    // The readback of the output's region.
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT out_fp{}; UINT64 out_total = 0;
    { auto d = output->GetDesc(); dev->GetCopyableFootprints(&d, 0, 1, 0, &out_fp, nullptr, nullptr, &out_total); }
    ID3D12Resource *readback = make_buffer(dev, out_total, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);

    // NGX, through the module that answers it: OptiScaler, loaded as dxgi.dll
    HMODULE ngx = GetModuleHandleW(L"dxgi.dll");
    auto Init = (PFN_Init)GetProcAddress(ngx, "NVSDK_NGX_D3D12_Init");
    auto Caps = (PFN_Caps)GetProcAddress(ngx, "NVSDK_NGX_D3D12_GetCapabilityParameters");
    auto Create = (PFN_Create)GetProcAddress(ngx, "NVSDK_NGX_D3D12_CreateFeature");
    auto Eval = (PFN_Eval)GetProcAddress(ngx, "NVSDK_NGX_D3D12_EvaluateFeature");
    if (!Init || !Caps || !Create || !Eval) { puts("the NGX D3D12 entry points are not exported by dxgi.dll"); return 5; }
    NVSDK_NGX_Result r = Init(0x1234, L".", dev, nullptr, NVSDK_NGX_Version_API);
    printf("init %x\n", (unsigned)r); fflush(stdout);
    NVSDK_NGX_Parameter *params = nullptr; r = Caps(&params);
    printf("caps %x %p\n", (unsigned)r, (void *)params); fflush(stdout);
    params->Set(NVSDK_NGX_Parameter_Width, (unsigned)RW); params->Set(NVSDK_NGX_Parameter_Height, (unsigned)RH);
    params->Set(NVSDK_NGX_Parameter_OutWidth, (unsigned)W); params->Set(NVSDK_NGX_Parameter_OutHeight, (unsigned)H);
    params->Set(NVSDK_NGX_Parameter_PerfQualityValue, (int)NVSDK_NGX_PerfQuality_Value_Balanced);
    if (preset_hint) {
        for (const char *name : {NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality,
                                 NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Balanced, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance,
                                 NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraPerformance, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraQuality})
            params->Set(name, (unsigned)preset_hint);
    }
    int flags = NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;
    if (!with_exposure) flags |= NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;
    params->Set(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags, flags);

    CHECK(list->Close());
    CHECK(alloc->Reset()); CHECK(list->Reset(alloc, nullptr));
    NVSDK_NGX_Handle *handle = nullptr;
    r = Create(list, rr ? NVSDK_NGX_Feature_RayReconstruction : NVSDK_NGX_Feature_SuperSampling, params, &handle);
    printf("NGX create: %x\n", (unsigned)r); fflush(stdout);
    CHECK(list->Close());
    { ID3D12CommandList *l[] = {list}; queue->ExecuteCommandLists(1, l); }
    wait_idle();
    if (!handle) return 6;

    using clk = std::chrono::steady_clock; auto start = clk::now(), tick = start; long frames = 0, lastFrames = 0; bool shot = false;
    MSG msg; int index = 0;
    std::vector<float> flat_normals;
    for (;;) {
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) { if (msg.message == WM_QUIT) return 0; TranslateMessage(&msg); DispatchMessageW(&msg); }
        double t = std::chrono::duration<double>(clk::now() - start).count();
        if (t > seconds) break;
        const float jx = halton(index % 16 + 1, 2) - 0.5f, jy = halton(index % 16 + 1, 3) - 0.5f; ++index;
        const float st = (float)(freeze >= 0 ? freeze : t), moving = freeze >= 0 ? 0.f : 1.f;

        // The scene, in the rendered region of the textures (the rest is garbage that must not be read).
        for (UINT y = 0; y < TH; ++y) {
            uint8_t *crow = color.mapped + (size_t)y * color.footprint.Footprint.RowPitch;
            uint16_t *hrow = (uint16_t *)crow;
            float *drow = (float *)(depth.mapped + (size_t)y * depth.footprint.Footprint.RowPitch);
            uint16_t *mrow = (uint16_t *)(motion.mapped + (size_t)y * motion.footprint.Footprint.RowPitch);
            for (UINT x = 0; x < TW; ++x) {
                const bool inside = x >= ox && y >= oy && x < ox + (UINT)DW && y < oy + (UINT)DH;
                if (!inside && fp16) { for (int k = 0; k < 4; ++k) hrow[x * 4 + k] = 0x7b7b; drow[x] = 0.0f; mrow[x * 2] = to_half(100.f); mrow[x * 2 + 1] = to_half(100.f); continue; }
                if (!inside) { crow[x * 4 + 0] = 0x7b; crow[x * 4 + 1] = 0x7b; crow[x * 4 + 2] = 0x7b; crow[x * 4 + 3] = 0x7b; drow[x] = 0.0f; mrow[x * 2] = to_half(100.f); mrow[x * 2 + 1] = to_half(100.f); continue; }
                Pixel p = scene((float)(x - ox) + 0.5f + jx, (float)(y - oy) + 0.5f + jy, (float)DW, (float)DH, st, moving);
                auto q = [](float v) { v = v < 0 ? 0 : v > 1 ? 1 : v; return (uint8_t)(v * 255.f + 0.5f); };
                if (fp16) { hrow[x * 4 + 0] = to_half(p.r); hrow[x * 4 + 1] = to_half(p.g); hrow[x * 4 + 2] = to_half(p.b); hrow[x * 4 + 3] = to_half(1.0f); }
                else { crow[x * 4 + 0] = q(p.r); crow[x * 4 + 1] = q(p.g); crow[x * 4 + 2] = q(p.b); crow[x * 4 + 3] = 255; }
                drow[x] = p.depth; mrow[x * 2] = to_half(p.mvx); mrow[x * 2 + 1] = to_half(p.mvy);
            }
        }
        auto fill = [&](Upload &u, uint8_t value) {
            for (UINT y = 0; y < u.height; ++y) memset(u.mapped + (size_t)y * u.footprint.Footprint.RowPitch, value, (size_t)u.width * u.bytes);
        };
        if (with_exposure) *(float *)exposure_u.mapped = getenv("EXPOSURE_VALUE") ? (float)atof(getenv("EXPOSURE_VALUE")) : 1.0f;
        if (with_masks) { fill(reactive_u, 0); fill(transparency_u, 0); }
        if (rr) {
            fill(diffuse_u, 0xb0); fill(specular_u, 0x30);
            for (UINT y = 0; y < normals_u.height; ++y) {
                uint16_t *row = (uint16_t *)(normals_u.mapped + (size_t)y * normals_u.footprint.Footprint.RowPitch);
                for (UINT x = 0; x < normals_u.width; ++x) { row[x * 4] = to_half(0); row[x * 4 + 1] = to_half(0); row[x * 4 + 2] = to_half(1); row[x * 4 + 3] = to_half(0.5f); }
            }
        }

        CHECK(alloc->Reset()); CHECK(list->Reset(alloc, nullptr));
        record_upload(list, color); record_upload(list, depth); record_upload(list, motion);
        if (with_exposure) record_upload(list, exposure_u);
        if (with_masks) { record_upload(list, reactive_u); record_upload(list, transparency_u); }
        if (rr) { record_upload(list, diffuse_u); record_upload(list, specular_u); record_upload(list, normals_u); }

        params->Set(NVSDK_NGX_Parameter_Color, (ID3D12Resource *)color.texture); params->Set(NVSDK_NGX_Parameter_Output, (ID3D12Resource *)output);
        params->Set(NVSDK_NGX_Parameter_Depth, (ID3D12Resource *)depth.texture); params->Set(NVSDK_NGX_Parameter_MotionVectors, (ID3D12Resource *)motion.texture);
        if (with_exposure) params->Set(NVSDK_NGX_Parameter_ExposureTexture, (ID3D12Resource *)exposure_u.texture);
        if (with_masks) {
            params->Set(NVSDK_NGX_Parameter_DLSS_Input_Bias_Current_Color_Mask, (ID3D12Resource *)reactive_u.texture);
            params->Set(NVSDK_NGX_Parameter_TransparencyMask, (ID3D12Resource *)transparency_u.texture);
        }
        if (rr) {
            params->Set("GBuffer.DiffuseAlbedo", (ID3D12Resource *)diffuse_u.texture);
            params->Set("GBuffer.SpecularAlbedo", (ID3D12Resource *)specular_u.texture);
            params->Set(NVSDK_NGX_Parameter_GBuffer_Normals, (ID3D12Resource *)normals_u.texture);
            params->Set("DLSS.Roughness.Mode", 1u);
            static float world_to_view[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
            static float view_to_clip[16] = {1.2f, 0, 0, 0, 0, 2.1f, 0, 0, 0, 0, 0, -1, 0, 0, 0.1f, 0};
            params->Set("WorldToViewMatrix", (void *)world_to_view);
            params->Set("ViewToClipMatrix", (void *)view_to_clip);
        }
        params->Set(NVSDK_NGX_Parameter_Jitter_Offset_X, jx); params->Set(NVSDK_NGX_Parameter_Jitter_Offset_Y, jy);
        params->Set(NVSDK_NGX_Parameter_Sharpness, 0.0f); params->Set(NVSDK_NGX_Parameter_Reset, frames == 0 ? 1 : 0);
        params->Set(NVSDK_NGX_Parameter_MV_Scale_X, 1.0f); params->Set(NVSDK_NGX_Parameter_MV_Scale_Y, 1.0f);
        if (getenv("PRE_EXPOSURE")) params->Set(NVSDK_NGX_Parameter_DLSS_Pre_Exposure, (float)atof(getenv("PRE_EXPOSURE")));
        if (getenv("EXPOSURE_SCALE")) params->Set(NVSDK_NGX_Parameter_DLSS_Exposure_Scale, (float)atof(getenv("EXPOSURE_SCALE")));
        params->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width, (unsigned)DW); params->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height, (unsigned)DH);
        if (subrect) {
            params->Set(NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_X, ox); params->Set(NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_Y, oy);
            params->Set(NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_X, ox); params->Set(NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_Y, oy);
            params->Set(NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_X, ox); params->Set(NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_Y, oy);
            params->Set(NVSDK_NGX_Parameter_DLSS_Input_Bias_Current_Color_SubrectBase_X, ox); params->Set(NVSDK_NGX_Parameter_DLSS_Input_Bias_Current_Color_SubrectBase_Y, oy);
            params->Set(NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_X, out_x); params->Set(NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_Y, out_y);
            if (rr) { params->Set("DLSS.Input.DiffuseAlbedo.Subrect.Base.X", ox); params->Set("DLSS.Input.DiffuseAlbedo.Subrect.Base.Y", oy); }
        }
        r = Eval(list, handle, params, nullptr);
        if (r != NVSDK_NGX_Result_Success && frames < 3) printf("NGX evaluate: %x\n", (unsigned)r);

        static const bool series = getenv("SERIES") != nullptr;   // SERIES=1: print the mean colour of every 40th frame
        const bool capture = (!shot && t >= capture_at) || (series && (frames < 10 || frames % 40 == 5));
        if (capture) {
            barrier(list, output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
            D3D12_TEXTURE_COPY_LOCATION dst = {}, src = {};
            dst.pResource = readback; dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; dst.PlacedFootprint = out_fp;
            src.pResource = output; src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            barrier(list, output, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
        // The textures go back to where the next upload expects them.
        auto back = [&](Upload &u) { if (u.texture) barrier(list, u.texture, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON); };
        back(color); back(depth); back(motion); back(exposure_u); back(reactive_u); back(transparency_u); back(diffuse_u); back(specular_u); back(normals_u);
        CHECK(list->Close());
        { ID3D12CommandList *l[] = {list}; queue->ExecuteCommandLists(1, l); }
        wait_idle();

        if (capture) {
            uint8_t *data = nullptr; D3D12_RANGE all = {0, (SIZE_T)out_total};
            readback->Map(0, &all, (void **)&data);
            if (series) {
                double sum[3] = {0, 0, 0};
                for (int y = 0; y < H; y += 4) for (int x = 0; x < W; x += 4) {
                    const uint8_t *row = data + (size_t)(y + out_y) * out_fp.Footprint.RowPitch + (size_t)(x + out_x) * (fp16 ? 8 : 4);
                    for (int c = 0; c < 3; ++c) sum[c] += fp16 ? from_half(((const uint16_t *)row)[c]) : row[c] / 255.0;
                }
                const double n = (double)((H + 3) / 4) * ((W + 3) / 4);
                printf("frame %ld mean %.4f %.4f %.4f\n", frames, sum[0] / n, sum[1] / n, sum[2] / n); fflush(stdout);
            }
            if (!shot) write_bmp(out, data + (size_t)out_y * out_fp.Footprint.RowPitch + out_x * (fp16 ? 8 : 4), out_fp.Footprint.RowPitch, W, H, fp16);
            D3D12_RANGE none = {0, 0}; readback->Unmap(0, &none);
            if (!shot) {
                // What the game rendered, for comparison: the colour texture's region.
                write_bmp((std::string(out) + ".in.bmp").c_str(), color.mapped + (size_t)oy * color.footprint.Footprint.RowPitch + ox * (fp16 ? 8 : 4),
                          color.footprint.Footprint.RowPitch, DW, DH, fp16);
                printf("captured %s\n", out); shot = true;
            }
        }
        sc->Present(getenv("NOVSYNC") ? 0 : 1, 0); ++frames;
        double dt = std::chrono::duration<double>(clk::now() - tick).count();
        if (dt >= 2.0) { printf("t=%.1f  %.1f fps\n", t, (frames - lastFrames) / dt); fflush(stdout); lastFrames = frames; tick = clk::now(); }
    }
    // A game releases its features and shuts NGX down before it exits.
    wait_idle();
    if (!getenv("NO_RELEASE")) {
        typedef NVSDK_NGX_Result (*PFN_Release)(NVSDK_NGX_Handle *);
        typedef NVSDK_NGX_Result (*PFN_Shutdown)(ID3D12Device *);
        auto Release = (PFN_Release)GetProcAddress(ngx, "NVSDK_NGX_D3D12_ReleaseFeature");
        auto Shutdown = (PFN_Shutdown)GetProcAddress(ngx, "NVSDK_NGX_D3D12_Shutdown1");
        if (Release) printf("release %x\n", (unsigned)Release(handle));
        Sleep(3000);
        if (Shutdown) printf("shutdown %x\n", (unsigned)Shutdown(dev));
        fflush(stdout);
    }
    return 0;
}
