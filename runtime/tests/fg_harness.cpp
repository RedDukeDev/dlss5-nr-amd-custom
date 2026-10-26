// dlss5nr_fg_harness.exe: NVIDIA's frame generation on this GPU, on real Direct3D 12 textures.
//
// Drives nvngx_dlssg.dll through the NGX host the way the OptiScaler provider does, over the NVAPI stand-in
// (nvapi64.dll), with the command lists tracked and split by d4r's command hooks. It moves a bright disc across a
// background, one step per frame, hands the library the frames as the game would, and writes the frame it
// generates between the last two as a PNG: the disc should be about halfway.
//
//   dlss5nr_fg_harness.exe <dir> <nvngx_dlssg.dll> [width height frames] [--out file.png]
//
// <dir> holds nvapi64.dll (the stand-in), zluda_nvapi.dll (ZLUDA's NVAPI), zluda_real.dll (ZLUDA's nvcuda.dll),
// nvcuda.dll (the gate) and nvngx.dll (the NGX host), and cubin_fatbin_map.txt beside the stand-in.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "d3d12_command_hooks.h"   // d4r's command hooks, from the OptiScaler tree
#include "../common/ngx_cuda.h"

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

using Microsoft::WRL::ComPtr;

namespace {

typedef const char *(*ngxrt_load_t)(const wchar_t *);
typedef NVSDK_NGX_Result (*ngxrt_d3d12_init_t)(unsigned long long, const wchar_t *, ID3D12Device *, NVSDK_NGX_Version,
                                               unsigned long long, unsigned);
typedef NVSDK_NGX_Result (*ngxrt_d3d12_populate_t)(NVSDK_NGX_Parameter **);
typedef NVSDK_NGX_Result (*ngxrt_d3d12_create_t)(ID3D12GraphicsCommandList *, NVSDK_NGX_Feature, NVSDK_NGX_Parameter *,
                                                 NVSDK_NGX_Handle **);
typedef NVSDK_NGX_Result (*ngxrt_d3d12_evaluate_t)(ID3D12GraphicsCommandList *, const NVSDK_NGX_Handle *,
                                                   const NVSDK_NGX_Parameter *, void *);
typedef NVSDK_NGX_Result (*ngxrt_d3d12_release_t)(NVSDK_NGX_Handle *);
typedef int (*fgnv_attach_t)(ID3D12Device *, bool (*)(ID3D12GraphicsCommandList *, d4r::win::commands::BoundaryC, void *),
                             uint64_t (*)(ID3D12GraphicsCommandList *),
                             uint32_t (*)(ID3D12GraphicsCommandList *, ID3D12Resource *, uint32_t));
typedef int32_t (*fgnv_evaluate_t)(ID3D12GraphicsCommandList *, const void *, NVSDK_NGX_Parameter *, void *,
                                   int32_t (*)(ID3D12GraphicsCommandList *, const void *, NVSDK_NGX_Parameter *, void *));
typedef void (*fgnv_scope_t)(int);
typedef void (*fgnv_report_t)();
typedef void (*fgnv_purge_t)();

constexpr NVSDK_NGX_Feature kFrameGeneration = (NVSDK_NGX_Feature)11;

void ngx_log(const char *message, int level, unsigned feature) {
    if (level > 1) return;
    printf("  [ngx %d/%u] %s", level, feature, message);
    const size_t n = strlen(message);
    if (!n || message[n - 1] != '\n') printf("\n");
}

bool ok(HRESULT hr, const char *what) {
    if (SUCCEEDED(hr)) return true;
    printf("[FAIL] %s: 0x%08x\n", what, (unsigned)hr);
    return false;
}

HMODULE load(const std::wstring &path) {
    HMODULE h = LoadLibraryExW(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!h) printf("[FAIL] could not load %ls (error %lu)\n", path.c_str(), GetLastError());
    return h;
}

// ---- a PNG, with stored (uncompressed) deflate blocks ------------------------------------------------------------

uint32_t crc32(const uint8_t *data, size_t size, uint32_t crc = 0) {
    static uint32_t table[256];
    if (!table[1]) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xedb88320u ^ (c >> 1) : c >> 1;
            table[i] = c;
        }
    }
    crc = ~crc;
    for (size_t i = 0; i < size; ++i) crc = table[(crc ^ data[i]) & 0xff] ^ (crc >> 8);
    return ~crc;
}

void put32(std::vector<uint8_t> &v, uint32_t x) {
    for (int s = 24; s >= 0; s -= 8) v.push_back((uint8_t)(x >> s));
}

void chunk(std::vector<uint8_t> &png, const char *type, const std::vector<uint8_t> &body) {
    put32(png, (uint32_t)body.size());
    std::vector<uint8_t> typed(type, type + 4);
    typed.insert(typed.end(), body.begin(), body.end());
    png.insert(png.end(), typed.begin(), typed.end());
    put32(png, crc32(typed.data(), typed.size()));
}

bool write_png(const wchar_t *path, const uint8_t *rgba, uint32_t w, uint32_t h) {
    std::vector<uint8_t> raw;
    raw.reserve((size_t)(w * 3 + 1) * h);
    for (uint32_t y = 0; y < h; ++y) {
        raw.push_back(0);
        for (uint32_t x = 0; x < w; ++x) {
            const uint8_t *p = rgba + ((size_t)y * w + x) * 4;
            raw.push_back(p[0]);
            raw.push_back(p[1]);
            raw.push_back(p[2]);
        }
    }
    std::vector<uint8_t> z = {0x78, 0x01};
    uint32_t a = 1, b = 0;
    for (uint8_t c : raw) { a = (a + c) % 65521; b = (b + a) % 65521; }
    for (size_t at = 0; at < raw.size(); at += 65535) {
        const size_t n = std::min<size_t>(65535, raw.size() - at);
        z.push_back(at + n >= raw.size() ? 1 : 0);
        z.push_back((uint8_t)(n & 0xff));
        z.push_back((uint8_t)(n >> 8));
        z.push_back((uint8_t)(~n & 0xff));
        z.push_back((uint8_t)((~n >> 8) & 0xff));
        z.insert(z.end(), raw.begin() + at, raw.begin() + at + n);
    }
    put32(z, (b << 16) | a);
    std::vector<uint8_t> png = {0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};
    std::vector<uint8_t> ihdr;
    put32(ihdr, w);
    put32(ihdr, h);
    ihdr.push_back(8);
    ihdr.push_back(2);
    ihdr.push_back(0);
    ihdr.push_back(0);
    ihdr.push_back(0);
    chunk(png, "IHDR", ihdr);
    chunk(png, "IDAT", z);
    chunk(png, "IEND", {});
    FILE *f = _wfopen(path, L"wb");
    if (!f) return false;
    fwrite(png.data(), 1, png.size(), f);
    fclose(f);
    return true;
}

// ---- the scene ---------------------------------------------------------------------------------------------------

constexpr float kStep = 24.0f;   // pixels the disc moves per frame

struct Scene {
    uint32_t w, h;
    float disc_x(int frame) const { return 0.25f * w + kStep * frame; }
    float disc_y() const { return 0.5f * h; }

    std::vector<uint8_t> color(int frame) const {
        std::vector<uint8_t> px((size_t)w * h * 4);
        const float cx = disc_x(frame), cy = disc_y(), r = 0.09f * h;
        for (uint32_t y = 0; y < h; ++y)
            for (uint32_t x = 0; x < w; ++x) {
                uint8_t *p = &px[((size_t)y * w + x) * 4];
                const bool grid = (x % 64) < 2 || (y % 64) < 2;
                p[0] = (uint8_t)(40 + 90 * x / w);
                p[1] = (uint8_t)(60 + 80 * y / h);
                p[2] = grid ? 200 : 90;
                p[3] = 255;
                const float d = std::hypot(x - cx, y - cy);
                if (d < r) { p[0] = 250; p[1] = 230; p[2] = 60; }
            }
        return px;
    }
    std::vector<uint16_t> motion(int frame) const {
        // Half floats, per pixel, in pixels: where the surface was in the previous frame, relative to here.
        std::vector<uint16_t> px((size_t)w * h * 2, 0);
        const float cx = disc_x(frame), cy = disc_y(), r = 0.09f * h;
        auto half = [](float f) {
            uint32_t b;
            memcpy(&b, &f, 4);
            const uint32_t s = (b >> 16) & 0x8000, e = (b >> 23) & 0xff, m = b & 0x7fffff;
            if (e <= 112) return (uint16_t)s;
            if (e >= 143) return (uint16_t)(s | 0x7bff);
            return (uint16_t)(s | ((e - 112) << 10) | (m >> 13));
        };
        const uint16_t mv = half(-kStep);
        for (uint32_t y = 0; y < h; ++y)
            for (uint32_t x = 0; x < w; ++x)
                if (std::hypot(x - cx, y - cy) < r) px[((size_t)y * w + x) * 2] = mv;
        return px;
    }
};

// ---- the backbuffer format ---------------------------------------------------------------------------------------

DXGI_FORMAT g_format = DXGI_FORMAT_R8G8B8A8_UNORM;   // FG_HARNESS_FORMAT=bgra8 | rgb10a2 changes it

// The scene is made as RGBA8; what goes to the GPU, and what comes back, is in the backbuffer's format.
std::vector<uint8_t> to_format(std::vector<uint8_t> rgba) {
    if (g_format == DXGI_FORMAT_R8G8B8A8_UNORM) return rgba;
    for (size_t i = 0; i < rgba.size(); i += 4) {
        const uint8_t r = rgba[i], g = rgba[i + 1], b = rgba[i + 2], a = rgba[i + 3];
        if (g_format == DXGI_FORMAT_B8G8R8A8_UNORM) {
            rgba[i] = b;
            rgba[i + 2] = r;
        } else {
            const uint32_t v = ((uint32_t)(r * 1023 / 255)) | ((uint32_t)(g * 1023 / 255) << 10) |
                               ((uint32_t)(b * 1023 / 255) << 20) | ((uint32_t)(a * 3 / 255) << 30);
            memcpy(&rgba[i], &v, 4);
        }
    }
    return rgba;
}

std::vector<uint8_t> from_format(std::vector<uint8_t> bytes) {
    if (g_format == DXGI_FORMAT_R8G8B8A8_UNORM) return bytes;
    for (size_t i = 0; i < bytes.size(); i += 4) {
        if (g_format == DXGI_FORMAT_B8G8R8A8_UNORM) {
            std::swap(bytes[i], bytes[i + 2]);
        } else {
            uint32_t v;
            memcpy(&v, &bytes[i], 4);
            bytes[i] = (uint8_t)(((v & 1023) * 255 + 511) / 1023);
            bytes[i + 1] = (uint8_t)((((v >> 10) & 1023) * 255 + 511) / 1023);
            bytes[i + 2] = (uint8_t)((((v >> 20) & 1023) * 255 + 511) / 1023);
            bytes[i + 3] = 255;
        }
    }
    return bytes;
}

// ---- Direct3D 12 plumbing --------------------------------------------------------------------------------------

struct Gpu {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    HANDLE event = nullptr;
    uint64_t value = 0;

    bool init() {
        if (!ok(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)), "allocator")) return false;
        D3D12_COMMAND_QUEUE_DESC qd{};
        qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        if (!ok(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)), "queue")) return false;
        if (!ok(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&list)), "list")) return false;
        list->Close();
        if (!ok(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "fence")) return false;
        event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        return true;
    }
    bool open() { return ok(allocator->Reset(), "allocator reset") && ok(list->Reset(allocator.Get(), nullptr), "list reset"); }
    bool run() {
        if (!ok(list->Close(), "close")) return false;
        ID3D12CommandList *lists[] = {list.Get()};
        queue->ExecuteCommandLists(1, lists);
        queue->Signal(fence.Get(), ++value);
        if (fence->GetCompletedValue() < value) {
            fence->SetEventOnCompletion(value, event);
            if (WaitForSingleObject(event, 120000) != WAIT_OBJECT_0) { printf("[FAIL] the GPU did not finish\n"); return false; }
        }
        const HRESULT removed = device->GetDeviceRemovedReason();
        if (FAILED(removed)) { printf("[FAIL] device removed: 0x%08x\n", (unsigned)removed); return false; }
        return true;
    }
};

ComPtr<ID3D12Resource> texture(ID3D12Device *device, uint32_t w, uint32_t h, DXGI_FORMAT format, bool shared = true) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = w;
    d.Height = h;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.Format = format;
    d.SampleDesc.Count = 1;
    d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ComPtr<ID3D12Resource> r;
    // Shared, so that HIP can map the memory; a game's textures are not, which is what the session is for.
    if (FAILED(device->CreateCommittedResource(&heap, shared ? D3D12_HEAP_FLAG_SHARED : D3D12_HEAP_FLAG_NONE, &d,
                                               D3D12_RESOURCE_STATE_COMMON, nullptr,
                                               IID_PPV_ARGS(&r))))
        printf("[FAIL] texture %ux%u format %d\n", w, h, (int)format);
    return r;
}

ComPtr<ID3D12Resource> buffer(ID3D12Device *device, UINT64 size, D3D12_HEAP_TYPE type) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = type;
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = size;
    d.Height = d.DepthOrArraySize = d.MipLevels = 1;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ComPtr<ID3D12Resource> r;
    device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d,
                                    type == D3D12_HEAP_TYPE_UPLOAD ? D3D12_RESOURCE_STATE_GENERIC_READ : D3D12_RESOURCE_STATE_COPY_DEST,
                                    nullptr, IID_PPV_ARGS(&r));
    return r;
}

void barrier(ID3D12GraphicsCommandList *list, ID3D12Resource *r, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = r;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = from;
    b.Transition.StateAfter = to;
    list->ResourceBarrier(1, &b);
}

// Pixels into a texture, on a list of its own, finished before this returns.
bool upload(Gpu &gpu, ID3D12Resource *target, const void *pixels, uint32_t row_bytes) {
    const D3D12_RESOURCE_DESC desc = target->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
    UINT rows = 0;
    UINT64 row_size = 0, total = 0;
    gpu.device->GetCopyableFootprints(&desc, 0, 1, 0, &fp, &rows, &row_size, &total);
    auto staging = buffer(gpu.device.Get(), total, D3D12_HEAP_TYPE_UPLOAD);
    uint8_t *mapped = nullptr;
    staging->Map(0, nullptr, (void **)&mapped);
    for (UINT y = 0; y < rows; ++y) memcpy(mapped + (size_t)y * fp.Footprint.RowPitch, (const uint8_t *)pixels + (size_t)y * row_bytes, row_bytes);
    staging->Unmap(0, nullptr);
    if (!gpu.open()) return false;
    D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
    dst.pResource = target;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.pResource = staging.Get();
    src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint = fp;
    gpu.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    return gpu.run();
}

bool read_back(Gpu &gpu, ID3D12Resource *source, std::vector<uint8_t> &rgba) {
    const D3D12_RESOURCE_DESC desc = source->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
    UINT rows = 0;
    UINT64 row_size = 0, total = 0;
    gpu.device->GetCopyableFootprints(&desc, 0, 1, 0, &fp, &rows, &row_size, &total);
    auto staging = buffer(gpu.device.Get(), total, D3D12_HEAP_TYPE_READBACK);
    if (!gpu.open()) return false;
    D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
    dst.pResource = staging.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = fp;
    src.pResource = source;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    gpu.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    if (!gpu.run()) return false;
    uint8_t *mapped = nullptr;
    staging->Map(0, nullptr, (void **)&mapped);
    rgba.resize((size_t)desc.Width * desc.Height * 4);
    for (UINT y = 0; y < rows; ++y) memcpy(&rgba[(size_t)y * desc.Width * 4], mapped + (size_t)y * fp.Footprint.RowPitch, (size_t)desc.Width * 4);
    staging->Unmap(0, nullptr);
    return true;
}

void set_subrect(NVSDK_NGX_Parameter *p, const char *base, uint32_t w, uint32_t h) {
    char name[96];
    snprintf(name, sizeof name, "DLSSG.%sSubrectBaseX", base);
    p->Set(name, 0u);
    snprintf(name, sizeof name, "DLSSG.%sSubrectBaseY", base);
    p->Set(name, 0u);
    snprintf(name, sizeof name, "DLSSG.%sSubrectWidth", base);
    p->Set(name, w);
    snprintf(name, sizeof name, "DLSSG.%sSubrectHeight", base);
    p->Set(name, h);
}

} // namespace

// A crash names where it was: the module and offset of each frame, for an exit that ends in a fail-fast.
LONG WINAPI report_crash(EXCEPTION_POINTERS *info) {
    const DWORD code = info->ExceptionRecord->ExceptionCode;
    if (code == 0x40010006 || code == 0x4001000A || code == 0x406D1388 || (code >> 28) == 4) return EXCEPTION_CONTINUE_SEARCH;   // debug output and thread names
    void *frames[48];
    const USHORT n = RtlCaptureStackBackTrace(0, 48, frames, nullptr);
    printf("CRASH 0x%08lx at %p\n", info->ExceptionRecord->ExceptionCode, info->ExceptionRecord->ExceptionAddress);
    for (USHORT i = 0; i < n; ++i) {
        HMODULE module = nullptr;
        wchar_t name[MAX_PATH] = L"?";
        if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               (LPCWSTR)frames[i], &module))
            GetModuleFileNameW(module, name, MAX_PATH);
        const wchar_t *base = wcsrchr(name, L'\\');
        printf("   %2u %ls+0x%llx\n", i, base ? base + 1 : name, (unsigned long long)((uint8_t *)frames[i] - (uint8_t *)module));
    }
    fflush(stdout);
    return EXCEPTION_CONTINUE_SEARCH;
}

int wmain(int argc, wchar_t **argv) {
    AddVectoredExceptionHandler(1, report_crash);
    if (argc < 3) {
        printf("usage: dlss5nr_fg_harness <dir> <nvngx_dlssg.dll> [width height frames] [--out file.png]\n");
        return 1;
    }
    const std::wstring dir = std::wstring(argv[1]) + L"\\", snippet = argv[2];
    const uint32_t width = argc > 3 ? (uint32_t)_wtoi(argv[3]) : 960;
    const uint32_t height = argc > 4 ? (uint32_t)_wtoi(argv[4]) : 540;
    const int frames = argc > 5 ? _wtoi(argv[5]) : 6;
    std::wstring out_path = dir + L"fg_out.png";
    for (int i = 1; i + 1 < argc; ++i)
        if (!wcscmp(argv[i], L"--out")) out_path = argv[i + 1];

    if (const wchar_t *f = _wgetenv(L"FG_HARNESS_FORMAT")) {
        if (!wcscmp(f, L"bgra8")) g_format = DXGI_FORMAT_B8G8R8A8_UNORM;
        else if (!wcscmp(f, L"rgb10a2")) g_format = DXGI_FORMAT_R10G10B10A2_UNORM;
        printf("backbuffer format %d\n", (int)g_format);
    }
    // FG_HARNESS_DIST=1 runs as the package does: nothing is set here, so the stand-in's own defaults are what is tested.
    if (!_wgetenv(L"FG_HARNESS_DIST")) {
        SetEnvironmentVariableW(L"ZLUDA_CACHE_DIR", (dir + L"ComputeCache").c_str());
        CreateDirectoryW((dir + L"ComputeCache").c_str(), nullptr);
        SetEnvironmentVariableW(L"ZLUDA_NVAPI_GPU_ARCH", L"0x190");
        SetEnvironmentVariableW(L"DLSS5NR_GATE_LOG", (dir + L"gate.log").c_str());
        SetEnvironmentVariableW(L"__NGX_LOG_LEVEL", L"2");
        SetEnvironmentVariableW(L"__NGX_ENABLE_OVERRIDE_LOG_PATH", L"1");
        SetEnvironmentVariableW(L"__NGX_LOG_PATH_OVERRIDE", dir.c_str());
        SetEnvironmentVariableW(L"__NV_SIGNED_LOAD_CHECK", L"none");
    }

    printf("frame generation harness: %ux%u, %d frames\n", width, height, frames);

    // The order matters: the stand-in first, under the name the library asks for.
    HMODULE nvapi = load(dir + L"nvapi64.dll");
    if (!nvapi || !load(dir + L"zluda_real.dll") || !load(dir + L"nvcuda.dll")) return 1;
    auto attach = (fgnv_attach_t)GetProcAddress(nvapi, "fgnv_attach");
    auto session_evaluate = (fgnv_evaluate_t)GetProcAddress(nvapi, "fgnv_evaluate");
    const bool use_session = _wgetenv(L"FG_HARNESS_SESSION") != nullptr;   // the game's textures, as they come
    auto scope = (fgnv_scope_t)GetProcAddress(nvapi, "fgnv_scope");
    auto report = (fgnv_report_t)GetProcAddress(nvapi, "fgnv_report");
    auto purge = (fgnv_purge_t)GetProcAddress(nvapi, "fgnv_purge");
    if (!attach || !scope) { printf("[FAIL] nvapi64.dll is not the frame generation stand-in\n"); return 1; }

    Gpu gpu;
    ComPtr<IDXGIFactory4> factory;
    if (!ok(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)), "factory")) return 1;
    ComPtr<IDXGIAdapter1> adapter;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) == S_OK; ++i) {
        DXGI_ADAPTER_DESC1 d;
        adapter->GetDesc1(&d);
        if (!(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) { printf("adapter: %ls\n", d.Description); break; }
        adapter.Reset();
    }
    if (_wgetenv(L"FG_DEBUG_LAYER")) {
        ComPtr<ID3D12Debug> debug;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) { debug->EnableDebugLayer(); printf("debug layer on\n"); }
    }
    if (!ok(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&gpu.device)), "D3D12CreateDevice")) return 1;
    if (!gpu.init()) return 1;

    // The hooks that split command lists, installed as the host would, before any list is made that matters.
    try {
        d4r::win::commands::install(gpu.device.Get());
    } catch (const std::exception &e) {
        printf("[FAIL] command hooks: %s\n", e.what());
        return 1;
    }
    if (!attach(gpu.device.Get(), &d4r::win::commands::record_boundary_c, &d4r::win::commands::command_mark,
                &d4r::win::commands::resource_state_c)) {
        printf("[FAIL] fgnv_attach\n");
        return 1;
    }
    // A list made now is tracked; the one made before the hooks was not.
    gpu.list.Reset();
    gpu.allocator.Reset();
    {
        D3D12_COMMAND_QUEUE_DESC qd{};
        (void)qd;
        if (!ok(gpu.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&gpu.allocator)), "allocator")) return 1;
        if (!ok(gpu.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, gpu.allocator.Get(), nullptr, IID_PPV_ARGS(&gpu.list)), "list")) return 1;
        gpu.list->Close();
    }

    HMODULE host = load(dir + L"nvngx.dll");
    if (!host) return 1;
    auto ngx_load = (ngxrt_load_t)GetProcAddress(host, "ngxrt_load");
    auto ngx_init = (ngxrt_d3d12_init_t)GetProcAddress(host, "ngxrt_d3d12_init");
    auto ngx_populate = (ngxrt_d3d12_populate_t)GetProcAddress(host, "ngxrt_d3d12_populate");
    auto ngx_create = (ngxrt_d3d12_create_t)GetProcAddress(host, "ngxrt_d3d12_create");
    auto ngx_evaluate = (ngxrt_d3d12_evaluate_t)GetProcAddress(host, "ngxrt_d3d12_evaluate");
    auto ngx_release = (ngxrt_d3d12_release_t)GetProcAddress(host, "ngxrt_d3d12_release");
    if (!ngx_load || !ngx_init || !ngx_populate || !ngx_create || !ngx_evaluate || !ngx_release) {
        printf("[FAIL] nvngx.dll is not our NGX host\n");
        return 1;
    }
    if (const char *problem = ngx_load(snippet.c_str())) { printf("[FAIL] loading the library: %s\n", problem); return 1; }

    scope(1);
    const NVSDK_NGX_Result init = ngx_init(0x24480451, dir.c_str(), gpu.device.Get(), NVSDK_NGX_Version_API,
                                           (unsigned long long)(void *)&ngx_log, 2);
    scope(0);
    if (!NVSDK_NGX_SUCCEED(init)) { printf("[FAIL] Init -> 0x%08x\n", init); return 1; }
    NVSDK_NGX_Parameter *p = nullptr;
    scope(1);
    ngx_populate(&p);
    scope(0);
    if (!p) return 1;

    p->Set("Width", width);
    p->Set("Height", height);
    p->Set("DLSSG.Width", width);
    p->Set("DLSSG.Height", height);
    // FG_HARNESS_GAMECREATE=1: the creation parameters as OptiScaler gives them in a game (no internal size, the extra keys)
    const bool game_create = _wgetenv(L"FG_HARNESS_GAMECREATE") != nullptr;
    p->Set("DLSSG.InternalWidth", game_create ? 0u : width);
    p->Set("DLSSG.InternalHeight", game_create ? 0u : height);
    if (game_create) {
        p->Set("DLSSG.EnableInterp", 1u);
        p->Set("DLSSG.DynamicResolution", 0u);
        p->Set("DLSSG.UseReflexMatrices", 0u);
        p->Set("DLSSG.ColorBuffersHDR", 1u);
        p->Set("DLSSG.DepthInverted", 1u);
        p->Set("DLSSG.MvecJittered", 0u);
        p->Set("DLSSG.MvecDilated", 0u);
        p->Set("DLSSG.OrthoProjection", 0u);
    }
    p->Set("DLSSG.BackbufferFormat", (unsigned)g_format);
    p->Set("DLSSG.MultiFrameCountMax", 1u);
    p->Set("DLSSG.MultiFrameCount", 1u);
    p->Set("CreationNodeMask", 1u);
    p->Set("VisibilityNodeMask", 1u);

    NVSDK_NGX_Handle *feature = nullptr;
    auto t0 = std::chrono::steady_clock::now();
    if (!gpu.open()) return 1;
    scope(1);
    NVSDK_NGX_Result r = ngx_create(gpu.list.Get(), kFrameGeneration, p, &feature);
    scope(0);
    if (!NVSDK_NGX_SUCCEED(r)) { printf("[FAIL] CreateFeature -> 0x%08x\n", r); return 1; }
    if (!gpu.run()) return 1;
    printf("[ OK ] feature created and its work executed in %.1f s\n",
           std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    if (report) report();

    const bool shared = !use_session;
    // As a game gives them: depth and motion vectors at the render size, depth inverted, the interface only in the
    // frame that has it (FG_HARNESS_RENDER=WxH, _INVERTED, _HDR, _UI)
    uint32_t render_w = width, render_h = height;
    if (const wchar_t *r = _wgetenv(L"FG_HARNESS_RENDER")) swscanf(r, L"%ux%u", &render_w, &render_h);
    const bool inverted = _wgetenv(L"FG_HARNESS_INVERTED") != nullptr, hdr = _wgetenv(L"FG_HARNESS_HDR") != nullptr,
               with_ui = _wgetenv(L"FG_HARNESS_UI") != nullptr;
    printf("render %ux%u, inverted %d, hdr %d, ui %d%c", render_w, render_h, inverted, hdr, with_ui, 10);
    // FG_HARNESS_CAPTURE=<dir>: the inputs the stand-in captured in a game (cap_<frame>_<n>.bin), played back in place of the
    // scene. The textures are told by their formats: depth is a float, motion two halves, the others colour in the order
    // backbuffer, hudless, interface.
    struct Capture { uint32_t w = 0, h = 0, format = 0, row = 0; std::vector<uint8_t> data; };
    std::vector<Capture> depths, motions;
    std::vector<std::vector<Capture>> colours;
    uint32_t depth_w = render_w, depth_h = render_h, motion_w = render_w, motion_h = render_h;
    if (const wchar_t *c = _wgetenv(L"FG_HARNESS_CAPTURE")) {
        for (int f = 0; f < 16; ++f) {
            std::vector<Capture> found;
            for (int n = 0; n < 8; ++n) {
                wchar_t name[MAX_PATH];
                swprintf(name, MAX_PATH, L"%s/cap_%d_%d.bin", c, f, n);
                FILE *file = _wfopen(name, L"rb");
                if (!file) break;
                Capture cap;
                uint32_t header[4];
                if (fread(header, 4, 4, file) == 4) {
                    cap.w = header[0]; cap.h = header[1]; cap.format = header[2]; cap.row = header[3];
                    cap.data.resize((size_t)cap.row * cap.h);
                    fread(cap.data.data(), 1, cap.data.size(), file);
                    found.push_back(std::move(cap));
                }
                fclose(file);
            }
            if (found.empty()) break;
            std::vector<Capture> frame_colours;
            for (auto &cap : found) {
                if (cap.format == DXGI_FORMAT_R32_FLOAT) { depth_w = cap.w; depth_h = cap.h; depths.push_back(std::move(cap)); }
                else if (cap.format == DXGI_FORMAT_R16G16_FLOAT) { motion_w = cap.w; motion_h = cap.h; motions.push_back(std::move(cap)); }
                else frame_colours.push_back(std::move(cap));
            }
            colours.push_back(std::move(frame_colours));
        }
        printf("capture: %zu frames, depth %ux%u, motion %ux%u, %zu colour textures a frame%c", colours.size(), depth_w, depth_h,
               motion_w, motion_h, colours.empty() ? (size_t)0 : colours[0].size(), 10);
    }
    const bool captured = !colours.empty() && !depths.empty() && !motions.empty();
    auto color = texture(gpu.device.Get(), width, height, g_format, shared);
    auto hudless = texture(gpu.device.Get(), width, height, g_format, shared);
    auto depth = texture(gpu.device.Get(), depth_w, depth_h, DXGI_FORMAT_R32_FLOAT, shared);
    auto motion = texture(gpu.device.Get(), motion_w, motion_h, DXGI_FORMAT_R16G16_FLOAT, shared);
    auto interface_texture = texture(gpu.device.Get(), width, height, g_format, shared);
    auto output = texture(gpu.device.Get(), width, height, g_format, shared);
    if (use_session) printf("session: the library is handed copies of the game's textures (%d format)\n", (int)g_format);
    if (!color || !hudless || !depth || !motion || !output) return 1;

    Scene scene{width, height};
    Scene motion_scene{render_w, render_h};
    std::vector<float> depth_pixels((size_t)render_w * render_h, inverted ? 0.02f : 0.5f);
    if (!captured) upload(gpu, depth.Get(), depth_pixels.data(), render_w * 4);

    float identity[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

    // FG_HARNESS_PARAMS=<file>: every parameter as OptiScaler gave it to the library in a game (its fg_params.txt), set
    // on top of the harness's own: the same values, to see whether any of them is what a game does differently
    struct Applied { std::string key; bool matrix; double value; float m[16]; };
    std::vector<Applied> applied;
    if (const wchar_t *params_file = _wgetenv(L"FG_HARNESS_PARAMS")) {
        FILE *file = _wfopen(params_file, L"r");
        char line[1024];
        while (file && fgets(line, sizeof line, file)) {
            char key[128], kind[16];
            if (sscanf(line, "%127s %15s", key, kind) != 2) continue;
            Applied a{key, false, 0, {}};
            const char *rest = strstr(line, kind) + strlen(kind);
            if (!strcmp(kind, "number")) { a.value = atof(rest); applied.push_back(a); }
            else if (!strcmp(kind, "matrix")) {
                a.matrix = true;
                char *end = nullptr;
                const char *at = rest;
                for (int i = 0; i < 16; ++i) { a.m[i] = (float)strtod(at, &end); at = end; }
                applied.push_back(a);
            }
        }
        if (file) fclose(file);
        printf("parameters from the file: %zu%c", applied.size(), 10);
    }
    double total = 0;
    int timed = 0;
    for (int frame = 0; frame < frames; ++frame) {
        auto pixels = to_format(scene.color(frame));
        auto vectors = motion_scene.motion(frame);
        auto with_hud = pixels;
        if (with_ui)   // a bar of interface in the frame that is presented, not in the one without it
            for (uint32_t y = 40; y < 120 && y < height; ++y)
                for (uint32_t x = 40; x < 340 && x < width; ++x) std::memset(&with_hud[((size_t)y * width + x) * 4], 0xF0, 4);
        if (captured) {
            const size_t fi = (size_t)frame % colours.size();
            const auto &set = colours[fi];
            if (set.size() < 2 || !upload(gpu, color.Get(), set[0].data.data(), set[0].row) ||
                !upload(gpu, hudless.Get(), set[1].data.data(), set[1].row) ||
                !upload(gpu, depth.Get(), depths[fi % depths.size()].data.data(), depths[0].row) ||
                !upload(gpu, motion.Get(), motions[fi % motions.size()].data.data(), motions[0].row) ||
                (set.size() > 2 && !upload(gpu, interface_texture.Get(), set[2].data.data(), set[2].row)))
                return 1;
        } else if (!upload(gpu, color.Get(), with_hud.data(), width * 4) || !upload(gpu, hudless.Get(), pixels.data(), width * 4) ||
                   !upload(gpu, motion.Get(), vectors.data(), render_w * 4))
            return 1;

        p->Set("DLSSG.Backbuffer", color.Get());
        p->Set("DLSSG.HUDLess", hudless.Get());
        p->Set("DLSSG.Depth", depth.Get());
        p->Set("DLSSG.MVecs", motion.Get());
        p->Set("DLSSG.OutputInterpolated", output.Get());
        for (const char *base : {"Backbuffer", "InputBackbuffer", "HUDLess", "OutputInterpolated"})
            set_subrect(p, base, width, height);
        for (const char *base : {"Depth", "MVecs"})
            set_subrect(p, base, captured ? motion_w : render_w, captured ? motion_h : render_h);
        if (captured && colours[0].size() > 2) {
            p->Set("DLSSG.UI", interface_texture.Get());
            set_subrect(p, "UI", width, height);
        }
        p->Set("DLSSG.Reset", frame == 0 ? 1 : 0);
        p->Set("DLSSG.MultiFrameCount", 1u);
        p->Set("DLSSG.MultiFrameIndex", 1u);
        p->Set("DLSSG.MvecScaleX", 1.0f);
        p->Set("DLSSG.MvecScaleY", 1.0f);
        p->Set("DLSSG.MvecJittered", 0);
        p->Set("DLSSG.MvecDilated", 0);
        p->Set("DLSSG.DepthInverted", inverted ? 1 : 0);
        p->Set("DLSSG.ColorBuffersHDR", hdr ? 1 : 0);
        p->Set("DLSSG.CameraMotionIncluded", 1);
        p->Set("DLSSG.OrthoProjection", 0);
        // FG_HARNESS_GAMECAM=1: what the game's camera is, from the Streamline log of a session in Hogwarts Legacy
        const bool gamecam = _wgetenv(L"FG_HARNESS_GAMECAM") != nullptr;
        static float game_projection[16], game_inverse[16];
        if (gamecam) {
            const float near_ = 10.0f, far_ = 1.0e6f, f = 1.0f / tanf(0.7766f * 0.5f), aspect = (float)width / (float)height;
            const float a = near_ / (near_ - far_), b = near_ * far_ / (far_ - near_);
            const float proj[16] = {f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, a, 1, 0, 0, b, 0};
            std::memcpy(game_projection, proj, sizeof proj);
            // the inverse of this projection, written out
            const float inv[16] = {aspect / f, 0, 0, 0, 0, 1 / f, 0, 0, 0, 0, 0, 1 / b, 0, 0, 1, -a / b};
            std::memcpy(game_inverse, inv, sizeof inv);
        }
        p->Set("DLSSG.JitterOffsetX", gamecam ? 0.1875f : 0.0f);
        p->Set("DLSSG.JitterOffsetY", gamecam ? -0.0185f : 0.0f);
        p->Set("DLSSG.CameraNear", gamecam ? 0.01f : 0.1f);
        p->Set("DLSSG.CameraFar", gamecam ? 75000.0f : 1000.0f);
        p->Set("DLSSG.CameraFOV", gamecam ? 1.2576f : 1.0f);
        p->Set("DLSSG.CameraAspectRatio", (float)width / (float)height);
        for (const char *m : {"DLSSG.ClipToLensClip", "DLSSG.ClipToPrevClip", "DLSSG.PrevClipToClip"})
            p->Set(m, (void *)identity);
        p->Set("DLSSG.CameraViewToClip", (void *)(gamecam ? game_projection : identity));
        p->Set("DLSSG.ClipToCameraView", (void *)(gamecam ? game_inverse : identity));
        p->Set("DLSSG.CameraPosX", gamecam ? 339760.0f : 0.0f);
        p->Set("DLSSG.CameraPosY", gamecam ? -248890.0f : 0.0f);
        p->Set("DLSSG.CameraPosZ", gamecam ? -88229.0f : 0.0f);
        p->Set("DLSSG.CameraUpX", 0.0f);
        p->Set("DLSSG.CameraUpY", 1.0f);
        p->Set("DLSSG.CameraUpZ", 0.0f);
        p->Set("DLSSG.CameraRightX", 1.0f);
        p->Set("DLSSG.CameraRightY", 0.0f);
        p->Set("DLSSG.CameraRightZ", 0.0f);
        p->Set("DLSSG.CameraFwdX", 0.0f);
        p->Set("DLSSG.CameraFwdY", 0.0f);
        p->Set("DLSSG.CameraFwdZ", 1.0f);

        for (auto &a : applied) {
            if (a.matrix) p->Set(a.key.c_str(), (void *)a.m);
            else if (a.key == "DLSSG.Reset" && frame == 0) p->Set(a.key.c_str(), 1u);
            else if (a.value == (double)(unsigned)a.value && a.value >= 0) p->Set(a.key.c_str(), (unsigned)a.value);
            else p->Set(a.key.c_str(), (float)a.value);
        }
        // FG_HARNESS_SLEEP=<ms>: frames as slow as a game's, which the library may notice
        if (const wchar_t *sleep_ms = _wgetenv(L"FG_HARNESS_SLEEP")) Sleep((DWORD)_wtoi(sleep_ms));
        if (!gpu.open()) return 1;
        const auto start = std::chrono::steady_clock::now();
        if (use_session) {
            // As a game leaves them: what the library reads, readable; what it writes, writable.
            for (ID3D12Resource *in : {color.Get(), hudless.Get(), depth.Get(), motion.Get()})
                barrier(gpu.list.Get(), in, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            barrier(gpu.list.Get(), output.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            r = session_evaluate(gpu.list.Get(), feature, p, nullptr,
                                 (int32_t(*)(ID3D12GraphicsCommandList *, const void *, NVSDK_NGX_Parameter *, void *))ngx_evaluate);
            for (ID3D12Resource *in : {color.Get(), hudless.Get(), depth.Get(), motion.Get()})
                barrier(gpu.list.Get(), in, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COMMON);
            barrier(gpu.list.Get(), output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
        } else {
            scope(1);
            r = ngx_evaluate(gpu.list.Get(), feature, p, nullptr);
            scope(0);
        }
        const double record = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        if (!NVSDK_NGX_SUCCEED(r)) { printf("[FAIL] EvaluateFeature, frame %d -> 0x%08x\n", frame, r); return 1; }
        if (!gpu.run()) return 1;
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        printf("frame %2d: recorded in %.2f ms, with the list executed %.2f ms\n", frame, record, ms);
        {
            std::vector<uint8_t> shot;
            if (read_back(gpu, output.Get(), shot)) {
                shot = from_format(std::move(shot));
                double mean[3] = {0, 0, 0};
                for (size_t i = 0; i < (size_t)width * height; ++i)
                    for (int c = 0; c < 3; ++c) mean[c] += shot[i * 4 + c];
                const double n = (double)width * height;
                printf("          output mean rgb %.1f %.1f %.1f\n", mean[0] / n, mean[1] / n, mean[2] / n);
                wchar_t name[MAX_PATH];
                swprintf(name, MAX_PATH, L"%sfg_out_%d.png", dir.c_str(), frame);
                write_png(name, shot.data(), width, height);
            }
        }
        fflush(stdout);
        if (frame >= 2) { total += ms; ++timed; }
    }
    if (timed) printf("average from frame 2: %.2f ms per generated frame\n", total / timed);
    if (report) report();

    std::vector<uint8_t> rgba;
    if (read_back(gpu, output.Get(), rgba)) {
        rgba = from_format(std::move(rgba));
        // Where is the disc in what was generated? The bright yellow pixels.
        double sx = 0, n = 0;
        for (uint32_t y = 0; y < height; ++y)
            for (uint32_t x = 0; x < width; ++x) {
                const uint8_t *px = &rgba[((size_t)y * width + x) * 4];
                if (px[0] > 220 && px[1] > 200 && px[2] < 110) { sx += x; ++n; }
            }
        const int last = frames - 1;
        printf("disc: last real frames at x=%.0f and x=%.0f; generated frame at x=%.0f (%0.f pixels)\n",
               scene.disc_x(last - 1), scene.disc_x(last), n ? sx / n : -1.0, n);
        if (write_png(out_path.c_str(), rgba.data(), width, height)) printf("wrote %ls\n", out_path.c_str());
    }
    if (_wgetenv(L"FG_HARNESS_SKIP_RELEASE")) { printf("skipping the release\n"); fflush(stdout); return 0; }
    scope(1);
    ngx_release(feature);
    scope(0);
    if (purge && !_wgetenv(L"FG_HARNESS_NO_PURGE")) purge();
    printf("released\n");
    fflush(stdout);
    if (report) report();
    // ZLUDA and HIP do not survive being unloaded with the process: the plugin ends its process the same way.
    fflush(stdout);
    TerminateProcess(GetCurrentProcess(), 0);
    return 0;
}
