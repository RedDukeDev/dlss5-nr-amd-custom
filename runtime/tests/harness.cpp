// dlss5nr_harness.exe: the runtime driven the way a host drives it, without a
// game.
//
// A still image stands in for the game's frame: every frame it goes through
// dlss5nr_process in a command list of the harness's own, the list is
// submitted and dlss5nr_present is called, exactly as OptiScaler does at the
// end of a frame. Depth is constant (--depth, zero by default, as the image
// tools leave it); motion vectors are zero, or a constant
// value (--motion) that only exercises the tracking path: the image does
// not move, so with motion the composed result is expected to smear.
// --scroll moves the image itself by that many pixels every frame, wrapping
// around, with motion vectors to match: the output then has to be the still
// image's output moved by the total, or the result lags behind the scene.
// --object moves a 400 x 400 piece of the image across the rest, which stays
// still, with its own depth and motion vectors: what it uncovers is what the
// network never saw. --orbit is the other way round, a camera turning around a
// character: the piece stays where it is, with no motion of its own, and the
// background behind it scrolls by that many pixels a frame, farther away, with
// motion vectors to match. --bare hands over the image alone, without depth or
// motion vectors, as a host that only has the finished frame does. --churn
// changes the settings every 90 frames, for watching what the runtime holds on
// to over a long run: 1 the style, 2 the network's size between two, 4 between
// three, 8 a different one every ten frames (add them up).
//
// With the image already display encoded (an ordinary PNG), the network at
// full resolution and the answer composed as it is, the output is the
// network's own image: it can be compared with what the image enhancer
// produces for the same picture. --ratio composes it the way a game gets it
// instead, and --hdr hands the image over as scene-referred linear light, so
// the runtime's white point, knee and encoding are exercised.
//
//   dlss5nr_harness --image in.png --out out.png [--snippet nvngx_dlssnr.dll]
//                   [--runtime-dir dir] [--data-dir dir] [--frames 60]
//                   [--scale 1.0] [--motion dx,dy] [--depth 0.0] [--evaluations 3]
//                   [--linear] [--hdr] [--ratio] [--scroll dx,dy]
//                   [--object dx,dy] [--orbit dx,dy] [--show-tracking] [--wait] [--bare]
//                   [--churn n]
//                   [--skin v] [--local-tone v] [--local-structure v]
//                   [--intensity v] [--style n]

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wincodec.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "dlss5nr.h"

#pragma comment(lib, "windowscodecs.lib")
#pragma comment(lib, "ole32.lib")

namespace {

template <typename T>
void release(T *&o) {
    if (o) o->Release();
    o = nullptr;
}

bool fail(const char *what) {
    fprintf(stderr, "harness: %s\n", what);
    return false;
}

// RGBA8, straight from the file.
bool load_png(const std::wstring &path, std::vector<uint8_t> &pixels, UINT &width, UINT &height) {
    IWICImagingFactory *wic = nullptr;
    if (FAILED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&wic))))
        return fail("WIC is not available");
    IWICBitmapDecoder *decoder = nullptr;
    IWICBitmapFrameDecode *frame = nullptr;
    IWICFormatConverter *converter = nullptr;
    bool ok = SUCCEEDED(wic->CreateDecoderFromFilename(path.c_str(), nullptr, GENERIC_READ,
                                                       WICDecodeMetadataCacheOnDemand, &decoder)) &&
              SUCCEEDED(decoder->GetFrame(0, &frame)) && SUCCEEDED(wic->CreateFormatConverter(&converter)) &&
              SUCCEEDED(converter->Initialize(frame, GUID_WICPixelFormat32bppRGBA, WICBitmapDitherTypeNone,
                                              nullptr, 0.0, WICBitmapPaletteTypeCustom)) &&
              SUCCEEDED(converter->GetSize(&width, &height));
    if (ok) {
        pixels.resize((size_t)width * height * 4);
        ok = SUCCEEDED(converter->CopyPixels(nullptr, width * 4, (UINT)pixels.size(), pixels.data()));
    }
    release(converter);
    release(frame);
    release(decoder);
    release(wic);
    return ok || fail("could not read the image");
}

bool save_png(const std::wstring &path, const std::vector<uint8_t> &pixels, UINT width, UINT height) {
    IWICImagingFactory *wic = nullptr;
    if (FAILED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&wic))))
        return fail("WIC is not available");
    IWICStream *stream = nullptr;
    IWICBitmapEncoder *encoder = nullptr;
    IWICBitmapFrameEncode *frame = nullptr;
    // The encoder may answer SetPixelFormat with a format of its own choosing
    // -- 32bppBGRA, for PNG -- and then reads the bytes that way.
    WICPixelFormatGUID format = GUID_WICPixelFormat32bppRGBA;
    std::vector<uint8_t> bytes = pixels;
    bool ok = SUCCEEDED(wic->CreateStream(&stream)) &&
              SUCCEEDED(stream->InitializeFromFilename(path.c_str(), GENERIC_WRITE)) &&
              SUCCEEDED(wic->CreateEncoder(GUID_ContainerFormatPng, nullptr, &encoder)) &&
              SUCCEEDED(encoder->Initialize(stream, WICBitmapEncoderNoCache)) &&
              SUCCEEDED(encoder->CreateNewFrame(&frame, nullptr)) && SUCCEEDED(frame->Initialize(nullptr)) &&
              SUCCEEDED(frame->SetSize(width, height)) && SUCCEEDED(frame->SetPixelFormat(&format));
    if (ok && IsEqualGUID(format, GUID_WICPixelFormat32bppBGRA))
        for (size_t i = 0; i + 3 < bytes.size(); i += 4) std::swap(bytes[i], bytes[i + 2]);
    else if (ok && !IsEqualGUID(format, GUID_WICPixelFormat32bppRGBA))
        ok = fail("the PNG encoder wants a pixel format this harness does not write");
    ok = ok && SUCCEEDED(frame->WritePixels(height, width * 4, (UINT)bytes.size(), bytes.data())) &&
         SUCCEEDED(frame->Commit()) && SUCCEEDED(encoder->Commit());
    release(frame);
    release(encoder);
    release(stream);
    release(wic);
    return ok || fail("could not write the image");
}

struct Gpu {
    ID3D12Device *device = nullptr;
    ID3D12CommandQueue *queue = nullptr;
    ID3D12CommandAllocator *allocator = nullptr;
    ID3D12GraphicsCommandList *cmd = nullptr;
    ID3D12Fence *fence = nullptr;
    HANDLE event = nullptr;
    UINT64 value = 0;

    bool init() {
        IDXGIFactory4 *factory = nullptr;
        if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)))) return fail("no DXGI");
        IDXGIAdapter1 *adapter = nullptr;
        for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_ADAPTER_DESC1 desc{};
            adapter->GetDesc1(&desc);
            const bool hardware = !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE);
            if (hardware && SUCCEEDED(D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)))) {
                wprintf(L"adapter: %ls\n", desc.Description);
                release(adapter);
                break;
            }
            release(adapter);
        }
        release(factory);
        if (!device) return fail("no Direct3D 12 device");
        D3D12_COMMAND_QUEUE_DESC q{};
        q.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        if (FAILED(device->CreateCommandQueue(&q, IID_PPV_ARGS(&queue))) ||
            FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))) ||
            FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, nullptr, IID_PPV_ARGS(&cmd))) ||
            FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))))
            return fail("could not create the command objects");
        cmd->Close();
        event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        return true;
    }

    void begin() {
        allocator->Reset();
        cmd->Reset(allocator, nullptr);
    }

    void submit_and_wait() {
        cmd->Close();
        ID3D12CommandList *lists[] = {cmd};
        queue->ExecuteCommandLists(1, lists);
        queue->Signal(fence, ++value);
        if (fence->GetCompletedValue() < value) {
            fence->SetEventOnCompletion(value, event);
            WaitForSingleObject(event, INFINITE);
        }
    }

    ID3D12Resource *texture(UINT w, UINT h, DXGI_FORMAT format, bool uav, D3D12_RESOURCE_STATES state) {
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
        d.Flags = uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
        ID3D12Resource *r = nullptr;
        device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, state, nullptr, IID_PPV_ARGS(&r));
        return r;
    }

    ID3D12Resource *buffer(UINT64 bytes, D3D12_HEAP_TYPE type, D3D12_RESOURCE_STATES state) {
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = type;
        D3D12_RESOURCE_DESC d{};
        d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        d.Width = bytes;
        d.Height = 1;
        d.DepthOrArraySize = 1;
        d.MipLevels = 1;
        d.SampleDesc.Count = 1;
        d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ID3D12Resource *r = nullptr;
        device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, state, nullptr, IID_PPV_ARGS(&r));
        return r;
    }

    // Uploads rows of `bpp` bytes into a texture in COPY_DEST.
    void upload(ID3D12Resource *target, const void *data, UINT w, UINT h, UINT bpp, ID3D12Resource *staging) {
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
        const D3D12_RESOURCE_DESC d = target->GetDesc();
        device->GetCopyableFootprints(&d, 0, 1, 0, &fp, nullptr, nullptr, nullptr);
        uint8_t *mapped = nullptr;
        staging->Map(0, nullptr, (void **)&mapped);
        for (UINT y = 0; y < h; ++y)
            memcpy(mapped + fp.Offset + (size_t)y * fp.Footprint.RowPitch, (const uint8_t *)data + (size_t)y * w * bpp,
                   (size_t)w * bpp);
        staging->Unmap(0, nullptr);
        D3D12_TEXTURE_COPY_LOCATION to{}, from{};
        to.pResource = target;
        to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        from.pResource = staging;
        from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        from.PlacedFootprint = fp;
        cmd->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    }
};

void barrier(ID3D12GraphicsCommandList *cmd, ID3D12Resource *r, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b) {
    D3D12_RESOURCE_BARRIER x{};
    x.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    x.Transition.pResource = r;
    x.Transition.StateBefore = a;
    x.Transition.StateAfter = b;
    x.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmd->ResourceBarrier(1, &x);
}

void log_sink(void *, int level, const char *message) {
    static const char *const kLevel[] = {"error", "warning", "info", "debug"};
    if (level <= 2) printf("  [runtime %s] %s\n", kLevel[level < 0 ? 0 : level > 3 ? 3 : level], message);
}

float from_half(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    const int e = (h >> 10) & 31;
    const uint32_t m = h & 0x3FF;
    uint32_t x;
    if (e == 0) {
        const float v = m / 16777216.0f;   // m * 2^-24
        return (h & 0x8000) ? -v : v;
    }
    if (e == 31) x = sign | 0x7F800000 | (m << 13);
    else x = sign | ((uint32_t)(e - 15 + 127) << 23) | (m << 13);
    float f;
    memcpy(&f, &x, 4);
    return f;
}

float srgb_to_linear(float c) { return c <= 0.04045f ? c / 12.92f : powf((c + 0.055f) / 1.055f, 2.4f); }
float linear_to_srgb(float c) {
    c = c < 0.0f ? 0.0f : c > 1.0f ? 1.0f : c;
    return c <= 0.0031308f ? c * 12.92f : 1.055f * powf(c, 1.0f / 2.4f) - 0.055f;
}

uint16_t to_half(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000;
    int e = (int)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t m = x & 0x7FFFFF;
    if (e <= 0) return (uint16_t)sign;
    if (e >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | (e << 10) | ((m + 0x1000) >> 13));
}

} // namespace

int wmain(int argc, wchar_t **argv) {
    std::wstring image = L"in.png", out = L"out.png", snippet, runtime_dir, data_dir;
    int frames = 60, evaluations = 3;
    float scale = 1.0f, motion_x = 0.0f, motion_y = 0.0f, depth_value = 0.0f;
    int scroll_x = 0, scroll_y = 0, object_x = 0, object_y = 0, orbit_x = 0, orbit_y = 0;
    int churn = 0;
    // The network's own controls, when given; the runtime's defaults otherwise.
    float skin = -2.0f, local_tone = -2.0f, local_structure = -2.0f, intensity = -2.0f;
    int style = -1;
    bool linear = false, hdr = false, ratio = false, show_tracking = false, wait = false, bare = false;
    for (int i = 1; i < argc; ++i) {
        const std::wstring flag = argv[i];
        linear |= flag == L"--linear" || flag == L"--hdr";
        hdr |= flag == L"--hdr";
        ratio |= flag == L"--ratio";
        show_tracking |= flag == L"--show-tracking";
        wait |= flag == L"--wait";
        bare |= flag == L"--bare";
    }
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::wstring key = argv[i], value = argv[i + 1];
        if (key == L"--linear" || key == L"--hdr" || key == L"--ratio" || key == L"--show-tracking" ||
            key == L"--wait" || key == L"--bare") {
            --i;
            continue;
        }
        if (key == L"--image") image = value;
        else if (key == L"--out") out = value;
        else if (key == L"--snippet") snippet = value;
        else if (key == L"--runtime-dir") runtime_dir = value;
        else if (key == L"--data-dir") data_dir = value;
        else if (key == L"--frames") frames = _wtoi(value.c_str());
        else if (key == L"--evaluations") evaluations = _wtoi(value.c_str());
        else if (key == L"--scale") scale = (float)_wtof(value.c_str());
        else if (key == L"--motion") swscanf(value.c_str(), L"%f,%f", &motion_x, &motion_y);
        else if (key == L"--scroll") swscanf(value.c_str(), L"%d,%d", &scroll_x, &scroll_y);
        else if (key == L"--object") swscanf(value.c_str(), L"%d,%d", &object_x, &object_y);
        else if (key == L"--orbit") swscanf(value.c_str(), L"%d,%d", &orbit_x, &orbit_y);
        else if (key == L"--churn") churn = _wtoi(value.c_str());
        else if (key == L"--skin") skin = (float)_wtof(value.c_str());
        else if (key == L"--local-tone") local_tone = (float)_wtof(value.c_str());
        else if (key == L"--local-structure") local_structure = (float)_wtof(value.c_str());
        else if (key == L"--intensity") intensity = (float)_wtof(value.c_str());
        else if (key == L"--style") style = _wtoi(value.c_str());
        else if (key == L"--depth") depth_value = (float)_wtof(value.c_str());
    }
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);

    std::vector<uint8_t> rgba;
    UINT width = 0, height = 0;
    if (!load_png(image, rgba, width, height)) return 1;
    printf("image: %ux%u\n", width, height);

    Gpu gpu;
    if (!gpu.init()) return 1;

    // The game's buffers: colour as R8G8B8A8 (display encoded), or with
    // --linear as half floats decoded from sRGB -- what the research tools fed
    // the network -- then depth, motion vectors, and the output the runtime
    // composes into, in the colour's format.
    const DXGI_FORMAT color_format = linear ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_R8G8B8A8_UNORM;
    const UINT color_bytes = linear ? 8 : 4;
    ID3D12Resource *color = gpu.texture(width, height, color_format, false, D3D12_RESOURCE_STATE_COPY_DEST);
    ID3D12Resource *output = gpu.texture(width, height, color_format, true, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ID3D12Resource *depth = gpu.texture(width, height, DXGI_FORMAT_R32_FLOAT, false, D3D12_RESOURCE_STATE_COPY_DEST);
    ID3D12Resource *motion = gpu.texture(width, height, DXGI_FORMAT_R16G16_FLOAT, false, D3D12_RESOURCE_STATE_COPY_DEST);
    ID3D12Resource *staging = gpu.buffer((UINT64)width * height * 8 + 65536 * 8, D3D12_HEAP_TYPE_UPLOAD,
                                         D3D12_RESOURCE_STATE_GENERIC_READ);
    ID3D12Resource *staging2 = gpu.buffer((UINT64)width * height * 4 + 65536 * 4, D3D12_HEAP_TYPE_UPLOAD,
                                          D3D12_RESOURCE_STATE_GENERIC_READ);
    ID3D12Resource *staging3 = gpu.buffer((UINT64)width * height * 4 + 65536 * 4, D3D12_HEAP_TYPE_UPLOAD,
                                          D3D12_RESOURCE_STATE_GENERIC_READ);
    if (!color || !output || !depth || !motion || !staging) return fail("could not create the frame's buffers"), 1;

    const bool scrolling = scroll_x || scroll_y;
    if (scrolling) {
        motion_x = (float)scroll_x;
        motion_y = (float)scroll_y;
    }
    // The colour as uploaded, kept for scrolling.
    std::vector<uint8_t> pixels;
    {
        std::vector<float> depth_values((size_t)width * height, depth_value);
        std::vector<uint16_t> motion_values((size_t)width * height * 2);
        for (size_t i = 0; i < (size_t)width * height; ++i) {
            motion_values[2 * i] = to_half(-motion_x);   // current -> previous
            motion_values[2 * i + 1] = to_half(-motion_y);
        }
        gpu.begin();
        if (linear) {
            std::vector<uint16_t> halves((size_t)width * height * 4);
            for (size_t i = 0; i < halves.size(); ++i)
                halves[i] = to_half((i & 3) == 3 ? rgba[i] / 255.0f : srgb_to_linear(rgba[i] / 255.0f));
            pixels.assign((const uint8_t *)halves.data(), (const uint8_t *)(halves.data() + halves.size()));
        } else {
            pixels = rgba;
        }
        gpu.upload(color, pixels.data(), width, height, color_bytes, staging);
        gpu.upload(depth, depth_values.data(), width, height, 4, staging2);
        gpu.upload(motion, motion_values.data(), width, height, 4, staging3);
        barrier(gpu.cmd, color, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        barrier(gpu.cmd, depth, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        barrier(gpu.cmd, motion, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        gpu.submit_and_wait();
    }

    dlss5nr_create_info info{};
    info.struct_size = sizeof info;
    info.device = gpu.device;
    info.runtime_dir = runtime_dir.empty() ? nullptr : runtime_dir.c_str();
    info.snippet_path = snippet.empty() ? nullptr : snippet.c_str();
    info.data_dir = data_dir.empty() ? nullptr : data_dir.c_str();
    info.log = &log_sink;
    dlss5nr_context *nr = nullptr;
    if (dlss5nr_create(&info, &nr) != 0) return fail("dlss5nr_create failed"), 1;

    dlss5nr_settings settings;
    dlss5nr_default_settings(&settings);
    settings.resolution_scale = scale;
    if (!ratio) settings.composition = DLSS5NR_COMPOSITION_REPLACE;   // the network's answer, to compare
    settings.max_age = 1000;
    settings.show_tracking = show_tracking ? 1 : 0;
    if (skin > -2.0f) settings.skin_structure = skin;
    if (local_tone > -2.0f) settings.local_tone = local_tone;
    if (local_structure > -2.0f) settings.local_structure = local_structure;
    if (intensity > -2.0f) settings.intensity = intensity;
    if (style >= 0) settings.style = style;
    settings.wait_for_network = wait ? 1 : 0;
    dlss5nr_set_settings(nr, &settings);

    // Until the network runs, frames are copies; wait for it rather than
    // spend the frames on nothing.
    dlss5nr_status status{};
    status.struct_size = sizeof status;
    const auto started = std::chrono::steady_clock::now();
    for (int last = -1;;) {
        dlss5nr_get_status(nr, &status);
        if (status.state != last) {
            printf("state %d: %s\n", status.state, status.message);
            last = status.state;
        }
        if (status.state == DLSS5NR_STATE_RUNNING || status.state == DLSS5NR_STATE_FAILED) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    printf("startup: %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
    if (status.state == DLSS5NR_STATE_FAILED) {
        dlss5nr_destroy(nr);
        return 1;
    }

    // Frames. The network's results arrive on its own thread; a frame with no
    // fresh result reprojects the last one. With --evaluations the loop ends
    // once that many results have been composed, however many frames it took.
    int composed = 0;
    double record_ms = 0.0;
    int frame = 0;
    // The moving object: a piece of the image, nearer than the rest.
    const bool orbit = orbit_x || orbit_y;
    const bool object = object_x || object_y || orbit;
    const UINT kObject = 400, kObjectFromX = 200, kObjectFromY = 200, kObjectAtX = 300, kObjectAtY = 500;
    const float object_depth = 0.5f, scene_depth = 0.1f;
    std::vector<float> depth_frame(object ? (size_t)width * height : 0);
    std::vector<uint16_t> motion_frame(object ? (size_t)width * height * 2 : 0);
    UINT object_now_x = 0, object_now_y = 0;
    std::vector<uint8_t> moved(scrolling || object ? pixels.size() : 0);
    for (; frame < frames || composed < evaluations; ++frame) {
        if ((churn & 8) && frame > 0 && frame % 10 == 0) {
            // A slider being dragged: a different size every few frames.
            settings.resolution_scale = 0.5f + 0.05f * (float)((frame / 10) % 11);
            dlss5nr_set_settings(nr, &settings);
        }
        if (churn && frame > 0 && frame % 90 == 0) {
            if (churn & 1) settings.style = (settings.style + 1) % 3;
            if (churn & 2) settings.resolution_scale = ((frame / 90) % 2) ? 0.75f : 1.0f;
            if (churn & 4) settings.resolution_scale = 1.0f - 0.25f * (float)((frame / 90) % 3);
            dlss5nr_set_settings(nr, &settings);
        }
        gpu.begin();
        if (scrolling && frame > 0) {
            // What was at x is now at x + frame * scroll.
            const int sx = (int)(((long long)frame * scroll_x) % (long long)width + width) % (int)width;
            const int sy = (int)(((long long)frame * scroll_y) % (long long)height + height) % (int)height;
            for (UINT y = 0; y < height; ++y) {
                const UINT from_y = (y + height - sy) % height;
                for (UINT x = 0; x < width; ++x) {
                    const UINT from_x = (x + width - sx) % width;
                    memcpy(&moved[((size_t)y * width + x) * color_bytes],
                           &pixels[((size_t)from_y * width + from_x) * color_bytes], color_bytes);
                }
            }
            barrier(gpu.cmd, color, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
            gpu.upload(color, moved.data(), width, height, color_bytes, staging);
            barrier(gpu.cmd, color, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        if (object) {
            object_now_x = (UINT)((kObjectAtX + (long long)frame * object_x) % (width - kObject));
            object_now_y = (UINT)((kObjectAtY + (long long)frame * object_y) % (height - kObject));
            moved = pixels;
            if (orbit) {
                // The background has moved by frame * orbit since the first frame.
                const int sx = (int)(((long long)frame * orbit_x) % (long long)width + width) % (int)width;
                const int sy = (int)(((long long)frame * orbit_y) % (long long)height + height) % (int)height;
                for (UINT y = 0; y < height; ++y)
                    for (UINT x = 0; x < width; ++x)
                        memcpy(&moved[((size_t)y * width + x) * color_bytes],
                               &pixels[((size_t)((y + height - sy) % height) * width + (x + width - sx) % width) *
                                       color_bytes],
                               color_bytes);
            }
            std::fill(depth_frame.begin(), depth_frame.end(), scene_depth);
            for (size_t i = 0; i < (size_t)width * height; ++i) {
                motion_frame[2 * i] = to_half(-(float)orbit_x);   // current -> previous
                motion_frame[2 * i + 1] = to_half(-(float)orbit_y);
            }
            for (UINT y = 0; y < kObject; ++y) {
                const size_t to = (size_t)(object_now_y + y) * width + object_now_x;
                memcpy(&moved[to * color_bytes], &pixels[((size_t)(kObjectFromY + y) * width + kObjectFromX) * color_bytes],
                       (size_t)kObject * color_bytes);
                for (UINT x = 0; x < kObject; ++x) {
                    depth_frame[to + x] = object_depth;
                    motion_frame[2 * (to + x)] = to_half(-(float)object_x);   // current -> previous
                    motion_frame[2 * (to + x) + 1] = to_half(-(float)object_y);
                }
            }
            const D3D12_RESOURCE_STATES shown = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            for (ID3D12Resource *r : {color, depth, motion}) barrier(gpu.cmd, r, shown, D3D12_RESOURCE_STATE_COPY_DEST);
            gpu.upload(color, moved.data(), width, height, color_bytes, staging);
            gpu.upload(depth, depth_frame.data(), width, height, 4, staging2);
            gpu.upload(motion, motion_frame.data(), width, height, 4, staging3);
            for (ID3D12Resource *r : {color, depth, motion}) barrier(gpu.cmd, r, D3D12_RESOURCE_STATE_COPY_DEST, shown);
        }
        dlss5nr_frame f{};
        f.struct_size = sizeof f;
        f.command_list = gpu.cmd;
        f.input = color;
        f.input_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        f.output = output;
        f.output_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        f.width = width;
        f.height = height;
        f.depth = bare ? nullptr : depth;
        f.depth_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        f.motion = bare ? nullptr : motion;
        f.motion_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        f.depth_width = f.motion_width = width;
        f.depth_height = f.motion_height = height;
        f.motion_scale_x = f.motion_scale_y = 1.0f;
        f.pre_exposure = 1.0f;
        f.hdr = hdr ? 1 : 0;
        f.reset = frame == 0;
        const auto t0 = std::chrono::steady_clock::now();
        if (dlss5nr_process(nr, &f) != 0) return fail("dlss5nr_process failed"), 1;
        record_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        gpu.submit_and_wait();
        dlss5nr_present(nr, gpu.queue);

        dlss5nr_get_status(nr, &status);
        composed = (int)status.results;
        if (frame > 100000) break;
        // A game's frame takes time; without it the harness would outrun the
        // network by thousands of frames and compose only reprojections.
        std::this_thread::sleep_for(std::chrono::milliseconds(8));
    }
    dlss5nr_get_status(nr, &status);
    printf("frames %d, results composed %d, network %ux%u, last evaluation %.1f ms, %.1f per second, "
           "%u native kernels, recording %.3f ms per frame\n",
           frame, composed, status.network_width, status.network_height, status.evaluation_ms,
           status.evaluations_per_second, status.native_kernels, record_ms / frame);
    printf("the last result was captured %u frames before it was composed\n", status.result_latency);
    if (object)
        printf("the object is at %u,%u, and moves by %d,%d a frame\n", object_now_x, object_now_y, object_x, object_y);
    if (scrolling)
        printf("scrolled by %lld,%lld in all\n", (long long)(frame - 1) * scroll_x, (long long)(frame - 1) * scroll_y);

    // Read back the output.
    {
        ID3D12Resource *readback = gpu.buffer((UINT64)width * height * color_bytes + 65536 * 8, D3D12_HEAP_TYPE_READBACK,
                                              D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
        const D3D12_RESOURCE_DESC d = output->GetDesc();
        gpu.device->GetCopyableFootprints(&d, 0, 1, 0, &fp, nullptr, nullptr, nullptr);
        gpu.begin();
        barrier(gpu.cmd, output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION to{}, from{};
        to.pResource = readback;
        to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        to.PlacedFootprint = fp;
        from.pResource = output;
        from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        gpu.cmd->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
        barrier(gpu.cmd, output, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        gpu.submit_and_wait();
        uint8_t *mapped = nullptr;
        readback->Map(0, nullptr, (void **)&mapped);
        std::vector<uint8_t> result((size_t)width * height * 4);
        for (UINT y = 0; y < height; ++y) {
            const uint8_t *row = mapped + fp.Offset + (size_t)y * fp.Footprint.RowPitch;
            if (!linear) {
                memcpy(result.data() + (size_t)y * width * 4, row, (size_t)width * 4);
                continue;
            }
            const uint16_t *halves = (const uint16_t *)row;
            for (UINT i = 0; i < width * 4; ++i) {
                const float v = from_half(halves[i]);
                const float e = (i & 3) == 3 ? v : linear_to_srgb(v);
                result[(size_t)y * width * 4 + i] = (uint8_t)(e * 255.0f + 0.5f);
            }
        }
        readback->Unmap(0, nullptr);
        release(readback);
        if (!save_png(out, result, width, height)) return 1;
        wprintf(L"written %ls\n", out.c_str());
    }

    dlss5nr_destroy(nr);
    release(color);
    release(output);
    release(depth);
    release(motion);
    release(staging);
    release(staging2);
    release(staging3);
    return 0;
}
