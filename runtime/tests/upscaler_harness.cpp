// dlss5up_harness.exe: NVIDIA's upscalers (super resolution, ray reconstruction) driven through the
// runtime's C interface the way a host does it: ordinary game textures (not shared, in states of their
// own), a queue, and nothing known of how the network gets its pixels.
//
// The frames are the synthetic ones of dlss5sr_harness and tools' sr_probe, so the output can be compared
// with theirs: it has to be the same, whatever the texture formats and states the host presents.
//
//   dlss5up_harness --kind sr|rr --runtime-dir DIR --data-dir DIR --snippet DLL
//                   [--render 1280x720] [--output 1920x1080] [--frames 8] [--out prefix]
//                   [--depth-format r32|d32] [--cpu-wait]
//
// <prefix>.raw is the last frame's output as half RGBA.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "../include/dlss5nr.h"

namespace {

template <class T> void release(T *&p) {
    if (p) p->Release();
    p = nullptr;
}

uint16_t to_half(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000, e = (x >> 23) & 0xff, m = x & 0x7fffff;
    if (e == 0) return (uint16_t)sign;
    const int he = (int)e - 127 + 15;
    if (he >= 31) return (uint16_t)(sign | 0x7c00);
    if (he <= 0) return (uint16_t)sign;
    return (uint16_t)(sign | (he << 10) | (m >> 13));
}

struct Gpu {
    ID3D12Device *device = nullptr;
    ID3D12CommandQueue *queue = nullptr;
    ID3D12CommandAllocator *allocator = nullptr;
    ID3D12GraphicsCommandList *list = nullptr;
    ID3D12Fence *fence = nullptr;
    HANDLE event = nullptr;
    uint64_t value = 0;
};

bool start_gpu(Gpu &gpu) {
    IDXGIFactory4 *factory = nullptr;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return false;
    IDXGIAdapter1 *adapter = nullptr;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);
        if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) break;
        release(adapter);
    }
    factory->Release();
    if (!adapter) return false;
    DXGI_ADAPTER_DESC1 desc;
    adapter->GetDesc1(&desc);
    printf("adapter: %ls\n", desc.Description);
    const HRESULT made = D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&gpu.device));
    adapter->Release();
    if (FAILED(made)) return false;
    D3D12_COMMAND_QUEUE_DESC queue{};
    queue.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(gpu.device->CreateCommandQueue(&queue, IID_PPV_ARGS(&gpu.queue))) ||
        FAILED(gpu.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&gpu.allocator))) ||
        FAILED(gpu.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, gpu.allocator, nullptr,
                                             IID_PPV_ARGS(&gpu.list))) ||
        FAILED(gpu.device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gpu.fence))))
        return false;
    gpu.list->Close();
    gpu.event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    return true;
}

void wait_idle(Gpu &gpu) {
    gpu.queue->Signal(gpu.fence, ++gpu.value);
    if (gpu.fence->GetCompletedValue() < gpu.value) {
        gpu.fence->SetEventOnCompletion(gpu.value, gpu.event);
        WaitForSingleObject(gpu.event, 60000);
    }
}

template <class Record> bool run(Gpu &gpu, Record record) {
    gpu.allocator->Reset();
    gpu.list->Reset(gpu.allocator, nullptr);
    record(gpu.list);
    gpu.list->Close();
    ID3D12CommandList *lists[] = {gpu.list};
    gpu.queue->ExecuteCommandLists(1, lists);
    wait_idle(gpu);
    return true;
}

ID3D12Resource *make_texture(Gpu &gpu, uint32_t w, uint32_t h, DXGI_FORMAT format, bool uav) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = w;
    desc.Height = h;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Flags = uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
    ID3D12Resource *out = nullptr;
    gpu.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                        IID_PPV_ARGS(&out));
    return out;
}

ID3D12Resource *make_buffer(Gpu &gpu, UINT64 size, D3D12_HEAP_TYPE type) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = type;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource *out = nullptr;
    gpu.device->CreateCommittedResource(
        &heap, D3D12_HEAP_FLAG_NONE, &desc,
        type == D3D12_HEAP_TYPE_READBACK ? D3D12_RESOURCE_STATE_COPY_DEST : D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
        IID_PPV_ARGS(&out));
    return out;
}

void barrier(ID3D12GraphicsCommandList *cmd, ID3D12Resource *r, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = r;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = from;
    b.Transition.StateAfter = to;
    cmd->ResourceBarrier(1, &b);
}

bool upload(Gpu &gpu, ID3D12Resource *texture, uint32_t w, uint32_t h, const void *pixels, uint32_t bpp) {
    D3D12_RESOURCE_DESC desc = texture->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 total = 0;
    gpu.device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, &total);
    ID3D12Resource *buffer = make_buffer(gpu, total, D3D12_HEAP_TYPE_UPLOAD);
    uint8_t *mapped = nullptr;
    buffer->Map(0, nullptr, (void **)&mapped);
    for (uint32_t y = 0; y < h; ++y)
        memcpy(mapped + (size_t)y * footprint.Footprint.RowPitch, (const uint8_t *)pixels + (size_t)y * w * bpp, (size_t)w * bpp);
    buffer->Unmap(0, nullptr);
    run(gpu, [&](ID3D12GraphicsCommandList *cmd) {
        barrier(cmd, texture, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_TEXTURE_COPY_LOCATION to{}, from{};
        to.pResource = texture;
        to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        from.pResource = buffer;
        from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        from.PlacedFootprint = footprint;
        cmd->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
        barrier(cmd, texture, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    });
    buffer->Release();
    return true;
}

bool download(Gpu &gpu, ID3D12Resource *texture, uint32_t w, uint32_t h, std::vector<uint8_t> &pixels, uint32_t bpp) {
    D3D12_RESOURCE_DESC desc = texture->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 total = 0;
    gpu.device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, &total);
    ID3D12Resource *buffer = make_buffer(gpu, total, D3D12_HEAP_TYPE_READBACK);
    run(gpu, [&](ID3D12GraphicsCommandList *cmd) {
        barrier(cmd, texture, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION to{}, from{};
        to.pResource = buffer;
        to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        to.PlacedFootprint = footprint;
        from.pResource = texture;
        from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        cmd->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
        barrier(cmd, texture, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    });
    const uint8_t *mapped = nullptr;
    buffer->Map(0, nullptr, (void **)&mapped);
    pixels.resize((size_t)w * h * bpp);
    for (uint32_t y = 0; y < h; ++y)
        memcpy(pixels.data() + (size_t)y * w * bpp, mapped + (size_t)y * footprint.Footprint.RowPitch, (size_t)w * bpp);
    buffer->Unmap(0, nullptr);
    buffer->Release();
    return true;
}

std::wstring wide(const char *s) {
    wchar_t b[1024];
    MultiByteToWideChar(CP_UTF8, 0, s, -1, b, 1024);
    return b;
}

void log_sink(void *, int level, const char *message) {
    if (level <= 2) printf("  [runtime %d] %s\n", level, message);
}

} // namespace

int main(int argc, char **argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::string kind = "sr", snippet, runtime_dir, data_dir, out_prefix, depth_format = "r32";
    uint32_t rw = 1280, rh = 720, ow = 1920, oh = 1080;
    int frames = 8;
    bool subrect = false, with_exposure = false, with_masks = false;
    int flags_override = -1, quality = 2, preset = 0;   // creation flags and quality, when the options say so
    bool reset_every = false;   // every frame is a first one (the history is not used)
    int reset_at = -1;          // and the frame that is one besides the first
    int reset_parity = -1;      // and every frame of this parity (0 even, 1 odd)
    bool means = false;   // print the output mean colour after every frame
    std::string scene = "gradient";   // gradient | const | corner | topleft: what the colour holds (for probing)
    uint32_t dynamic_w = 0, dynamic_h = 0;
    float exposure_value = 1.0f, mask_value = 0.0f;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--cpu-wait") {
            SetEnvironmentVariableA("DLSS5NR_UPSCALER_CPU_WAIT", "1");
            continue;
        }
        if (key == "--scene" && i + 1 < argc) {
            scene = argv[++i];
            continue;
        }
        if (key == "--reset-every") {
            reset_every = true;
            continue;
        }
        if (key == "--means") {
            means = true;
            continue;
        }
        if (key == "--subrect") {   // the game's textures are larger, the rendered region in a corner of its own
            subrect = true;
            continue;
        }
        if (key == "--exposure") {   // the game gives an exposure texture (and no auto exposure)
            with_exposure = true;
            continue;
        }
        if (key == "--masks") {   // and a reactive and a transparency mask
            with_masks = true;
            continue;
        }
        if (i + 1 >= argc) break;
        const std::string value = argv[++i];
        if (key == "--kind") kind = value;
        else if (key == "--snippet") snippet = value;
        else if (key == "--runtime-dir") runtime_dir = value;
        else if (key == "--data-dir") data_dir = value;
        else if (key == "--render") sscanf(value.c_str(), "%ux%u", &rw, &rh);
        else if (key == "--output") sscanf(value.c_str(), "%ux%u", &ow, &oh);
        else if (key == "--frames") frames = atoi(value.c_str());
        else if (key == "--out") out_prefix = value;
        else if (key == "--depth-format") depth_format = value;
        else if (key == "--flags") flags_override = (int)strtol(value.c_str(), nullptr, 0);   // NVIDIA's creation flags, as they are
        else if (key == "--reset-at") reset_at = atoi(value.c_str());
        else if (key == "--reset-parity") reset_parity = atoi(value.c_str());
        else if (key == "--quality") quality = atoi(value.c_str());
        else if (key == "--preset") preset = atoi(value.c_str());   // 1 A .. 11 K, 12 L, 13 M; 0 the library's own
        else if (key == "--exposure-value") exposure_value = (float)atof(value.c_str());
        else if (key == "--mask-value") mask_value = (float)atof(value.c_str());
        else if (key == "--dynamic") sscanf(value.c_str(), "%ux%u", &dynamic_w, &dynamic_h);   // render this much of it
        else {
            printf("unknown option %s\n", key.c_str());
            return 2;
        }
    }
    if (snippet.empty() || runtime_dir.empty() || data_dir.empty()) {
        printf("usage: dlss5up_harness --kind sr|rr --snippet DLL --runtime-dir DIR --data-dir DIR [--render WxH] "
               "[--output WxH] [--frames N] [--out prefix] [--cpu-wait]\n");
        return 2;
    }
    const bool rr = kind == "rr";

    Gpu gpu;
    if (!start_gpu(gpu)) {
        printf("[FAIL] no Direct3D 12 device\n");
        return 1;
    }
    const std::wstring runtime = wide(runtime_dir.c_str()), snippet_w = wide(snippet.c_str()), data = wide(data_dir.c_str());
    dlss5nr_upscaler_create_info info{};
    info.struct_size = sizeof info;
    info.device = gpu.device;
    info.kind = rr ? DLSS5NR_UPSCALER_RAY_RECONSTRUCTION : DLSS5NR_UPSCALER_SUPER_RESOLUTION;
    info.runtime_dir = runtime.c_str();
    info.snippet_path = snippet_w.c_str();
    info.data_dir = data.c_str();
    info.log = log_sink;
    info.render_width = rw;
    info.render_height = rh;
    info.output_width = ow;
    info.output_height = oh;
    info.flags = DLSS5NR_UPSCALER_HDR | DLSS5NR_UPSCALER_MOTION_LOW_RES | DLSS5NR_UPSCALER_DEPTH_INVERTED | (with_exposure ? 0u : DLSS5NR_UPSCALER_AUTO_EXPOSURE);
    if (flags_override >= 0) info.flags = (uint32_t)flags_override;
    info.quality = quality;
    info.preset = preset;
    const auto began = std::chrono::steady_clock::now();
    dlss5nr_upscaler *upscaler = nullptr;
    if (dlss5nr_upscaler_create(&info, &upscaler) != 0) {
        printf("[FAIL] create: %s\n", dlss5nr_upscaler_last_error());
        return 1;
    }
    dlss5nr_upscaler_status status;
    dlss5nr_upscaler_get_status(upscaler, &status);
    printf("[ OK ] upscaler created in %.1f s, %u native kernels, %s synchronisation\n",
           std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count(), status.native_kernels,
           status.gpu_synchronised ? "GPU" : "CPU");

    // The game's textures: no sharing, and a depth that may be typeless as games make it.
    const DXGI_FORMAT depth_dxgi = depth_format == "d32" ? DXGI_FORMAT_R32_TYPELESS : DXGI_FORMAT_R32_FLOAT;
    // With --subrect every texture is larger than the region in it, which starts at an origin of its own.
    const uint32_t pad_w = subrect ? 64 : 0, pad_h = subrect ? 48 : 0;
    const uint32_t tw = rw + pad_w, th = rh + pad_h;   // the game's input textures
    const uint32_t gw = ow + (subrect ? 40 : 0), gh = oh + (subrect ? 30 : 0);   // and its output
    const dlss5nr_point origin = subrect ? dlss5nr_point{37, 23} : dlss5nr_point{0, 0};
    const dlss5nr_point out_origin = subrect ? dlss5nr_point{17, 9} : dlss5nr_point{0, 0};
    ID3D12Resource *color = make_texture(gpu, tw, th, DXGI_FORMAT_R16G16B16A16_FLOAT, false);
    ID3D12Resource *depth = make_texture(gpu, tw, th, depth_dxgi, false);
    ID3D12Resource *motion = make_texture(gpu, tw, th, DXGI_FORMAT_R16G16_FLOAT, false);
    ID3D12Resource *output = make_texture(gpu, gw, gh, DXGI_FORMAT_R16G16B16A16_FLOAT, true);
    ID3D12Resource *diffuse = nullptr, *specular = nullptr, *normals = nullptr, *hit = nullptr;
    ID3D12Resource *exposure = nullptr, *reactive = nullptr, *transparency = nullptr;
    if (rr) {
        diffuse = make_texture(gpu, tw, th, DXGI_FORMAT_R16G16B16A16_FLOAT, false);
        specular = make_texture(gpu, tw, th, DXGI_FORMAT_R16G16B16A16_FLOAT, false);
        normals = make_texture(gpu, tw, th, DXGI_FORMAT_R16G16B16A16_FLOAT, false);
        hit = make_texture(gpu, tw, th, DXGI_FORMAT_R16_FLOAT, false);
    }
    if (with_exposure) exposure = make_texture(gpu, 1, 1, DXGI_FORMAT_R32_FLOAT, false);
    if (with_masks) {
        reactive = make_texture(gpu, tw, th, DXGI_FORMAT_R8_UNORM, false);
        transparency = make_texture(gpu, tw, th, DXGI_FORMAT_R16_FLOAT, false);
    }
    // A texture of the game's size with the frame's region put at the origin and garbage around it.
    auto embed = [&](const void *pixels, uint32_t bpp, uint8_t garbage) {
        std::vector<uint8_t> big((size_t)tw * th * bpp, garbage);
        for (uint32_t y = 0; y < rh; ++y)
            memcpy(big.data() + ((size_t)(y + origin.y) * tw + origin.x) * bpp, (const uint8_t *)pixels + (size_t)y * rw * bpp,
                   (size_t)rw * bpp);
        return big;
    };
    float world_to_view[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    float view_to_clip[16] = {1.2f, 0, 0, 0, 0, 2.1f, 0, 0, 0, 0, 0, -1, 0, 0, 0.1f, 0};
    std::vector<uint16_t> col((size_t)rw * rh * 4), mv((size_t)rw * rh * 2, 0), hd((size_t)rw * rh, to_half(1.0f));
    std::vector<uint16_t> nrm((size_t)rw * rh * 4), alb((size_t)rw * rh * 4);
    std::vector<float> dep((size_t)rw * rh);
    for (size_t i = 0; i < (size_t)rw * rh; ++i) {
        nrm[i * 4 + 0] = to_half(0.0f);
        nrm[i * 4 + 1] = to_half(0.0f);
        nrm[i * 4 + 2] = to_half(1.0f);
        nrm[i * 4 + 3] = to_half(0.5f);
        alb[i * 4 + 0] = alb[i * 4 + 1] = alb[i * 4 + 2] = to_half(0.5f);
        alb[i * 4 + 3] = to_half(1.0f);
    }
    double total = 0;
    int timed = 0;
    for (int frame = 0; frame < frames; ++frame) {
        for (uint32_t y = 0; y < rh; ++y)
            for (uint32_t x = 0; x < rw; ++x) {
                const int sq = (int)rh / 4, x0 = (frame * 8) % (int)rw, y0 = (int)rh / 3;
                const bool in_sq = ((int)x - x0 + (int)rw) % (int)rw < sq && (int)y >= y0 && (int)y < y0 + sq;
                const float tex = 0.12f * sinf(x * 0.9f) * sinf(y * 0.7f) + 0.05f * sinf((x + y) * 0.23f);
                float r = in_sq ? 1.0f : (float)x / rw + tex, g = in_sq ? 1.0f : (float)y / rh + tex,
                      b = in_sq ? 1.0f : 0.4f + tex;
                if (scene != "gradient") {
                    r = g = b = (scene == "ramp") ? 0.25f + 0.1f * frame : 0.25f;
                    if (scene == "corner" && x + 8 >= rw && y + 8 >= rh) r = g = b = 0.9f;
                    if (scene == "topleft" && x < 8 && y < 8) r = g = b = 0.9f;
                }
                col[((size_t)y * rw + x) * 4 + 0] = to_half(r);
                col[((size_t)y * rw + x) * 4 + 1] = to_half(g);
                col[((size_t)y * rw + x) * 4 + 2] = to_half(b);
                col[((size_t)y * rw + x) * 4 + 3] = to_half(1.0f);
                dep[(size_t)y * rw + x] = in_sq ? 0.8f : 0.2f;
                mv[((size_t)y * rw + x) * 2] = to_half(in_sq && frame ? -8.0f : 0.0f);
            }
        upload(gpu, color, tw, th, embed(col.data(), 8, 0x7b).data(), 8);
        upload(gpu, depth, tw, th, embed(dep.data(), 4, 0xff).data(), 4);
        upload(gpu, motion, tw, th, embed(mv.data(), 4, 0x7b).data(), 4);
        if (rr) {
            upload(gpu, diffuse, tw, th, embed(alb.data(), 8, 0x7b).data(), 8);
            upload(gpu, specular, tw, th, embed(alb.data(), 8, 0x7b).data(), 8);
            upload(gpu, normals, tw, th, embed(nrm.data(), 8, 0x7b).data(), 8);
            upload(gpu, hit, tw, th, embed(hd.data(), 2, 0x7b).data(), 2);
        }
        if (exposure) {
            upload(gpu, exposure, 1, 1, &exposure_value, 4);
        }
        if (reactive) {
            std::vector<uint8_t> zeros((size_t)rw * rh, (uint8_t)(mask_value * 255.0f + 0.5f));
            upload(gpu, reactive, tw, th, embed(zeros.data(), 1, 0xff).data(), 1);
            std::vector<uint16_t> none((size_t)rw * rh, to_half(mask_value));
            upload(gpu, transparency, tw, th, embed(none.data(), 2, 0x7b).data(), 2);
        }
        dlss5nr_upscaler_frame f{};
        f.struct_size = sizeof f;
        f.color = {color, D3D12_RESOURCE_STATE_COMMON};
        f.depth = {depth, D3D12_RESOURCE_STATE_COMMON};
        f.motion = {motion, D3D12_RESOURCE_STATE_COMMON};
        f.output = {output, D3D12_RESOURCE_STATE_COMMON};
        if (rr) {
            f.diffuse_albedo = {diffuse, D3D12_RESOURCE_STATE_COMMON};
            f.specular_albedo = {specular, D3D12_RESOURCE_STATE_COMMON};
            f.normals = {normals, D3D12_RESOURCE_STATE_COMMON};
            f.hit_distance = {hit, D3D12_RESOURCE_STATE_COMMON};
            f.world_to_view = world_to_view;
            f.view_to_clip = view_to_clip;
        }
        f.motion_scale_x = f.motion_scale_y = 1.0f;
        f.pre_exposure = f.exposure_scale = 1.0f;
        f.color_origin = f.depth_origin = f.motion_origin = f.guide_origin = f.mask_origin = origin;
        f.output_origin = out_origin;
        f.render_width = dynamic_w;
        f.render_height = dynamic_h;
        if (exposure) f.exposure = {exposure, D3D12_RESOURCE_STATE_COMMON};
        if (reactive) {
            f.reactive_mask = {reactive, D3D12_RESOURCE_STATE_COMMON};
            f.transparency_mask = {transparency, D3D12_RESOURCE_STATE_COMMON};
        }
        f.reset = frame == 0 || reset_every || (reset_at >= 0 && frame == reset_at) || (reset_parity >= 0 && frame % 2 == reset_parity);
        const auto start = std::chrono::steady_clock::now();
        if (dlss5nr_upscaler_evaluate(upscaler, gpu.queue, &f) != 0) {
            printf("[FAIL] evaluate (frame %d): %s\n", frame, dlss5nr_upscaler_last_error());
            return 1;
        }
        wait_idle(gpu);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        printf("frame %2d: %.2f ms\n", frame, ms);
        if (means) {
            std::vector<uint8_t> shot;
            download(gpu, output, gw, gh, shot, 8);
            double sum[3] = {0, 0, 0};
            for (uint32_t y = 0; y < oh; y += 4)
                for (uint32_t x = 0; x < ow; x += 4)
                    for (int c = 0; c < 3; ++c) {
                        const uint16_t h = *(const uint16_t *)(shot.data() + (((size_t)(y + out_origin.y) * gw + x + out_origin.x) * 4 + c) * 2);
                        // half to float
                        const uint32_t sign = (h & 0x8000u) << 16;
                        const int exp = (h >> 10) & 31;
                        const uint32_t mant = h & 1023;
                        uint32_t bits;
                        if (exp == 0) bits = sign;   // zero, or too small to matter here
                        else if (exp == 31) bits = sign | 0x7f800000u | (mant << 13);
                        else bits = sign | ((uint32_t)(exp + 112) << 23) | (mant << 13);
                        float v;
                        memcpy(&v, &bits, 4);
                        sum[c] += v;
                    }
            const double n = (double)((oh + 3) / 4) * ((ow + 3) / 4);
            printf("          mean %.4f %.4f %.4f\n", sum[0] / n, sum[1] / n, sum[2] / n);
        }
        if (frame >= 3) total += ms, ++timed;
    }
    if (timed) printf("average from frame 3: %.2f ms per upscaled frame\n", total / timed);
    if (!out_prefix.empty()) {
        std::vector<uint8_t> pixels;
        std::vector<uint8_t> whole;
        download(gpu, output, gw, gh, whole, 8);
        pixels.resize((size_t)ow * oh * 8);
        for (uint32_t y = 0; y < oh; ++y)
            memcpy(pixels.data() + (size_t)y * ow * 8, whole.data() + ((size_t)(y + out_origin.y) * gw + out_origin.x) * 8, (size_t)ow * 8);
        char path[512];
        snprintf(path, sizeof path, "%s.raw", out_prefix.c_str());
        if (FILE *file = fopen(path, "wb")) {
            fwrite(pixels.data(), 1, pixels.size(), file);
            fclose(file);
        }
        printf("wrote %s.raw\n", out_prefix.c_str());
    }
    dlss5nr_upscaler_destroy(upscaler);
    TerminateProcess(GetCurrentProcess(), 0);
    return 0;
}
