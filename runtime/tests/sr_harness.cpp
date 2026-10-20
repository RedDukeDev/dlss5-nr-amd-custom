// dlss5sr_harness.exe: NVIDIA's upscalers (super resolution, ray reconstruction) driven through the
// runtime's Network class on Direct3D 12 textures, without a game.
//
// Synthetic frames (a gradient with fine detail and a moving square, constant depth, motion that
// follows the square) are uploaded to shared textures, the network runs, and the output is read back.
// The same frames go through tools' sr_probe on plain CUDA arrays, so the two outputs can be compared:
// they have to be identical, or the sharing of the textures with CUDA is losing something.
//
//   dlss5sr_harness --kind sr|rr --snippet nvngx_dlss.dll|nvngx_dlssd.dll --runtime-dir DIR --data-dir DIR
//                   [--render 1280x720] [--output 1920x1080] [--frames 8] [--out prefix]
//
// <prefix>.raw is the last frame's output as half RGBA, <prefix>.ppm the same as an 8 bit picture.

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
#include <vector>

#include "../core/network.h"

using namespace dlss5nr;

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

float from_half(uint16_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1f, m = h & 0x3ff, x;
    if (e == 0) {
        if (!m) x = sign;
        else {
            e = 127 - 14;
            while (!(m & 0x400)) m <<= 1, --e;
            m &= 0x3ff;
            x = sign | (e << 23) | (m << 13);
        }
    } else if (e == 31) x = sign | 0x7f800000u | (m << 13);
    else x = sign | ((e + 112) << 23) | (m << 13);
    float f;
    memcpy(&f, &x, 4);
    return f;
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

template <class Record> bool run(Gpu &gpu, Record record) {
    gpu.allocator->Reset();
    gpu.list->Reset(gpu.allocator, nullptr);
    record(gpu.list);
    gpu.list->Close();
    ID3D12CommandList *lists[] = {gpu.list};
    gpu.queue->ExecuteCommandLists(1, lists);
    gpu.queue->Signal(gpu.fence, ++gpu.value);
    if (gpu.fence->GetCompletedValue() < gpu.value) {
        gpu.fence->SetEventOnCompletion(gpu.value, gpu.event);
        if (WaitForSingleObject(gpu.event, 20000) != WAIT_OBJECT_0) return false;
    }
    return true;
}

ID3D12Resource *make_texture(Gpu &gpu, uint32_t w, uint32_t h, DXGI_FORMAT format) {
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
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ID3D12Resource *out = nullptr;
    gpu.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_SHARED, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                        IID_PPV_ARGS(&out));
    return out;
}

bool make_image(Gpu &gpu, SharedImage &image, uint32_t w, uint32_t h, DXGI_FORMAT format, bool writable) {
    image.resource = make_texture(gpu, w, h, format);
    if (!image.resource) return false;
    HANDLE handle = nullptr;
    if (FAILED(gpu.device->CreateSharedHandle(image.resource, nullptr, GENERIC_ALL, nullptr, &handle))) return false;
    image.nt_handle = handle;
    image.width = w;
    image.height = h;
    image.dxgi_format = format;
    image.writable = writable;
    return true;
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

// Pixels into a texture and back, through a buffer, with the GPU waited for each time.
bool upload(Gpu &gpu, const SharedImage &image, const void *pixels, uint32_t bytes_per_pixel) {
    D3D12_RESOURCE_DESC desc = image.resource->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 total = 0;
    gpu.device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, &total);
    ID3D12Resource *buffer = make_buffer(gpu, total, D3D12_HEAP_TYPE_UPLOAD);
    uint8_t *mapped = nullptr;
    buffer->Map(0, nullptr, (void **)&mapped);
    for (uint32_t y = 0; y < image.height; ++y)
        memcpy(mapped + (size_t)y * footprint.Footprint.RowPitch, (const uint8_t *)pixels + (size_t)y * image.width * bytes_per_pixel,
               (size_t)image.width * bytes_per_pixel);
    buffer->Unmap(0, nullptr);
    const bool ok = run(gpu, [&](ID3D12GraphicsCommandList *cmd) {
        barrier(cmd, image.resource, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_TEXTURE_COPY_LOCATION to{}, from{};
        to.pResource = image.resource;
        to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        from.pResource = buffer;
        from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        from.PlacedFootprint = footprint;
        cmd->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
        barrier(cmd, image.resource, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    });
    buffer->Release();
    return ok;
}

bool download(Gpu &gpu, const SharedImage &image, std::vector<uint8_t> &pixels, uint32_t bytes_per_pixel) {
    D3D12_RESOURCE_DESC desc = image.resource->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 total = 0;
    gpu.device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, &total);
    ID3D12Resource *buffer = make_buffer(gpu, total, D3D12_HEAP_TYPE_READBACK);
    const bool ok = run(gpu, [&](ID3D12GraphicsCommandList *cmd) {
        barrier(cmd, image.resource, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION to{}, from{};
        to.pResource = buffer;
        to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        to.PlacedFootprint = footprint;
        from.pResource = image.resource;
        from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        cmd->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
        barrier(cmd, image.resource, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    });
    if (ok) {
        const uint8_t *mapped = nullptr;
        buffer->Map(0, nullptr, (void **)&mapped);
        pixels.resize((size_t)image.width * image.height * bytes_per_pixel);
        for (uint32_t y = 0; y < image.height; ++y)
            memcpy(pixels.data() + (size_t)y * image.width * bytes_per_pixel, mapped + (size_t)y * footprint.Footprint.RowPitch,
                   (size_t)image.width * bytes_per_pixel);
        buffer->Unmap(0, nullptr);
    }
    buffer->Release();
    return ok;
}

std::wstring wide(const char *s) {
    wchar_t b[1024];
    MultiByteToWideChar(CP_UTF8, 0, s, -1, b, 1024);
    return b;
}

void log_sink(int level, const char *message) {
    if (level <= 2) printf("  [runtime %d] %s\n", level, message);
}

} // namespace

int main(int argc, char **argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);   // the run ends by TerminateProcess, which flushes nothing
    std::string kind = "sr", snippet, runtime_dir, data_dir, out_prefix;
    uint32_t rw = 1280, rh = 720, ow = 1920, oh = 1080;
    int frames = 8;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string key = argv[i], value = argv[i + 1];
        if (key == "--kind") kind = value;
        else if (key == "--snippet") snippet = value;
        else if (key == "--runtime-dir") runtime_dir = value;
        else if (key == "--data-dir") data_dir = value;
        else if (key == "--render") sscanf(value.c_str(), "%ux%u", &rw, &rh);
        else if (key == "--output") sscanf(value.c_str(), "%ux%u", &ow, &oh);
        else if (key == "--frames") frames = atoi(value.c_str());
        else if (key == "--out") out_prefix = value;
        else {
            printf("unknown option %s\n", key.c_str());
            return 2;
        }
    }
    if (snippet.empty() || runtime_dir.empty() || data_dir.empty()) {
        printf("usage: dlss5sr_harness --kind sr|rr --snippet DLL --runtime-dir DIR --data-dir DIR [--render WxH] "
               "[--output WxH] [--frames N] [--out prefix]\n");
        return 2;
    }
    const bool rr = kind == "rr";
    set_network_log(log_sink);

    Gpu gpu;
    if (!start_gpu(gpu)) {
        printf("[FAIL] no Direct3D 12 device\n");
        return 1;
    }
    Network network;
    NetworkPaths paths;
    paths.runtime_dir = wide(runtime_dir.c_str());
    paths.snippet = wide(snippet.c_str());
    paths.data_dir = wide(data_dir.c_str());
    paths.kind = rr ? Kind::RayReconstruction : Kind::SuperResolution;
    std::string error;
    if (!network.load(paths, error)) {
        printf("[FAIL] load: %s\n", error.c_str());
        return 1;
    }
    printf("[ OK ] network loaded on %s, %u native kernels\n", network.device_name().c_str(), network.native_kernels());

    UpscalerCreate create;
    create.render_width = rw;
    create.render_height = rh;
    create.output_width = ow;
    create.output_height = oh;
    if (!network.create_upscaler(create, error)) {
        printf("[FAIL] create: %s\n", error.c_str());
        return 1;
    }

    SharedImage color, depth, motion, output, diffuse, specular, normals, hit;
    bool ok = make_image(gpu, color, rw, rh, DXGI_FORMAT_R16G16B16A16_FLOAT, false) &&
              make_image(gpu, depth, rw, rh, DXGI_FORMAT_R32_FLOAT, false) &&
              make_image(gpu, motion, rw, rh, DXGI_FORMAT_R16G16_FLOAT, false) &&
              make_image(gpu, output, ow, oh, DXGI_FORMAT_R16G16B16A16_FLOAT, true);
    if (rr)
        ok = ok && make_image(gpu, diffuse, rw, rh, DXGI_FORMAT_R16G16B16A16_FLOAT, false) &&
             make_image(gpu, specular, rw, rh, DXGI_FORMAT_R16G16B16A16_FLOAT, false) &&
             make_image(gpu, normals, rw, rh, DXGI_FORMAT_R16G16B16A16_FLOAT, false) &&
             make_image(gpu, hit, rw, rh, DXGI_FORMAT_R16_FLOAT, false);
    if (!ok) {
        printf("[FAIL] could not create the shared textures\n");
        return 1;
    }
    SharedImage *all[] = {&color, &depth, &motion, &output, &diffuse, &specular, &normals, &hit};
    for (SharedImage *image : all) {
        if (!image->resource) continue;
        if (!network.import(*image, error)) {
            printf("[FAIL] import: %s\n", error.c_str());
            return 1;
        }
    }

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
                const float r = in_sq ? 1.0f : (float)x / rw + tex, g = in_sq ? 1.0f : (float)y / rh + tex,
                            b = in_sq ? 1.0f : 0.4f + tex;
                col[((size_t)y * rw + x) * 4 + 0] = to_half(r);
                col[((size_t)y * rw + x) * 4 + 1] = to_half(g);
                col[((size_t)y * rw + x) * 4 + 2] = to_half(b);
                col[((size_t)y * rw + x) * 4 + 3] = to_half(1.0f);
                dep[(size_t)y * rw + x] = in_sq ? 0.8f : 0.2f;
                mv[((size_t)y * rw + x) * 2] = to_half(in_sq && frame ? -8.0f : 0.0f);
            }
        if (!upload(gpu, color, col.data(), 8) || !upload(gpu, depth, dep.data(), 4) || !upload(gpu, motion, mv.data(), 4) ||
            (rr && (!upload(gpu, diffuse, alb.data(), 8) || !upload(gpu, specular, alb.data(), 8) ||
                    !upload(gpu, normals, nrm.data(), 8) || !upload(gpu, hit, hd.data(), 2)))) {
            printf("[FAIL] upload\n");
            return 1;
        }
        UpscalerDesc desc;
        desc.color = &color;
        desc.depth = &depth;
        desc.motion = &motion;
        desc.output = &output;
        if (rr) {
            desc.diffuse_albedo = &diffuse;
            desc.specular_albedo = &specular;
            desc.normals = &normals;
            desc.hit_distance = &hit;
            desc.world_to_view = world_to_view;
            desc.view_to_clip = view_to_clip;
        }
        desc.reset = frame == 0;
        const auto start = std::chrono::steady_clock::now();
        if (!network.evaluate_upscaler(desc, error)) {
            printf("[FAIL] evaluate (frame %d): %s\n", frame, error.c_str());
            return 1;
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        printf("frame %2d: %.2f ms\n", frame, ms);
        if (frame >= 3) total += ms, ++timed;
    }
    if (timed) printf("average from frame 3: %.2f ms per upscaled frame\n", total / timed);

    if (!out_prefix.empty()) {
        std::vector<uint8_t> pixels;
        if (!download(gpu, output, pixels, 8)) {
            printf("[FAIL] download\n");
            return 1;
        }
        char path[512];
        snprintf(path, sizeof path, "%s.raw", out_prefix.c_str());
        if (FILE *f = fopen(path, "wb")) {
            fwrite(pixels.data(), 1, pixels.size(), f);
            fclose(f);
        }
        snprintf(path, sizeof path, "%s.ppm", out_prefix.c_str());
        if (FILE *f = fopen(path, "wb")) {
            fprintf(f, "P6\n%u %u\n255\n", ow, oh);
            const uint16_t *h = (const uint16_t *)pixels.data();
            std::vector<uint8_t> row((size_t)ow * 3);
            for (uint32_t y = 0; y < oh; ++y) {
                for (uint32_t x = 0; x < ow; ++x)
                    for (int k = 0; k < 3; ++k) {
                        float v = from_half(h[((size_t)y * ow + x) * 4 + k]);
                        v = v < 0 ? 0 : (v > 1 ? 1 : v);
                        v = v <= 0.0031308f ? 12.92f * v : 1.055f * powf(v, 1.0f / 2.4f) - 0.055f;
                        row[(size_t)x * 3 + k] = (uint8_t)(v * 255.0f + 0.5f);
                    }
                fwrite(row.data(), 1, row.size(), f);
            }
            fclose(f);
        }
        printf("wrote %s.raw and .ppm\n", out_prefix.c_str());
    }
    network.shutdown();
    TerminateProcess(GetCurrentProcess(), 0);   // the libraries keep threads that do not survive a normal exit
    return 0;
}
