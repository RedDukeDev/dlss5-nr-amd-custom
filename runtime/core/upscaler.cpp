// The upscalers (super resolution, ray reconstruction) as the runtime's C interface: see the end of
// dlss5nr.h for what a host does and why.
//
// The game's textures cannot be given to the network as they are: they are in formats and states the
// network's CUDA side knows nothing of, and in memory that was never shared. So each frame this file
//
//   1. converts the game's inputs, on the queue, into textures of its own in the formats the network
//      reads (half float colour, float depth, half motion), which are shared with CUDA;
//   2. makes the network wait for that to be done, runs it, and makes the queue wait for it: on the GPU,
//      through a shared fence the CUDA side waits on and signals, or, where that is not possible, on the
//      CPU;
//   3. converts the network's output into the game's output texture, on the queue.
//
// The conversions are one small compute shader in three variants, compiled when the upscaler is created.

#define DLSS5NR_BUILD
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <d3dcompiler.h>

#include <atomic>
#include <chrono>
#include <algorithm>
#include <cstddef>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

#include "../include/dlss5nr.h"
#include "network.h"
#include "convert.h"
#include "passes.h"

#pragma comment(lib, "d3dcompiler.lib")

using namespace dlss5nr;

namespace {

std::atomic<uint32_t> g_loader_thread{0};
thread_local std::string t_error;
dlss5nr_upscaler *g_upscaler = nullptr;
std::mutex g_log_lock;
dlss5nr_log_fn g_host_log = nullptr;
void *g_host_log_user = nullptr;

template <class T> void release(T *&p) {
    if (p) p->Release();
    p = nullptr;
}

void log_to_host(int level, const char *fmt, ...) {
    char line[1200];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(line, sizeof line, fmt, ap);
    va_end(ap);
    std::lock_guard<std::mutex> guard(g_log_lock);
    if (g_host_log) g_host_log(g_host_log_user, level, line);
}

void network_log(int level, const char *message) { log_to_host(level, "%s", message); }

constexpr uint32_t kRead = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
constexpr uint32_t kWrite = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
constexpr uint32_t kCommon = D3D12_RESOURCE_STATE_COMMON;

bool make_image(ID3D12Device *device, SharedImage &image, uint32_t w, uint32_t h, DXGI_FORMAT format, bool writable) {
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
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_SHARED, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                               IID_PPV_ARGS(&image.resource))))
        return false;
    HANDLE handle = nullptr;
    if (FAILED(device->CreateSharedHandle(image.resource, nullptr, GENERIC_ALL, nullptr, &handle))) return false;
    image.nt_handle = handle;
    image.width = w;
    image.height = h;
    image.dxgi_format = format;
    image.writable = writable;
    return true;
}

} // namespace

struct dlss5nr_upscaler {
    ID3D12Device *device = nullptr;
    Network net;
    Kind kind = Kind::SuperResolution;
    UpscalerCreate create;
    SharedImage color, depth, motion, output, diffuse, specular, normals, roughness, hit, exposure, bias, transparency;
    Converter converter;

    // A ring of command lists: the one for a frame is reused only once the GPU is past the frame that
    // used it three frames ago.
    static constexpr int kRing = 3;
    ID3D12CommandAllocator *allocators[kRing][2] = {};
    ID3D12GraphicsCommandList *lists[kRing][2] = {};
    uint64_t ring_value[kRing] = {};
    uint32_t ring_next = 0;

    ID3D12Fence *fence = nullptr;
    HANDLE fence_handle = nullptr;
    void *semaphore = nullptr;
    uint64_t value = 0;
    HANDLE event = nullptr;
    bool gpu_sync = false;

    float evaluation_ms = 0.0f;
    uint32_t native_kernels = 0;
    std::string message;

    SharedImage *images[12] = {&color, &depth, &motion, &output, &diffuse, &specular, &normals, &roughness, &hit,
                               &exposure, &bias, &transparency};

    void wait_cpu(uint64_t at_least) {
        if (fence->GetCompletedValue() >= at_least) return;
        fence->SetEventOnCompletion(at_least, event);
        WaitForSingleObject(event, 20000);
    }

    void teardown() {
        if (fence && value) wait_cpu(value);
        std::string ignored;
        if (net.make_current(ignored)) {
            for (SharedImage *image : images)
                if (image->resource) net.release(*image);
            if (semaphore) net.destroy_fence(semaphore);
        }
        for (SharedImage *image : images) {
            if (image->nt_handle) CloseHandle((HANDLE)image->nt_handle);
            release(image->resource);
            *image = SharedImage{};
        }
        for (auto &row : allocators)
            for (auto &a : row) release(a);
        for (auto &row : lists)
            for (auto &l : row) release(l);
        converter.shutdown();
        if (fence_handle) CloseHandle(fence_handle);
        release(fence);
        if (event) CloseHandle(event);
        net.shutdown();
        release(device);
    }
};

namespace {

bool build(dlss5nr_upscaler &u, const dlss5nr_upscaler_create_info &info, std::string &error) {
    const bool rr = info.kind == DLSS5NR_UPSCALER_RAY_RECONSTRUCTION;
    u.kind = rr ? Kind::RayReconstruction : Kind::SuperResolution;
    u.device = info.device;
    u.device->AddRef();

    NetworkPaths paths;
    wchar_t self[MAX_PATH] = {};
    std::wstring own_dir;
    if (!info.runtime_dir) {
        HMODULE module = nullptr;
        GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCWSTR)&build, &module);
        GetModuleFileNameW(module, self, MAX_PATH);
        own_dir = self;
        own_dir = own_dir.substr(0, own_dir.find_last_of(L"\\/"));
    }
    paths.runtime_dir = info.runtime_dir ? info.runtime_dir : own_dir;
    paths.snippet = info.snippet_path ? info.snippet_path
                                      : paths.runtime_dir + (rr ? L"\\nvngx_dlssd.dll" : L"\\nvngx_dlss.dll");
    paths.data_dir = info.data_dir ? info.data_dir : paths.runtime_dir + L"\\cache";
    paths.kind = u.kind;
    // Each network has an NGX host of its own, so that this one can run beside the neural rendering.
    paths.host_dir = rr ? L"ngx_rr" : L"ngx_sr";
    CreateDirectoryW(paths.data_dir.c_str(), nullptr);
    set_network_log(&network_log);

    if (!u.net.load(paths, error)) return false;
    u.native_kernels = u.net.native_kernels();
    u.create.render_width = info.render_width;
    u.create.render_height = info.render_height;
    u.create.output_width = info.output_width;
    u.create.output_height = info.output_height;
    u.create.flags = info.flags;
    u.create.quality = info.quality;
    u.create.preset = info.preset;
    if (!u.net.create_upscaler(u.create, error)) return false;

    // The network accepts a render region from half the output to the whole of it (dynamic resolution),
    // whatever size it was created for, so what it reads is kept at the largest.
    const uint32_t rw = std::max(info.render_width, info.output_width), rh = std::max(info.render_height, info.output_height);
    // Motion vectors the game made at the output resolution are kept at it: the network reads them that way.
    const bool motion_low_res = (info.flags & DLSS5NR_UPSCALER_MOTION_LOW_RES) != 0;
    const uint32_t motion_w = motion_low_res ? rw : info.output_width, motion_h = motion_low_res ? rh : info.output_height;
    bool ok = make_image(u.device, u.color, rw, rh, DXGI_FORMAT_R16G16B16A16_FLOAT, false) &&
              make_image(u.device, u.depth, rw, rh, DXGI_FORMAT_R32_FLOAT, false) &&
              make_image(u.device, u.motion, motion_w, motion_h, DXGI_FORMAT_R16G16_FLOAT, false) &&
              make_image(u.device, u.output, info.output_width, info.output_height, DXGI_FORMAT_R16G16B16A16_FLOAT, true);
    // The optional inputs every upscaler may be given: the exposure (one texel) and the two masks.
    ok = ok && make_image(u.device, u.exposure, 1, 1, DXGI_FORMAT_R32_FLOAT, false) &&
         make_image(u.device, u.bias, rw, rh, DXGI_FORMAT_R16_FLOAT, false) &&
         make_image(u.device, u.transparency, rw, rh, DXGI_FORMAT_R16_FLOAT, false);
    if (rr)
        ok = ok && make_image(u.device, u.diffuse, rw, rh, DXGI_FORMAT_R16G16B16A16_FLOAT, false) &&
             make_image(u.device, u.specular, rw, rh, DXGI_FORMAT_R16G16B16A16_FLOAT, false) &&
             make_image(u.device, u.normals, rw, rh, DXGI_FORMAT_R16G16B16A16_FLOAT, false) &&
             make_image(u.device, u.roughness, rw, rh, DXGI_FORMAT_R16_FLOAT, false) &&
             make_image(u.device, u.hit, rw, rh, DXGI_FORMAT_R16_FLOAT, false);
    if (!ok) {
        error = "could not create the shared textures";
        return false;
    }
    for (SharedImage *image : u.images)
        if (image->resource && !u.net.import(*image, error)) return false;

    if (!u.converter.init(u.device, error)) return false;
    for (int i = 0; i < dlss5nr_upscaler::kRing; ++i)
        for (int j = 0; j < 2; ++j)
            if (FAILED(u.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&u.allocators[i][j]))) ||
                FAILED(u.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, u.allocators[i][j], nullptr,
                                                   IID_PPV_ARGS(&u.lists[i][j])))) {
                error = "could not create the conversion command lists";
                return false;
            } else {
                u.lists[i][j]->Close();
            }
    if (FAILED(u.device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&u.fence)))) {
        error = "could not create the fence";
        return false;
    }
    u.event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    // The GPU waits for the GPU where it can. DLSS5NR_UPSCALER_CPU_WAIT=1 forces the CPU to.
    char forced[8] = {};
    const bool cpu_forced = GetEnvironmentVariableA("DLSS5NR_UPSCALER_CPU_WAIT", forced, sizeof forced) && forced[0] == '1';
    if (!cpu_forced && SUCCEEDED(u.device->CreateSharedHandle(u.fence, nullptr, GENERIC_ALL, nullptr, &u.fence_handle))) {
        std::string why;
        u.gpu_sync = u.net.import_fence(u.fence_handle, &u.semaphore, why);
        if (!u.gpu_sync) log_to_host(1, "the GPU cannot wait for the GPU here (%s): the CPU waits", why.c_str());
    }
    return true;
}

bool run(dlss5nr_upscaler &u, ID3D12CommandQueue *queue, const dlss5nr_upscaler_frame &f, std::string &error) {
    const bool rr = u.kind == Kind::RayReconstruction;
    if (!f.color.resource || !f.depth.resource || !f.motion.resource || !f.output.resource) {
        error = "colour, depth, motion vectors and output are all needed";
        return false;
    }
    // The camera matrices are optional: a game may leave them out (Unreal's DLSS passes null pointers) and the
    // network works from the G-buffer without them.
    if (rr && (!f.diffuse_albedo.resource || !f.specular_albedo.resource || !f.normals.resource)) {
        error = "ray reconstruction needs diffuse and specular albedo and normals";
        return false;
    }
    const uint32_t slot = u.ring_next++ % dlss5nr_upscaler::kRing;
    u.wait_cpu(u.ring_value[slot]);   // the lists of this slot were last used three frames ago

    // 1. the game's inputs into the network's textures
    ID3D12GraphicsCommandList *in = u.lists[slot][0];
    u.allocators[slot][0]->Reset();
    in->Reset(u.allocators[slot][0], nullptr);
    // The region the game rendered this frame (dynamic resolution renders less than the size the network was
    // made for) is copied to the corner of the network's own textures, whatever corner of the game's it was
    // in: the network is then told a region that starts at 0 in every texture.
    const uint32_t max_w = std::max(u.create.render_width, u.create.output_width),
                   max_h = std::max(u.create.render_height, u.create.output_height);
    const uint32_t region_w = f.render_width ? std::min(f.render_width, max_w) : u.create.render_width;
    const uint32_t region_h = f.render_height ? std::min(f.render_height, max_h) : u.create.render_height;
    struct Input {
        const dlss5nr_resource *game;
        SharedImage *image;
        Variant variant;
        dlss5nr_point origin;
        uint32_t width, height;
    };
    Input inputs[12];
    int count = 0;
    auto add = [&](const dlss5nr_resource &game, SharedImage &image, Variant variant, dlss5nr_point origin,
                   uint32_t w = 0, uint32_t h = 0) {
        inputs[count++] = {&game, &image, variant, origin, w ? w : region_w, h ? h : region_h};
    };
    add(f.color, u.color, ToFloat4, f.color_origin);
    add(f.depth, u.depth, ToFloat, f.depth_origin);
    if (u.create.flags & DLSS5NR_UPSCALER_MOTION_LOW_RES)
        add(f.motion, u.motion, ToFloat2, f.motion_origin);
    else
        add(f.motion, u.motion, ToFloat2, f.motion_origin, u.create.output_width, u.create.output_height);
    SharedImage *hit_image = nullptr, *exposure_image = nullptr, *bias_image = nullptr, *transparency_image = nullptr;
    if (rr) {
        add(f.diffuse_albedo, u.diffuse, ToFloat4, f.guide_origin);
        add(f.specular_albedo, u.specular, ToFloat4, f.guide_origin);
        add(f.normals, u.normals, ToFloat4, f.guide_origin);
        if (f.roughness.resource) add(f.roughness, u.roughness, ToFloat, f.guide_origin);
        if (f.hit_distance.resource) {
            add(f.hit_distance, u.hit, ToFloat, f.guide_origin);
            hit_image = &u.hit;
        }
    }
    // One texel is all the exposure is.
    if (f.exposure.resource) {
        add(f.exposure, u.exposure, ToFloat, dlss5nr_point{0, 0}, 1, 1);
        exposure_image = &u.exposure;
    }
    if (f.reactive_mask.resource) {
        add(f.reactive_mask, u.bias, ToFloat, f.mask_origin);
        bias_image = &u.bias;
    }
    if (f.transparency_mask.resource) {
        add(f.transparency_mask, u.transparency, ToFloat, f.mask_origin);
        transparency_image = &u.transparency;
    }
    auto convert_in = [&](const Input &input) {
        transition(in, input.game->resource, input.game->state, kRead);
        transition(in, input.image->resource, kCommon, kWrite);
        const bool ok = u.converter.dispatch(in, input.variant, input.game->resource, input.image->resource,
                                             input.origin, dlss5nr_point{0, 0}, input.width, input.height, error);
        transition(in, input.image->resource, kWrite, kCommon);
        transition(in, input.game->resource, kRead, input.game->state);
        return ok;
    };
    bool ok = true;
    for (int i = 0; i < count && ok; ++i) ok = convert_in(inputs[i]);
    in->Close();
    if (!ok) return false;
    ID3D12CommandList *first[] = {in};
    queue->ExecuteCommandLists(1, first);

    // 2. the network, in order with the GPU's work
    const uint64_t before = ++u.value;
    queue->Signal(u.fence, before);
    UpscalerDesc desc;
    desc.color = &u.color;
    desc.depth = &u.depth;
    desc.motion = &u.motion;
    desc.output = &u.output;
    if (rr) {
        desc.diffuse_albedo = &u.diffuse;
        desc.specular_albedo = &u.specular;
        desc.normals = &u.normals;
        desc.roughness = f.roughness.resource ? &u.roughness : nullptr;
        desc.roughness_packed_in_normals = !f.roughness.resource;
        desc.hit_distance = hit_image;
        desc.world_to_view = f.world_to_view;
        desc.view_to_clip = f.view_to_clip;
    }
    desc.exposure = exposure_image;
    desc.bias_mask = bias_image;
    desc.transparency_mask = transparency_image;
    desc.render_width = region_w;
    desc.render_height = region_h;
    desc.jitter_x = f.jitter_x;
    desc.jitter_y = f.jitter_y;
    desc.motion_scale_x = f.motion_scale_x != 0.0f ? f.motion_scale_x : 1.0f;
    desc.motion_scale_y = f.motion_scale_y != 0.0f ? f.motion_scale_y : 1.0f;
    desc.pre_exposure = f.pre_exposure != 0.0f ? f.pre_exposure : 1.0f;
    desc.exposure_scale = f.exposure_scale != 0.0f ? f.exposure_scale : 1.0f;
    desc.sharpness = f.sharpness;
    desc.reset = f.reset != 0;
    if (!u.net.make_current(error)) return false;
    u.net.set_thread_high_priority(true);   // this thread's frame is the one the game is waiting for
    // Read on every frame: the frame generation may start after the upscaler did (see below).
    char cpu_wait[8] = {};
    const bool wait_on_cpu = GetEnvironmentVariableA("DLSS5NR_UPSCALER_CPU_WAIT", cpu_wait, sizeof cpu_wait) && cpu_wait[0] == '1';
    if (u.gpu_sync && !wait_on_cpu) {
        const uint64_t after = ++u.value;
        if (!u.net.wait_fence(u.semaphore, before, error) || !u.net.evaluate_upscaler(desc, error, false) ||
            !u.net.signal_fence(u.semaphore, after, error))
            return false;
        queue->Wait(u.fence, after);
    } else {
        u.wait_cpu(before);
        if (!u.net.evaluate_upscaler(desc, error, true)) return false;
    }

    // 3. the network's output into the game's
    ID3D12GraphicsCommandList *out = u.lists[slot][1];
    u.allocators[slot][1]->Reset();
    out->Reset(u.allocators[slot][1], nullptr);
    transition(out, f.output.resource, f.output.state, kWrite);
    transition(out, u.output.resource, kCommon, kRead);
    ok = u.converter.dispatch(out, ToFloat4, u.output.resource, f.output.resource, dlss5nr_point{0, 0}, f.output_origin,
                              u.create.output_width, u.create.output_height, error);
    transition(out, u.output.resource, kRead, kCommon);
    transition(out, f.output.resource, kWrite, f.output.state);
    out->Close();
    if (!ok) return false;
    ID3D12CommandList *second[] = {out};
    queue->ExecuteCommandLists(1, second);
    u.ring_value[slot] = ++u.value;
    queue->Signal(u.fence, u.ring_value[slot]);
    return true;
}

} // namespace

DLSS5NR_API int dlss5nr_upscaler_create(const dlss5nr_upscaler_create_info *info, dlss5nr_upscaler **out) {
    if (out) *out = nullptr;
    if (!info || !out || !info->device || info->struct_size < sizeof(dlss5nr_upscaler_create_info)) {
        t_error = "bad arguments";
        return -1;
    }
    if (g_upscaler) {
        t_error = "one upscaler per process";
        return -2;
    }
    {
        std::lock_guard<std::mutex> guard(g_log_lock);
        g_host_log = info->log;
        g_host_log_user = info->log_user;
    }
    auto *u = new dlss5nr_upscaler;
    std::string error;
    bool built = false;
    // The libraries load on a thread of its own, whose id a host that filters library loads can ask for.
    std::thread loader([&] {
        g_loader_thread = GetCurrentThreadId();
        built = build(*u, *info, error);
    });
    // The id is published before the thread does anything a host could intercept.
    loader.join();
    g_loader_thread = 0;
    if (!built) {
        t_error = error;
        log_to_host(0, "the upscaler could not be created: %s", error.c_str());
        u->message = error;
        u->teardown();
        delete u;
        return -3;
    }
    g_upscaler = u;
    *out = u;
    log_to_host(2, "upscaler created (%s synchronisation, %u native kernels)", u->gpu_sync ? "GPU" : "CPU",
                u->native_kernels);
    return 0;
}

DLSS5NR_API void dlss5nr_upscaler_destroy(dlss5nr_upscaler *upscaler) {
    if (!upscaler) return;
    upscaler->teardown();
    if (g_upscaler == upscaler) g_upscaler = nullptr;
    delete upscaler;
}

DLSS5NR_API int dlss5nr_upscaler_evaluate(dlss5nr_upscaler *upscaler, ID3D12CommandQueue *queue,
                                          const dlss5nr_upscaler_frame *frame) {
    // The first version of the frame ends at view_to_clip: anything a host built for it did not set is zero.
    constexpr size_t kFirstVersion = offsetof(dlss5nr_upscaler_frame, view_to_clip) + sizeof(const float *);
    if (!upscaler || !queue || !frame || frame->struct_size < kFirstVersion) {
        t_error = "bad arguments";
        return -1;
    }
    dlss5nr_upscaler_frame whole{};
    memcpy(&whole, frame, frame->struct_size < sizeof whole ? frame->struct_size : sizeof whole);
    whole.struct_size = sizeof whole;
    const auto began = std::chrono::steady_clock::now();
    std::string error;
    if (!run(*upscaler, queue, whole, error)) {
        t_error = error;
        log_to_host(0, "upscaler evaluation failed: %s", error.c_str());
        return -3;
    }
    upscaler->evaluation_ms = (float)std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - began).count();
    return 0;
}

DLSS5NR_API void dlss5nr_upscaler_get_status(dlss5nr_upscaler *upscaler, dlss5nr_upscaler_status *status) {
    if (!status) return;
    memset(status, 0, sizeof *status);
    status->struct_size = sizeof *status;
    if (!upscaler) return;
    snprintf(status->message, sizeof status->message, "%s", upscaler->message.c_str());
    status->native_kernels = upscaler->native_kernels;
    status->evaluation_ms = upscaler->evaluation_ms;
    status->gpu_synchronised = upscaler->gpu_sync ? 1 : 0;
}

DLSS5NR_API const char *dlss5nr_upscaler_last_error(void) { return t_error.c_str(); }

DLSS5NR_API uint32_t dlss5nr_upscaler_loader_thread(void) { return g_loader_thread.load(); }
