// The runtime: slots, the network's worker thread, and the C interface.
//
// How a frame moves (docs/DESIGN.md has the why):
//
//   game thread, dlss5nr_process    exposure, displacement maps moved on by a
//                                   frame, capture into a free slot, a fresh
//                                   result taken, compose
//   game thread, dlss5nr_present    signals the fence that covers the capture
//   worker thread                   waits for that fence, runs the network,
//                                   marks the slot ready
//
// Why no GPU work ever waits: the capture writes a slot only when it is free,
// and the worker touches a slot only after the fence of the frame that
// captured it -- which the game's queue reaches after every frame before it,
// including those that read the slot. A ready slot is read once, into the
// residual, and is free again from then on.
//
// Why results do not lag behind: the network answers for a frame captured
// several frames ago. Each capture keeps a displacement map from the current
// frame back to its own, moved on every frame along the game's motion
// vectors, and a result is composed through the map of its capture -- so it
// lands where its pixels are now, not where they were.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#define DLSS5NR_BUILD
#include "../include/dlss5nr.h"
#include "network.h"
#include "passes.h"

using namespace dlss5nr;

namespace {

constexpr int kSlots = 3;
// Consecutive failed evaluations before the network is given up on.
constexpr int kMaxFailures = 3;

template <typename T>
void release(T *&object) {
    if (object) object->Release();
    object = nullptr;
}

std::wstring module_dir() {
    HMODULE self = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       (LPCWSTR)&module_dir, &self);
    wchar_t path[MAX_PATH];
    const DWORD n = GetModuleFileNameW(self, path, MAX_PATH);
    std::wstring dir(path, n);
    const size_t slash = dir.find_last_of(L"\\/");
    return slash == std::wstring::npos ? L"." : dir.substr(0, slash);
}

void transition(ID3D12GraphicsCommandList *cmd, ID3D12Resource *resource, D3D12_RESOURCE_STATES from,
                D3D12_RESOURCE_STATES to) {
    if (!resource || from == to) return;
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = resource;
    b.Transition.StateBefore = from;
    b.Transition.StateAfter = to;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    cmd->ResourceBarrier(1, &b);
}

void uav_barrier(ID3D12GraphicsCommandList *cmd, ID3D12Resource *resource) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    b.UAV.pResource = resource;
    cmd->ResourceBarrier(1, &b);
}

// A game resource used as a shader resource: left alone when its state
// already allows reading, transitioned there and back otherwise.
D3D12_RESOURCE_STATES readable(D3D12_RESOURCE_STATES state) {
    return (state & D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE) ? state
                                                                    : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
}

ID3D12Resource *make_texture(ID3D12Device *device, uint32_t width, uint32_t height, DXGI_FORMAT format,
                             bool shared, D3D12_RESOURCE_STATES state) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ID3D12Resource *out = nullptr;
    device->CreateCommittedResource(&heap, shared ? D3D12_HEAP_FLAG_SHARED : D3D12_HEAP_FLAG_NONE, &desc, state,
                                    nullptr, IID_PPV_ARGS(&out));
    return out;
}

// Under Wine a shared handle is not something ROCm can import (what
// tests/interop_probe.cpp found: the handle is not a file descriptor), so the
// textures are not shared and the network copies them through host memory
// instead. DLSS5NR_COPY=1 forces that on Windows, =0 forbids it.
bool g_copy_interop = false;

bool copy_interop_wanted() {
    char value[8];
    const DWORD n = GetEnvironmentVariableA("DLSS5NR_COPY", value, sizeof value);
    if (n && n < sizeof value) return value[0] == '1';
    HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    return ntdll && GetProcAddress(ntdll, "wine_get_version");
}

bool make_shared(ID3D12Device *device, SharedImage &image, uint32_t width, uint32_t height, DXGI_FORMAT format,
                 bool writable) {
    image.resource = make_texture(device, width, height, format, !g_copy_interop, D3D12_RESOURCE_STATE_COMMON);
    if (!image.resource) return false;
    if (!g_copy_interop) {
        HANDLE handle = nullptr;
        if (FAILED(device->CreateSharedHandle(image.resource, nullptr, GENERIC_ALL, nullptr, &handle))) return false;
        image.nt_handle = handle;
    }
    image.width = width;
    image.height = height;
    image.dxgi_format = format;
    image.writable = writable;
    return true;
}

void destroy_shared(SharedImage &image) {
    if (image.nt_handle) CloseHandle((HANDLE)image.nt_handle);
    release(image.resource);
    image = SharedImage{};
}

struct Slot {
    enum State { Free, Captured, Evaluating, Ready };
    SharedImage color, depth, motion, output;
    State state = Free;
    uint64_t frame = 0;          // the game frame it was captured in
    uint64_t fence = 0;          // the present fence value covering that capture; 0 until presented
    uint64_t sequence = 0;       // order of readiness
    // The frame of the capture before it, which its motion vectors lead back
    // to; ~0 when there is none to lead back to.
    uint64_t previous_frame = ~0ull;
    bool reset = false;
    bool depth_inverted = false;
};

// The slots' images at one size. Letting go of a set and importing another
// into CUDA leaks: measured with the size changed over and over, the device
// memory of every set that was let go of never came back, though both the
// CUDA and the Direct3D side reported it released -- 265 MB at 2560 x 1440
// for the three slots, until the game ran out of memory. So a set that is let
// go of is parked, imports and all, and the next time that size is wanted it
// is taken out again. Two are parked: a game goes between a few sizes, and a
// size that keeps changing is not followed at all (see where the size is
// worked out).
constexpr size_t kParkedSets = 2;
struct SlotSet {
    uint32_t width = 0, height = 0;
    uint64_t parked_at = 0;   // the game frame, to find the oldest
    SharedImage image[kSlots][4];
};

// A displacement map, from each pixel of the current frame to where it was in
// an earlier one (shaders.h, "track"), moved on by one frame at a time: two
// textures, read one and write the other. Game thread only.
struct Flow {
    ID3D12Resource *map[2] = {};
    D3D12_RESOURCE_STATES state[2] = {};
    int current = 0;
    bool identity = true;        // the earlier frame is this one: nothing has moved yet
    bool valid = false;          // it describes something worth tracking
};

// Retired D3D12 objects, released once the GPU is past the frame that last
// used them.
struct Retired {
    ID3D12Resource *resource;
    void *nt_handle;
    uint64_t fence;
};

} // namespace

struct dlss5nr_context {
    // Fixed at creation.
    ID3D12Device *device = nullptr;
    NetworkPaths paths;
    dlss5nr_log_fn host_log = nullptr;
    void *host_log_user = nullptr;
    FILE *log_file = nullptr;
    std::mutex log_lock;

    Passes passes;
    ID3D12Resource *exposure_buffer = nullptr;
    ID3D12Fence *fence = nullptr;     // signalled at present, read by the worker
    uint64_t fence_value = 0;

    // Game thread only.
    dlss5nr_settings settings{};
    uint64_t frame = 0;
    int pending_slot = -1;            // captured this frame, waiting for present
    // The result being composed: the network's change, in the pixels of the
    // frame it was captured in, and the map from now back to that frame.
    ID3D12Resource *residual = nullptr;
    D3D12_RESOURCE_STATES residual_state = {};
    // The network's gain, output over input, each averaged over 16 x 16
    // pixels, for what the network never saw.
    ID3D12Resource *residual_soft = nullptr;
    D3D12_RESOURCE_STATES residual_soft_state = {};
    uint32_t residual_width = 0, residual_height = 0;
    bool residual_valid = false;
    uint32_t residual_age = 0;
    uint64_t applied_sequence = 0;
    uint64_t applied_frame = ~0ull;   // the frame the composed result was captured in
    Flow applied_flow;
    // The result before it, and its map: what the newer capture did not see,
    // this one may have.
    ID3D12Resource *residual_previous = nullptr;
    D3D12_RESOURCE_STATES residual_previous_state = {};
    Flow previous_flow;
    // One per slot, from the frame it was captured in. The one of the capture
    // the network last took becomes the next capture's motion vectors.
    Flow slot_flow[kSlots];
    // Tracked only when nothing else is, so that the depth history is kept.
    Flow idle_flow;
    bool guided = true;               // the last frame came with depth and motion vectors
    // The previous frame's depth at the network's size, for the occlusion
    // test while tracking.
    ID3D12Resource *depth_history[2] = {};
    D3D12_RESOURCE_STATES depth_history_state[2] = {};
    int depth_current = 0;
    bool depth_history_valid = false;
    std::vector<Retired> retired;

    // Shared with the worker, under lock.
    std::mutex lock;
    std::condition_variable wake;
    Slot slots[kSlots];
    uint32_t slot_width = 0, slot_height = 0;
    // The size the network was last asked for when it was not the one it has,
    // and since when.
    uint32_t wanted_width = 0, wanted_height = 0;
    std::chrono::steady_clock::time_point wanted_since;
    bool slots_imported_size_changed = false;
    std::vector<SlotSet> slot_cache;   // parked, game thread only
    std::vector<SlotSet> evicting;     // handed to the worker to release the CUDA side of, under lock
    std::vector<SlotSet> evicted;      // released by the worker, back for the game thread to retire, under lock
    bool resize_requested = false;     // game thread wants new slots
    bool resize_released = false;      // worker has let go of the old ones
    uint64_t sequence = 0;
    NetworkControls controls;
    bool stop = false;

    // Status, written by the worker, read anywhere.
    std::atomic<int> state{DLSS5NR_STATE_STARTING};
    std::mutex status_lock;
    std::string message;
    std::atomic<float> evaluation_ms{0.0f};
    std::atomic<float> evaluations_per_second{0.0f};
    std::atomic<uint32_t> native_kernels{0};
    std::atomic<uint32_t> result_age{0};
    std::atomic<uint32_t> result_latency{0};
    std::atomic<uint32_t> results{0};
    std::atomic<uint32_t> worker_thread_id{0};
    // Set by the host's first call after dlss5nr_create. Nothing is loaded
    // before it: a host that filters library loads by thread has by then had
    // the chance to learn which thread to let through.
    std::atomic<bool> host_ready{false};

    void mark_host_ready() {
        if (host_ready.exchange(true)) return;
        std::lock_guard<std::mutex> guard(lock);
        wake.notify_all();
    }

    Network network;
    std::thread worker;

    void log(int level, const char *fmt, ...) {
        char line[1200];
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(line, sizeof line, fmt, ap);
        va_end(ap);
        std::lock_guard<std::mutex> guard(log_lock);
        if (log_file) {
            static const char *const kLevel[] = {"error", "warning", "info", "debug"};
            fprintf(log_file, "[%s] %s\n", kLevel[level < 0 ? 0 : level > 3 ? 3 : level], line);
            fflush(log_file);
        }
        if (host_log) host_log(host_log_user, level, line);
    }

    void set_message(const std::string &text) {
        std::lock_guard<std::mutex> guard(status_lock);
        message = text;
    }

    void run_worker();

    // Failed, the worker waits here for dlss5nr_destroy instead of ending. A
    // thread that ends while the game runs has the thread-detach notification of
    // every library loaded in the process called on it, ZLUDA's and ROCm's among
    // them, with the loader lock held; in a game (Cyberpunk 2077 under Wine, with
    // a network that could not be found) that took the whole process down, with
    // nothing in any log after the thread's own error.
    void park() {
        std::unique_lock<std::mutex> guard(lock);
        wake.wait(guard, [&] { return stop; });
    }
};

namespace {
// The network's log goes through the one context there is.
dlss5nr_context *g_context = nullptr;
void network_log(int level, const char *message) {
    if (g_context) g_context->log(level, "%s", message);
}
} // namespace

void dlss5nr_context::run_worker() {
    {
        std::unique_lock<std::mutex> guard(lock);
        wake.wait(guard, [&] { return stop || host_ready.load(); });
        if (stop) return;
    }
    {
        std::lock_guard<std::mutex> guard(lock);
        if (stop) return;
    }
    set_message("loading the network");
    std::string error;
    if (!network.load(paths, error)) {
        log(0, "the network could not be loaded: %s", error.c_str());
        set_message(error);
        state = DLSS5NR_STATE_FAILED;
        park();
        return;
    }
    native_kernels = network.native_kernels();
    log(2, "network loaded on %s, %u native kernels", network.device_name().c_str(), network.native_kernels());
    set_message("running");
    state = DLSS5NR_STATE_RUNNING;

    HANDLE fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    int failures = 0;
    // The capture the network's own history holds: the one it last evaluated.
    // A capture whose motion vectors lead back to another one -- the network
    // skipped it, or failed on it -- starts the history over.
    uint64_t history_frame = ~0ull;
    auto last_finish = std::chrono::steady_clock::now();
    // The controls the feature was created with, and those asked for since.
    // A new feature is only made once they have stayed put for a while: the
    // driver is known to stop answering when features are rebuilt on every
    // frame, which a slider being dragged would otherwise do.
    NetworkControls created_controls, asked_controls;
    bool created = false;
    auto asked_at = std::chrono::steady_clock::now();
    constexpr auto kSettle = std::chrono::milliseconds(500);
    for (;;) {
        int chosen = -1;
        NetworkControls controls_now;
        {
            std::unique_lock<std::mutex> guard(lock);
            wake.wait(guard, [&] {
                if (stop || (resize_requested && !resize_released) || !evicting.empty()) return true;
                for (const Slot &s : slots)
                    if (s.state == Slot::Captured && s.fence) return true;
                return false;
            });
            if (stop) break;
            if (!evicting.empty()) {
                // Sets given up for good: only this thread has the CUDA context.
                for (SlotSet &set : evicting) {
                    for (auto &images : set.image)
                        for (SharedImage &image : images) network.release(image);
                    evicted.push_back(std::move(set));
                }
                evicting.clear();
                wake.notify_all();
                continue;
            }
            if (resize_requested && !resize_released) {
                // Nothing is evaluating: the slots' images are let go of as they
                // are, still imported, for the game thread to park.
                resize_released = true;
                wake.notify_all();
                continue;
            }
            // The newest presented capture; older ones are stale by now and
            // go back to free, which also tells the next evaluation that
            // frames were skipped.
            uint64_t newest = 0;
            for (int i = 0; i < kSlots; ++i)
                if (slots[i].state == Slot::Captured && slots[i].fence && slots[i].frame >= newest) {
                    newest = slots[i].frame;
                    chosen = i;
                }
            for (int i = 0; i < kSlots; ++i)
                if (i != chosen && slots[i].state == Slot::Captured && slots[i].fence) slots[i].state = Slot::Free;
            slots[chosen].state = Slot::Evaluating;
            controls_now = controls;
        }

        Slot &slot = slots[chosen];
        // The capture has to have run before the network reads it.
        if (fence->GetCompletedValue() < slot.fence) {
            fence->SetEventOnCompletion(slot.fence, fence_event);
            while (WaitForSingleObject(fence_event, 100) == WAIT_TIMEOUT) {
                std::lock_guard<std::mutex> guard(lock);
                if (stop) break;
            }
        }

        const auto began = std::chrono::steady_clock::now();
        if (controls_now != asked_controls) {
            asked_controls = controls_now;
            asked_at = began;
        }
        if (!created || (asked_controls != created_controls && began - asked_at >= kSettle)) {
            created_controls = asked_controls;
            created = true;
        }
        bool ok = network.import(slot.color, error) && network.import(slot.depth, error) &&
                  network.import(slot.motion, error) && network.import(slot.output, error) &&
                  network.create_feature(slot.color.width, slot.color.height, created_controls, error);
        if (ok) {
            EvaluateDesc desc;
            desc.color = &slot.color;
            desc.depth = &slot.depth;
            desc.motion = &slot.motion;
            desc.output = &slot.output;
            desc.depth_inverted = slot.depth_inverted;
            desc.reset = slot.reset || slot.previous_frame != history_frame;
            desc.controls = created_controls;
            ok = network.evaluate(desc, error);
        }
        const auto finished = std::chrono::steady_clock::now();

        std::lock_guard<std::mutex> guard(lock);
        history_frame = ok ? slot.frame : ~0ull;
        if (ok) {
            failures = 0;
            slot.state = Slot::Ready;
            slot.sequence = ++sequence;
            evaluation_ms = (float)std::chrono::duration<double, std::milli>(finished - began).count();
            const double period = std::chrono::duration<double>(finished - last_finish).count();
            last_finish = finished;
            if (period > 0.0) evaluations_per_second = (float)(1.0 / period);
        } else {
            slot.state = Slot::Free;
            log(0, "evaluation failed: %s", error.c_str());
            if (++failures >= kMaxFailures) {
                set_message("the network failed: " + error);
                state = DLSS5NR_STATE_FAILED;
                wake.notify_all();
                break;
            }
        }
        wake.notify_all();   // dlss5nr_present may be waiting for this result
    }
    CloseHandle(fence_event);
    network.shutdown();
    park();
}

extern "C" {

DLSS5NR_API void dlss5nr_default_settings(dlss5nr_settings *s) {
    if (!s) return;
    *s = dlss5nr_settings{};
    s->struct_size = sizeof *s;
    s->enabled = 1;
    s->resolution_scale = 1.0f;
    s->style = 0;
    s->intensity = 1.0f;
    s->local_tone = 1.0f;
    s->local_structure = 1.0f;
    s->skin_structure = -1.0f;  // automatic: follows the structure strength
    s->detail_strength = 1.0f;
    s->max_ratio = 2.0f;
    s->max_age = 8;
    s->encoding = DLSS5NR_ENCODING_AUTO;
    s->colour_strength = 1.0f;
    s->exposure_mode = DLSS5NR_EXPOSURE_AUTO;   // the game's own exposure showed artifacts around moving things
    s->exposure = 1.0f;
    s->follow_motion = 1;
    s->network_history = 0;
    s->wait_for_network = 0;
}

DLSS5NR_API int dlss5nr_create(const dlss5nr_create_info *info, dlss5nr_context **out) {
    if (!info || !out || !info->device || info->struct_size < sizeof(uint32_t) + sizeof(void *)) return -1;
    if (g_context) return -2;   // one per process: the network itself is a process-wide singleton

    auto *c = new dlss5nr_context;
    c->device = info->device;
    c->device->AddRef();
    c->host_log = info->log;
    c->host_log_user = info->log_user;
    c->paths.runtime_dir = info->runtime_dir ? info->runtime_dir : module_dir();
    c->paths.snippet = info->snippet_path ? info->snippet_path : c->paths.runtime_dir + L"\\nvngx_dlssnr.dll";
    c->paths.data_dir = info->data_dir ? info->data_dir : c->paths.runtime_dir + L"\\cache";
    CreateDirectoryW(c->paths.data_dir.c_str(), nullptr);
    c->log_file = _wfopen((c->paths.data_dir + L"\\dlss5nr.log").c_str(), L"w");
    g_context = c;
    set_network_log(&network_log);
    dlss5nr_default_settings(&c->settings);

    std::string error;
    if (!c->passes.init(c->device, error)) {
        c->log(0, "the compute passes could not be built: %s", error.c_str());
        dlss5nr_destroy(c);
        return -3;
    }
    // The exposure: one float and a valid flag. Committed resources start out
    // zeroed, so it starts out invalid and the first pass sets it outright.
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = 16;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    if (FAILED(c->device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                  D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                  IID_PPV_ARGS(&c->exposure_buffer))) ||
        FAILED(c->device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&c->fence)))) {
        c->log(0, "could not create the runtime's buffers");
        dlss5nr_destroy(c);
        return -4;
    }

    g_copy_interop = copy_interop_wanted();
    c->network.set_copy_mode(g_copy_interop);
    c->log(2, "runtime starting: files in %ls, network %ls, images %s", c->paths.runtime_dir.c_str(),
           c->paths.snippet.c_str(), g_copy_interop ? "copied through host memory" : "shared with CUDA");
    c->worker = std::thread([c] {
        c->worker_thread_id = GetCurrentThreadId();
        c->run_worker();
    });
    // Known before anything is loaded on it: the host may start filtering
    // library loads as soon as this returns.
    while (!c->worker_thread_id) std::this_thread::yield();
    *out = c;
    return 0;
}

DLSS5NR_API void dlss5nr_destroy(dlss5nr_context *c) {
    if (!c) return;
    {
        std::lock_guard<std::mutex> guard(c->lock);
        c->stop = true;
    }
    c->wake.notify_all();
    if (c->worker.joinable()) c->worker.join();
    for (Slot &s : c->slots) {
        destroy_shared(s.color);
        destroy_shared(s.depth);
        destroy_shared(s.motion);
        destroy_shared(s.output);
    }
    for (std::vector<SlotSet> *sets : {&c->slot_cache, &c->evicting, &c->evicted})
        for (SlotSet &set : *sets)
            for (auto &images : set.image)
                for (SharedImage &image : images) destroy_shared(image);
    for (Retired &r : c->retired) {
        if (r.nt_handle) CloseHandle((HANDLE)r.nt_handle);
        release(r.resource);
    }
    release(c->residual);
    release(c->residual_soft);
    release(c->residual_previous);
    for (ID3D12Resource *&map : c->previous_flow.map) release(map);
    for (ID3D12Resource *&map : c->applied_flow.map) release(map);
    for (ID3D12Resource *&map : c->idle_flow.map) release(map);
    for (Flow &flow : c->slot_flow)
        for (ID3D12Resource *&map : flow.map) release(map);
    for (ID3D12Resource *&depth : c->depth_history) release(depth);
    release(c->exposure_buffer);
    release(c->fence);
    c->passes.shutdown();
    if (c->log_file) fclose(c->log_file);
    release(c->device);
    if (g_context == c) g_context = nullptr;
    delete c;
}

DLSS5NR_API void dlss5nr_set_settings(dlss5nr_context *c, const dlss5nr_settings *s) {
    if (!c || !s) return;
    c->mark_host_ready();
    dlss5nr_settings next;
    dlss5nr_default_settings(&next);
    memcpy(&next, s, s->struct_size < sizeof next ? s->struct_size : sizeof next);
    next.struct_size = sizeof next;
    if (next.resolution_scale < 0.25f) next.resolution_scale = 0.25f;
    if (next.resolution_scale > 1.0f) next.resolution_scale = 1.0f;
    c->settings = next;
    std::lock_guard<std::mutex> guard(c->lock);
    c->controls.style = next.style;
    c->controls.intensity = next.intensity;
    c->controls.local_tone = next.local_tone;
    c->controls.local_structure = next.local_structure;
    c->controls.skin_structure = next.skin_structure;
}

DLSS5NR_API void dlss5nr_get_status(dlss5nr_context *c, dlss5nr_status *s) {
    if (!c || !s) return;
    c->mark_host_ready();
    const uint32_t size = s->struct_size;
    dlss5nr_status out{};
    out.struct_size = sizeof out;
    out.state = c->state;
    {
        std::lock_guard<std::mutex> guard(c->status_lock);
        snprintf(out.message, sizeof out.message, "%s", c->message.c_str());
    }
    {
        std::lock_guard<std::mutex> guard(c->lock);
        out.network_width = c->slot_width;
        out.network_height = c->slot_height;
    }
    out.evaluation_ms = c->evaluation_ms;
    out.evaluations_per_second = c->evaluations_per_second;
    out.result_age = c->result_age;
    out.result_latency = c->result_latency;
    out.native_kernels = c->native_kernels;
    out.results = c->results;
    memcpy(s, &out, size < sizeof out ? size : sizeof out);
}

DLSS5NR_API uint32_t dlss5nr_loader_thread(dlss5nr_context *c) {
    if (!c) return 0;
    c->mark_host_ready();
    return c->worker_thread_id.load();
}

DLSS5NR_API void dlss5nr_present(dlss5nr_context *c, ID3D12CommandQueue *queue) {
    if (!c || !queue) return;
    c->mark_host_ready();
    ++c->frame;
    const uint64_t value = ++c->fence_value;
    queue->Signal(c->fence, value);

    {
        std::vector<SlotSet> released;
        {
            std::lock_guard<std::mutex> guard(c->lock);
            released.swap(c->evicted);
        }
        for (SlotSet &set : released)
            for (auto &images : set.image)
                for (SharedImage &image : images)
                    if (image.resource) c->retired.push_back({image.resource, image.nt_handle, value});
    }

    // Retired resources the GPU is done with.
    const uint64_t done = c->fence->GetCompletedValue();
    for (size_t i = 0; i < c->retired.size();) {
        if (c->retired[i].fence <= done) {
            if (c->retired[i].nt_handle) CloseHandle((HANDLE)c->retired[i].nt_handle);
            release(c->retired[i].resource);
            c->retired[i] = c->retired.back();
            c->retired.pop_back();
        } else {
            ++i;
        }
    }

    if (c->pending_slot < 0) return;
    const int captured = c->pending_slot;
    {
        std::lock_guard<std::mutex> guard(c->lock);
        c->slots[captured].fence = value;
    }
    c->pending_slot = -1;
    c->wake.notify_all();

    // Waiting for the network: the game does not go on until the frame it
    // just presented has its result, which the next frame composes one frame
    // late instead of many. The frame rate becomes the network's. Bounded, so
    // that a network that stops answering cannot stop the game.
    if (c->settings.wait_for_network && c->state == DLSS5NR_STATE_RUNNING) {
        std::unique_lock<std::mutex> guard(c->lock);
        const bool answered = c->wake.wait_for(guard, std::chrono::seconds(2), [&] {
            const Slot::State state = c->slots[captured].state;
            return c->stop || c->state != DLSS5NR_STATE_RUNNING || (state != Slot::Captured && state != Slot::Evaluating);
        });
        guard.unlock();
        if (!answered) c->log(1, "waited 2 s for the network's result and went on without it");
    }
}

DLSS5NR_API int dlss5nr_process(dlss5nr_context *c, const dlss5nr_frame *f) {
    if (!c || !f || f->struct_size < offsetof(dlss5nr_frame, exposure) || !f->command_list || !f->input ||
        !f->output || !f->width || !f->height)
        return -1;
    c->mark_host_ready();
    ID3D12GraphicsCommandList *cmd = f->command_list;
    const dlss5nr_settings &s = c->settings;
    const D3D12_RESOURCE_STATES input_state = (D3D12_RESOURCE_STATES)f->input_state;
    const D3D12_RESOURCE_STATES output_state = (D3D12_RESOURCE_STATES)f->output_state;
    if (view_format(f->input->GetDesc().Format) == DXGI_FORMAT_UNKNOWN ||
        view_format(f->output->GetDesc().Format) == DXGI_FORMAT_UNKNOWN)
        return -2;

    const bool running = c->state == DLSS5NR_STATE_RUNNING && s.enabled;
    // Without depth and motion vectors -- a host that only has the finished
    // image -- there is nothing to follow the scene with: the network gets
    // none, keeps no history, and its result is composed where it put it.
    const bool guided = f->depth && f->motion;
    const int encoding = s.encoding != DLSS5NR_ENCODING_AUTO ? (s.encoding == DLSS5NR_ENCODING_DISPLAY ? 1 : 0)
                         : f->hdr                             ? 0
                                                              : 1;

    auto region_scale = [](ID3D12Resource *r, uint32_t w, uint32_t h, float out[2]) {
        const D3D12_RESOURCE_DESC d = r->GetDesc();
        out[0] = (float)w / (float)d.Width;
        out[1] = (float)h / (float)d.Height;
    };

    Constants base{};
    base.encoding = (uint32_t)encoding;
    base.colour_strength = s.colour_strength;
    base.exposure_mul = s.exposure;
    base.pre_exposure = f->pre_exposure > 0.0f ? f->pre_exposure : 1.0f;
    base.exposure_mode = (uint32_t)(s.exposure_mode == DLSS5NR_EXPOSURE_GAME && !f->exposure ? DLSS5NR_EXPOSURE_AUTO
                                                                                            : s.exposure_mode);
    // The game's exposure is already smoothed the way it wants; the measured one is not.
    base.adaptation = base.exposure_mode == DLSS5NR_EXPOSURE_GAME ? 1.0f : 0.05f;
    base.flags = (f->reset ? FLAG_RESET : 0) | (f->depth_inverted ? FLAG_DEPTH_INVERTED : 0);

    const D3D12_RESOURCE_STATES input_read = readable(input_state);
    transition(cmd, f->input, input_state, input_read);

    // ------------------------------------------------------------ network size
    // Rounded to a multiple of 8: the network works in 8x8 windows.
    uint32_t nw = (uint32_t)(f->width * s.resolution_scale + 0.5f) & ~7u;
    uint32_t nh = (uint32_t)(f->height * s.resolution_scale + 0.5f) & ~7u;
    if (nw < 64) nw = 64;
    if (nh < 64) nh = 64;
    // A size that keeps changing -- the slider for it being dragged -- is not
    // followed frame by frame: every size is a set of images and a network to
    // build, and a drag crosses dozens in a few seconds. The size it has is kept
    // until the one asked for has stayed the same for a moment.
    if (c->slot_width && (nw != c->slot_width || nh != c->slot_height)) {
        const auto now = std::chrono::steady_clock::now();
        if (c->wanted_width != nw || c->wanted_height != nh) {
            c->wanted_width = nw;
            c->wanted_height = nh;
            c->wanted_since = now;
        }
        if (now - c->wanted_since < std::chrono::milliseconds(750)) {
            nw = c->slot_width;
            nh = c->slot_height;
        }
    } else {
        c->wanted_width = c->wanted_height = 0;
    }

    bool can_capture = running;
    if (running && (nw != c->slot_width || nh != c->slot_height)) {
        // New slots. The worker lets go of the old ones first; until it has,
        // nothing is captured and the residual is dropped.
        std::unique_lock<std::mutex> guard(c->lock);
        bool busy = false;
        for (const Slot &slot : c->slots) busy |= slot.state == Slot::Evaluating;
        if (!c->resize_requested) {
            c->resize_requested = true;
            c->resize_released = false;
            c->wake.notify_all();
        }
        can_capture = false;
        if (!busy && c->resize_released) {
            // The set in use is parked, and the one for the new size taken out of
            // the parked ones if there is one.
            if (c->slot_width) {
                SlotSet old;
                old.width = c->slot_width;
                old.height = c->slot_height;
                old.parked_at = c->frame;
                for (int i = 0; i < kSlots; ++i) {
                    SharedImage *images[4] = {&c->slots[i].color, &c->slots[i].depth, &c->slots[i].motion,
                                              &c->slots[i].output};
                    for (int k = 0; k < 4; ++k) {
                        old.image[i][k] = *images[k];
                        *images[k] = SharedImage{};
                    }
                }
                c->slot_cache.push_back(old);
            }
            for (Slot &slot : c->slots) {
                slot.state = Slot::Free;
                slot.fence = 0;
            }
            bool ok = true;
            bool reused = false;
            for (size_t n = 0; n < c->slot_cache.size(); ++n) {
                if (c->slot_cache[n].width != nw || c->slot_cache[n].height != nh) continue;
                for (int i = 0; i < kSlots; ++i) {
                    SharedImage *images[4] = {&c->slots[i].color, &c->slots[i].depth, &c->slots[i].motion,
                                              &c->slots[i].output};
                    for (int k = 0; k < 4; ++k) *images[k] = c->slot_cache[n].image[i][k];
                }
                c->slot_cache.erase(c->slot_cache.begin() + n);
                reused = true;
                break;
            }
            if (!reused) {
                for (Slot &slot : c->slots) {
                    ok = ok && make_shared(c->device, slot.color, nw, nh, DXGI_FORMAT_R16G16B16A16_FLOAT, false) &&
                         make_shared(c->device, slot.depth, nw, nh, DXGI_FORMAT_R32_FLOAT, false) &&
                         make_shared(c->device, slot.motion, nw, nh, DXGI_FORMAT_R16G16_FLOAT, false) &&
                         make_shared(c->device, slot.output, nw, nh, DXGI_FORMAT_R16G16B16A16_FLOAT, true);
                }
            }
            // Beyond what is kept, the oldest is given up: its CUDA side is let
            // go of by the worker, the Direct3D side retired when it is back.
            while (c->slot_cache.size() > kParkedSets) {
                size_t oldest = 0;
                for (size_t n = 1; n < c->slot_cache.size(); ++n)
                    if (c->slot_cache[n].parked_at < c->slot_cache[oldest].parked_at) oldest = n;
                // c->lock is held here already, by the guard above.
                c->evicting.push_back(c->slot_cache[oldest]);
                c->slot_cache.erase(c->slot_cache.begin() + oldest);
                c->wake.notify_all();
            }
            if (!ok) {
                c->log(0, "could not create the network's %ux%u images", nw, nh);
                return -3;
            }
            c->slot_width = nw;
            c->slot_height = nh;
            c->resize_requested = false;
            c->log(2, "network images at %ux%u for a %ux%u image%s", nw, nh, f->width, f->height,
                   reused ? " (kept from before)" : "");
        }
        c->residual_valid = false;
    }
    if (running && (c->residual_width != nw || c->residual_height != nh || !c->residual)) {
        auto replace = [&](ID3D12Resource *&texture, D3D12_RESOURCE_STATES &state, DXGI_FORMAT format) {
            if (texture) c->retired.push_back({texture, nullptr, c->fence_value + 1});
            texture = make_texture(c->device, nw, nh, format, false, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        };
        replace(c->residual, c->residual_state, DXGI_FORMAT_R16G16B16A16_FLOAT);
        if (c->residual_soft) c->retired.push_back({c->residual_soft, nullptr, c->fence_value + 1});
        c->residual_soft = make_texture(c->device, (nw + 15) / 16, (nh + 15) / 16, DXGI_FORMAT_R16G16B16A16_FLOAT,
                                        false, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        c->residual_soft_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        // Displacements reach thousands of pixels and have to stay exact to a
        // fraction of one: 32-bit floats.
        auto remake = [&](Flow &flow) {
            for (int m = 0; m < 2; ++m) replace(flow.map[m], flow.state[m], DXGI_FORMAT_R32G32_FLOAT);
            flow.valid = false;
            flow.identity = true;
        };
        remake(c->applied_flow);
        remake(c->previous_flow);
        replace(c->residual_previous, c->residual_previous_state, DXGI_FORMAT_R16G16B16A16_FLOAT);
        remake(c->idle_flow);
        for (Flow &flow : c->slot_flow) remake(flow);
        for (int m = 0; m < 2; ++m) replace(c->depth_history[m], c->depth_history_state[m], DXGI_FORMAT_R32_FLOAT);
        c->depth_history_valid = false;
        c->residual_width = nw;
        c->residual_height = nh;
        c->residual_valid = false;
    }

    // ---------------------------------------------------------------- exposure
    if (running && encoding != 1 && base.exposure_mode != DLSS5NR_EXPOSURE_FIXED) {
        Constants k = base;
        region_scale(f->input, f->width, f->height, k.uv0_scale);
        Bindings b;
        b.srv[0] = f->input;
        D3D12_RESOURCE_STATES exposure_state = (D3D12_RESOURCE_STATES)f->exposure_state;
        if (base.exposure_mode == DLSS5NR_EXPOSURE_GAME) {
            b.srv[1] = f->exposure;
            transition(cmd, f->exposure, exposure_state, readable(exposure_state));
        }
        b.exposure_buffer = c->exposure_buffer;
        c->passes.dispatch(cmd, Pass::Exposure, k, b, 1, 1);
        uav_barrier(cmd, c->exposure_buffer);
        if (base.exposure_mode == DLSS5NR_EXPOSURE_GAME)
            transition(cmd, f->exposure, readable(exposure_state), exposure_state);
    }

    const D3D12_RESOURCE_STATES depth_state = (D3D12_RESOURCE_STATES)f->depth_state;
    const D3D12_RESOURCE_STATES motion_state = (D3D12_RESOURCE_STATES)f->motion_state;
    if (running) {
        transition(cmd, f->depth, depth_state, readable(depth_state));
        transition(cmd, f->motion, motion_state, readable(motion_state));
    }
    const float motion_to_px[2] = {
        f->motion_width ? f->motion_scale_x * (float)nw / (float)f->motion_width : 0.0f,
        f->motion_height ? f->motion_scale_y * (float)nh / (float)f->motion_height : 0.0f};

    // Constants for a pass over the network's size.
    auto network_constants = [&]() {
        Constants k = base;
        k.size[0] = nw;
        k.size[1] = nh;
        k.inv_size[0] = 1.0f / nw;
        k.inv_size[1] = 1.0f / nh;
        region_scale(f->input, f->width, f->height, k.uv0_scale);
        if (guided) {
            region_scale(f->depth, f->depth_width, f->depth_height, k.uv1_scale);
            region_scale(f->motion, f->motion_width, f->motion_height, k.uv2_scale);
        }
        k.motion_to_px[0] = motion_to_px[0];
        k.motion_to_px[1] = motion_to_px[1];
        return k;
    };
    auto to_state = [&](ID3D12Resource *texture, D3D12_RESOURCE_STATES &state, D3D12_RESOURCE_STATES next) {
        transition(cmd, texture, state, next);
        state = next;
    };
    const D3D12_RESOURCE_STATES read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    const D3D12_RESOURCE_STATES write = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

    // ------------------------------------------------------------------- track
    // Every map moves on by this frame's motion, before anything reads it: a
    // capture made now starts from here, and the result composed now is found
    // through its map as of this frame.
    if (running && (f->reset || guided != c->guided)) {
        // A cut, or motion vectors coming or going: nothing before it
        // describes what is on screen now.
        c->guided = guided;
        c->residual_valid = false;
        c->applied_flow.valid = false;
        c->previous_flow.valid = false;
        for (Flow &flow : c->slot_flow) flow.valid = false;
        c->depth_history_valid = false;
    }
    if (running && guided) {
        bool in_flight[kSlots] = {};
        {
            std::lock_guard<std::mutex> guard(c->lock);
            for (int i = 0; i < kSlots; ++i) in_flight[i] = c->slots[i].state != Slot::Free;
        }
        std::vector<Flow *> flows;
        for (int i = 0; i < kSlots; ++i)
            if (in_flight[i] && c->slot_flow[i].valid) flows.push_back(&c->slot_flow[i]);
            else c->slot_flow[i].valid = false;
        if (c->residual_valid && c->applied_flow.valid) flows.push_back(&c->applied_flow);
        if (c->residual_valid && c->previous_flow.valid) flows.push_back(&c->previous_flow);
        // The last one also keeps this frame's depth for the next one, so
        // there has to be one.
        if (flows.empty()) {
            c->idle_flow.identity = true;
            flows.push_back(&c->idle_flow);
        }

        const int depth_next = 1 - c->depth_current;
        for (Flow *flow : flows) {
            const bool keep_depth = flow == flows.back();
            const int next = 1 - flow->current;
            Constants k = network_constants();
            if (flow->identity) k.flags |= FLAG_IDENTITY;
            if (c->depth_history_valid) k.flags |= FLAG_DEPTH_CHECK;
            if (keep_depth) k.flags |= FLAG_KEEP_DEPTH;
            to_state(flow->map[flow->current], flow->state[flow->current], read);
            to_state(flow->map[next], flow->state[next], write);
            to_state(c->depth_history[c->depth_current], c->depth_history_state[c->depth_current], read);
            if (keep_depth) to_state(c->depth_history[depth_next], c->depth_history_state[depth_next], write);
            Bindings b;
            b.srv[0] = flow->identity ? nullptr : flow->map[flow->current];
            b.srv[1] = f->depth;
            b.srv[2] = f->motion;
            b.srv[3] = c->depth_history_valid ? c->depth_history[c->depth_current] : nullptr;
            b.uav[0] = flow->map[next];
            if (keep_depth) b.uav[1] = c->depth_history[depth_next];
            b.exposure_buffer = c->exposure_buffer;
            c->passes.dispatch(cmd, Pass::Track, k, b, nw, nh);
            to_state(flow->map[next], flow->state[next], read);
            flow->current = next;
            flow->identity = false;
        }
        to_state(c->depth_history[depth_next], c->depth_history_state[depth_next], read);
        c->depth_current = depth_next;
        c->depth_history_valid = true;
    }

    // ----------------------------------------------------------------- capture
    if (can_capture && c->pending_slot < 0) {
        // A capture the network has not taken yet is replaced by this frame:
        // when the network is free again it starts from the newest frame
        // there is, not from one that waited through a whole evaluation. That
        // wait would come back as the same number of frames of lag.
        //
        // The motion vectors the network gets lead back to the capture its
        // own history holds when it takes this one: the one it is evaluating
        // now, or else the newest it has finished. However many frames ago
        // that was, that capture's map is the way back to it.
        int free_slot = -1;
        const Flow *previous = nullptr;
        uint64_t previous_frame = ~0ull;
        {
            std::lock_guard<std::mutex> guard(c->lock);
            for (int i = 0; i < kSlots && free_slot < 0; ++i)
                if (c->slots[i].state == Slot::Captured) free_slot = i;
            for (int i = 0; i < kSlots && free_slot < 0; ++i)
                if (c->slots[i].state == Slot::Free) free_slot = i;

            int taken = -1;
            for (int i = 0; i < kSlots; ++i)
                if (c->slots[i].state == Slot::Evaluating) taken = i;
            if (taken < 0)
                for (int i = 0; i < kSlots; ++i)
                    if (c->slots[i].state == Slot::Ready && (taken < 0 || c->slots[i].frame > c->slots[taken].frame))
                        taken = i;
            if (!guided) {
                // No way back to any capture.
            } else if (taken >= 0 && c->slot_flow[taken].valid && !c->slot_flow[taken].identity) {
                previous = &c->slot_flow[taken];
                previous_frame = c->slots[taken].frame;
            } else if (taken < 0 && c->residual_valid && c->applied_flow.valid && !c->applied_flow.identity) {
                previous = &c->applied_flow;
                previous_frame = c->applied_frame;
            }

            if (free_slot >= 0) {
                Slot &slot = c->slots[free_slot];
                slot.state = Slot::Captured;
                slot.fence = 0;
                slot.frame = c->frame;
                slot.previous_frame = previous_frame;
                slot.reset = f->reset || !previous || !s.network_history;
                slot.depth_inverted = f->depth_inverted != 0;
            }
        }
        const bool has_previous = previous != nullptr;
        if (free_slot >= 0) {
            Slot &slot = c->slots[free_slot];
            c->pending_slot = free_slot;
            Constants k = network_constants();
            if (!has_previous) k.flags |= FLAG_IDENTITY;
            for (ID3D12Resource *r : {slot.color.resource, slot.depth.resource, slot.motion.resource})
                transition(cmd, r, D3D12_RESOURCE_STATE_COMMON, write);
            Bindings b;
            b.srv[0] = f->input;
            b.srv[1] = f->depth;
            b.srv[3] = has_previous ? previous->map[previous->current] : nullptr;
            b.uav[0] = slot.color.resource;
            b.uav[1] = slot.depth.resource;
            b.uav[2] = slot.motion.resource;
            b.exposure_buffer = c->exposure_buffer;
            c->passes.dispatch(cmd, Pass::Capture, k, b, nw, nh);
            for (ID3D12Resource *r : {slot.color.resource, slot.depth.resource, slot.motion.resource})
                transition(cmd, r, write, D3D12_RESOURCE_STATE_COMMON);
            // Its map starts here.
            c->slot_flow[free_slot].identity = true;
            c->slot_flow[free_slot].valid = true;
        }
    }

    // ------------------------------------------------------------------ result
    if (running) {
        int ready = -1;
        {
            std::lock_guard<std::mutex> guard(c->lock);
            uint64_t best = c->applied_sequence;
            for (int i = 0; i < kSlots; ++i)
                if (c->slots[i].state == Slot::Ready && c->slots[i].sequence > best) {
                    best = c->slots[i].sequence;
                    ready = i;
                }
        }
        if (ready >= 0) {
            Slot &slot = c->slots[ready];
            // Captured before a cut, or at another size: nothing to show.
            const bool usable = c->slot_flow[ready].valid && slot.output.width == nw && slot.output.height == nh;
            if (usable) {
                Constants k = network_constants();
                transition(cmd, slot.output.resource, D3D12_RESOURCE_STATE_COMMON, read);
                transition(cmd, slot.color.resource, D3D12_RESOURCE_STATE_COMMON, read);
                transition(cmd, slot.depth.resource, D3D12_RESOURCE_STATE_COMMON, read);
                // The result composed until now becomes the one before: its
                // residual and its map change places with the spare ones,
                // which the new result then takes.
                if (c->residual_valid && c->applied_flow.valid) {
                    std::swap(c->residual, c->residual_previous);
                    std::swap(c->residual_state, c->residual_previous_state);
                    std::swap(c->applied_flow, c->previous_flow);
                } else {
                    c->previous_flow.valid = false;
                }
                to_state(c->residual, c->residual_state, write);
                Bindings b;
                b.srv[0] = slot.output.resource;
                b.srv[1] = slot.color.resource;
                b.srv[2] = slot.depth.resource;
                b.uav[0] = c->residual;
                b.exposure_buffer = c->exposure_buffer;
                c->passes.dispatch(cmd, Pass::Refresh, k, b, nw, nh);
                to_state(c->residual, c->residual_state, read);
                {
                    const uint32_t sw = (nw + 15) / 16, sh = (nh + 15) / 16;
                    Constants soft = base;
                    soft.size[0] = sw;
                    soft.size[1] = sh;
                    soft.uv1_scale[0] = 1.0f / nw;
                    soft.uv1_scale[1] = 1.0f / nh;
                    to_state(c->residual_soft, c->residual_soft_state, write);
                    Bindings sb;
                    sb.srv[0] = slot.output.resource;
                    sb.srv[1] = slot.color.resource;
                    sb.uav[0] = c->residual_soft;
                    sb.exposure_buffer = c->exposure_buffer;
                    c->passes.dispatch(cmd, Pass::Soften, soft, sb, sw, sh);
                    to_state(c->residual_soft, c->residual_soft_state, read);
                }
                transition(cmd, slot.output.resource, read, D3D12_RESOURCE_STATE_COMMON);
                transition(cmd, slot.color.resource, read, D3D12_RESOURCE_STATE_COMMON);
                transition(cmd, slot.depth.resource, read, D3D12_RESOURCE_STATE_COMMON);
                // The slot's map now finds this result; the old one goes back
                // to the slot, free to start over at its next capture.
                std::swap(c->applied_flow, c->slot_flow[ready]);
                c->slot_flow[ready].valid = false;
                c->residual_valid = true;
                c->applied_frame = slot.frame;
                c->residual_age = 0;
                c->result_latency = (uint32_t)(c->frame - slot.frame);
            }
            std::lock_guard<std::mutex> guard(c->lock);
            c->applied_sequence = slot.sequence;
            if (usable) ++c->results;
            slot.state = Slot::Free;
        } else if (c->residual_valid) {
            ++c->residual_age;
        }
        c->result_age = c->residual_age;
        transition(cmd, f->depth, readable(depth_state), depth_state);
        transition(cmd, f->motion, readable(motion_state), motion_state);
    }

    // ----------------------------------------------------------------- compose
    {
        Constants k = base;
        k.size[0] = f->width;
        k.size[1] = f->height;
        k.inv_size[0] = 1.0f / f->width;
        k.inv_size[1] = 1.0f / f->height;
        k.uv1_scale[0] = 1.0f / nw;
        k.uv1_scale[1] = 1.0f / nh;
        k.strength = s.detail_strength;
        k.max_ratio = s.max_ratio;
        // A result nothing new has replaced for max_age frames fades out.
        const int over = (int)c->residual_age - s.max_age;
        k.fade = over <= 0 ? 1.0f : powf(0.8f, (float)over);
        const bool apply = running && c->residual_valid && c->applied_flow.valid && k.fade > 0.01f &&
                           s.detail_strength != 0.0f;
        if (!apply) k.flags |= FLAG_COPY;
        if (apply && (c->applied_flow.identity || !s.follow_motion)) k.flags |= FLAG_IDENTITY;
        if (apply && c->depth_history_valid) k.flags |= FLAG_FILL;
        if (s.show_tracking) k.flags |= FLAG_SHOW_TRACKING;
        const bool older = apply && c->previous_flow.valid && !c->previous_flow.identity && s.follow_motion;
        if (older) k.flags |= FLAG_PREVIOUS;
        if (s.composition == DLSS5NR_COMPOSITION_REPLACE) k.flags |= FLAG_REPLACE;
        const D3D12_RESOURCE_STATES output_write = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        transition(cmd, f->output, output_state, output_write);
        Bindings b;
        b.srv[0] = f->input;
        b.srv[1] = apply ? c->residual_soft : nullptr;
        b.srv[3] = apply ? c->residual : nullptr;
        b.srv[4] = apply && !c->applied_flow.identity ? c->applied_flow.map[c->applied_flow.current] : nullptr;
        // This frame's depth at the network's size, kept while tracking.
        b.srv[5] = apply && c->depth_history_valid ? c->depth_history[c->depth_current] : nullptr;
        b.srv[2] = older ? c->residual_previous : nullptr;
        b.srv[6] = older ? c->previous_flow.map[c->previous_flow.current] : nullptr;
        b.uav[0] = f->output;
        b.exposure_buffer = c->exposure_buffer;
        c->passes.dispatch(cmd, Pass::Compose, k, b, f->width, f->height);
        // Left as an unordered access view, the next reader may be another
        // unordered access, which only a UAV barrier orders after this write.
        if (output_state == output_write) uav_barrier(cmd, f->output);
        transition(cmd, f->output, output_write, output_state);
    }
    transition(cmd, f->input, input_read, input_state);
    return 0;
}

} // extern "C"
