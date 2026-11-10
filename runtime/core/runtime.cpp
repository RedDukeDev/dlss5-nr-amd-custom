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
#include <emmintrin.h>
#include <tlhelp32.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
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

bool copy_interop_wanted(Network &network, ID3D12Device *device) {
    char value[8];
    const DWORD n = GetEnvironmentVariableA("DLSS5NR_COPY", value, sizeof value);
    if (n && n < sizeof value) return value[0] == '1';
    HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    return ntdll && GetProcAddress(ntdll, "wine_get_version") && !network.can_share(device);
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

// The linear mode (dlss5nr_process_queue): the images one frame goes through at one network size, and the
// residual its answer leaves. Made by the game thread; the worker imports them into CUDA and builds the
// network's feature for `controls`, and says so in `ready`.
struct LinearSet {
    uint32_t width = 0, height = 0;
    SharedImage color, depth, motion, output;
    ID3D12Resource *residual = nullptr;
    D3D12_RESOURCE_STATES residual_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    NetworkControls controls;
    std::atomic<int> ready{0};      // 0 being prepared, 1 usable, -1 could not be prepared
    std::atomic<int> pending{0};    // evaluations queued on its images and not finished
};
constexpr int kLinearRing = 3;

// An evaluation for the linear mode's thread: wait (on the CPU) until the capture has been done, run the network on
// it, and tell the game's queue.
struct LinearJob {
    LinearSet *set = nullptr;
    uint64_t before = 0, after = 0;   // the fence values: the capture is done / the network is
    uint64_t frame = 0;
    EvaluateDesc desc;
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
    // Frames between the last two results, and the frame the last one arrived in.
    uint32_t result_interval = 1000;
    uint64_t arrival_frame = 0;
    uint64_t applied_sequence = 0;
    uint64_t applied_frame = ~0ull;   // the frame the composed result was captured in
    Flow applied_flow;
    // The result before it, and its map: what the newer capture did not see,
    // this one may have.
    ID3D12Resource *residual_previous = nullptr;
    D3D12_RESOURCE_STATES residual_previous_state = {};
    // How far each pixel of those two was from an edge in depth, in its capture.
    ID3D12Resource *residual_edges = nullptr;
    D3D12_RESOURCE_STATES residual_edges_state = {};
    ID3D12Resource *residual_previous_edges = nullptr;
    D3D12_RESOURCE_STATES residual_previous_edges_state = {};
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

    // The linear mode (dlss5nr_process_queue). Game thread only, but for what is said.
    // The network has one user at a time: the worker evaluating a slot, a feature being built, the linear mode.
    std::mutex net_lock;
    std::unique_ptr<LinearSet> lin_set;           // the images in use
    std::unique_ptr<LinearSet> lin_next;          // being prepared by the worker, for another size
    LinearSet *lin_request = nullptr;             // what the worker is to prepare next, under `lock`
    NetworkControls lin_asked;                    // the controls last asked for, and since when
    std::chrono::steady_clock::time_point lin_asked_at{};
    uint32_t lin_wanted_width = 0, lin_wanted_height = 0;
    std::chrono::steady_clock::time_point lin_wanted_since{};
    ID3D12CommandAllocator *lin_allocator[kLinearRing][2] = {};
    ID3D12GraphicsCommandList *lin_list[kLinearRing][2] = {};
    ID3D12Fence *lin_fence = nullptr;
    void *lin_fence_handle = nullptr;
    void *lin_semaphore = nullptr;                // lin_fence in CUDA
    bool lin_gpu = false;                         // the GPU waits (else the CPU does)
    bool lin_made = false, lin_failed = false;
    std::atomic<int> lin_trace{0};                            // calls left to log step by step (after a new set is first used)
    bool lin_cpu_forced = false;                  // DLSS5NR_LINEAR_CPU_WAIT=1
    bool lin_import_tried = false;                // the fence given to the network, once it is loaded
    HANDLE lin_event = nullptr;
    uint64_t lin_value = 0, lin_ring_value[kLinearRing] = {};
    uint32_t lin_ring = 0;
    std::atomic<uint64_t> lin_last_frame{~0ull};  // the frame of the last evaluation: the network's history is it
    std::atomic<int> lin_failures{0};             // evaluations in a row that failed
    // The thread that runs the network for the linear mode. It is the only one that waits for the capture, on the CPU,
    // so that nothing is ever queued on the network's side that waits for the game's queue: anything else that
    // waits for the network's side to be idle (the frame generation does) would hold up the very queue that
    // wait is for.
    std::thread lin_thread;
    std::mutex lin_job_lock;
    std::condition_variable lin_job_wake;
    std::deque<LinearJob> lin_jobs;
    bool lin_job_stop = false;
    std::vector<std::unique_ptr<LinearSet>> lin_graveyard;   // retired while evaluations were still queued on them
    double lin_profile_ms = 0.0;
    uint64_t lin_profile_calls = 0;
    std::chrono::steady_clock::time_point lin_retry_at{};   // when the evaluations are tried again after failing
    std::chrono::steady_clock::time_point lin_last_began{};  // the last evaluation, for how often they come
    std::atomic<uint32_t> lin_width{0}, lin_height{0};   // for the status

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
    // Not part of the controls: changing them leaves the feature as it is. Under lock.
    int network_passes = 1;
    bool keep_local_tone = true;
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
    // Watches for the game standing still (no frame presented for seconds): then it writes where every thread of the
    // process is to the log, once, which is what a freeze needs to be understood.
    std::thread watchdog;
    std::atomic<bool> watchdog_stop{false};
    std::atomic<int64_t> last_present_ms{0};
    std::atomic<uint64_t> presents{0};

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
    // Decided here, with the network loaded, and before the state is running: the images are made after that.
    g_copy_interop = copy_interop_wanted(network, device);
    network.set_copy_mode(g_copy_interop);
    log(2, "images %s", g_copy_interop ? "copied through host memory" : "shared with CUDA");
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
        int passes_now = 1;
        bool keep_tone_now = true;
        {
            std::unique_lock<std::mutex> guard(lock);
            wake.wait(guard, [&] {
                if (stop || (resize_requested && !resize_released) || !evicting.empty() || lin_request) return true;
                for (const Slot &s : slots)
                    if (s.state == Slot::Captured && s.fence) return true;
                return false;
            });
            if (stop) break;
            if (lin_request) {
                // The linear mode's images for another size or other controls: into CUDA, and the feature built.
                // Seconds, and the network is nobody else's meanwhile.
                LinearSet *set = lin_request;
                lin_request = nullptr;
                guard.unlock();
                std::string why;
                std::lock_guard<std::mutex> use(net_lock);
                bool ok = network.import(set->color, why) && network.import(set->depth, why) &&
                          network.import(set->motion, why) && network.import(set->output, why) &&
                          network.create_feature(set->width, set->height, set->controls, why);
                if (ok) {
                    // One evaluation on whatever the images hold, so that the first frames do not pay for the
                    // network's first launches (measured: ~9 ms of the game thread's time per frame, for the first
                    // dozens of them).
                    EvaluateDesc warm;
                    warm.color = &set->color;
                    warm.depth = &set->depth;
                    warm.motion = &set->motion;
                    warm.output = &set->output;
                    warm.reset = true;
                    warm.controls = set->controls;
                    ok = network.evaluate(warm, why);
                }
                if (ok) log(2, "the network is ready at %ux%u for the linear mode", set->width, set->height);
                else log(0, "the network could not be prepared for the linear mode: %s", why.c_str());
                set->ready = ok ? 1 : -1;
                continue;
            }
            if (!evicting.empty()) {
                // Sets given up for good. Letting go of a CUDA image waits for the GPU, so it is done with neither
                // the context's lock held (the game's thread may need it to finish what the GPU waits for) nor
                // while the game's thread is using the network (the linear mode evaluates on it): net_lock first,
                // and the lock only again when it is over.
                std::vector<SlotSet> mine = std::move(evicting);
                evicting.clear();
                guard.unlock();
                {
                    std::lock_guard<std::mutex> use(net_lock);
                    for (SlotSet &set : mine)
                        for (auto &images : set.image)
                            for (SharedImage &image : images) network.release(image);
                }
                guard.lock();
                for (SlotSet &set : mine) evicted.push_back(std::move(set));
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
            passes_now = network_passes;
            keep_tone_now = keep_local_tone;
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

        // The network is this evaluation's for as long as it takes (the linear mode may be using it too).
        std::unique_lock<std::mutex> use_network(net_lock);
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
            desc.passes = passes_now;
            desc.keep_local_tone = keep_tone_now;
            ok = network.evaluate(desc, error);
        }
        use_network.unlock();
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
        wake.notify_all();
    }
    CloseHandle(fence_event);
    {
        // The linear mode's CUDA side: only this thread's context may let go of it.
        std::lock_guard<std::mutex> use(net_lock);
        if (lin_semaphore) {
            network.destroy_fence(lin_semaphore);
            lin_semaphore = nullptr;
        }
        for (LinearSet *set : {lin_set.get(), lin_next.get()})
            if (set)
                for (SharedImage *image : {&set->color, &set->depth, &set->motion, &set->output}) network.release(*image);
    }
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
    s->passes = 1;
    s->keep_local_tone = 1;
    s->blend_frames = 4;
    s->network_preset = 0;
}

namespace {

int64_t steady_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// Where every other thread of the process is, as module + offset, to the log. The threads are stopped one at a time and
// only long enough to read their stacks into memory that is already there (a thread that is stopped may hold a lock
// this one would need to log), and the names are put to the addresses after they run again.
void dump_stacks(dlss5nr_context *c) {
    constexpr int kThreads = 96, kFrames = 28;
    static DWORD ids[kThreads];
    static DWORD64 frames[kThreads][kFrames];
    static int depth[kThreads];
    static ULONGLONG user_ms[kThreads], kernel_ms[kThreads];
    int count = 0;
    const DWORD self = GetCurrentThreadId(), pid = GetCurrentProcessId();
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snap != INVALID_HANDLE_VALUE) {
        THREADENTRY32 te{};
        te.dwSize = sizeof te;
        for (BOOL ok = Thread32First(snap, &te); ok && count < kThreads; ok = Thread32Next(snap, &te))
            if (te.th32OwnerProcessID == pid && te.th32ThreadID != self) ids[count++] = te.th32ThreadID;
        CloseHandle(snap);
    }
    for (int i = 0; i < count; ++i) {
        depth[i] = 0;
        HANDLE th = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, ids[i]);
        if (!th) continue;
        FILETIME created, exited, kernel, user;
        if (GetThreadTimes(th, &created, &exited, &kernel, &user)) {
            user_ms[i] = ((ULONGLONG)user.dwHighDateTime << 32 | user.dwLowDateTime) / 10000;
            kernel_ms[i] = ((ULONGLONG)kernel.dwHighDateTime << 32 | kernel.dwLowDateTime) / 10000;
        }
        if (SuspendThread(th) != (DWORD)-1) {
            CONTEXT ctx{};
            ctx.ContextFlags = CONTEXT_FULL;
            if (GetThreadContext(th, &ctx)) {
                for (int f = 0; f < kFrames && ctx.Rip; ++f) {
                    frames[i][depth[i]++] = ctx.Rip;
                    DWORD64 base = 0;
                    PRUNTIME_FUNCTION entry = RtlLookupFunctionEntry(ctx.Rip, &base, nullptr);
                    if (!entry) {
                        // A leaf: the return address is on top of the stack.
                        if (!ctx.Rsp) break;
                        ctx.Rip = *(DWORD64 *)ctx.Rsp;
                        ctx.Rsp += 8;
                    } else {
                        PVOID handler_data = nullptr;
                        DWORD64 frame_base = 0;
                        RtlVirtualUnwind(UNW_FLAG_NHANDLER, base, ctx.Rip, entry, &ctx, &handler_data, &frame_base,
                                         nullptr);
                    }
                }
            }
            ResumeThread(th);
        }
        CloseHandle(th);
    }
    c->log(0, "STALL: no frame presented for seconds; frame %llu, linear fence at %llu of %llu, state %d",
           (unsigned long long)c->frame, c->lin_fence ? (unsigned long long)c->lin_fence->GetCompletedValue() : 0ull,
           (unsigned long long)c->lin_value, (int)c->state.load());
    for (int i = 0; i < count; ++i) {
        c->log(0, "STALL: thread %lu%s (user %llu ms, kernel %llu ms)", ids[i],
               ids[i] == c->worker_thread_id.load() ? " [dlss5nr worker]" : "", (unsigned long long)user_ms[i],
               (unsigned long long)kernel_ms[i]);
        for (int f = 0; f < depth[i]; ++f) {
            HMODULE module = nullptr;
            wchar_t path[MAX_PATH] = {};
            if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                   (LPCWSTR)frames[i][f], &module) &&
                GetModuleFileNameW(module, path, MAX_PATH)) {
                const wchar_t *name = wcsrchr(path, L'\\');
                c->log(0, "STALL:   %ls+0x%llx", name ? name + 1 : path,
                       (unsigned long long)(frames[i][f] - (DWORD64)module));
            } else {
                c->log(0, "STALL:   0x%llx", (unsigned long long)frames[i][f]);
            }
        }
    }
}

} // namespace

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
    const auto began = std::chrono::steady_clock::now();
    const bool built = c->passes.init(c->device, error);
    c->log(2, "compute passes built in %.0f ms",
           std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - began).count());
    if (!built) {
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
    c->log(2, "runtime starting: files in %ls, network %ls", c->paths.runtime_dir.c_str(), c->paths.snippet.c_str());
    c->worker = std::thread([c] {
        c->worker_thread_id = GetCurrentThreadId();
        c->run_worker();
    });
    c->watchdog = std::thread([c] {
        bool dumped = false;
        char value[16] = {};
        const int64_t stall_ms = GetEnvironmentVariableA("DLSS5NR_STALL_MS", value, sizeof value) ? atoi(value) : 6000;
        while (!c->watchdog_stop) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            const int64_t last = c->last_present_ms;
            if (!last || c->presents < 120) continue;
            const int64_t idle = steady_ms() - last;
            if (idle > stall_ms && !dumped) {
                dumped = true;
                dump_stacks(c);
            } else if (idle < 1000) {
                dumped = false;
            }
        }
    });
    // Known before anything is loaded on it: the host may start filtering
    // library loads as soon as this returns.
    while (!c->worker_thread_id) std::this_thread::yield();
    *out = c;
    return 0;
}

DLSS5NR_API void dlss5nr_destroy(dlss5nr_context *c) {
    if (!c) return;
    // The linear mode's thread first: it uses the network and the fence, and signals what is still queued without
    // running the network.
    {
        std::lock_guard<std::mutex> guard(c->lin_job_lock);
        c->lin_job_stop = true;
    }
    c->lin_job_wake.notify_all();
    if (c->lin_thread.joinable()) c->lin_thread.join();
    {
        std::lock_guard<std::mutex> guard(c->lock);
        c->stop = true;
    }
    c->wake.notify_all();
    c->watchdog_stop = true;
    if (c->watchdog.joinable()) c->watchdog.join();
    if (c->worker.joinable()) c->worker.join();
    for (Slot &s : c->slots) {
        destroy_shared(s.color);
        destroy_shared(s.depth);
        destroy_shared(s.motion);
        destroy_shared(s.output);
    }
    std::vector<LinearSet *> linear_sets = {c->lin_set.get(), c->lin_next.get()};
    for (auto &gone : c->lin_graveyard) linear_sets.push_back(gone.get());
    for (LinearSet *set : linear_sets)
        if (set) {
            destroy_shared(set->color);
            destroy_shared(set->depth);
            destroy_shared(set->motion);
            destroy_shared(set->output);
            release(set->residual);
        }
    for (int i = 0; i < kLinearRing; ++i)
        for (int k = 0; k < 2; ++k) {
            release(c->lin_list[i][k]);
            release(c->lin_allocator[i][k]);
        }
    release(c->lin_fence);
    if (c->lin_fence_handle) CloseHandle((HANDLE)c->lin_fence_handle);
    if (c->lin_event) CloseHandle(c->lin_event);
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
    release(c->residual_edges);
    release(c->residual_previous_edges);
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
    c->controls.preset = next.network_preset < 0 ? 0 : next.network_preset;
    c->network_passes = next.passes < 1 ? 1 : next.passes > 8 ? 8 : next.passes;
    c->keep_local_tone = next.keep_local_tone != 0;
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
        out.network_width = c->lin_width ? c->lin_width.load() : c->slot_width;
        out.network_height = c->lin_width ? c->lin_height.load() : c->slot_height;
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
    c->last_present_ms = steady_ms();
    ++c->presents;
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
        replace(c->residual_edges, c->residual_edges_state, DXGI_FORMAT_R16G16B16A16_FLOAT);
        replace(c->residual_previous_edges, c->residual_previous_edges_state, DXGI_FORMAT_R16G16B16A16_FLOAT);
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
                // With more than one pass the network starts afresh on every one,
                // so there is no history for the next capture to continue from.
                slot.reset = f->reset || !previous || !s.network_history || s.passes > 1;
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
                    std::swap(c->residual_edges, c->residual_previous_edges);
                    std::swap(c->residual_edges_state, c->residual_previous_edges_state);
                    std::swap(c->applied_flow, c->previous_flow);
                } else {
                    c->previous_flow.valid = false;
                }
                to_state(c->residual, c->residual_state, write);
                to_state(c->residual_edges, c->residual_edges_state, write);
                Bindings b;
                b.srv[0] = slot.output.resource;
                b.srv[1] = slot.color.resource;
                b.srv[2] = slot.depth.resource;
                b.uav[0] = c->residual;
                b.uav[1] = c->residual_edges;
                b.exposure_buffer = c->exposure_buffer;
                c->passes.dispatch(cmd, Pass::Refresh, k, b, nw, nh);
                to_state(c->residual, c->residual_state, read);
                to_state(c->residual_edges, c->residual_edges_state, read);
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
                c->result_interval = (uint32_t)std::max<uint64_t>(1, c->frame - c->arrival_frame);
                c->arrival_frame = c->frame;
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
        // A result that has just arrived takes over from the one before over a
        // few frames, but not for longer than the next one takes to come: with
        // a result every frame (waiting for the network) there is nothing to
        // blend, and the new one is shown as it is.
        const int blend_frames =
            std::min(std::min(std::max(s.blend_frames, 0), 8), (int)std::min<uint32_t>(c->result_interval, 8));
        k.blend = blend_frames > 0 ? std::min(1.0f, (float)(c->residual_age + 1) / (float)blend_frames) : 1.0f;
        const bool apply = running && c->residual_valid && c->applied_flow.valid && k.fade > 0.01f &&
                           s.detail_strength != 0.0f;
        if (!apply) k.flags |= FLAG_COPY;
        if (apply && (c->applied_flow.identity || !s.follow_motion)) k.flags |= FLAG_IDENTITY;
        if (apply && c->depth_history_valid) k.flags |= FLAG_FILL;
        if (s.show_tracking) k.flags |= FLAG_SHOW_TRACKING;
        const bool older = apply && c->previous_flow.valid && !c->previous_flow.identity &&
                           s.follow_motion;
        if (older) k.flags |= FLAG_PREVIOUS;
        // Halos around edges that have moved since: needs this frame's depth.
        const bool edges = apply && c->depth_history_valid && s.follow_motion &&
                           !(c->applied_flow.identity);
        if (edges) k.flags |= FLAG_EDGES;
        if (s.composition == DLSS5NR_COMPOSITION_REPLACE) k.flags |= FLAG_REPLACE;
        const D3D12_RESOURCE_STATES output_write = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        transition(cmd, f->output, output_state, output_write);
        Bindings b;
        b.srv[0] = f->input;
        b.srv[1] = apply ? c->residual_soft : nullptr;
        b.srv[3] = apply ? c->residual : nullptr;
        b.srv[4] = apply && !c->applied_flow.identity ? c->applied_flow.map[c->applied_flow.current]
                                                                      : nullptr;
        // This frame's depth at the network's size, kept while tracking.
        b.srv[5] = apply && c->depth_history_valid ? c->depth_history[c->depth_current] : nullptr;
        b.srv[2] = older ? c->residual_previous : nullptr;
        b.srv[8] = edges ? c->residual_edges : nullptr;
        b.srv[9] = edges && older ? c->residual_previous_edges : nullptr;
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

// ------------------------------------------------------------------------------------------------------------
// The linear mode: dlss5nr_process_queue. The frame goes through the network in order and is composed from its
// own answer: capture on the queue, the network when the capture is done (the GPU waits on a fence the network's
// side signals), the composition on the queue after it. No slots, no worker thread in the way, no displacement
// maps, no result carried to a later frame. The worker only makes the images and the network's feature ready.
// ------------------------------------------------------------------------------------------------------------

namespace {

constexpr auto kLinearSizeSettle = std::chrono::milliseconds(750);
constexpr auto kLinearControlsSettle = std::chrono::milliseconds(500);
constexpr auto kLinearRetry = std::chrono::seconds(10);
constexpr int kLinearMaxFailures = 3;

// False if the fence did not reach the value in time: the GPU is stuck behind something, and the game must
// not be held for it frame after frame.
constexpr DWORD kLinearWaitMs = 2000;

bool linear_wait_cpu(dlss5nr_context *c, uint64_t value, HANDLE event) {
    if (!value || c->lin_fence->GetCompletedValue() >= value) return true;
    c->lin_fence->SetEventOnCompletion(value, event);
    if (WaitForSingleObject(event, kLinearWaitMs) == WAIT_OBJECT_0) return true;
    c->log(0, "the GPU did not reach fence value %llu in %lu ms", (unsigned long long)value, kLinearWaitMs);
    return false;
}

void linear_thread_main(dlss5nr_context *c);

// The command lists, the fence both sides wait on. False: the linear mode cannot be had.
bool linear_init(dlss5nr_context *c) {
    if (c->lin_made) return true;
    if (c->lin_failed) return false;
    c->lin_failed = true;   // until it is all made
    ID3D12Device *device = c->device;
    for (int i = 0; i < kLinearRing; ++i)
        for (int k = 0; k < 2; ++k) {
            if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                      IID_PPV_ARGS(&c->lin_allocator[i][k]))) ||
                FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, c->lin_allocator[i][k], nullptr,
                                                 IID_PPV_ARGS(&c->lin_list[i][k]))))
                return false;
            c->lin_list[i][k]->Close();
        }
    if (FAILED(device->CreateFence(0, g_copy_interop ? D3D12_FENCE_FLAG_NONE : D3D12_FENCE_FLAG_SHARED,
                                   IID_PPV_ARGS(&c->lin_fence))))
        return false;
    c->lin_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    char cpu_wait[8] = {};
    c->lin_cpu_forced =
        GetEnvironmentVariableA("DLSS5NR_LINEAR_CPU_WAIT", cpu_wait, sizeof cpu_wait) && cpu_wait[0] == '1';
    c->lin_thread = std::thread(linear_thread_main, c);
    c->lin_made = true;
    c->lin_failed = false;
    return true;
}

// Gives the fence to the network, once it is loaded (it is not when the first frame comes): the GPU then waits
// for the capture and the game's queue waits for the network, with no wait on the CPU. Otherwise the CPU does both.
void linear_import(dlss5nr_context *c) {
    c->lin_import_tried = true;
    if (!c->lin_cpu_forced && !g_copy_interop) {
        HANDLE handle = nullptr;
        std::string why;
        if (SUCCEEDED(c->device->CreateSharedHandle(c->lin_fence, nullptr, GENERIC_ALL, nullptr, &handle)) &&
            c->network.make_current(why) && c->network.import_fence(handle, &c->lin_semaphore, why)) {
            c->lin_fence_handle = handle;
            c->lin_gpu = true;
        } else {
            if (handle) CloseHandle(handle);
            c->log(1, "the linear mode will wait on the CPU: the fence could not be shared with the network (%s)",
                   why.c_str());
        }
    }
    c->log(2, "the linear mode waits for the network on the %s", c->lin_gpu ? "GPU" : "CPU");
}

void linear_thread_main(dlss5nr_context *c) {
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    for (;;) {
        LinearJob job;
        {
            std::unique_lock<std::mutex> guard(c->lin_job_lock);
            c->lin_job_wake.wait(guard, [&] { return c->lin_job_stop || !c->lin_jobs.empty(); });
            if (c->lin_jobs.empty()) break;
            job = std::move(c->lin_jobs.front());
            c->lin_jobs.pop_front();
        }
        auto trace = [&](const char *step) {
            if (c->lin_trace > 0) c->log(2, "linear job %llu: %s", (unsigned long long)job.frame, step);
        };
        std::string error;
        bool ok = !c->lin_job_stop;
        // The capture is made by the game's queue, which is behind whatever the game gave it: this waits as long as
        // that takes, here and not where the game's thread is.
        if (ok && !linear_wait_cpu(c, job.before, event)) {
            error = "the capture did not finish";
            ok = false;
        }
        trace(ok ? "capture done" : "no evaluation");
        const auto began = std::chrono::steady_clock::now();
        bool signalled = false;
        {
            std::unique_lock<std::mutex> use(c->net_lock, std::try_to_lock);
            // A feature being built holds the network for seconds: not waited for, the frame goes without.
            for (int i = 0; ok && !use.owns_lock() && i < 25; ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                use.try_lock();
            }
            std::string current;
            const bool have_context = c->network.make_current(current);
            if (ok && !use.owns_lock()) {
                error = "the network is busy";
                ok = false;
            }
            if (ok) ok = have_context && c->network.evaluate(job.desc, error, false);
            // Always: the game's queue is waiting for it, and what it shows is the last result if there is none.
            if (have_context) {
                std::string why;
                signalled = c->network.signal_fence(c->lin_semaphore, job.after, why);
                if (!signalled) c->log(0, "could not signal the fence: %s", why.c_str());
            }
        }
        trace("network queued");
        if (signalled && !linear_wait_cpu(c, job.after, event)) ok = false;
        if (ok) {
            c->evaluation_ms = (float)std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - began).count();
        } else {
            c->log(0, "the network failed on a frame: %s", error.c_str());
            c->lin_last_frame = ~0ull;
            if (++c->lin_failures >= kLinearMaxFailures)
                c->log(1, "the network failed %d times in a row: the frames go through as they are for a while",
                       kLinearMaxFailures);
        }
        trace("job done");
        --job.set->pending;
    }
    CloseHandle(event);
}

// Gives a set up: its Direct3D side once the GPU is past this frame, its CUDA side to the worker.
void linear_retire(dlss5nr_context *c, std::unique_ptr<LinearSet> set) {
    if (!set) return;
    if (set->pending > 0) {
        // The thread still has the network on these images.
        c->lin_graveyard.push_back(std::move(set));
        return;
    }
    if (set->residual) c->retired.push_back({set->residual, nullptr, c->fence_value + 1});
    set->residual = nullptr;
    SlotSet gone;
    gone.width = set->width;
    gone.height = set->height;
    gone.image[0][0] = set->color;
    gone.image[0][1] = set->depth;
    gone.image[0][2] = set->motion;
    gone.image[0][3] = set->output;
    std::lock_guard<std::mutex> guard(c->lock);
    c->evicting.push_back(std::move(gone));
    c->wake.notify_all();
}

std::unique_ptr<LinearSet> linear_make(dlss5nr_context *c, uint32_t nw, uint32_t nh, const NetworkControls &controls) {
    auto set = std::make_unique<LinearSet>();
    set->width = nw;
    set->height = nh;
    set->controls = controls;
    auto give_up = [&]() {
        destroy_shared(set->color);
        destroy_shared(set->depth);
        destroy_shared(set->motion);
        destroy_shared(set->output);
        release(set->residual);
        return std::unique_ptr<LinearSet>();
    };
    if (!make_shared(c->device, set->color, nw, nh, DXGI_FORMAT_R16G16B16A16_FLOAT, false) ||
        !make_shared(c->device, set->depth, nw, nh, DXGI_FORMAT_R32_FLOAT, false) ||
        !make_shared(c->device, set->motion, nw, nh, DXGI_FORMAT_R16G16_FLOAT, false) ||
        !make_shared(c->device, set->output, nw, nh, DXGI_FORMAT_R16G16B16A16_FLOAT, true)) {
        c->log(0, "could not create the network's %ux%u images", nw, nh);
        return give_up();
    }
    set->residual = make_texture(c->device, nw, nh, DXGI_FORMAT_R16G16B16A16_FLOAT, false,
                                 D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (!set->residual) return give_up();
    return set;
}

void linear_request(dlss5nr_context *c, LinearSet *set) {
    set->ready = 0;
    std::lock_guard<std::mutex> guard(c->lock);
    c->lin_request = set;
    c->wake.notify_all();
}

// The set to run the network on at this size, with the controls asked for, or null while there is none ready.
// A new size or new controls are not followed at once (a slider being dragged would rebuild the network dozens of
// times), and the network takes seconds to be ready for them: until it is, frames pass through as they came.
LinearSet *linear_prepare(dlss5nr_context *c, uint32_t &nw, uint32_t &nh) {
    const auto now = std::chrono::steady_clock::now();
    for (size_t i = 0; i < c->lin_graveyard.size();) {
        if (c->lin_graveyard[i]->pending == 0) {
            std::unique_ptr<LinearSet> done = std::move(c->lin_graveyard[i]);
            c->lin_graveyard.erase(c->lin_graveyard.begin() + i);
            linear_retire(c, std::move(done));
        } else {
            ++i;
        }
    }
    NetworkControls want;
    {
        std::lock_guard<std::mutex> guard(c->lock);
        want = c->controls;
    }
    if (c->lin_set && (nw != c->lin_set->width || nh != c->lin_set->height)) {
        if (c->lin_wanted_width != nw || c->lin_wanted_height != nh) {
            c->lin_wanted_width = nw;
            c->lin_wanted_height = nh;
            c->lin_wanted_since = now;
        }
        if (now - c->lin_wanted_since < kLinearSizeSettle) {
            nw = c->lin_set->width;
            nh = c->lin_set->height;
        }
    } else {
        c->lin_wanted_width = c->lin_wanted_height = 0;
    }

    if (c->lin_next) {
        const int state = c->lin_next->ready;
        if (state == 0) return nullptr;
        if (state == 1) {
            linear_retire(c, std::move(c->lin_set));
            c->lin_set = std::move(c->lin_next);
            c->lin_trace = 12;
            c->lin_width = c->lin_set->width;
            c->lin_height = c->lin_set->height;
        } else {
            linear_retire(c, std::move(c->lin_next));
            c->lin_asked_at = now;   // tried again in a while
            return nullptr;
        }
    }
    if (!c->lin_set || c->lin_set->width != nw || c->lin_set->height != nh) {
        if (!c->lin_set && c->lin_asked_at != std::chrono::steady_clock::time_point{} &&
            now - c->lin_asked_at < kLinearRetry)
            return nullptr;
        c->lin_next = linear_make(c, nw, nh, want);
        if (!c->lin_next) {
            c->lin_asked_at = now;
            return nullptr;
        }
        c->lin_asked_at = {};
        linear_request(c, c->lin_next.get());
        return nullptr;
    }
    LinearSet *set = c->lin_set.get();
    const int state = set->ready;
    if (state == 1) {
        if (set->controls != want) {
            if (want != c->lin_asked) {
                c->lin_asked = want;
                c->lin_asked_at = now;
            } else if (now - c->lin_asked_at >= kLinearControlsSettle) {
                set->controls = want;
                linear_request(c, set);   // the feature again, for the new controls
                return nullptr;
            }
        }
        return set;
    }
    if (state == -1 && now - c->lin_asked_at >= kLinearRetry) {
        c->lin_asked_at = now;
        set->controls = want;
        linear_request(c, set);
    }
    return nullptr;
}

} // namespace

extern "C" {

DLSS5NR_API int dlss5nr_process_queue(dlss5nr_context *c, ID3D12CommandQueue *queue, const dlss5nr_frame *f) {
    if (!c || !queue || !f || f->struct_size < offsetof(dlss5nr_frame, exposure) || !f->input || !f->output ||
        !f->width || !f->height)
        return -1;
    c->mark_host_ready();
    const dlss5nr_settings &s = c->settings;
    const D3D12_RESOURCE_STATES input_state = (D3D12_RESOURCE_STATES)f->input_state;
    const D3D12_RESOURCE_STATES output_state = (D3D12_RESOURCE_STATES)f->output_state;
    if (view_format(f->input->GetDesc().Format) == DXGI_FORMAT_UNKNOWN ||
        view_format(f->output->GetDesc().Format) == DXGI_FORMAT_UNKNOWN)
        return -2;

    const bool running = c->state == DLSS5NR_STATE_RUNNING && s.enabled;
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
    base.adaptation = base.exposure_mode == DLSS5NR_EXPOSURE_GAME ? 1.0f : 0.05f;
    base.flags = (f->reset ? FLAG_RESET : 0) | (f->depth_inverted ? FLAG_DEPTH_INVERTED : 0);

    // Rounded to a multiple of 8: the network works in 8x8 windows.
    uint32_t nw = (uint32_t)(f->width * s.resolution_scale + 0.5f) & ~7u;
    uint32_t nh = (uint32_t)(f->height * s.resolution_scale + 0.5f) & ~7u;
    if (nw < 64) nw = 64;
    if (nh < 64) nh = 64;

    // DLSS5NR_LINEAR_PROFILE=1: how long this call takes on the CPU, in the log, every 60 frames.
    static const bool profile = [] {
        char v[8] = {};
        return GetEnvironmentVariableA("DLSS5NR_LINEAR_PROFILE", v, sizeof v) && v[0] == '1';
    }();
    struct ProfileScope {
        dlss5nr_context *c;
        bool on;
        std::chrono::steady_clock::time_point began = std::chrono::steady_clock::now();
        ~ProfileScope() {
            if (!on) return;
            c->lin_profile_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - began).count();
            if (++c->lin_profile_calls % 60 == 0) {
                c->log(2, "linear mode: %.2f ms of CPU per call over the last 60 calls", c->lin_profile_ms / 60.0);
                c->lin_profile_ms = 0.0;
            }
        }
    } profile_scope{c, profile};

    auto trace = [&](const char *step) {
        if (c->lin_trace > 0) c->log(2, "linear frame %llu: %s", (unsigned long long)c->frame, step);
    };
    trace("begin");
    if (!linear_init(c)) return -3;
    if (c->lin_failures >= kLinearMaxFailures) {
        // Given up on for a while: it may have been the network busy building a feature.
        if (c->lin_retry_at == std::chrono::steady_clock::time_point{})
            c->lin_retry_at = std::chrono::steady_clock::now() + kLinearRetry;
        else if (std::chrono::steady_clock::now() >= c->lin_retry_at) {
            c->lin_failures = 0;
            c->lin_retry_at = {};
        }
    }
    LinearSet *set = running && c->lin_failures < kLinearMaxFailures ? linear_prepare(c, nw, nh) : nullptr;

    trace(set ? "set ready" : "no set: the frame passes through");
    const uint32_t slot = c->lin_ring++ % kLinearRing;
    linear_wait_cpu(c, c->lin_ring_value[slot], c->lin_event);
    trace("slot free");   // the lists of this slot were last used a few frames ago

    const D3D12_RESOURCE_STATES input_read = readable(input_state);
    const D3D12_RESOURCE_STATES read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    const D3D12_RESOURCE_STATES write = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    const D3D12_RESOURCE_STATES depth_state = (D3D12_RESOURCE_STATES)f->depth_state;
    const D3D12_RESOURCE_STATES motion_state = (D3D12_RESOURCE_STATES)f->motion_state;

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
        k.motion_to_px[0] = f->motion_width ? f->motion_scale_x * (float)nw / (float)f->motion_width : 0.0f;
        k.motion_to_px[1] = f->motion_height ? f->motion_scale_y * (float)nh / (float)f->motion_height : 0.0f;
        return k;
    };

    // ---------------------------------------------------------------------------------- the network's turn
    bool evaluated = false;
    std::unique_lock<std::mutex> use_network(c->net_lock, std::try_to_lock);
    if (set && use_network.owns_lock() && s.detail_strength != 0.0f) {
        if (!c->lin_import_tried) linear_import(c);
        const auto began = std::chrono::steady_clock::now();
        // The network keeps its history from the frame before, along the game's own motion vectors: every
        // frame is evaluated, so there is always one, unless there was a cut or the frames were not consecutive.
        static const bool history_allowed = [] {
            char v[8] = {};
            return !(GetEnvironmentVariableA("DLSS5NR_LINEAR_HISTORY", v, sizeof v) && v[0] == '0');
        }();
        const bool has_history = history_allowed && guided && f->reset == 0 && s.passes <= 1 &&
                                 c->lin_last_frame != ~0ull && c->lin_last_frame + 1 == c->frame;
        ID3D12GraphicsCommandList *in = c->lin_list[slot][0];
        c->lin_allocator[slot][0]->Reset();
        in->Reset(c->lin_allocator[slot][0], nullptr);
        transition(in, f->input, input_state, input_read);
        if (encoding != 1 && base.exposure_mode != DLSS5NR_EXPOSURE_FIXED) {
            Constants k = base;
            region_scale(f->input, f->width, f->height, k.uv0_scale);
            Bindings b;
            b.srv[0] = f->input;
            const D3D12_RESOURCE_STATES exposure_state = (D3D12_RESOURCE_STATES)f->exposure_state;
            if (base.exposure_mode == DLSS5NR_EXPOSURE_GAME) {
                b.srv[1] = f->exposure;
                transition(in, f->exposure, exposure_state, readable(exposure_state));
            }
            b.exposure_buffer = c->exposure_buffer;
            c->passes.dispatch(in, Pass::Exposure, k, b, 1, 1);
            uav_barrier(in, c->exposure_buffer);
            if (base.exposure_mode == DLSS5NR_EXPOSURE_GAME)
                transition(in, f->exposure, readable(exposure_state), exposure_state);
        }
        if (guided) {
            transition(in, f->depth, depth_state, readable(depth_state));
            transition(in, f->motion, motion_state, readable(motion_state));
        }
        Constants k = network_constants();
        k.flags |= has_history ? FLAG_RAWMV : FLAG_IDENTITY;
        for (ID3D12Resource *r : {set->color.resource, set->depth.resource, set->motion.resource})
            transition(in, r, D3D12_RESOURCE_STATE_COMMON, write);
        Bindings capture;
        capture.srv[0] = f->input;
        capture.srv[1] = f->depth;
        capture.srv[2] = has_history ? f->motion : nullptr;
        capture.uav[0] = set->color.resource;
        capture.uav[1] = set->depth.resource;
        capture.uav[2] = set->motion.resource;
        capture.exposure_buffer = c->exposure_buffer;
        c->passes.dispatch(in, Pass::Capture, k, capture, nw, nh);
        for (ID3D12Resource *r : {set->color.resource, set->depth.resource, set->motion.resource})
            transition(in, r, write, D3D12_RESOURCE_STATE_COMMON);
        if (guided) {
            transition(in, f->depth, readable(depth_state), depth_state);
            transition(in, f->motion, readable(motion_state), motion_state);
        }
        transition(in, f->input, input_read, input_state);
        in->Close();
        ID3D12CommandList *first[] = {in};
        trace("capture list ready");
        queue->ExecuteCommandLists(1, first);
        const uint64_t before = ++c->lin_value;
        queue->Signal(c->lin_fence, before);
        trace("capture submitted");

        EvaluateDesc desc;
        desc.color = &set->color;
        desc.depth = &set->depth;
        desc.motion = &set->motion;
        desc.output = &set->output;
        desc.depth_inverted = f->depth_inverted != 0;
        desc.reset = !has_history;
        desc.controls = set->controls;
        {
            std::lock_guard<std::mutex> guard(c->lock);
            desc.passes = c->network_passes;
            desc.keep_local_tone = c->keep_local_tone;
        }
        std::string error;
        bool ok = true;
        if (c->lin_gpu) {
            // The network runs on the thread made for it, once the capture is done; the game's queue waits for the
            // answer on the GPU. Nothing waits here.
            use_network.unlock();
            const uint64_t after = ++c->lin_value;
            LinearJob job;
            job.set = set;
            job.before = before;
            job.after = after;
            job.frame = c->frame;
            job.desc = desc;
            ++set->pending;
            {
                std::lock_guard<std::mutex> guard(c->lin_job_lock);
                c->lin_jobs.push_back(std::move(job));
            }
            c->lin_job_wake.notify_one();
            queue->Wait(c->lin_fence, after);
            trace("network queued for the thread");
        } else {
            trace("network on the CPU's wait");
            ok = c->network.make_current(error);
            if (ok && !linear_wait_cpu(c, before, c->lin_event)) {
                error = "the capture did not finish";
                ok = false;
            }
            if (ok) {
                const auto network_began = std::chrono::steady_clock::now();
                ok = c->network.evaluate(desc, error, true);
                if (ok)
                    c->evaluation_ms = (float)std::chrono::duration<double, std::milli>(
                                           std::chrono::steady_clock::now() - network_began).count();
            }
        }
        if (ok) {
            c->lin_failures = 0;
            c->lin_last_frame = c->frame;
            evaluated = true;
            ++c->results;
            c->result_age = 0;
            c->result_latency = 0;
            if (c->lin_last_began != std::chrono::steady_clock::time_point{}) {
                const double period = std::chrono::duration<double>(began - c->lin_last_began).count();
                if (period > 0.0)
                    c->evaluations_per_second = 0.9f * c->evaluations_per_second.load() + 0.1f * (float)(1.0 / period);
            }
            c->lin_last_began = began;
        } else {
            c->log(0, "the network failed on a frame: %s", error.c_str());
            c->lin_last_frame = ~0ull;
            if (++c->lin_failures >= kLinearMaxFailures)
                c->log(1, "the network failed %d times in a row: the frames go through as they are for a while",
                       kLinearMaxFailures);
        }
    }
    if (use_network.owns_lock()) use_network.unlock();
    trace("network done");

    // ------------------------------------------------------------------------------------- the composition
    ID3D12GraphicsCommandList *out = c->lin_list[slot][1];
    c->lin_allocator[slot][1]->Reset();
    out->Reset(c->lin_allocator[slot][1], nullptr);
    transition(out, f->input, input_state, input_read);
    if (evaluated) {
        // What the network made of it, as a luminance ratio against the image it was shown.
        transition(out, set->output.resource, D3D12_RESOURCE_STATE_COMMON, read);
        transition(out, set->color.resource, D3D12_RESOURCE_STATE_COMMON, read);
        transition(out, set->depth.resource, D3D12_RESOURCE_STATE_COMMON, read);
        transition(out, set->residual, set->residual_state, write);
        Bindings b;
        b.srv[0] = set->output.resource;
        b.srv[1] = set->color.resource;
        b.srv[2] = set->depth.resource;
        b.uav[0] = set->residual;
        b.exposure_buffer = c->exposure_buffer;
        c->passes.dispatch(out, Pass::Refresh, network_constants(), b, nw, nh);
        transition(out, set->residual, write, read);
        set->residual_state = read;
        transition(out, set->output.resource, read, D3D12_RESOURCE_STATE_COMMON);
        transition(out, set->color.resource, read, D3D12_RESOURCE_STATE_COMMON);
        transition(out, set->depth.resource, read, D3D12_RESOURCE_STATE_COMMON);
    }
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
        k.fade = 1.0f;
        k.blend = 1.0f;
        // The network's own answer for this frame, laid on it where it was made; or the frame as it came.
        k.flags |= evaluated ? FLAG_IDENTITY : FLAG_COPY;
        if (s.composition == DLSS5NR_COMPOSITION_REPLACE) k.flags |= FLAG_REPLACE;
        transition(out, f->output, output_state, write);
        Bindings b;
        b.srv[0] = f->input;
        b.srv[3] = evaluated ? set->residual : nullptr;
        b.uav[0] = f->output;
        b.exposure_buffer = c->exposure_buffer;
        c->passes.dispatch(out, Pass::Compose, k, b, f->width, f->height);
        if (output_state == write) uav_barrier(out, f->output);
        transition(out, f->output, write, output_state);
    }
    transition(out, f->input, input_read, input_state);
    out->Close();
    ID3D12CommandList *second[] = {out};
    queue->ExecuteCommandLists(1, second);
    c->lin_ring_value[slot] = ++c->lin_value;
    queue->Signal(c->lin_fence, c->lin_ring_value[slot]);
    trace("composition submitted");
    if (c->lin_trace > 0) --c->lin_trace;
    return 0;
}

} // extern "C"
