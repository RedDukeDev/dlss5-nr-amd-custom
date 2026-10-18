// dlss5nr_host.exe: the 64-bit half of the bridge for 32-bit games. The client (a d3d9.dll in
// the game) leaves the finished frame in shared memory; this process runs the network on it,
// through the same runtime the 64-bit games use, and leaves what the network did to the picture
// as a gain per channel. See ../common/protocol.h.
//
//   dlss5nr_host.exe --client <process id of the game>
//
// The runtime, ZLUDA, the kernels and the network are looked for beside this executable
// (OptiScaler\dlss5nr in a game's folder).
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <stdlib.h>
#include <chrono>
#include <string>
#include <thread>
#include <vector>
#include "../common/protocol.h"
#include "../../runtime/include/dlss5nr.h"

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

namespace {

FILE *g_log = nullptr;

void log(const char *format, ...) {
    if (!g_log) return;
    SYSTEMTIME t;
    GetLocalTime(&t);
    fprintf(g_log, "[%02d:%02d:%02d.%03d] ", t.wHour, t.wMinute, t.wSecond, t.wMilliseconds);
    va_list args;
    va_start(args, format);
    vfprintf(g_log, format, args);
    va_end(args);
    fputc('\n', g_log);
    fflush(g_log);
}

template <class T> void release(T *&p) {
    if (p) p->Release();
    p = nullptr;
}

// ------------------------------------------------------------------ the runtime

struct Runtime {
    HMODULE dll = nullptr;
    decltype(&dlss5nr_create) create = nullptr;
    decltype(&dlss5nr_destroy) destroy = nullptr;
    decltype(&dlss5nr_default_settings) default_settings = nullptr;
    decltype(&dlss5nr_set_settings) set_settings = nullptr;
    decltype(&dlss5nr_process) process = nullptr;
    decltype(&dlss5nr_present) present = nullptr;
    decltype(&dlss5nr_get_status) get_status = nullptr;

    bool load(const std::wstring &dir) {
        dll = LoadLibraryExW((dir + L"dlss5nr_runtime.dll").c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (!dll) return false;
#define BIND(name) name = (decltype(name))GetProcAddress(dll, "dlss5nr_" #name)
        BIND(create);
        BIND(destroy);
        BIND(default_settings);
        BIND(set_settings);
        BIND(process);
        BIND(present);
        BIND(get_status);
#undef BIND
        return create && destroy && default_settings && set_settings && process && present && get_status;
    }
};

void runtime_log(void *, int level, const char *message) {
    static const char *const kLevel[] = {"error", "warning", "info", "debug"};
    if (level <= 2) log("runtime %s: %s", kLevel[level < 0 ? 0 : level > 3 ? 3 : level], message);
}

// ------------------------------------------------------------------ Direct3D 12

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
        if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)))) return false;
        // The card with the most memory of its own: the one the game is on, as a rule.
        IDXGIAdapter1 *best = nullptr;
        SIZE_T best_memory = 0;
        IDXGIAdapter1 *adapter = nullptr;
        for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_ADAPTER_DESC1 desc{};
            adapter->GetDesc1(&desc);
            if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) && desc.DedicatedVideoMemory >= best_memory) {
                if (best) best->Release();
                best = adapter;
                best_memory = desc.DedicatedVideoMemory;
            } else {
                adapter->Release();
            }
        }
        if (best) {
            DXGI_ADAPTER_DESC1 desc{};
            best->GetDesc1(&desc);
            log("adapter: %ls", desc.Description);
            D3D12CreateDevice(best, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device));
            best->Release();
        }
        factory->Release();
        if (!device) return false;
        D3D12_COMMAND_QUEUE_DESC q{};
        q.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        if (FAILED(device->CreateCommandQueue(&q, IID_PPV_ARGS(&queue))) ||
            FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))) ||
            FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, nullptr, IID_PPV_ARGS(&cmd))) ||
            FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))))
            return false;
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

// The buffers of one frame size.
struct Frame {
    UINT w = 0, h = 0;
    ID3D12Resource *input = nullptr, *output = nullptr, *upload = nullptr, *readback = nullptr;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 bytes = 0;
    std::vector<uint8_t> rgba;   // the capture as the network reads it

    void release_all() {
        release(input);
        release(output);
        release(upload);
        release(readback);
    }

    bool make(Gpu &gpu, UINT width, UINT height) {
        release_all();
        w = width;
        h = height;
        input = gpu.texture(w, h, DXGI_FORMAT_R8G8B8A8_UNORM, false, D3D12_RESOURCE_STATE_COPY_DEST);
        output = gpu.texture(w, h, DXGI_FORMAT_R8G8B8A8_UNORM, true, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        const D3D12_RESOURCE_DESC d = input->GetDesc();
        gpu.device->GetCopyableFootprints(&d, 0, 1, 0, &footprint, nullptr, nullptr, &bytes);
        upload = gpu.buffer(bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        readback = gpu.buffer(bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        rgba.assign((size_t)w * h * 4, 0);
        return input && output && upload && readback;
    }
};

// ------------------------------------------------------------------ the settings

float g_smoothing = bridge::kDefaultSmoothing;   // how much of a result's change is held back where the picture is still

void apply_settings(const bridge::Settings &user, dlss5nr_settings &s) {
    dlss5nr_settings defaults = s;
    (void)defaults;
    auto f = [](float v, float &to) { if (v > bridge::kAutoFloat / 2) to = v; };
    auto i = [](int32_t v, int32_t &to) { if (v != bridge::kAutoInt) to = v; };
    f(user.resolutionScale, s.resolution_scale);
    i(user.style, s.style);
    f(user.intensity, s.intensity);
    f(user.localTone, s.local_tone);
    f(user.localStructure, s.local_structure);
    f(user.skinStructure, s.skin_structure);
    f(user.detailStrength, s.detail_strength);
    f(user.colourStrength, s.colour_strength);
    f(user.maxRatio, s.max_ratio);
    i(user.passes, s.passes);
    i(user.keepLocalTone, s.keep_local_tone);
    i(user.preset, s.network_preset);
    g_smoothing = user.smoothing > bridge::kAutoFloat / 2 ? std::min(1.0f, std::max(0.0f, user.smoothing)) : bridge::kDefaultSmoothing;
}

// What the runtime is running with, for the client's menu to show where the user chose nothing.
void fill_effective(const dlss5nr_settings &s, bridge::Settings &e) {
    e.enabled = 1;
    e.resolutionScale = s.resolution_scale;
    e.style = s.style;
    e.intensity = s.intensity;
    e.localTone = s.local_tone;
    e.localStructure = s.local_structure;
    e.skinStructure = s.skin_structure;
    e.detailStrength = s.detail_strength;
    e.colourStrength = s.colour_strength;
    e.maxRatio = s.max_ratio;
    e.passes = s.passes;
    e.keepLocalTone = s.keep_local_tone;
    e.preset = s.network_preset;
    e.smoothing = g_smoothing;
}

// What the host needs of the runtime, whatever the user chose: the frame waits for the
// network and carries nothing along (there are no motion vectors), because the result is
// turned into a gain and applied by the client to a later frame.
void force_settings(dlss5nr_settings &s) {
    s.enabled = 1;
    // The client sends the picture at the network's resolution already, and the gain it gets back is
    // stretched over the frame: the runtime works at the size it is given.
    s.resolution_scale = 1.0f;
    s.composition = DLSS5NR_COMPOSITION_RATIO;
    s.encoding = DLSS5NR_ENCODING_DISPLAY;
    s.wait_for_network = 1;
    s.wait_inside = 0;
    s.follow_motion = 0;
    s.network_history = 0;
    s.blend_frames = 0;
    s.max_age = 1000;
    s.show_tracking = 0;
}

// ------------------------------------------------------------------ the gain

// Runs fn(first_row, end_row) over the rows on several threads: at 1440p the loops over pixels
// below take longer than the network does.
template <class F> void parallel_rows(UINT rows, F fn) {
    const unsigned threads = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
    if (threads == 1 || rows < 64) {
        fn(0u, rows, 0u);
        return;
    }
    const UINT per = (rows + threads - 1) / threads;
    std::vector<std::thread> pool;
    for (unsigned i = 1; i < threads; ++i) {
        const UINT a = i * per, b = std::min(rows, a + per);
        if (a < b) pool.emplace_back([=, &fn] { fn(a, b, i); });
    }
    fn(0u, std::min(rows, per), 0u);
    for (std::thread &t : pool) t.join();
}

// What the network did to each pixel, as a gain on the capture: out = in * gain. A small offset
// keeps dark pixels, where one step of the 8-bit image is a large fraction, from producing wild
// gains.
//
// The gain is also steadied over time. The network decides some of its detail and tone anew with every
// picture, so a picture that has hardly changed can still get a gain that has: that is the flicker.
// `gain` holds the result of the last call, and where the picture is within a few steps of what it
// was then (`previous`), the new gain moves it only part of the way; where the picture has changed,
// all the way. `hold` is how much is held back in the pixels that have not changed at all.
//
// Returns the rms error, in 8-bit steps, between in * gain and the network's own output, before the
// steadying: what the gain loses by being a gain (its clamp at kGainRange, its 16 bits, the offset).
double compute_gain(const uint8_t *in, const uint8_t *previous, const uint8_t *out, UINT w, UINT h, UINT out_pitch, uint16_t *gain,
                    float hold) {
    constexpr float kOffset = 2.0f;
    static float inverse[256];
    static bool ready = false;
    if (!ready) {
        for (int v = 0; v < 256; ++v) inverse[v] = 1.0f / ((float)v + kOffset);
        ready = true;
    }
    // How much of the new gain to take, by how much the pixel has changed: held back below kStill
    // steps, taken whole from kMoved steps on.
    constexpr int kStill = 2, kMoved = 12;
    float take[256];
    for (int d = 0; d < 256; ++d) {
        const float t = d <= kStill ? 0.0f : d >= kMoved ? 1.0f : (float)(d - kStill) / (float)(kMoved - kStill);
        take[d] = (1.0f - hold) + hold * t;
    }
    const float scale = 65535.0f / bridge::kGainRange;
    double errors[8] = {};
    parallel_rows(h, [&](UINT first, UINT end, unsigned slot) {
        double error = 0.0;
        for (UINT y = first; y < end; ++y) {
            const uint8_t *a = in + (size_t)y * w * 4;
            const uint8_t *p = previous ? previous + (size_t)y * w * 4 : nullptr;
            const uint8_t *b = out + (size_t)y * out_pitch;
            uint16_t *g = gain + (size_t)y * w * 4;
            for (UINT x = 0; x < w; ++x) {
                float keep = 1.0f;   // how much of the new gain is taken
                if (p) {
                    int change = 0;
                    for (int c = 0; c < 3; ++c) change = std::max(change, std::abs((int)a[4 * x + c] - (int)p[4 * x + c]));
                    keep = take[change];
                }
                for (int c = 0; c < 3; ++c) {
                    float r = ((float)b[4 * x + c] + kOffset) * inverse[a[4 * x + c]];
                    if (r > bridge::kGainRange) r = bridge::kGainRange;
                    const uint16_t q = (uint16_t)(r * scale + 0.5f);
                    const float back = (float)a[4 * x + c] * (float)q / scale;
                    const float d = back - (float)b[4 * x + c];
                    error += (double)d * d;
                    g[4 * x + c] = keep >= 1.0f ? q : (uint16_t)((float)g[4 * x + c] + ((float)q - (float)g[4 * x + c]) * keep + 0.5f);
                }
                g[4 * x + 3] = 65535;
            }
        }
        errors[slot] += error;
    });
    double total = 0.0;
    for (double e : errors) total += e;
    return sqrt(total / ((double)w * h * 3));
}

} // namespace

int wmain(int argc, wchar_t **argv) {
    DWORD client_pid = 0;
    for (int i = 1; i + 1 < argc; ++i)
        if (!wcscmp(argv[i], L"--client")) client_pid = (DWORD)_wtoi(argv[i + 1]);

    wchar_t path[MAX_PATH];
    GetModuleFileNameW(nullptr, path, MAX_PATH);
    std::wstring dir = path;
    dir = dir.substr(0, dir.find_last_of(L'\\') + 1);
    g_log = _wfsopen((dir + L"dlss5nr_host.log").c_str(), L"w", _SH_DENYNO);
    log("host started for client %lu", client_pid);
    if (!client_pid) return 2;

    HANDLE client = OpenProcess(SYNCHRONIZE, FALSE, client_pid);
    wchar_t name[96];
    swprintf(name, 96, L"Local\\dlss5nr_bridge_%lu", client_pid);
    HANDLE mapping = OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, name);
    swprintf(name, 96, L"Local\\dlss5nr_bridge_capture_%lu", client_pid);
    HANDLE capture_event = OpenEventW(SYNCHRONIZE, FALSE, name);
    swprintf(name, 96, L"Local\\dlss5nr_bridge_result_%lu", client_pid);
    HANDLE result_event = OpenEventW(EVENT_MODIFY_STATE, FALSE, name);
    if (!client || !mapping || !capture_event || !result_event) {
        log("could not open the client's objects (error %lu)", GetLastError());
        return 3;
    }
    uint8_t *view = (uint8_t *)MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    bridge::Header *h = (bridge::Header *)view;
    if (!view || h->magic != bridge::kMagic || h->version != bridge::kVersion) {
        log("the shared memory is not what this host speaks (magic %08x version %u)", view ? h->magic : 0, view ? h->version : 0);
        return 4;
    }
    auto fail = [&](const char *message) {
        log("failed: %s", message);
        snprintf(h->message, sizeof h->message, "%s", message);
        InterlockedExchange(&h->hostState, bridge::kFailed);
        return 1;
    };

    Runtime rt;
    if (!rt.load(dir)) return fail("dlss5nr_runtime.dll could not be loaded");
    Gpu gpu;
    if (!gpu.init()) return fail("no Direct3D 12 device");

    dlss5nr_create_info info{};
    info.struct_size = sizeof info;
    info.device = (ID3D12Device *)gpu.device;
    info.runtime_dir = dir.c_str();
    info.log = &runtime_log;
    dlss5nr_context *nr = nullptr;
    if (rt.create(&info, &nr) != 0) return fail("dlss5nr_create failed");

    dlss5nr_settings settings;
    rt.default_settings(&settings);
    int32_t seen_settings = -1;
    auto refresh_settings = [&]() {
        rt.default_settings(&settings);
        apply_settings(h->settings, settings);
        force_settings(settings);
        rt.set_settings(nr, &settings);
        fill_effective(settings, h->effective);
        h->effective.resolutionScale = h->settings.resolutionScale > bridge::kAutoFloat / 2 ? h->settings.resolutionScale : bridge::kDefaultScale;
        seen_settings = h->settingsSeq;
        log("settings: scale %.2f, detail %.2f, colour %.2f, preset %d", settings.resolution_scale,
            settings.detail_strength, settings.colour_strength, settings.network_preset);
    };
    refresh_settings();

    // The network loads on a thread of its own, on the first calls into the runtime.
    dlss5nr_status status{};
    status.struct_size = sizeof status;
    const auto started = std::chrono::steady_clock::now();
    snprintf(h->message, sizeof h->message, "loading the network");
    for (;;) {
        rt.get_status(nr, &status);
        if (status.state == DLSS5NR_STATE_RUNNING || status.state == DLSS5NR_STATE_FAILED) break;
        if (WaitForSingleObject(client, 100) == WAIT_OBJECT_0) return 0;
        if (h->shutdown) return 0;
    }
    if (status.state == DLSS5NR_STATE_FAILED) return fail(status.message);
    log("the network is running after %.1f s: %ux%u",
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count(), status.network_width,
        status.network_height);
    InterlockedExchange(&h->hostState, bridge::kRunning);

    Frame frame;
    std::vector<uint16_t> gain;
    std::vector<uint8_t> previous;    // the picture the last result was made from
    bool previous_valid = false;
    std::chrono::steady_clock::time_point previous_at;
    unsigned cycles = 0;
    for (;;) {
        InterlockedExchange(&h->hostReady, 1);
        HANDLE waits[2] = {capture_event, client};
        const DWORD woke = WaitForMultipleObjects(2, waits, FALSE, 500);
        if (woke == WAIT_OBJECT_0 + 1) break;   // the game ended
        if (h->shutdown) break;
        if (woke != WAIT_OBJECT_0) continue;
        if (seen_settings != h->settingsSeq) refresh_settings();

        const UINT w = h->width, hh = h->height;
        if (!w || !hh || w > bridge::kMaxWidth || hh > bridge::kMaxHeight) continue;
        if (w != frame.w || hh != frame.h) {
            if (!frame.make(gpu, w, hh)) return fail("could not create the frame's buffers");
            gain.assign((size_t)w * hh * 4, 0);
            previous_valid = false;
            log("frame size %ux%u", w, hh);
        }
        const auto t0 = std::chrono::steady_clock::now();

        // BGRA to RGBA, into the picture the gain is measured against and into the upload buffer.
        const uint8_t *capture = view + bridge::capture_offset();
        uint8_t *mapped = nullptr;
        frame.upload->Map(0, nullptr, (void **)&mapped);
        parallel_rows(hh, [&](UINT first, UINT end, unsigned) {
            for (UINT y = first; y < end; ++y) {
                const uint8_t *src = capture + (size_t)y * w * 4;
                uint8_t *rgba = &frame.rgba[(size_t)y * w * 4];
                uint8_t *row = mapped + frame.footprint.Offset + (size_t)y * frame.footprint.Footprint.RowPitch;
                for (UINT x = 0; x < w; ++x) {
                    rgba[4 * x + 0] = row[4 * x + 0] = src[4 * x + 2];
                    rgba[4 * x + 1] = row[4 * x + 1] = src[4 * x + 1];
                    rgba[4 * x + 2] = row[4 * x + 2] = src[4 * x + 0];
                    rgba[4 * x + 3] = row[4 * x + 3] = 255;
                }
            }
        });
        frame.upload->Unmap(0, nullptr);

        const auto t_swizzled = std::chrono::steady_clock::now();
        int frames_run = 0;
        // The same picture until the network has answered for it: the first frame hands it
        // over, the ones after compose what comes back.
        rt.get_status(nr, &status);
        const uint32_t before = status.results;
        bool composed = false;
        for (int n = 0; n < 200 && !composed; ++n) {
            gpu.begin();
            if (n == 0) {
                D3D12_TEXTURE_COPY_LOCATION to{}, from{};
                to.pResource = frame.input;
                to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                from.pResource = frame.upload;
                from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                from.PlacedFootprint = frame.footprint;
                gpu.cmd->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
                barrier(gpu.cmd, frame.input, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            }
            dlss5nr_frame f{};
            f.struct_size = sizeof f;
            f.command_list = gpu.cmd;
            f.input = frame.input;
            f.input_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            f.output = frame.output;
            f.output_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            f.width = w;
            f.height = hh;
            f.pre_exposure = 1.0f;
            f.reset = 0;
            if (rt.process(nr, &f) != 0) return fail("dlss5nr_process failed");
            gpu.submit_and_wait();
            rt.present(nr, gpu.queue);
            rt.get_status(nr, &status);
            if (status.state == DLSS5NR_STATE_FAILED) return fail(status.message);
            ++frames_run;
            composed = status.results > before;
            if (!composed) std::this_thread::sleep_for(std::chrono::milliseconds(2));
            if (h->shutdown || WaitForSingleObject(client, 0) == WAIT_OBJECT_0) return 0;
        }
        if (!composed) {
            log("the network did not answer");
            continue;
        }

        const auto t_network = std::chrono::steady_clock::now();
        // The last frame's output, back to the CPU.
        gpu.begin();
        barrier(gpu.cmd, frame.output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION to{}, from{};
        to.pResource = frame.readback;
        to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        to.PlacedFootprint = frame.footprint;
        from.pResource = frame.output;
        from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        gpu.cmd->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
        barrier(gpu.cmd, frame.output, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        gpu.submit_and_wait();
        uint8_t *out = nullptr;
        frame.readback->Map(0, nullptr, (void **)&out);
        // The steadying needs the picture of the last result, close enough in time: not after a pause.
        const bool have_previous = previous_valid && previous.size() == frame.rgba.size() &&
                                   std::chrono::duration<double, std::milli>(t_swizzled - previous_at).count() < 600.0;
        const double error = compute_gain(frame.rgba.data(), have_previous ? previous.data() : nullptr, out + frame.footprint.Offset, w, hh,
                                          frame.footprint.Footprint.RowPitch, gain.data(), g_smoothing);
        previous = frame.rgba;
        previous_valid = true;
        previous_at = t_swizzled;
        frame.readback->Unmap(0, nullptr);

        // For testing: a slower network, to see what the client does with an old result.
        static const int delay_ms = GetEnvironmentVariableW(L"DLSS5NR_HOST_DELAY_MS", nullptr, 0) ? _wtoi(_wgetenv(L"DLSS5NR_HOST_DELAY_MS")) : 0;
        if (delay_ms > 0) Sleep(delay_ms);
        const auto t_gain = std::chrono::steady_clock::now();
        memcpy(view + bridge::result_offset(w, hh), gain.data(), (size_t)w * hh * 8);
        h->resultWidth = w;
        h->resultHeight = hh;
        h->evaluationMs = status.evaluation_ms;
        h->evaluationsPerSecond = status.evaluations_per_second;
        h->networkWidth = status.network_width;
        h->networkHeight = status.network_height;
        h->nativeKernels = status.native_kernels;
        InterlockedIncrement(&h->resultSeq);
        SetEvent(result_event);
        ++cycles;
        if (cycles <= 3 || cycles % 20 == 0) {
            const auto t_end = std::chrono::steady_clock::now();
            auto ms = [](std::chrono::steady_clock::time_point x, std::chrono::steady_clock::time_point y) {
                return std::chrono::duration<double, std::milli>(y - x).count();
            };
            log("cycle %u: %.0f ms = swizzle+upload %.0f, runtime (%d frames) %.0f, readback+gain %.0f, write %.0f; network %.0f ms; gain error %.2f steps rms",
                cycles, ms(t0, t_end), ms(t0, t_swizzled), frames_run, ms(t_swizzled, t_network), ms(t_network, t_gain), ms(t_gain, t_end),
                status.evaluation_ms, error);
        }
    }
    log("the game ended; stopping");
    rt.destroy(nr);
    return 0;
}
