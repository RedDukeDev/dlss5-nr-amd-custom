#include "network.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>

#include <cstdarg>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "../common/cuda_min.h"
#include "../common/ngx_cuda.h"

namespace dlss5nr {
namespace {

void (*g_log)(int, const char *) = nullptr;

void log(int level, const char *fmt, ...) {
    if (!g_log) return;
    char line[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(line, sizeof line, fmt, ap);
    va_end(ap);
    g_log(level, line);
}

// The snippet reports fatal conditions to its log and then terminates the
// process, so these lines are often the only explanation there is.
void __cdecl ngx_log_sink(const char *message, int level, unsigned feature) {
    (void)feature;
    if (!g_log || !message) return;
    char line[1024];
    snprintf(line, sizeof line, "[ngx %d] %s", level, message);
    const size_t n = strlen(line);
    if (n && line[n - 1] == '\n') line[n - 1] = 0;
    g_log(3, line);
}

struct State {
    HMODULE nvapi = nullptr;
    HMODULE zluda = nullptr;
    HMODULE gate = nullptr;
    HMODULE ngx = nullptr;
    CudaApi cu{};
    CUcontext context = nullptr;

    const char *(*ngx_load)(const wchar_t *) = nullptr;
    NVSDK_NGX_Result (*ngx_init)(unsigned long long, const wchar_t *, NVSDK_NGX_Version,
                                 unsigned long long, unsigned, void *) = nullptr;
    NVSDK_NGX_Result (*ngx_populate)(NVSDK_NGX_Parameter **) = nullptr;
    NVSDK_NGX_Result (*ngx_create)(NVSDK_NGX_Feature, NVSDK_NGX_Parameter *, NVSDK_NGX_Handle **) = nullptr;
    NVSDK_NGX_Result (*ngx_evaluate)(NVSDK_NGX_Handle *, const NVSDK_NGX_Parameter *) = nullptr;
    NVSDK_NGX_Result (*ngx_release)(NVSDK_NGX_Handle *) = nullptr;
    NVSDK_NGX_Result (*ngx_shutdown)(void) = nullptr;

    NVSDK_NGX_Parameter *params = nullptr;
    NVSDK_NGX_Handle *feature = nullptr;
    std::wstring data_dir;
};

State g;

// Copy mode: the queue the pixels are moved on, and the wait for it.
struct Copier {
    ID3D12CommandQueue *queue = nullptr;
    ID3D12CommandAllocator *allocator = nullptr;
    ID3D12GraphicsCommandList *list = nullptr;
    ID3D12Fence *fence = nullptr;
    HANDLE event = nullptr;
    uint64_t value = 0;
} copier;

bool copier_start(ID3D12Device *device, std::string &error) {
    if (copier.queue) return true;
    D3D12_COMMAND_QUEUE_DESC queue{};
    queue.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(device->CreateCommandQueue(&queue, IID_PPV_ARGS(&copier.queue))) ||
        FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&copier.allocator))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, copier.allocator, nullptr,
                                         IID_PPV_ARGS(&copier.list))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&copier.fence)))) {
        error = "could not create the copy queue";
        return false;
    }
    copier.list->Close();
    copier.event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    return true;
}

// Records what `record` says, runs it, and waits until the GPU is done.
template <class Record> bool copier_run(Record record, std::string &error) {
    copier.allocator->Reset();
    copier.list->Reset(copier.allocator, nullptr);
    record(copier.list);
    copier.list->Close();
    ID3D12CommandList *lists[] = {copier.list};
    copier.queue->ExecuteCommandLists(1, lists);
    copier.queue->Signal(copier.fence, ++copier.value);
    if (copier.fence->GetCompletedValue() < copier.value) {
        copier.fence->SetEventOnCompletion(copier.value, copier.event);
        if (WaitForSingleObject(copier.event, 10000) != WAIT_OBJECT_0) {
            error = "the copy queue did not finish";
            return false;
        }
    }
    return true;
}

void copier_barrier(ID3D12GraphicsCommandList *cmd, ID3D12Resource *resource, D3D12_RESOURCE_STATES from,
                    D3D12_RESOURCE_STATES to) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = resource;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = from;
    b.Transition.StateAfter = to;
    cmd->ResourceBarrier(1, &b);
}

D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint_of(const SharedImage &t) {
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT f{};
    f.Footprint.Format = (DXGI_FORMAT)t.dxgi_format;
    f.Footprint.Width = t.width;
    f.Footprint.Height = t.height;
    f.Footprint.Depth = 1;
    f.Footprint.RowPitch = t.row_pitch;
    return f;
}

bool cu_ok(CUresult r, const char *what, std::string &error) {
    if (r == CUDA_SUCCESS) return true;
    const char *text = nullptr;
    if (g.cu.cuGetErrorString) g.cu.cuGetErrorString(r, &text);
    char line[256];
    snprintf(line, sizeof line, "%s failed: %d (%s)", what, r, text ? text : "?");
    error = line;
    return false;
}

bool ngx_ok(NVSDK_NGX_Result r, const char *what, std::string &error) {
    if (NVSDK_NGX_SUCCEED(r)) return true;
    char line[256];
    snprintf(line, sizeof line, "%s failed: 0x%08X", what, r);
    error = line;
    return false;
}

std::string narrow(const std::wstring &text) {
    std::string out;
    for (wchar_t c : text) out += (c < 128) ? (char)c : '?';
    return out;
}

// The runtime never translates the network's code itself: a game would wait
// tens of minutes for it. It runs from the cache and the kernels it ships with,
// and refuses to start when either is missing or does not fit this machine.
// Whoever builds the cache, or has none, sets DLSS5NR_ALLOW_TRANSLATION=1 to
// let ZLUDA translate as the network loads.
bool translation_allowed() {
    char value[8] = {};
    return GetEnvironmentVariableA("DLSS5NR_ALLOW_TRANSLATION", value, sizeof value) > 0 && value[0] == '1';
}

// The GPU targets the shipped cache holds, from the coverage.txt that
// prepare_cache.py writes beside the database.
bool read_coverage(const std::wstring &cache, std::set<std::string> &targets, std::string &error) {
    const std::wstring database = cache + L"\\zluda2.db";
    if (GetFileAttributesW(database.c_str()) == INVALID_FILE_ATTRIBUTES) {
        error = "the translation cache is missing: " + narrow(database);
        return false;
    }
    FILE *f = _wfopen((cache + L"\\coverage.txt").c_str(), L"r");
    if (!f) {
        error = "the translation cache has no coverage.txt: " + narrow(cache);
        return false;
    }
    char line[256];
    while (fgets(line, sizeof line, f)) {
        char word[64] = {};
        if (sscanf(line, "target %63s", word) == 1) targets.insert(word);
    }
    fclose(f);
    if (targets.empty()) {
        error = "the translation cache's coverage.txt names no GPU target: " + narrow(cache);
        return false;
    }
    return true;
}

// The gfx names of the GPUs the HIP runtime ZLUDA sits on can see. Read from
// the device properties by looking for the name in the structure, which spares
// this file the layout of a structure that changes between ROCm releases.
std::vector<std::string> gpu_architectures() {
    std::vector<std::string> found;
    HMODULE hip = nullptr;
    for (const wchar_t *name : {L"amdhip64_7.dll", L"amdhip64_6.dll", L"amdhip64.dll"})
        if ((hip = GetModuleHandleW(name))) break;
    if (!hip) return found;
    auto count = reinterpret_cast<int (*)(int *)>(GetProcAddress(hip, "hipGetDeviceCount"));
    auto properties = reinterpret_cast<int (*)(void *, int)>(GetProcAddress(hip, "hipGetDevicePropertiesR0600"));
    int devices = 0;
    if (!count || !properties || count(&devices) != 0) return found;
    for (int i = 0; i < devices; ++i) {
        std::vector<char> buffer(16384, 0);
        if (properties(buffer.data(), i) != 0) continue;
        for (size_t k = 0; k + 4 < buffer.size(); ++k) {
            if (memcmp(&buffer[k], "gfx", 3) != 0 || buffer[k + 3] < '0' || buffer[k + 3] > '9') continue;
            std::string arch;
            for (size_t j = k; j < buffer.size() && (isalnum((unsigned char)buffer[j]) || buffer[j] == '-'); ++j)
                arch += buffer[j];
            found.push_back(arch);
            break;
        }
    }
    return found;
}

// An architecture is served by its own translation, or by the generic one of
// its family.
bool cache_serves(const std::set<std::string> &targets, const std::string &arch) {
    if (targets.count(arch)) return true;
    const char *generic = arch.compare(0, 6, "gfx103") == 0 ? "gfx10-3-generic"
                          : arch.compare(0, 5, "gfx11") == 0 ? "gfx11-generic"
                          : arch.compare(0, 5, "gfx12") == 0 ? "gfx12-generic"
                                                              : nullptr;
    return generic && targets.count(generic);
}

// Loads a library from exactly this path. Its own dependencies are looked for
// beside it first (LOAD_WITH_ALTERED_SEARCH_PATH), so nothing has to be added
// to the game's DLL search path -- which the game, and OptiScaler, rely on.
HMODULE load_exact(const std::wstring &path, std::string &error) {
    HMODULE module = LoadLibraryExW(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!module) {
        char line[512];
        snprintf(line, sizeof line, "could not load %s (Windows error %lu)", narrow(path).c_str(),
                 GetLastError());
        error = line;
        return nullptr;
    }
    // LoadLibrary matches an already loaded module by base name, so a module
    // of the same name loaded earlier by someone else is what comes back. Say
    // so: the failure it leads to is far from here.
    wchar_t actual[MAX_PATH] = L"";
    GetModuleFileNameW(module, actual, MAX_PATH);
    if (_wcsicmp(actual, path.c_str()) != 0)
        log(1, "asked for %s, the process already had %s", narrow(path).c_str(), narrow(actual).c_str());
    return module;
}

const char *load_cuda_api(HMODULE h, CudaApi &api) {
    // The _v2 name first: over ZLUDA several bare names are v1-ABI stubs that
    // return CUDA_ERROR_NOT_SUPPORTED.
#define LOAD(field, name)                                                  \
    do {                                                                   \
        void *p = (void *)GetProcAddress(h, name "_v2");                    \
        if (!p) p = (void *)GetProcAddress(h, name);                        \
        if (!p) return name;                                                \
        *reinterpret_cast<void **>(&api.field) = p;                         \
    } while (0)
    LOAD(cuInit, "cuInit");
    LOAD(cuDeviceGet, "cuDeviceGet");
    LOAD(cuDeviceGetCount, "cuDeviceGetCount");
    LOAD(cuDeviceGetName, "cuDeviceGetName");
    LOAD(cuDeviceGetAttribute, "cuDeviceGetAttribute");
    LOAD(cuCtxCreate, "cuCtxCreate");
    LOAD(cuCtxDestroy, "cuCtxDestroy");
    LOAD(cuCtxSetCurrent, "cuCtxSetCurrent");
    // The bare name only. CUDA has no cuCtxSynchronize_v2; what the _v2
    // lookup finds over ZLUDA is an internal variant that answers
    // INVALID_VALUE once NGX has run.
    *reinterpret_cast<void **>(&api.cuCtxSynchronize) = (void *)GetProcAddress(h, "cuCtxSynchronize");
    if (!api.cuCtxSynchronize) return "cuCtxSynchronize";
    LOAD(cuImportExternalMemory, "cuImportExternalMemory");
    LOAD(cuDestroyExternalMemory, "cuDestroyExternalMemory");
    LOAD(cuExternalMemoryGetMappedMipmappedArray, "cuExternalMemoryGetMappedMipmappedArray");
    LOAD(cuMipmappedArrayGetLevel, "cuMipmappedArrayGetLevel");
    LOAD(cuMipmappedArrayDestroy, "cuMipmappedArrayDestroy");
    LOAD(cuSurfObjectCreate, "cuSurfObjectCreate");
    LOAD(cuSurfObjectDestroy, "cuSurfObjectDestroy");
    LOAD(cuTexObjectCreate, "cuTexObjectCreate");
    LOAD(cuTexObjectDestroy, "cuTexObjectDestroy");
    LOAD(cuArrayGetDescriptor, "cuArrayGetDescriptor");
    LOAD(cuGetErrorString, "cuGetErrorString");
    *reinterpret_cast<void **>(&api.cuMemGetInfo) = (void *)GetProcAddress(h, "cuMemGetInfo_v2");
    *reinterpret_cast<void **>(&api.cuMemcpy2D) = (void *)GetProcAddress(h, "cuMemcpy2D_v2");
    *reinterpret_cast<void **>(&api.cuArray3DCreate) = (void *)GetProcAddress(h, "cuArray3DCreate_v2");
    *reinterpret_cast<void **>(&api.cuArrayDestroy) = (void *)GetProcAddress(h, "cuArrayDestroy");
#undef LOAD
    return nullptr;
}

bool cuda_format_of(int dxgi, CUarray_format &format, unsigned &channels) {
    switch (dxgi) {
    case DXGI_FORMAT_R16G16B16A16_FLOAT: format = CU_AD_FORMAT_HALF; channels = 4; return true;
    case DXGI_FORMAT_R32G32B32A32_FLOAT: format = CU_AD_FORMAT_FLOAT; channels = 4; return true;
    case DXGI_FORMAT_R16G16_FLOAT: format = CU_AD_FORMAT_HALF; channels = 2; return true;
    case DXGI_FORMAT_R32G32_FLOAT: format = CU_AD_FORMAT_FLOAT; channels = 2; return true;
    case DXGI_FORMAT_R32_FLOAT: format = CU_AD_FORMAT_FLOAT; channels = 1; return true;
    default: return false;
    }
}

// The snippet reads the first eight bytes of an image parameter as a CUDA
// object and asks cuTexObjectGetResourceDesc, then cuSurfObjectGetResourceDesc,
// what it is. A texture object answers "read only", a surface object
// "writable"; getting that wrong on the output is 0xBAD00009.
void *image_param(SharedImage &image) {
    image.descriptor.object = (!image.writable && image.texture) ? image.texture : image.surface;
    return &image.descriptor;
}

} // namespace

void set_network_log(void (*sink)(int, const char *)) { g_log = sink; }

bool Network::load(const NetworkPaths &paths, std::string &error) {
    const std::wstring zluda_dir = paths.runtime_dir + L"\\zluda\\";
    const std::wstring kernels = paths.runtime_dir + L"\\kernels\\kernels.txt";
    const std::wstring cache = paths.data_dir + L"\\ComputeCache";
    g.data_dir = paths.data_dir;

    const bool may_translate = translation_allowed();
    std::set<std::string> covered;
    if (!may_translate && !read_coverage(cache, covered, error)) return false;

    // Environment first: every consumer reads it on load.
    //
    //  - ZLUDA keeps the translated network in its cache, which is shipped
    //    already filled (tools\prepare_cache.py).
    //  - The network refuses anything below Blackwell and asks NVAPI; ZLUDA's
    //    stand-in answers with this architecture.
    //  - The gate reads its manifest from here, and logs beside our own log.
    //  - The snippet's own log names file, line and function of every refusal.
    CreateDirectoryW(cache.c_str(), nullptr);
    SetEnvironmentVariableW(L"ZLUDA_CACHE_DIR", cache.c_str());
    SetEnvironmentVariableW(L"ZLUDA_NVAPI_GPU_ARCH", L"0x1B0");
    if (GetFileAttributesW(kernels.c_str()) != INVALID_FILE_ATTRIBUTES) {
        SetEnvironmentVariableW(L"NVCUDA_PROXY_NATIVE", kernels.c_str());
        // One manifest line per native kernel, comments and blanks aside.
        FILE *f = _wfopen(kernels.c_str(), L"r");
        char line[512];
        while (f && fgets(line, sizeof line, f)) {
            const char *p = line;
            while (*p == ' ' || *p == '\t') ++p;
            if (*p && *p != '#' && *p != '\r' && *p != '\n') {
                ++native_kernels_;
                char name[256] = {}, object[256] = {};
                if (!may_translate && sscanf(p, "%255s %255s", name, object) == 2) {
                    const std::wstring file = paths.runtime_dir + L"\\kernels\\" + std::wstring(object, object + strlen(object));
                    if (GetFileAttributesW(file.c_str()) == INVALID_FILE_ATTRIBUTES) {
                        error = std::string("a native kernel is missing: ") + narrow(file);
                        fclose(f);
                        return false;
                    }
                }
            }
        }
        if (f) fclose(f);
    } else if (!may_translate) {
        error = "the native kernels are missing: " + narrow(kernels);
        return false;
    } else {
        log(1, "no native kernels at %s: the whole network runs translated", narrow(kernels).c_str());
    }
    SetEnvironmentVariableW(L"DLSS5NR_GATE_LOG", (paths.data_dir + L"\\gate.log").c_str());
    SetEnvironmentVariableW(L"__NGX_LOG_LEVEL", L"1");
    SetEnvironmentVariableW(L"__NGX_ENABLE_OVERRIDE_LOG_PATH", L"1");
    SetEnvironmentVariableW(L"__NGX_LOG_PATH_OVERRIDE", paths.data_dir.c_str());

    // In dependency order, each by full path, so that the names the others
    // look for by base name are already in the process: the gate's forwarders
    // name zluda_real, the snippet asks for nvcuda.dll and nvapi64.dll.
    if (!(g.nvapi = load_exact(zluda_dir + L"nvapi64.dll", error))) return false;
    if (!(g.zluda = load_exact(zluda_dir + L"zluda_real.dll", error))) return false;
    if (!(g.gate = load_exact(zluda_dir + L"nvcuda.dll", error))) return false;
    if (const char *missing = load_cuda_api(g.gate, g.cu)) {
        error = std::string("the CUDA driver does not export ") + missing;
        return false;
    }

    if (!cu_ok(g.cu.cuInit(0), "cuInit", error)) return false;
    CUdevice device = 0;
    if (!cu_ok(g.cu.cuDeviceGet(&device, 0), "cuDeviceGet", error)) return false;
    char name[256] = {};
    if (g.cu.cuDeviceGetName(name, sizeof name, device) == CUDA_SUCCESS) device_name_ = name;
    log(2, "device: %s", device_name_.c_str());
    if (!may_translate) {
        const std::vector<std::string> architectures = gpu_architectures();
        bool served = architectures.empty();   // not told: let the network try
        std::string seen, held;
        for (const std::string &arch : architectures) {
            served = served || cache_serves(covered, arch);
            seen += (seen.empty() ? "" : ", ") + arch;
        }
        for (const std::string &target : covered) held += (held.empty() ? "" : ", ") + target;
        log(2, "GPU %s; the shipped cache covers %s", seen.empty() ? "unknown" : seen.c_str(), held.c_str());
        if (!served) {
            error = "this GPU (" + seen + ") has no translation in the shipped cache, which covers " + held;
            return false;
        }
    }
    if (!cu_ok(g.cu.cuCtxCreate(&g.context, CU_CTX_SCHED_AUTO, device), "cuCtxCreate", error)) return false;
    if (!cu_ok(g.cu.cuCtxSetCurrent(g.context), "cuCtxSetCurrent", error)) return false;

    // Our NGX runtime. The snippet only accepts calls that come from a module
    // named nvngx.dll, so every NGX call is made from inside it.
    if (!(g.ngx = load_exact(zluda_dir + L"nvngx.dll", error))) return false;
#define NGX(field, symbol)                                                           \
    do {                                                                             \
        *reinterpret_cast<void **>(&g.field) = (void *)GetProcAddress(g.ngx, symbol); \
        if (!g.field) {                                                              \
            error = "nvngx.dll does not export " symbol                              \
                    " (another module named nvngx.dll answered?)";                   \
            return false;                                                            \
        }                                                                            \
    } while (0)
    NGX(ngx_load, "ngxrt_load");
    NGX(ngx_init, "ngxrt_init");
    NGX(ngx_create, "ngxrt_create_feature");
    NGX(ngx_evaluate, "ngxrt_evaluate");
    NGX(ngx_release, "ngxrt_release_feature");
    NGX(ngx_shutdown, "ngxrt_shutdown");
    NGX(ngx_populate, "ngxrt_populate_parameters");
#undef NGX

    if (const char *problem = g.ngx_load(paths.snippet.c_str())) {
        error = std::string("could not load the network: ") + problem;
        return false;
    }
    if (!ngx_ok(g.ngx_init(0, paths.data_dir.c_str(), NVSDK_NGX_Version_API,
                           (unsigned long long)(void *)&ngx_log_sink, 1, g.context),
                "NVSDK_NGX_CUDA_Init_Ext", error))
        return false;
    g.params = nullptr;
    if (!ngx_ok(g.ngx_populate(&g.params), "NVSDK_NGX_CUDA_PopulateParameters_Impl", error)) return false;
    if (!g.params) {
        error = "NGX returned no parameter block";
        return false;
    }
    loaded_ = true;
    return true;
}

bool Network::create_feature(uint32_t width, uint32_t height, const NetworkControls &controls,
                             std::string &error) {
    if (!loaded_) {
        error = "the network is not loaded";
        return false;
    }
    if (g.feature && width == feature_width_ && height == feature_height_ && controls == feature_controls_)
        return true;
    if (g.feature) {
        g.ngx_release(g.feature);
        g.feature = nullptr;
    }

    size_t free_bytes = 0, total_bytes = 0;
    if (g.cu.cuMemGetInfo && g.cu.cuMemGetInfo(&free_bytes, &total_bytes) == CUDA_SUCCESS)
        log(2, "device memory before the network: %zu of %zu MB free", free_bytes >> 20, total_bytes >> 20);

    // The network alters an image without resizing it, and refuses any other
    // arrangement: input and output are one size. Both spellings of the
    // output size are published, since the snippet's strings carry both.
    NVSDK_NGX_Parameter *p = g.params;
    p->Set(nr_param::Enabled, 1);
    p->Set(nr_param::Width, width);
    p->Set(nr_param::Height, height);
    p->Set(nr_param::InputWidth, width);
    p->Set(nr_param::InputHeight, height);
    p->Set(nr_param::OutputWidth, width);
    p->Set(nr_param::OutputHeight, height);
    p->Set("DLSSNR.Output.Width", width);
    p->Set("DLSSNR.Output.Height", height);
    p->Set(nr_param::Upscaling, 0);
    p->Set(nr_param::ScalingRatio, 1.0f);
    p->Set(ngx_param::CreationNodeMask, 1u);
    p->Set(ngx_param::VisibilityNodeMask, 1u);
    p->Set(nr_param::Style, controls.style);
    p->Set(nr_param::Intensity, controls.intensity);
    p->Set(nr_param::LocalToneStrength, controls.local_tone);
    p->Set(nr_param::LocalStructureStrength, controls.local_structure);
    p->Set(nr_param::SkinStructureStrength, controls.skin_structure);

    // On the stand-in the id is SuperSampling: a snippet implements one feature,
    // and the selection that matters is which DLL was loaded.
    if (!ngx_ok(g.ngx_create(NVSDK_NGX_Feature_SuperSampling, p, &g.feature), "NVSDK_NGX_CUDA_CreateFeature",
                error))
        return false;
    feature_width_ = width;
    feature_height_ = height;
    feature_controls_ = controls;
    log(2, "network created at %ux%u", width, height);
    return true;
}

bool Network::import(SharedImage &t, std::string &error) {
    if (copy_mode_) return import_copy(t, error);
    if (t.external_memory) return true;
    CUarray_format format;
    unsigned channels;
    if (!cuda_format_of(t.dxgi_format, format, channels)) {
        error = "unsupported format for sharing with CUDA";
        return false;
    }
    D3D12_RESOURCE_DESC desc = t.resource->GetDesc();
    ID3D12Device *device = nullptr;
    t.resource->GetDevice(IID_PPV_ARGS(&device));
    const D3D12_RESOURCE_ALLOCATION_INFO info = device->GetResourceAllocationInfo(0, 1, &desc);
    device->Release();

    CUDA_EXTERNAL_MEMORY_HANDLE_DESC memory{};
    memory.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
    memory.handle.win32.handle = t.nt_handle;
    memory.size = info.SizeInBytes;
    memory.flags = 1; // dedicated: the allocation backs exactly this resource
    CUexternalMemory external = nullptr;
    if (!cu_ok(g.cu.cuImportExternalMemory(&external, &memory), "cuImportExternalMemory", error)) return false;
    t.external_memory = external;

    CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC mapped{};
    mapped.arrayDesc.Width = t.width;
    mapped.arrayDesc.Height = t.height;
    mapped.arrayDesc.Format = format;
    mapped.arrayDesc.NumChannels = channels;
    mapped.arrayDesc.Flags = 0x2; // CUDA_ARRAY3D_SURFACE_LDST
    mapped.numLevels = 1;
    CUmipmappedArray mipmapped = nullptr;
    if (!cu_ok(g.cu.cuExternalMemoryGetMappedMipmappedArray(&mipmapped, external, &mapped),
               "cuExternalMemoryGetMappedMipmappedArray", error))
        return false;
    t.mipmapped_array = mipmapped;
    CUarray array = nullptr;
    if (!cu_ok(g.cu.cuMipmappedArrayGetLevel(&array, mipmapped, 0), "cuMipmappedArrayGetLevel", error))
        return false;
    t.array = array;

    CUDA_RESOURCE_DESC resource{};
    resource.resType = CU_RESOURCE_TYPE_ARRAY;
    resource.res.array.hArray = array;
    if (!cu_ok(g.cu.cuSurfObjectCreate(&t.surface, &resource), "cuSurfObjectCreate", error)) return false;
    // A texture object over the same array: the snippet asks the texture query
    // first, so read-only images answer on the first attempt.
    CUDA_TEXTURE_DESC sampling{};
    sampling.filterMode = 0; // point
    sampling.flags = 2;      // integer coordinates, no normalisation
    for (int i = 0; i < 3; ++i) sampling.addressMode[i] = 1; // clamp
    if (!cu_ok(g.cu.cuTexObjectCreate(&t.texture, &resource, &sampling, nullptr), "cuTexObjectCreate", error))
        return false;

    t.descriptor.width = t.width;
    t.descriptor.height = t.height;
    t.descriptor.pitch = 0;
    t.descriptor.format = (uint32_t)t.dxgi_format;
    return true;
}

// Copy mode: the image is a CUDA array of its own, and a buffer to move the
// pixels through.
bool Network::import_copy(SharedImage &t, std::string &error) {
    if (t.array) return true;
    CUarray_format format;
    unsigned channels;
    if (!cuda_format_of(t.dxgi_format, format, channels)) {
        error = "unsupported format for CUDA";
        return false;
    }
    if (!g.cu.cuMemcpy2D || !g.cu.cuArray3DCreate || !g.cu.cuArrayDestroy) {
        error = "copying through host memory needs cuMemcpy2D, cuArray3DCreate and cuArrayDestroy";
        return false;
    }
    ID3D12Device *device = nullptr;
    t.resource->GetDevice(IID_PPV_ARGS(&device));
    const bool started = copier_start(device, error);
    D3D12_RESOURCE_DESC desc = t.resource->GetDesc();
    UINT64 total = 0, row_bytes = 0;
    UINT rows = 0;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, &rows, &row_bytes, &total);
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = t.writable ? D3D12_HEAP_TYPE_UPLOAD : D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC buffer{};
    buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer.Width = total;
    buffer.Height = 1;
    buffer.DepthOrArraySize = 1;
    buffer.MipLevels = 1;
    buffer.SampleDesc.Count = 1;
    buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource *staging = nullptr;
    const HRESULT made = started ? device->CreateCommittedResource(
                                       &heap, D3D12_HEAP_FLAG_NONE, &buffer,
                                       t.writable ? D3D12_RESOURCE_STATE_GENERIC_READ : D3D12_RESOURCE_STATE_COPY_DEST,
                                       nullptr, IID_PPV_ARGS(&staging))
                                 : E_FAIL;
    device->Release();
    if (!started) return false;
    void *mapped = nullptr;
    if (FAILED(made) || FAILED(staging->Map(0, nullptr, &mapped))) {
        if (staging) staging->Release();
        error = "could not create the staging buffer";
        return false;
    }
    t.staging = staging;
    t.staging_ptr = mapped;
    t.host = malloc((size_t)footprint.Footprint.RowPitch * t.height);
    if (!t.host) {
        error = "out of memory for the staging copy";
        return false;
    }
    t.row_pitch = footprint.Footprint.RowPitch;
    t.bytes_per_pixel = (uint32_t)row_bytes / t.width;

    CUDA_ARRAY3D_DESCRIPTOR array_desc{};
    array_desc.Width = t.width;
    array_desc.Height = t.height;
    array_desc.Format = format;
    array_desc.NumChannels = channels;
    array_desc.Flags = 0x2; // CUDA_ARRAY3D_SURFACE_LDST
    CUarray array = nullptr;
    if (!cu_ok(g.cu.cuArray3DCreate(&array, &array_desc), "cuArray3DCreate", error)) return false;
    t.array = array;
    t.owns_array = true;

    CUDA_RESOURCE_DESC resource{};
    resource.resType = CU_RESOURCE_TYPE_ARRAY;
    resource.res.array.hArray = array;
    if (!cu_ok(g.cu.cuSurfObjectCreate(&t.surface, &resource), "cuSurfObjectCreate", error)) return false;
    CUDA_TEXTURE_DESC sampling{};
    sampling.filterMode = 0;
    sampling.flags = 2;
    for (int i = 0; i < 3; ++i) sampling.addressMode[i] = 1;
    if (!cu_ok(g.cu.cuTexObjectCreate(&t.texture, &resource, &sampling, nullptr), "cuTexObjectCreate", error))
        return false;

    t.descriptor.width = t.width;
    t.descriptor.height = t.height;
    t.descriptor.pitch = 0;
    t.descriptor.format = (uint32_t)t.dxgi_format;
    return true;
}

// The inputs, out of the game's textures and into the network's arrays.
bool Network::copy_in(const EvaluateDesc &d, std::string &error) {
    SharedImage *inputs[] = {d.color, d.depth, d.motion};
    if (!copier_run(
            [&](ID3D12GraphicsCommandList *cmd) {
                for (SharedImage *t : inputs) {
                    ID3D12Resource *texture = t->resource;
                    copier_barrier(cmd, texture, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
                    D3D12_TEXTURE_COPY_LOCATION to{}, from{};
                    to.pResource = (ID3D12Resource *)t->staging;
                    to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                    to.PlacedFootprint = footprint_of(*t);
                    from.pResource = texture;
                    from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                    cmd->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
                    copier_barrier(cmd, texture, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
                }
            },
            error))
        return false;
    for (SharedImage *t : inputs) {
        memcpy(t->host, t->staging_ptr, (size_t)t->row_pitch * t->height);
        CUDA_MEMCPY2D copy{};
        copy.srcMemoryType = CU_MEMORYTYPE_HOST;
        copy.srcHost = t->host;
        copy.srcPitch = t->row_pitch;
        copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        copy.dstArray = (CUarray)t->array;
        copy.WidthInBytes = (size_t)t->width * t->bytes_per_pixel;
        copy.Height = t->height;
        if (!cu_ok(g.cu.cuMemcpy2D(&copy), "cuMemcpy2D (in)", error)) return false;
    }
    return true;
}

// The output, out of the network's array and into the game's texture.
bool Network::copy_out(const EvaluateDesc &d, std::string &error) {
    SharedImage *t = d.output;
    CUDA_MEMCPY2D copy{};
    copy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.srcArray = (CUarray)t->array;
    copy.dstMemoryType = CU_MEMORYTYPE_HOST;
    copy.dstHost = t->host;
    copy.dstPitch = t->row_pitch;
    copy.WidthInBytes = (size_t)t->width * t->bytes_per_pixel;
    copy.Height = t->height;
    if (!cu_ok(g.cu.cuMemcpy2D(&copy), "cuMemcpy2D (out)", error)) return false;
    memcpy(t->staging_ptr, t->host, (size_t)t->row_pitch * t->height);
    return copier_run(
        [&](ID3D12GraphicsCommandList *cmd) {
            copier_barrier(cmd, t->resource, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
            D3D12_TEXTURE_COPY_LOCATION to{}, from{};
            to.pResource = t->resource;
            to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            from.pResource = (ID3D12Resource *)t->staging;
            from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            from.PlacedFootprint = footprint_of(*t);
            cmd->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
            copier_barrier(cmd, t->resource, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
        },
        error);
}

void Network::release(SharedImage &t) {
    if (t.texture) g.cu.cuTexObjectDestroy(t.texture);
    if (t.surface) g.cu.cuSurfObjectDestroy(t.surface);
    if (t.owns_array && t.array) g.cu.cuArrayDestroy((CUarray)t.array);
    if (t.staging) ((ID3D12Resource *)t.staging)->Release();
    free(t.host);
    t.host = nullptr;
    t.staging = t.staging_ptr = nullptr;
    t.owns_array = false;
    if (t.mipmapped_array) g.cu.cuMipmappedArrayDestroy((CUmipmappedArray)t.mipmapped_array);
    if (t.external_memory) g.cu.cuDestroyExternalMemory((CUexternalMemory)t.external_memory);
    t.texture = t.surface = 0;
    t.mipmapped_array = t.array = t.external_memory = nullptr;
}

bool Network::evaluate(const EvaluateDesc &d, std::string &error) {
    if (!g.feature) {
        error = "the network has no feature";
        return false;
    }
    if (!cu_ok(g.cu.cuCtxSetCurrent(g.context), "cuCtxSetCurrent", error)) return false;
    if (copy_mode_ && !copy_in(d, error)) return false;
    NVSDK_NGX_Parameter *p = g.params;
    const uint32_t w = feature_width_, h = feature_height_;

    p->Set(nr_param::Color, image_param(*d.color));
    p->Set(nr_param::Depth, image_param(*d.depth));
    p->Set(nr_param::MVec, image_param(*d.motion));
    p->Set(nr_param::Output, image_param(*d.output));
    // The network alters the finished image and asks for the buffer it lives
    // in; the output stands for it.
    p->Set(nr_param::Backbuffer, image_param(*d.output));

    // Motion vectors arrive already in pixels of the network's own size.
    p->Set(nr_param::MVecScaleX, 1.0f);
    p->Set(nr_param::MVecScaleY, 1.0f);
    p->Set(nr_param::DepthInverted, d.depth_inverted ? 1 : 0);
    p->Set(nr_param::Reset, d.reset ? 1 : 0);

    // Every subrect covers the whole image. They are read on every evaluation,
    // and an absent value is indistinguishable from a rejected one.
    for (const char *base : {"DLSSNR.Color", "DLSSNR.Depth", "DLSSNR.MVec", "DLSSNR.Output", "DLSSNR.Backbuffer"}) {
        char name[64];
        snprintf(name, sizeof name, "%sSubrectBaseX", base);
        p->Set(name, 0u);
        snprintf(name, sizeof name, "%sSubrectBaseY", base);
        p->Set(name, 0u);
        snprintf(name, sizeof name, "%sSubrectWidth", base);
        p->Set(name, w);
        snprintf(name, sizeof name, "%sSubrectHeight", base);
        p->Set(name, h);
    }

    p->Set(nr_param::Style, d.controls.style);
    p->Set(nr_param::Intensity, d.controls.intensity);
    p->Set(nr_param::LocalToneStrength, d.controls.local_tone);
    p->Set(nr_param::LocalStructureStrength, d.controls.local_structure);
    p->Set(nr_param::SkinStructureStrength, d.controls.skin_structure);
    // Off: the correction composites against the UI textures (DLSSNR.UI and
    // DLSSNR.UIAlpha), which nothing here supplies. On without them it
    // composites against nothing and returns black.
    p->Set(nr_param::UICorrection, 0);
    p->Set(nr_param::UseAutoMask, 1);
    p->Set(nr_param::ScalingRatio, 1.0f);
    p->Set("DLSS.Indicator.Invert.X.Axis", 0);
    p->Set("DLSS.Indicator.Invert.Y.Axis", 0);

    if (!ngx_ok(g.ngx_evaluate(g.feature, p), "NVSDK_NGX_CUDA_EvaluateFeature", error)) return false;
    if (!cu_ok(g.cu.cuCtxSynchronize(), "cuCtxSynchronize", error)) return false;
    return !copy_mode_ || copy_out(d, error);
}

void Network::shutdown() {
    if (g.feature && g.ngx_release) g.ngx_release(g.feature);
    g.feature = nullptr;
    if (loaded_ && g.ngx_shutdown) g.ngx_shutdown();
    if (g.context && g.cu.cuCtxDestroy) g.cu.cuCtxDestroy(g.context);
    g.context = nullptr;
    // The libraries stay loaded. The snippet, ZLUDA and the HIP runtime keep
    // threads and state of their own, and unloading them under a running game
    // is how processes crash on exit; the process ending releases them.
    loaded_ = false;
    feature_width_ = feature_height_ = 0;
}

} // namespace dlss5nr
