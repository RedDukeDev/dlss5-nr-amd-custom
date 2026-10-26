// A stand-in nvapi64.dll that runs NVIDIA's frame generation library (nvngx_dlssg.dll) on this GPU.
//
// The library does its work with CUDA kernels it launches through NVAPI inside a Direct3D 12 command list
// (NvAPI_D3D12_LaunchCuKernelChain and its relatives), on buffers and textures it creates on the device. This
// module answers those calls:
//
//   * every NVAPI function it does not know is forwarded to ZLUDA's own NVAPI (loaded as zluda_nvapi.dll);
//   * the kernels are loaded through ZLUDA (or run as native kernels), as before;
//   * the buffers the library creates are found by hooking the device (the hooks look at what the library does
//     and do nothing for anyone else), and their memory is handed to HIP as external memory, so that a GPU
//     virtual address in a kernel's arguments becomes the matching HIP pointer;
//   * the textures and surfaces the library asks for through a descriptor are the same: the resource behind
//     the descriptor becomes an array, with a texture and a surface object over it;
//   * a launch is not made when the library records it. It is held in a batch that runs when the command list
//     reaches the GPU, between the commands before it and the ones after, through a split of the list that the
//     host provides (d4r's record_boundary). Launches with nothing recorded between them share a split.
//
// The host (OptiScaler, or the test harness) attaches a device with fgnv_attach and brackets every call into
// the library with fgnv_scope.

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "../common/cuda_min.h"
#include "../common/ngx_cuda.h"
#include "../core/convert.h"

namespace {

using Microsoft::WRL::ComPtr;

typedef void *(*QueryInterface_t)(uint32_t);
QueryInterface_t g_real = nullptr;
FILE *g_log = nullptr;
std::mutex g_lock;
std::wstring g_dir;

// The first steps of the work, to see where a hang is: a handful of lines, not one per frame.
bool trace_step() {
    static std::atomic<int> count{0};
    return count.fetch_add(1) < 400;
}
#define TRACE(...) do { if (trace_step()) say(__VA_ARGS__); } while (0)

void say(const char *format, ...) {
    if (!g_log) return;
    SYSTEMTIME t;
    GetLocalTime(&t);
    char line[2048];
    int n = snprintf(line, sizeof line, "[%02d:%02d:%02d.%03d] ", t.wHour, t.wMinute, t.wSecond, t.wMilliseconds);
    va_list args;
    va_start(args, format);
    n += vsnprintf(line + n, sizeof line - n - 2, format, args);
    va_end(args);
    line[std::min<int>(n, (int)sizeof line - 2)] = 0;
    fprintf(g_log, "%s\n", line);
    fflush(g_log);
}

// ---- the CUDA driver, through the gate (nvcuda.dll) -------------------------------------------------------------

struct CUDA_EXTERNAL_MEMORY_BUFFER_DESC {
    unsigned long long offset;
    unsigned long long size;
    unsigned int flags;
    unsigned int reserved[16];
};

struct Cu {
    CUresult (*cuInit)(unsigned);
    CUresult (*cuDeviceGet)(CUdevice *, int);
    CUresult (*cuCtxCreate)(CUcontext *, unsigned, CUdevice);
    CUresult (*cuCtxGetCurrent)(CUcontext *);
    CUresult (*cuCtxSetCurrent)(CUcontext);
    CUresult (*cuCtxSynchronize)();
    CUresult (*cuStreamCreate)(void **, unsigned);
    CUresult (*cuStreamSynchronize)(void *);
    CUresult (*cuModuleLoadData)(void **, const void *);
    CUresult (*cuModuleUnload)(void *);
    CUresult (*cuModuleGetFunction)(void **, void *, const char *);
    CUresult (*cuLaunchKernel)(void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void *,
                               void **, void **);
    CUresult (*cuImportExternalMemory)(CUexternalMemory *, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC *);
    CUresult (*cuDestroyExternalMemory)(CUexternalMemory);
    CUresult (*cuExternalMemoryGetMappedBuffer)(CUdeviceptr *, CUexternalMemory, const CUDA_EXTERNAL_MEMORY_BUFFER_DESC *);
    CUresult (*cuExternalMemoryGetMappedMipmappedArray)(CUmipmappedArray *, CUexternalMemory,
                                                        const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC *);
    CUresult (*cuMipmappedArrayGetLevel)(CUarray *, CUmipmappedArray, unsigned);
    CUresult (*cuMipmappedArrayDestroy)(CUmipmappedArray);
    CUresult (*cuSurfObjectCreate)(CUsurfObject *, const CUDA_RESOURCE_DESC *);
    CUresult (*cuSurfObjectDestroy)(CUsurfObject);
    CUresult (*cuTexObjectCreate)(CUtexObject *, const CUDA_RESOURCE_DESC *, const CUDA_TEXTURE_DESC *, const void *);
    CUresult (*cuTexObjectDestroy)(CUtexObject);
    CUresult (*cuImportExternalSemaphore)(CUexternalSemaphore *, const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC *);
    CUresult (*cuDestroyExternalSemaphore)(CUexternalSemaphore);
    CUresult (*cuWaitExternalSemaphoresAsync)(const CUexternalSemaphore *, const CUDA_EXTERNAL_SEMAPHORE_PARAMS *,
                                              unsigned, void *);
    CUresult (*cuSignalExternalSemaphoresAsync)(const CUexternalSemaphore *, const CUDA_EXTERNAL_SEMAPHORE_PARAMS *,
                                                unsigned, void *);
    CUresult (*cuMemcpy2D)(const CUDA_MEMCPY2D *);
    CUresult (*cuMemcpyDtoH)(void *, CUdeviceptr, size_t);
    CUresult (*cuMemcpyHtoD)(CUdeviceptr, const void *, size_t);
} cu;

bool g_cu_ready = false;
CUcontext g_ctx = nullptr;
// Every launch of the library goes on this stream, whoever launches it (ZLUDA's modules or HIP's own) and whatever the
// default stream is made of in this process: the upscaler of the host gives each thread a default stream of its own,
// and launches split between that one and HIP's null stream would not wait for each other.
void *g_stream = nullptr;

// The gate is the nvcuda.dll the runtime loads; when only this module is in the process, it loads the one beside it.
bool cuda_ready() {
    if (g_cu_ready) return true;
    HMODULE h = GetModuleHandleW(L"nvcuda.dll");
    if (!h) h = LoadLibraryExW((g_dir + L"nvcuda.dll").c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!h) { say("nvcuda.dll is not loaded and not beside this module"); return false; }
#define L(name, optional)                                                                                    \
    do {                                                                                                     \
        *reinterpret_cast<void **>(&cu.name) = (void *)GetProcAddress(h, #name "_v2");                         \
        if (!cu.name) *reinterpret_cast<void **>(&cu.name) = (void *)GetProcAddress(h, #name);                 \
        if (!cu.name && !(optional)) { say("nvcuda.dll does not export %s", #name); return false; }           \
    } while (0)
    L(cuInit, false);
    L(cuDeviceGet, false);
    L(cuCtxCreate, false);
    L(cuCtxGetCurrent, false);
    L(cuCtxSetCurrent, false);
    // The bare name: ZLUDA's cuCtxSynchronize_v2 (a name CUDA does not have) answers INVALID_VALUE once NGX has run.
    *reinterpret_cast<void **>(&cu.cuCtxSynchronize) = (void *)GetProcAddress(h, "cuCtxSynchronize");
    if (!cu.cuCtxSynchronize) { say("nvcuda.dll does not export cuCtxSynchronize"); return false; }
    L(cuModuleLoadData, false);
    L(cuModuleUnload, false);
    L(cuStreamCreate, false);
    L(cuStreamSynchronize, false);
    L(cuModuleGetFunction, false);
    L(cuLaunchKernel, false);
    L(cuImportExternalMemory, false);
    L(cuDestroyExternalMemory, false);
    L(cuExternalMemoryGetMappedBuffer, false);
    L(cuExternalMemoryGetMappedMipmappedArray, false);
    L(cuMipmappedArrayGetLevel, false);
    L(cuMipmappedArrayDestroy, true);
    L(cuSurfObjectCreate, false);
    L(cuSurfObjectDestroy, true);
    L(cuTexObjectCreate, false);
    L(cuTexObjectDestroy, true);
    L(cuImportExternalSemaphore, true);
    L(cuDestroyExternalSemaphore, true);
    L(cuWaitExternalSemaphoresAsync, true);
    L(cuSignalExternalSemaphoresAsync, true);
    L(cuMemcpy2D, true);
    L(cuMemcpyDtoH, true);
    L(cuMemcpyHtoD, true);
#undef L
    g_cu_ready = true;
    return true;
}

// The library's calls come from whichever thread the host records on, and the kernels run wherever the
// command list is submitted: each makes the context current for itself. It is always this module's own: the upscaler
// of the host runs on the same threads with a context of its own, and modules, images and launches of different
// contexts do not meet (the generated frame is black, with no error).
bool attach_context(CUcontext wanted = nullptr) {
    if (!cuda_ready()) return false;
    std::lock_guard<std::mutex> lock(g_lock);
    static bool initialised = false;
    if (!initialised) {
        initialised = true;
        cu.cuInit(0);
    }
    if (wanted) return cu.cuCtxSetCurrent(wanted) == CUDA_SUCCESS;
    if (!g_ctx) {
        CUdevice device = 0;
        if (cu.cuDeviceGet(&device, 0) != CUDA_SUCCESS || cu.cuCtxCreate(&g_ctx, 0, device) != CUDA_SUCCESS) {
            say("could not create a CUDA context");
            return false;
        }
        say("CUDA context created");
    }
    if (cu.cuCtxSetCurrent(g_ctx) != CUDA_SUCCESS) return false;
    if (!g_stream && cu.cuStreamCreate(&g_stream, 1) != CUDA_SUCCESS) { say("could not create a stream"); g_stream = nullptr; }
    return true;
}

// ---- what the library creates on the device ---------------------------------------------------------------------
//
// Seen through the device's own methods, patched in the vtable all devices of the runtime share. They act only
// while the host says the library is running on this thread (fgnv_scope), so the game's own calls pass through
// untouched and unrecorded.

thread_local int t_scope = 0;

// A resource tells us when it is destroyed: it holds a reference to this, and releases it with its last one.
struct Alive {
    std::atomic<bool> value{true};
};
const GUID kAliveGuid = {0x6a3c1d52, 0x4b9a, 0x47f1, {0x9e, 0x6b, 0x2d, 0x84, 0x1c, 0x70, 0x5a, 0x33}};
struct Sentinel final : IUnknown {
    std::shared_ptr<Alive> alive;
    std::atomic<ULONG> references{1};
    explicit Sentinel(std::shared_ptr<Alive> a) : alive(std::move(a)) {}
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void **out) override {
        if (iid != IID_IUnknown) return E_NOINTERFACE;
        *out = static_cast<IUnknown *>(this);
        AddRef();
        return S_OK;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return ++references; }
    ULONG STDMETHODCALLTYPE Release() override {
        const ULONG n = --references;
        if (!n) {
            alive->value = false;
            delete this;
        }
        return n;
    }
};
std::shared_ptr<Alive> watch(ID3D12Resource *resource) {
    // One sentinel per resource: setting another would release the first, and so declare the resource dead.
    IUnknown *existing = nullptr;
    UINT size = sizeof existing;
    if (SUCCEEDED(resource->GetPrivateData(kAliveGuid, &size, &existing)) && existing) {
        auto alive = static_cast<Sentinel *>(existing)->alive;
        existing->Release();
        return alive;
    }
    auto alive = std::make_shared<Alive>();
    auto *sentinel = new Sentinel(alive);
    const HRESULT r = resource->SetPrivateDataInterface(kAliveGuid, sentinel);
    sentinel->Release();
    if (FAILED(r)) say("could not watch a resource (0x%08x)", (unsigned)r);
    return alive;
}

struct Buffer {
    ID3D12Resource *resource;
    std::shared_ptr<Alive> alive;
    uint64_t va, size;
    CUexternalMemory external = nullptr;
    CUdeviceptr pointer = 0;
    bool tried = false;
};
std::map<uint64_t, Buffer> g_buffers;                   // by GPU virtual address

struct View {
    ID3D12Resource *resource;
    std::shared_ptr<Alive> alive;
    DXGI_FORMAT format;
};
std::unordered_map<SIZE_T, View> g_views;               // by CPU descriptor handle

struct Image {
    std::shared_ptr<Alive> alive;
    CUexternalMemory external = nullptr;
    CUmipmappedArray mipmapped = nullptr;
    CUarray array = nullptr;
    CUsurfObject surface = 0;
    CUtexObject texture = 0;
    bool failed = false;
};
std::unordered_map<ID3D12Resource *, Image> g_images;
std::mutex g_world;

constexpr uint32_t kTagBase = 0x7f000000u;
struct Handle {
    ID3D12Resource *resource;
    bool surface;
};
std::vector<Handle> g_handles;                           // a tag minus kTagBase indexes this
std::map<std::pair<ID3D12Resource *, int>, uint32_t> g_tag_of;

template <class T> void vtable_patch(T *object, unsigned slot, void *hook, void **original) {
    void **table = *reinterpret_cast<void ***>(object);
    if (table[slot] == hook) return;
    if (!*original) *original = table[slot];
    DWORD old = 0;
    if (!VirtualProtect(&table[slot], sizeof(void *), PAGE_READWRITE, &old)) {
        say("could not patch vtable slot %u", slot);
        return;
    }
    table[slot] = hook;
    VirtualProtect(&table[slot], sizeof(void *), old, &old);
}

using CreateCommittedFn = HRESULT(STDMETHODCALLTYPE *)(ID3D12Device *, const D3D12_HEAP_PROPERTIES *, D3D12_HEAP_FLAGS,
                                                       const D3D12_RESOURCE_DESC *, D3D12_RESOURCE_STATES,
                                                       const D3D12_CLEAR_VALUE *, REFIID, void **);
using CreatePlacedFn = HRESULT(STDMETHODCALLTYPE *)(ID3D12Device *, ID3D12Heap *, UINT64, const D3D12_RESOURCE_DESC *,
                                                    D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE *, REFIID, void **);
using CreateViewFn = void(STDMETHODCALLTYPE *)(ID3D12Device *, ID3D12Resource *, const void *, D3D12_CPU_DESCRIPTOR_HANDLE);
using CreateUavFn = void(STDMETHODCALLTYPE *)(ID3D12Device *, ID3D12Resource *, ID3D12Resource *, const void *,
                                              D3D12_CPU_DESCRIPTOR_HANDLE);
using CopySimpleFn = void(STDMETHODCALLTYPE *)(ID3D12Device *, UINT, D3D12_CPU_DESCRIPTOR_HANDLE,
                                               D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_DESCRIPTOR_HEAP_TYPE);
using CopyFn = void(STDMETHODCALLTYPE *)(ID3D12Device *, UINT, const D3D12_CPU_DESCRIPTOR_HANDLE *, const UINT *, UINT,
                                         const D3D12_CPU_DESCRIPTOR_HANDLE *, const UINT *, D3D12_DESCRIPTOR_HEAP_TYPE);
CreateCommittedFn o_CreateCommitted = nullptr;
CreatePlacedFn o_CreatePlaced = nullptr;
CreateViewFn o_CreateSrv = nullptr;
CreateUavFn o_CreateUav = nullptr;
CopySimpleFn o_CopySimple = nullptr;
CopyFn o_Copy = nullptr;

// ID3D12Device, in header order after IUnknown (0-2) and ID3D12Object (3-6).
constexpr unsigned kSlotCreateSrv = 18, kSlotCreateUav = 19, kSlotCopyDescriptors = 23, kSlotCopySimple = 24,
                   kSlotCreateCommitted = 27, kSlotCreatePlaced = 29;

void record_resource(ID3D12Resource *resource) {
    const D3D12_RESOURCE_DESC desc = resource->GetDesc();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER) return;
    const uint64_t va = resource->GetGPUVirtualAddress();
    if (!va) return;
    auto alive = watch(resource);
    std::lock_guard<std::mutex> lock(g_world);
    g_buffers.insert_or_assign(va, Buffer{resource, alive, va, desc.Width});
}

HRESULT STDMETHODCALLTYPE hk_CreateCommitted(ID3D12Device *self, const D3D12_HEAP_PROPERTIES *heap, D3D12_HEAP_FLAGS flags,
                                             const D3D12_RESOURCE_DESC *desc, D3D12_RESOURCE_STATES state,
                                             const D3D12_CLEAR_VALUE *clear, REFIID riid, void **out) {
    if (!t_scope) return o_CreateCommitted(self, heap, flags, desc, state, clear, riid, out);
    // The memory of what the library makes is imported into HIP, which needs it shared. The library's textures
    // are, and its buffers may not be.
    D3D12_HEAP_FLAGS shared = flags;
    if (heap && desc && heap->Type == D3D12_HEAP_TYPE_DEFAULT && !(desc->Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL))
        shared = flags | D3D12_HEAP_FLAG_SHARED;
    HRESULT r = o_CreateCommitted(self, heap, shared, desc, state, clear, riid, out);
    if (FAILED(r) && shared != flags) r = o_CreateCommitted(self, heap, flags, desc, state, clear, riid, out);
    if (SUCCEEDED(r) && out && *out) {
        ComPtr<ID3D12Resource> resource;
        if (SUCCEEDED(static_cast<IUnknown *>(*out)->QueryInterface(IID_PPV_ARGS(resource.GetAddressOf()))))
            record_resource(resource.Get());
    }
    return r;
}

HRESULT STDMETHODCALLTYPE hk_CreatePlaced(ID3D12Device *self, ID3D12Heap *heap, UINT64 offset, const D3D12_RESOURCE_DESC *desc,
                                          D3D12_RESOURCE_STATES state, const D3D12_CLEAR_VALUE *clear, REFIID riid,
                                          void **out) {
    const HRESULT r = o_CreatePlaced(self, heap, offset, desc, state, clear, riid, out);
    if (t_scope && SUCCEEDED(r) && out && *out && desc && desc->Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
        say("the library placed a buffer in a heap: not handled (a heap would have to be imported whole)");
    return r;
}

void note_view(D3D12_CPU_DESCRIPTOR_HANDLE handle, ID3D12Resource *resource, DXGI_FORMAT format) {
    if (!resource) {
        std::lock_guard<std::mutex> lock(g_world);
        g_views.erase(handle.ptr);
        return;
    }
    auto alive = watch(resource);
    std::lock_guard<std::mutex> lock(g_world);
    g_views.insert_or_assign(handle.ptr, View{resource, alive, format});
}

void STDMETHODCALLTYPE hk_CreateSrv(ID3D12Device *self, ID3D12Resource *resource, const void *desc,
                                    D3D12_CPU_DESCRIPTOR_HANDLE handle) {
    o_CreateSrv(self, resource, desc, handle);
    if (!t_scope) return;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    if (desc) format = static_cast<const D3D12_SHADER_RESOURCE_VIEW_DESC *>(desc)->Format;
    note_view(handle, resource, format);
}

void STDMETHODCALLTYPE hk_CreateUav(ID3D12Device *self, ID3D12Resource *resource, ID3D12Resource *counter, const void *desc,
                                    D3D12_CPU_DESCRIPTOR_HANDLE handle) {
    o_CreateUav(self, resource, counter, desc, handle);
    if (!t_scope) return;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    if (desc) format = static_cast<const D3D12_UNORDERED_ACCESS_VIEW_DESC *>(desc)->Format;
    note_view(handle, resource, format);
}

void copy_views(ID3D12Device *device, D3D12_CPU_DESCRIPTOR_HANDLE to, D3D12_CPU_DESCRIPTOR_HANDLE from, UINT count,
                D3D12_DESCRIPTOR_HEAP_TYPE type) {
    if (type != D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) return;
    const SIZE_T step = device->GetDescriptorHandleIncrementSize(type);
    std::lock_guard<std::mutex> lock(g_world);
    for (UINT i = 0; i < count; ++i) {
        auto found = g_views.find(from.ptr + i * step);
        if (found == g_views.end()) g_views.erase(to.ptr + i * step);
        else {
            View view = found->second;
            g_views.insert_or_assign(to.ptr + i * step, view);
        }
    }
}

void STDMETHODCALLTYPE hk_CopySimple(ID3D12Device *self, UINT count, D3D12_CPU_DESCRIPTOR_HANDLE to,
                                     D3D12_CPU_DESCRIPTOR_HANDLE from, D3D12_DESCRIPTOR_HEAP_TYPE type) {
    o_CopySimple(self, count, to, from, type);
    if (t_scope) copy_views(self, to, from, count, type);
}

void STDMETHODCALLTYPE hk_Copy(ID3D12Device *self, UINT n_to, const D3D12_CPU_DESCRIPTOR_HANDLE *to, const UINT *to_sizes,
                               UINT n_from, const D3D12_CPU_DESCRIPTOR_HANDLE *from, const UINT *from_sizes,
                               D3D12_DESCRIPTOR_HEAP_TYPE type) {
    o_Copy(self, n_to, to, to_sizes, n_from, from, from_sizes, type);
    if (!t_scope || type != D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) return;
    // The ranges are walked together, as the runtime does.
    const SIZE_T step = self->GetDescriptorHandleIncrementSize(type);
    UINT di = 0, doff = 0, si = 0, soff = 0;
    while (di < n_to && si < n_from) {
        const UINT dsize = to_sizes ? to_sizes[di] : 1, ssize = from_sizes ? from_sizes[si] : 1;
        D3D12_CPU_DESCRIPTOR_HANDLE d{to[di].ptr + doff * step}, s{from[si].ptr + soff * step};
        copy_views(self, d, s, 1, type);
        if (++doff >= dsize) { ++di; doff = 0; }
        if (++soff >= ssize) { ++si; soff = 0; }
    }
}

// ---- importing into HIP -----------------------------------------------------------------------------------------

bool shared_handle(ID3D12Resource *resource, HANDLE *out) {
    ComPtr<ID3D12Device> device;
    if (FAILED(resource->GetDevice(IID_PPV_ARGS(device.GetAddressOf())))) return false;
    const HRESULT r = device->CreateSharedHandle(resource, nullptr, GENERIC_ALL, nullptr, out);
    if (FAILED(r)) say("CreateSharedHandle failed: 0x%08x", (unsigned)r);
    return SUCCEEDED(r);
}

UINT64 allocation_size(ID3D12Resource *resource) {
    ComPtr<ID3D12Device> device;
    resource->GetDevice(IID_PPV_ARGS(device.GetAddressOf()));
    D3D12_RESOURCE_DESC desc = resource->GetDesc();
    return device->GetResourceAllocationInfo(0, 1, &desc).SizeInBytes;
}

bool import_memory(ID3D12Resource *resource, CUexternalMemory *external) {
    HANDLE handle = nullptr;
    if (!shared_handle(resource, &handle)) return false;
    CUDA_EXTERNAL_MEMORY_HANDLE_DESC memory{};
    memory.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
    memory.handle.win32.handle = handle;
    memory.size = allocation_size(resource);
    memory.flags = 1;   // dedicated: the allocation backs exactly this resource
    const CUresult r = cu.cuImportExternalMemory(external, &memory);
    CloseHandle(handle);
    if (r != CUDA_SUCCESS) say("cuImportExternalMemory -> %d", r);
    return r == CUDA_SUCCESS;
}

bool import_buffer(Buffer &b) {
    if (b.pointer) return true;
    if (b.tried) return false;
    b.tried = true;
    if (!import_memory(b.resource, &b.external)) return false;
    CUDA_EXTERNAL_MEMORY_BUFFER_DESC desc{};
    desc.size = b.size;
    const CUresult r = cu.cuExternalMemoryGetMappedBuffer(&b.pointer, b.external, &desc);
    if (r != CUDA_SUCCESS) {
        say("cuExternalMemoryGetMappedBuffer(%llu bytes) -> %d", (unsigned long long)b.size, r);
        return false;
    }
    return true;
}

constexpr CUarray_format kUnsigned32 = (CUarray_format)0x03, kSigned32 = (CUarray_format)0x0a;

bool cuda_format_of(DXGI_FORMAT dxgi, CUarray_format &format, unsigned &channels) {
    switch (dxgi) {
    case DXGI_FORMAT_R16G16B16A16_FLOAT: format = CU_AD_FORMAT_HALF; channels = 4; return true;
    case DXGI_FORMAT_R32G32B32A32_FLOAT: format = CU_AD_FORMAT_FLOAT; channels = 4; return true;
    case DXGI_FORMAT_R16G16_FLOAT: format = CU_AD_FORMAT_HALF; channels = 2; return true;
    case DXGI_FORMAT_R32G32_FLOAT: format = CU_AD_FORMAT_FLOAT; channels = 2; return true;
    case DXGI_FORMAT_R32_TYPELESS:
    case DXGI_FORMAT_R32_FLOAT: format = CU_AD_FORMAT_FLOAT; channels = 1; return true;
    case DXGI_FORMAT_R16_FLOAT: format = CU_AD_FORMAT_HALF; channels = 1; return true;
    case DXGI_FORMAT_R8_UNORM: format = CU_AD_FORMAT_UNSIGNED_INT8; channels = 1; return true;
    case DXGI_FORMAT_R8G8_UNORM: format = CU_AD_FORMAT_UNSIGNED_INT8; channels = 2; return true;
    case DXGI_FORMAT_R8G8B8A8_TYPELESS:
    case DXGI_FORMAT_R8G8B8A8_UNORM: format = CU_AD_FORMAT_UNSIGNED_INT8; channels = 4; return true;
    case DXGI_FORMAT_R16_UNORM: format = CU_AD_FORMAT_UNSIGNED_INT16; channels = 1; return true;
    case DXGI_FORMAT_R16G16_UNORM: format = CU_AD_FORMAT_UNSIGNED_INT16; channels = 2; return true;
    case DXGI_FORMAT_R16G16B16A16_UNORM: format = CU_AD_FORMAT_UNSIGNED_INT16; channels = 4; return true;
    // The integer ones carry indices and flags between the library's kernels; the arrays hold the bits as they are.
    case DXGI_FORMAT_R32_UINT: format = kUnsigned32; channels = 1; return true;
    case DXGI_FORMAT_R32G32_UINT: format = kUnsigned32; channels = 2; return true;
    case DXGI_FORMAT_R32G32B32A32_UINT: format = kUnsigned32; channels = 4; return true;
    case DXGI_FORMAT_R32_SINT: format = kSigned32; channels = 1; return true;
    case DXGI_FORMAT_R32G32_SINT: format = kSigned32; channels = 2; return true;
    case DXGI_FORMAT_R32G32B32A32_SINT: format = kSigned32; channels = 4; return true;
    case DXGI_FORMAT_R16_UINT: format = CU_AD_FORMAT_UNSIGNED_INT16; channels = 1; return true;
    case DXGI_FORMAT_R16G16_UINT: format = CU_AD_FORMAT_UNSIGNED_INT16; channels = 2; return true;
    case DXGI_FORMAT_R16G16B16A16_UINT: format = CU_AD_FORMAT_UNSIGNED_INT16; channels = 4; return true;
    case DXGI_FORMAT_R8_UINT: format = CU_AD_FORMAT_UNSIGNED_INT8; channels = 1; return true;
    case DXGI_FORMAT_R8G8_UINT: format = CU_AD_FORMAT_UNSIGNED_INT8; channels = 2; return true;
    case DXGI_FORMAT_R8G8B8A8_UINT: format = CU_AD_FORMAT_UNSIGNED_INT8; channels = 4; return true;
    // Blue first in memory: the array is the same four bytes, and the library's kernels know the order.
    case DXGI_FORMAT_B8G8R8A8_TYPELESS:
    case DXGI_FORMAT_B8G8R8A8_UNORM: format = CU_AD_FORMAT_UNSIGNED_INT8; channels = 4; return true;
    // Packed formats are one 32 bit word per pixel, which the kernels pack and unpack themselves.
    case DXGI_FORMAT_R10G10B10A2_TYPELESS:
    case DXGI_FORMAT_R10G10B10A2_UNORM:
    case DXGI_FORMAT_R10G10B10A2_UINT:
    case DXGI_FORMAT_R11G11B10_FLOAT: format = kUnsigned32; channels = 1; return true;
    default: return false;
    }
}


bool is_integer_format(DXGI_FORMAT f) {
    switch (f) {
    case DXGI_FORMAT_R32G32B32A32_UINT: case DXGI_FORMAT_R32G32B32A32_SINT: case DXGI_FORMAT_R32G32_UINT:
    case DXGI_FORMAT_R32G32_SINT: case DXGI_FORMAT_R32_UINT: case DXGI_FORMAT_R32_SINT:
    case DXGI_FORMAT_R16G16B16A16_UINT: case DXGI_FORMAT_R16G16_UINT: case DXGI_FORMAT_R16_UINT:
    case DXGI_FORMAT_R8G8B8A8_UINT: case DXGI_FORMAT_R8G8_UINT: case DXGI_FORMAT_R8_UINT:
    case DXGI_FORMAT_R10G10B10A2_TYPELESS: case DXGI_FORMAT_R10G10B10A2_UNORM: case DXGI_FORMAT_R10G10B10A2_UINT:
    case DXGI_FORMAT_R11G11B10_FLOAT:
        return true;
    default:
        return false;
    }
}

bool is_unorm_format(DXGI_FORMAT f) {
    switch (f) {
    case DXGI_FORMAT_R8_UNORM: case DXGI_FORMAT_R8G8_UNORM: case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_TYPELESS: case DXGI_FORMAT_B8G8R8A8_UNORM: case DXGI_FORMAT_B8G8R8A8_TYPELESS:
    case DXGI_FORMAT_R16_UNORM: case DXGI_FORMAT_R16G16_UNORM:
    case DXGI_FORMAT_R16G16B16A16_UNORM:
        return true;
    default:
        return false;
    }
}

// HIP makes the surface of an 8 or 16 bit array an integer image, where Direct3D's texture is normalised: a store of
// floats ("sust.p") would be taken for integers and saturate. The texture object over the same array describes the
// image as normalised, and the two descriptors differ in two words only, the hardware format (1) and the channel type
// the kernels' loads and stores consult (8). The surface takes them from the texture; ZLUDA's raw stores and loads
// decode and encode by that channel type, so a raw store of bytes keeps working.
void make_surface_normalised(Image &image) {
    if (!cu.cuMemcpyDtoH || !cu.cuMemcpyHtoD) return;
    uint32_t surface[12] = {}, texture[12] = {};
    if (cu.cuMemcpyDtoH(surface, image.surface, sizeof surface) != CUDA_SUCCESS ||
        cu.cuMemcpyDtoH(texture, image.texture, sizeof texture) != CUDA_SUCCESS)
        return;
    for (int w = 0; w < 12; ++w)
        if (w != 1 && w != 8 && surface[w] != texture[w]) {
            // The page is not laid out as this expects: leave it, and say so.
            say("the descriptors of a normalised image differ in word %d too: not adapting its surface", w);
            return;
        }
    cu.cuMemcpyHtoD(image.surface + 4, &texture[1], 4);
    cu.cuMemcpyHtoD(image.surface + 32, &texture[8], 4);
}

// The array, and a texture and a surface object over it, for a texture the library made.
Image *image_of(ID3D12Resource *resource, DXGI_FORMAT view_format) {
    auto found = g_images.find(resource);
    if (found != g_images.end()) return found->second.failed ? nullptr : &found->second;
    Image &image = g_images[resource];
    image.alive = watch(resource);
    const D3D12_RESOURCE_DESC desc = resource->GetDesc();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.DepthOrArraySize != 1) {
        say("a texture the library uses is not a plain 2D one (dimension %d, %u layers)", (int)desc.Dimension,
            (unsigned)desc.DepthOrArraySize);
        image.failed = true;
        return nullptr;
    }
    CUarray_format format;
    unsigned channels;
    const DXGI_FORMAT use = view_format != DXGI_FORMAT_UNKNOWN ? view_format : desc.Format;
    if (!cuda_format_of(use, format, channels) && !cuda_format_of(desc.Format, format, channels)) {
        say("a texture the library uses has a format this cannot share with HIP: %d", (int)use);
        image.failed = true;
        return nullptr;
    }
    if (!import_memory(resource, &image.external)) { image.failed = true; return nullptr; }
    CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC mapped{};
    mapped.arrayDesc.Width = (size_t)desc.Width;
    mapped.arrayDesc.Height = desc.Height;
    mapped.arrayDesc.Format = format;
    mapped.arrayDesc.NumChannels = channels;
    mapped.arrayDesc.Flags = 0x2;   // CUDA_ARRAY3D_SURFACE_LDST
    mapped.numLevels = desc.MipLevels ? desc.MipLevels : 1;
    CUresult r = cu.cuExternalMemoryGetMappedMipmappedArray(&image.mipmapped, image.external, &mapped);
    if (r == CUDA_SUCCESS) r = cu.cuMipmappedArrayGetLevel(&image.array, image.mipmapped, 0);
    if (r != CUDA_SUCCESS) {
        say("mapping a texture of %llux%u (format %d) -> %d", (unsigned long long)desc.Width, desc.Height, (int)use, r);
        image.failed = true;
        return nullptr;
    }
    CUDA_RESOURCE_DESC resource_desc{};
    resource_desc.resType = CU_RESOURCE_TYPE_ARRAY;
    resource_desc.res.array.hArray = image.array;
    CUDA_TEXTURE_DESC sampling{};
    sampling.filterMode = 0;   // point
    // Normalised coordinates are what the flag says, and what ZLUDA expects of a texture object (it reads the
    // integer ones the kernels ask for from it); an integer array is read as what it holds, not as a fraction.
    sampling.flags = 2 | (is_integer_format(use) ? 1u : 0u);
    for (int i = 0; i < 3; ++i) sampling.addressMode[i] = 1;
    if (cu.cuSurfObjectCreate(&image.surface, &resource_desc) != CUDA_SUCCESS ||
        cu.cuTexObjectCreate(&image.texture, &resource_desc, &sampling, nullptr) != CUDA_SUCCESS) {
        say("creating the texture and surface objects failed");
        image.failed = true;
        return nullptr;
    }
    if (is_unorm_format(use)) make_surface_normalised(image);
    if (_wgetenv(L"FG_NVAPI_DUMP_SRD") && cu.cuMemcpyDtoH) {
        // The image descriptors of the two objects, side by side: where they differ is what a store has to know.
        uint32_t s[24] = {}, x[24] = {};
        cu.cuMemcpyDtoH(s, image.surface, sizeof s);
        cu.cuMemcpyDtoH(x, image.texture, sizeof x);
        char line[600];
        int n = snprintf(line, sizeof line, "descriptors, format %d (%u x %d):", (int)use, channels, (int)format);
        say("%s", line);
        for (int w = 0; w < 12; ++w) say("    dword %2d: surface %08x   texture %08x%s", w, s[w], x[w], s[w] != x[w] ? "   <-" : "");
        (void)n;
    }
    return &image;
}


// Minimum, maximum and mean of the first channel, and how many values are not numbers: what a kernel left in a texture.
void texture_statistics(ID3D12Resource *resource, char *out, size_t out_size) {
    snprintf(out, out_size, "?");
    if (!cu.cuMemcpy2D) return;
    std::lock_guard<std::mutex> lock(g_world);
    auto found = g_images.find(resource);
    if (found == g_images.end() || found->second.failed) return;
    const D3D12_RESOURCE_DESC desc = resource->GetDesc();
    CUarray_format format;
    unsigned channels;
    if (!cuda_format_of(desc.Format, format, channels)) return;
    const size_t element = format == CU_AD_FORMAT_UNSIGNED_INT8 ? 1 : (format == CU_AD_FORMAT_UNSIGNED_INT16 || format == CU_AD_FORMAT_HALF) ? 2 : 4;
    const size_t row = (size_t)desc.Width * channels * element;
    std::vector<uint8_t> host(row * desc.Height);
    CUDA_MEMCPY2D copy{};
    copy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.srcArray = found->second.array;
    copy.dstMemoryType = CU_MEMORYTYPE_HOST;
    copy.dstHost = host.data();
    copy.dstPitch = row;
    copy.WidthInBytes = row;
    copy.Height = desc.Height;
    if (cu.cuMemcpy2D(&copy) != CUDA_SUCCESS) { snprintf(out, out_size, "unreadable"); return; }
    double lo = 1e300, hi = -1e300, sum = 0;
    size_t bad = 0, n = 0;
    auto half_to_float = [](uint16_t h) {
        const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
        if (e == 31) return m ? NAN : (s ? -INFINITY : INFINITY);
        const float v = e ? std::ldexp(1.0f + m / 1024.0f, (int)e - 15) : std::ldexp(m / 1024.0f, -14);
        return s ? -v : v;
    };
    for (size_t i = 0; i < host.size() / element; i += channels) {
        double v = 0;
        if (format == CU_AD_FORMAT_FLOAT) { float f; memcpy(&f, &host[i * 4], 4); v = f; }
        else if (format == CU_AD_FORMAT_HALF) { uint16_t h; memcpy(&h, &host[i * 2], 2); v = half_to_float(h); }
        else if (format == CU_AD_FORMAT_UNSIGNED_INT16) { uint16_t h; memcpy(&h, &host[i * 2], 2); v = h; }
        else if (format == CU_AD_FORMAT_UNSIGNED_INT8) v = host[i];
        else { uint32_t u; memcpy(&u, &host[i * 4], 4); v = u; }
        if (!std::isfinite(v)) { ++bad; continue; }
        lo = std::min(lo, v); hi = std::max(hi, v); sum += v; ++n;
    }
    snprintf(out, out_size, "%llux%u fmt %d: min %.4g max %.4g mean %.4g, %zu not finite", (unsigned long long)desc.Width,
             desc.Height, (int)desc.Format, n ? lo : 0.0, n ? hi : 0.0, n ? sum / n : 0.0, bad);
}

void destroy_buffer(Buffer &b) {
    if (b.external) cu.cuDestroyExternalMemory(b.external);
}
void destroy_image(Image &i) {
    if (i.surface && cu.cuSurfObjectDestroy) cu.cuSurfObjectDestroy(i.surface);
    if (i.texture && cu.cuTexObjectDestroy) cu.cuTexObjectDestroy(i.texture);
    if (i.mipmapped && cu.cuMipmappedArrayDestroy) cu.cuMipmappedArrayDestroy(i.mipmapped);
    if (i.external) cu.cuDestroyExternalMemory(i.external);
}

// What the library has destroyed since the last call goes with it. Called with g_world held.
void purge_dead() {
    for (auto it = g_buffers.begin(); it != g_buffers.end();) {
        if (!it->second.alive->value) { destroy_buffer(it->second); it = g_buffers.erase(it); }
        else ++it;
    }
    for (auto it = g_views.begin(); it != g_views.end();) {
        if (!it->second.alive->value) it = g_views.erase(it);
        else ++it;
    }
    for (auto it = g_images.begin(); it != g_images.end();) {
        if (!it->second.alive->value) { destroy_image(it->second); it = g_images.erase(it); }
        else ++it;
    }
}

// The 32 bit number the library keeps for a texture or surface. A kernel takes it in a 64 bit parameter and uses it
// as the object, which in ZLUDA is 64 bits wide: the launch swaps the tag for the object.
uint32_t tag_for(ID3D12Resource *resource, bool surface) {
    const auto key = std::make_pair(resource, surface ? 1 : 0);
    auto found = g_tag_of.find(key);
    if (found != g_tag_of.end()) return found->second;
    const uint32_t tag = kTagBase + (uint32_t)g_handles.size();
    g_handles.push_back({resource, surface});
    g_tag_of[key] = tag;
    return tag;
}

uint32_t object_for_descriptor(SIZE_T descriptor, bool surface) {
    if (!attach_context()) return 0;
    std::lock_guard<std::mutex> lock(g_world);
    purge_dead();
    auto view = g_views.find(descriptor);
    if (view == g_views.end()) {
        say("a %s object was asked for a descriptor nobody described (0x%zx)", surface ? "surface" : "texture", (size_t)descriptor);
        return 0;
    }
    Image *image = image_of(view->second.resource, view->second.format);
    return image ? tag_for(view->second.resource, surface) : 0;
}

// Replaces, in a kernel's arguments, what means nothing to HIP: GPU virtual addresses, and the tags.
bool translate_arguments(std::vector<uint8_t> &args, std::vector<ID3D12Resource *> *images = nullptr) {
    static const bool dump = _wgetenv(L"FG_NVAPI_DUMP_PARAMS") != nullptr;
    std::lock_guard<std::mutex> lock(g_world);
    purge_dead();
    for (size_t o = 0; o + 8 <= args.size(); o += 8) {
        uint64_t v;
        memcpy(&v, &args[o], 8);
        if (dump) say("    arg +%zu = 0x%llx", o, (unsigned long long)v);
        if (v < 0x10000) continue;
        if ((v >> 32) == 0) {
            const uint32_t tag = (uint32_t)v;
            if (tag >= kTagBase && tag < kTagBase + g_handles.size()) {
                const Handle &h = g_handles[tag - kTagBase];
                auto image = g_images.find(h.resource);
                if (image == g_images.end() || image->second.failed) { say("a tag whose texture is gone"); return false; }
                const uint64_t object = h.surface ? image->second.surface : image->second.texture;
                if (images) images->push_back(h.resource);
                if (dump) say("      -> %s object 0x%llx", h.surface ? "surface" : "texture", (unsigned long long)object);
                memcpy(&args[o], &object, 8);
            }
            continue;
        }
        auto next = g_buffers.upper_bound(v);
        if (next == g_buffers.begin()) continue;
        Buffer &b = std::prev(next)->second;
        if (v >= b.va + b.size) continue;
        if (!import_buffer(b)) { say("a buffer the kernel uses could not be imported (0x%llx)", (unsigned long long)b.va); return false; }
        const uint64_t moved = b.pointer + (v - b.va);
        if (dump) say("      -> buffer of %llu bytes at 0x%llx, now 0x%llx", (unsigned long long)b.size, (unsigned long long)b.va, (unsigned long long)moved);
        memcpy(&args[o], &moved, 8);
    }
    return true;
}

// ---- when the GPU reaches a launch --------------------------------------------------------------------------------
// ---- the functions answered here -----------------------------------------

typedef int32_t NvAPI_Status;
constexpr NvAPI_Status NVAPI_OK = 0, NVAPI_ERROR = -1, NVAPI_INVALID_ARGUMENT = -5;

struct Dim3 { uint32_t x, y, z; };
struct LaunchParams {
    void *function;
    Dim3 grid, block;
    uint32_t dynamic_shared;
    const void *params;
    uint32_t param_size;
};

int g_modules = 0, g_functions = 0;

NvAPI_Status __cdecl my_CreateCuModule(void *device, const void *blob, uint32_t size, void **module) {
    say("CreateCuModule called, blob %p size %u", blob, size);
    // A call with nothing to load is the library asking whether the entry exists.
    if (!blob && !size) { if (module) *module = nullptr; return NVAPI_OK; }
    if (!cuda_ready() || !blob || !module) return NVAPI_INVALID_ARGUMENT;
    // Text is read as a C string: one terminating zero after the blob.
    std::vector<char> copy((const char *)blob, (const char *)blob + size);
    copy.push_back(0);
    void *m = nullptr;
    auto t0 = std::chrono::steady_clock::now();
    const CUresult r = cu.cuModuleLoadData(&m, copy.data());
    say("CreateCuModule #%d: %u bytes -> CUDA %d, %.0f ms", g_modules, size, r,
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    ++g_modules;
    if (r != CUDA_SUCCESS) return NVAPI_ERROR;
    *module = m;
    return NVAPI_OK;
}

NvAPI_Status __cdecl my_CreateCuFunction(void *device, void *module, const char *name, void **function) {
    say("CreateCuFunction called, %s", name ? name : "(null)");
    if (!cuda_ready() || !module || !name || !function) return NVAPI_INVALID_ARGUMENT;
    void *f = nullptr;
    const CUresult r = cu.cuModuleGetFunction(&f, module, name);
    say("CreateCuFunction(%s) -> CUDA %d", name, r);
    ++g_functions;
    if (r != CUDA_SUCCESS) return NVAPI_ERROR;
    *function = f;
    return NVAPI_OK;
}

NvAPI_Status __cdecl my_DestroyCuModule(void *device, void *module) {
    if (cuda_ready() && module) cu.cuModuleUnload(module);
    return NVAPI_OK;
}

NvAPI_Status __cdecl my_DestroyCuFunction(void *device, void *function) { return NVAPI_OK; }

std::map<void *, std::string> g_function_name;
std::map<void *, std::pair<double, int>> g_function_time;
int g_chains = 0;
struct Lazy { const void *blob; std::string name; };
std::map<void *, Lazy> g_lazy;
std::map<void *, void *> g_lazy_resolved;   // token -> the function translated for it
bool function_from_preceding_fatbin(const void *blob, const char *name, void **module_out, void **function_out);

// ---- native replacements -----------------------------------------------------
//
// Our own code objects (native\*.hip) launched in place of translated kernels. A kernel is
// replaced when its PTX variant is in the table; anything else is left to ZLUDA. The code
// object is loaded through the HIP runtime ZLUDA already uses, so buffers are shared.

struct NativeConv {
    uint32_t variant;     // RVA of the PTX fatbin the library paired with the machine code
    int kind;             // 3: 3x3, 1: 1x1
    int cin, cout;
    int rw, nt;           // rows per wave, 16-channel tiles per wave
    int kx_outer;
    const char *kernel;
};
const NativeConv kNativeConvs[] = {
#include "fg_native_table.inc"
};

struct NativeCustom {
    const char *name;      // the PTX kernel
    int type;              // 0 conv (+ residual), 1 conv + 2x2 average pooling, 2 upsample + conv with three outputs,
                           // 3 upsample + conv, three heads, 4 the same with skips and one 8 channel output
    int nt;                // 16-channel tiles per wave
    int wl;                // weights: 0 packed 8x8 tiles, 1 plain [co][ky][kx][ci]
    const char *kernel;
};
const NativeCustom kNativeCustoms[] = {
    {"custom_block0_convPre_kernel", 1, 2, 0, "fg_custom_pool_nt2"},
    {"custom_block0_conv0_c8_kernel", 1, 2, 0, "fg_custom_pool_nt2"},
    {"custom_block0_conv1_kernel", 1, 1, 0, "fg_custom_pool_nt1"},
    {"custom_block0_conv2_kernel", 0, 1, 0, "fg_custom_rw2_nt1"},
    {"custom_block0_conv_bot0_hf_kernel", 2, 2, 0, "fg_custom_up_nt2"},
    {"custom_block0_conv_bot1_hf_kernel", 3, 1, 0, "fg_bot1_rw2"},
    {"custom_block1_conv0_kernel", 1, 1, 1, "fg_custom_pool_nt1"},
    {"custom_block1_conv1_kernel", 1, 2, 0, "fg_custom_pool_nt2"},
    {"custom_block1_conv2_kernel", 0, 2, 0, "fg_custom_rw2_nt2"},
    {"custom_block1_conv_bot0_hf_kernel", 2, 1, 0, "fg_custom_up_nt1"},
    {"custom_block1_conv_bot1_hf_kernel", 4, 1, 0, "fg_bot1_rw2"},
};

struct Hip {
    int (*module_load_data)(void **, const void *) = nullptr;
    int (*module_get_function)(void **, void *, const char *) = nullptr;
    int (*launch)(void *, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void *, void **, void **) = nullptr;
    bool ready = false;
} g_hip;
void *g_conv_module = nullptr;
void *g_custom_module = nullptr;
std::map<void *, std::pair<const NativeCustom *, void *>> g_custom_by_function;
bool g_native_enabled = true;
std::map<void *, std::pair<const NativeConv *, void *>> g_native_by_function;   // translated function -> (config, native function)
int g_native_launches = 0;

bool hip_ready() {
    if (g_hip.ready) return true;
    HMODULE h = GetModuleHandleW(L"amdhip64_7.dll");
    if (!h) return false;
    g_hip.module_load_data = (decltype(g_hip.module_load_data))GetProcAddress(h, "hipModuleLoadData");
    g_hip.module_get_function = (decltype(g_hip.module_get_function))GetProcAddress(h, "hipModuleGetFunction");
    g_hip.launch = (decltype(g_hip.launch))GetProcAddress(h, "hipModuleLaunchKernel");
    g_hip.ready = g_hip.module_load_data && g_hip.module_get_function && g_hip.launch;
    return g_hip.ready;
}

// Called when a shader is created: if its variant has a native kernel, remember the pairing.
void *load_hsaco(const wchar_t *file) {
    std::wstring path = g_dir + file;
    FILE *f = _wfopen(path.c_str(), L"rb");
    if (!f) { say("native: %ls not found", path.c_str()); return nullptr; }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    static std::vector<std::vector<char>> images;
    images.emplace_back(size);
    fread(images.back().data(), 1, size, f);
    fclose(f);
    void *module = nullptr;
    if (g_hip.module_load_data(&module, images.back().data()) != 0) { say("native: could not load %ls", file); return nullptr; }
    say("native: %ls loaded", file);
    return module;
}

void register_native(void *function, const std::string &name, uint64_t variant) {
    if (!g_native_enabled || !hip_ready()) return;
    for (const NativeCustom &c : kNativeCustoms) {
        if (name != c.name) continue;
        if (!g_custom_module && !(g_custom_module = load_hsaco(L"fg_custom.hsaco"))) { g_native_enabled = false; return; }
        void *fn = nullptr;
        if (g_hip.module_get_function(&fn, g_custom_module, c.kernel) != 0) { say("native: no kernel %s", c.kernel); return; }
        g_custom_by_function[function] = {&c, fn};
        say("native: %s -> %s", name.c_str(), c.kernel);
        return;
    }
    if (name != "k_conv_fp16_nhwc") return;
    if (const wchar_t *only = _wgetenv(L"FG_NVAPI_NATIVE_ONLY")) {      // comma separated hex variants
        wchar_t tag[16];
        swprintf(tag, 16, L"%x", (unsigned)variant);
        if (!wcsstr(only, tag)) return;
    }
    for (const NativeConv &c : kNativeConvs) {
        if (c.variant != variant) continue;
        if (!g_conv_module && !(g_conv_module = load_hsaco(L"fg_conv.hsaco"))) { g_native_enabled = false; return; }
        void *fn = nullptr;
        if (g_hip.module_get_function(&fn, g_conv_module, c.kernel) != 0) { say("native: no kernel %s", c.kernel); return; }
        g_native_by_function[function] = {&c, fn};
        say("native: %s@%x -> %s", name.c_str(), (unsigned)variant, c.kernel);
        return;
    }
}

// Returns true if the launch was done natively.
template <class T> T field(const std::vector<uint8_t> &a, size_t at) { T v; memcpy(&v, &a[at], sizeof v); return v; }

bool launch_custom(const LaunchParams &k, const std::vector<uint8_t> &args) {
    auto it = g_custom_by_function.find(k.function);
    if (it == g_custom_by_function.end()) return false;
    const NativeCustom &c = *it->second.first;
    void *fn = it->second.second;
    int r;
    if (c.type <= 2) {
        const uint64_t w = field<uint64_t>(args, 0), bias = field<uint64_t>(args, 8);
        const uint32_t cout_ = field<uint32_t>(args, 16), cin = field<uint32_t>(args, 28);
        uint64_t in = field<uint64_t>(args, 32), skip = 0, out, out1 = 0, out2 = 0;
        uint32_t H, W, flags = 0, cout_head = 0, Hin = 0, Win = 0, cout_total = cout_;
        if (c.type == 1) {                       // w b dims in dims out dims
            H = field<uint32_t>(args, 44); W = field<uint32_t>(args, 48);
            out = field<uint64_t>(args, 56);
        } else if (c.type == 0) {                // w b dims in skip dims out dims flag
            skip = field<uint64_t>(args, 40);
            H = field<uint32_t>(args, 52); W = field<uint32_t>(args, 56);
            out = field<uint64_t>(args, 64);
            flags = field<uint32_t>(args, 88);
        } else {                                 // w b dims in dims out out out dims
            Hin = field<uint32_t>(args, 44); Win = field<uint32_t>(args, 48);
            out = field<uint64_t>(args, 56); out1 = field<uint64_t>(args, 64); out2 = field<uint64_t>(args, 72);
            H = field<uint32_t>(args, 84); W = field<uint32_t>(args, 88);
            cout_head = field<uint32_t>(args, 92);
            cout_total = cout_head * 3;
        }
        struct { uint64_t w, bias, in, skip, out; uint32_t H, W, cin, cout, flags, order, wl, pad; uint64_t out1, out2;
                 uint32_t cout_head, Hin, Win, pad2; } na =
            {w, bias, in, skip, out, H, W, cin, c.type == 2 ? cout_total : cout_, flags, 4u, (uint32_t)c.wl, 0, out1, out2, cout_head, Hin, Win, 0};
        size_t size = sizeof na;
        void *extra[] = {(void *)1, &na, (void *)2, &size, (void *)3};
        const int ch = (int)((cin + 15) / 16), ps = ch * 16 + 8;
        const unsigned shared = (unsigned)((4 * 2 + 2) * 18 * ps * 2);
        r = g_hip.launch(fn, (W + 15) / 16, (H + 7) / 8, (c.type == 2 ? cout_total : cout_) / (16 * c.nt), 128, 1, 1, shared, g_stream, nullptr, extra);
    } else {
        struct { uint64_t w, bias, in0, in1, in2, out0, out1, out2; uint32_t Hin, Win, cin, H, W, order; uint64_t sk0, sk1, sk2;
                 uint32_t final_stage, pad; } na = {};
        na.w = field<uint64_t>(args, 0); na.bias = field<uint64_t>(args, 8);
        na.in0 = field<uint64_t>(args, 32); na.in1 = field<uint64_t>(args, 40); na.in2 = field<uint64_t>(args, 48);
        na.cin = field<uint32_t>(args, c.type == 3 ? 68 : 92);
        na.Hin = field<uint32_t>(args, c.type == 3 ? 60 : 84); na.Win = field<uint32_t>(args, c.type == 3 ? 64 : 88);
        na.order = 4;
        if (c.type == 3) {
            na.out0 = field<uint64_t>(args, 72); na.out1 = field<uint64_t>(args, 96); na.out2 = field<uint64_t>(args, 120);
            na.H = field<uint32_t>(args, 84); na.W = field<uint32_t>(args, 88);
        } else {
            na.sk0 = field<uint64_t>(args, 56); na.sk1 = field<uint64_t>(args, 64); na.sk2 = field<uint64_t>(args, 72);
            na.out0 = field<uint64_t>(args, 96);
            na.H = field<uint32_t>(args, 108); na.W = field<uint32_t>(args, 112);
            na.final_stage = 1;
        }
        size_t size = sizeof na;
        void *extra[] = {(void *)1, &na, (void *)2, &size, (void *)3};
        const int ch = (int)((na.cin + 15) / 16), ps = ch * 16 + 8;
        const unsigned shared = (unsigned)((4 * 2 + 2) * 18 * ps * 2);
        r = g_hip.launch(fn, (na.W + 15) / 16, (na.H + 7) / 8, 3, 128, 1, 1, shared, g_stream, nullptr, extra);
    }
    if (r != 0) {
        say("native launch of %s failed (%d): falling back", c.kernel, r);
        g_custom_by_function.erase(it);
        return false;
    }
    ++g_native_launches;
    return true;
}

int g_native_mask = 3;   // bit 0: k_conv, bit 1: custom_*; FG_NVAPI_NATIVE_MASK

bool launch_native(const LaunchParams &k, const std::vector<uint8_t> &args) {
    static const bool refuse = _wgetenv(L"FG_NVAPI_REFUSE_NATIVE") != nullptr;   // test hook: exercise the fallback
    if (refuse) return false;
    if ((g_native_mask & 2) && launch_custom(k, args)) return true;
    if (!(g_native_mask & 1)) return false;
    auto it = g_native_by_function.find(k.function);
    if (it == g_native_by_function.end()) return false;
    const NativeConv &c = *it->second.first;
    uint32_t W, H;
    uint64_t w, in, out;
    memcpy(&w, &args[0], 8);
    memcpy(&in, &args[8], 8);
    memcpy(&W, &args[16], 4);
    memcpy(&H, &args[20], 4);
    const size_t out_at = args.size() == 48 ? 32 : 24;
    memcpy(&out, &args[out_at], 8);
    const int taps = c.kind == 3 ? 9 : 1;
    uint8_t na[48] = {};
    const uint32_t cin = c.cin, cout = c.cout, bias = 2 * taps * c.cin * c.cout;
    memcpy(na + 0, &w, 8);
    memcpy(na + 8, &in, 8);
    memcpy(na + 16, &W, 4);
    memcpy(na + 20, &H, 4);
    memcpy(na + 24, &out, 8);
    memcpy(na + 32, &cin, 4);
    memcpy(na + 36, &cout, 4);
    memcpy(na + 40, &bias, 4);
    size_t size = sizeof na;
    void *extra[] = {(void *)1, na, (void *)2, &size, (void *)3};
    const unsigned rb = 4 * c.rw;
    const int r = g_hip.launch(it->second.second, (W + 15) / 16, (H + rb - 1) / rb, c.cout / (16 * c.nt), 128, 1, 1, 0, g_stream, nullptr, extra);
    if (r != 0) {
        say("native launch of %s failed (%d): falling back", c.kernel, r);
        g_native_by_function.erase(it);
        return false;
    }
    ++g_native_launches;
    return true;
}


// ---- batches of launches -----------------------------------------------------------------------------------------
//
// A launch is copied, with its arguments translated, into the batch of the command list it was recorded on. The
// batch runs when the list reaches the GPU: the host split the list there (record_boundary), so everything
// recorded before it has been submitted when run_batch is called, and what is recorded after it is submitted
// when run_batch returns. HIP and the queue wait for each other on a shared fence, or, where that is not
// possible, the CPU does.

struct Launch {
    LaunchParams k;
    std::vector<uint8_t> args;
    std::vector<ID3D12Resource *> images;   // the textures its arguments name, for FG_NVAPI_PERKERNEL's report
};
struct Batch {
    std::vector<Launch> launches;
    CUcontext context = nullptr;
    uint64_t mark = 0;
    bool done = false;
};

typedef void (*BoundaryFn)(void *user, ID3D12CommandQueue *queue);
typedef bool (*RecordBoundaryFn)(ID3D12GraphicsCommandList *list, BoundaryFn callback, void *user);
typedef uint64_t (*MarkFn)(ID3D12GraphicsCommandList *list);
RecordBoundaryFn g_record_boundary = nullptr;
MarkFn g_mark = nullptr;

std::map<ID3D12GraphicsCommandList *, std::shared_ptr<Batch>> g_open;

ComPtr<ID3D12Fence> g_fence;
CUexternalSemaphore g_semaphore = nullptr;
HANDLE g_event = nullptr;
uint64_t g_value = 0;
bool g_gpu_sync = false;
std::mutex g_run;
int g_launched = 0, g_batches = 0, g_failed = 0;

bool launch_one(const Launch &launch) {
    const LaunchParams &k = launch.k;
    if (launch_native(k, launch.args)) return true;
    void *function = k.function;
    auto lazy = g_lazy.find(function);
    auto resolved = g_lazy_resolved.find(function);
    if (resolved != g_lazy_resolved.end()) {
        function = resolved->second;
    } else if (lazy != g_lazy.end()) {          // the native kernel was refused: translate after all
        void *m = nullptr, *f = nullptr;
        if (!function_from_preceding_fatbin(lazy->second.blob, lazy->second.name.c_str(), &m, &f)) {
            say("launch of %s: no native kernel and no PTX to translate", lazy->second.name.c_str());
            return false;
        }
        say("launch of %s: translated after all", lazy->second.name.c_str());
        g_lazy_resolved[function] = f;
        function = f;
    }
    size_t size = launch.args.size();
    void *extra[] = {(void *)1, (void *)launch.args.data(), (void *)2, &size, (void *)0};
    CUresult r = cu.cuLaunchKernel(function, k.grid.x, k.grid.y, k.grid.z, k.block.x, k.block.y, k.block.z,
                                   k.dynamic_shared, g_stream, nullptr, extra);
    // With FG_NVAPI_PERKERNEL a kernel that faults is named, not blamed on a later one.
    static const bool per_kernel = _wgetenv(L"FG_NVAPI_PERKERNEL") != nullptr;
    const CUresult launched = r;
    if (r == CUDA_SUCCESS && per_kernel) r = cu.cuCtxSynchronize();
    if (per_kernel) {
        auto name = g_function_name.find(k.function);
        say("  ran %s grid %ux%ux%u block %ux%ux%u args %zu: launch %d, sync %d", name != g_function_name.end() ? name->second.c_str() : "?",
            k.grid.x, k.grid.y, k.grid.z, k.block.x, k.block.y, k.block.z, launch.args.size(), launched, r);
    }
    if (per_kernel && r == CUDA_SUCCESS) {
        std::vector<ID3D12Resource *> seen;
        for (ID3D12Resource *image : launch.images) {
            if (std::find(seen.begin(), seen.end(), image) != seen.end()) continue;
            seen.push_back(image);
            char text[200];
            texture_statistics(image, text, sizeof text);
            say("      texture %p %s", (void *)image, text);
        }
    }
    if (r != CUDA_SUCCESS) {
        auto name = g_function_name.find(k.function);
        say("launch of %s: grid %ux%ux%u block %ux%ux%u -> CUDA %d", name != g_function_name.end() ? name->second.c_str() : "?",
            k.grid.x, k.grid.y, k.grid.z, k.block.x, k.block.y, k.block.z, r);
        return false;
    }
    return true;
}

void wait_on_cpu(uint64_t value) {
    if (g_fence->GetCompletedValue() >= value) return;
    g_fence->SetEventOnCompletion(value, g_event);
    WaitForSingleObject(g_event, 20000);
}

void sync_launches() {
    if (g_stream) cu.cuStreamSynchronize(g_stream);
    cu.cuCtxSynchronize();
}

void run_batch(void *user, ID3D12CommandQueue *queue) {
    std::unique_ptr<std::shared_ptr<Batch>> holder(static_cast<std::shared_ptr<Batch> *>(user));
    Batch &b = **holder;
    TRACE("run_batch: %zu launches, queue type %d", b.launches.size(), (int)queue->GetDesc().Type);
    std::lock_guard<std::mutex> lock(g_run);
    b.done = true;
    CUcontext previous = nullptr;
    cu.cuCtxGetCurrent(&previous);
    struct Restore { CUcontext context; ~Restore() { if (context && context != g_ctx) cu.cuCtxSetCurrent(context); } } restore{previous};
    if (!attach_context(b.context)) { ++g_failed; return; }
    if (g_fence) {
        const uint64_t before = ++g_value;
        queue->Signal(g_fence.Get(), before);
        if (g_gpu_sync) {
            CUDA_EXTERNAL_SEMAPHORE_PARAMS wait{};
            wait.params.fence.value = before;
            cu.cuWaitExternalSemaphoresAsync(&g_semaphore, &wait, 1, nullptr);
        } else {
            wait_on_cpu(before);
        }
    }
    for (const Launch &launch : b.launches) {
        if (!launch_one(launch)) { ++g_failed; break; }
        ++g_launched;
    }
    ++g_batches;
    TRACE("run_batch: launched, waiting");
    if (g_fence) {
        if (g_gpu_sync) {
            const uint64_t after = ++g_value;
            CUDA_EXTERNAL_SEMAPHORE_PARAMS signal{};
            signal.params.fence.value = after;
            cu.cuSignalExternalSemaphoresAsync(&g_semaphore, &signal, 1, nullptr);
            queue->Wait(g_fence.Get(), after);
        } else {
            sync_launches();
        }
    } else {
        sync_launches();
    }
}

NvAPI_Status __cdecl my_LaunchCuKernelChain(void *list, const LaunchParams *kernels, uint32_t count) {
    if (!list || !kernels) return NVAPI_INVALID_ARGUMENT;
    if (!g_record_boundary) {
        say("LaunchCuKernelChain: no host has attached to split the command list");
        return NVAPI_ERROR;
    }
    if (!attach_context()) return NVAPI_ERROR;
    ID3D12GraphicsCommandList *cmd = static_cast<ID3D12GraphicsCommandList *>(list);

    std::vector<Launch> prepared(count);
    for (uint32_t i = 0; i < count; ++i) {
        prepared[i].k = kernels[i];
        prepared[i].k.params = nullptr;
        prepared[i].args.assign(static_cast<const uint8_t *>(kernels[i].params),
                                static_cast<const uint8_t *>(kernels[i].params) + kernels[i].param_size);
        if (_wgetenv(L"FG_NVAPI_DUMP_PARAMS")) {
            auto named = g_function_name.find(kernels[i].function);
            say("  launch %u/%u: %s grid %ux%ux%u block %ux%ux%u, %u bytes of arguments", i, count,
                named != g_function_name.end() ? named->second.c_str() : "?", kernels[i].grid.x, kernels[i].grid.y,
                kernels[i].grid.z, kernels[i].block.x, kernels[i].block.y, kernels[i].block.z, kernels[i].param_size);
        }
        if (!translate_arguments(prepared[i].args, &prepared[i].images)) return NVAPI_ERROR;
    }

    const uint64_t mark = g_mark ? g_mark(cmd) : 0;
    TRACE("LaunchCuKernelChain: %u launches on list %p (type %d)", count, (void *)cmd, (int)cmd->GetType());
    {
        // Which kernels a frame runs: what two ways of driving the library do differently is here
        static std::atomic<int> named{0};
        if (named.fetch_add(1) < 160) {
            auto first = g_function_name.find(kernels[0].function);
            say("  kernel %s grid %ux%ux%u block %ux%ux%u", first != g_function_name.end() ? first->second.c_str() : "?",
                kernels[0].grid.x, kernels[0].grid.y, kernels[0].grid.z, kernels[0].block.x, kernels[0].block.y, kernels[0].block.z);
        }
    }
    std::shared_ptr<Batch> batch;
    {
        std::lock_guard<std::mutex> lock(g_lock);
        for (auto it = g_open.begin(); it != g_open.end();) it = it->second->done ? g_open.erase(it) : std::next(it);
        auto found = g_open.find(cmd);
        if (found != g_open.end() && mark && found->second->mark == mark) batch = found->second;
    }
    if (!batch) {
        batch = std::make_shared<Batch>();
        batch->context = g_ctx;
        auto *holder = new std::shared_ptr<Batch>(batch);
        if (!g_record_boundary(cmd, run_batch, holder)) {
            delete holder;
            say("the command list cannot be split here");
            return NVAPI_ERROR;
        }
        batch->mark = g_mark ? g_mark(cmd) : 0;
        std::lock_guard<std::mutex> lock(g_lock);
        g_open[cmd] = batch;
    }
    for (Launch &l : prepared) batch->launches.push_back(std::move(l));
    return NVAPI_OK;
}


// ---- the older "cubin compute shader" family (documented in nvapi.h) -----
//
// One shader is one kernel of a module: the module is the blob the library
// hands over, the kernel its name. The same blob comes back for many shaders,
// so a module is loaded once per distinct blob.

struct Shader {
    void *module;
    void *function;
    uint32_t block[3];
    uint32_t dynamic_shared;
    std::string name;
};

uint64_t fnv1a(const void *data, size_t size) {
    const uint8_t *b = (const uint8_t *)data;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < size; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}
std::map<std::pair<uint64_t, uint32_t>, void *> g_blob_modules;

// ---- cubins that have to become PTX ---------------------------------------
//
// Some kernels reach the library's cubin entry points as machine code for an
// NVIDIA GPU, which nothing here can run. The same fatbin in the library holds
// their PTX next to that code, so the fatbin is found by the code it carries
// and handed to ZLUDA whole: it translates the PTX.

// Positions of every fatbin in the library, in order.
std::vector<const uint8_t *> g_fatbin_starts;
std::vector<size_t> g_fatbin_sizes;
bool g_fatbins_scanned = false;

void scan_library_fatbins() {
    if (g_fatbins_scanned) return;
    g_fatbins_scanned = true;
    HMODULE lib = GetModuleHandleW(L"nvngx_dlssg.dll");
    if (!lib) {
        say("fatbin scan: nvngx_dlssg.dll is not loaded under that name");
        return;
    }
    const uint8_t *base = (const uint8_t *)lib;
    const IMAGE_NT_HEADERS *nt = (const IMAGE_NT_HEADERS *)(base + ((const IMAGE_DOS_HEADER *)base)->e_lfanew);
    const size_t image = nt->OptionalHeader.SizeOfImage;
    for (size_t o = 0; o + 16 < image; o += 8) {
        if (*(const uint32_t *)(base + o) != 0xba55ed50u) continue;
        const uint16_t header = *(const uint16_t *)(base + o + 6);
        const uint64_t body = *(const uint64_t *)(base + o + 8);
        if (header < 16 || body < 32 || o + header + body > image) continue;
        g_fatbin_starts.push_back(base + o);
        g_fatbin_sizes.push_back((size_t)(header + body));
    }
    say("fatbin scan: %zu fatbins", g_fatbin_starts.size());
}

int g_shaders = 0;
std::map<const uint8_t *, void *> g_fatbin_modules;
uint64_t g_last_variant = 0;
std::map<uint64_t, uint64_t> g_cubin_to_fatbin; // RVA of the machine code -> RVA of its PTX fatbin

void load_cubin_map() {
    FILE *f = _wfopen((g_dir + L"cubin_fatbin_map.txt").c_str(), L"r");
    if (!f) { say("no cubin_fatbin_map.txt: kernels are matched by position only"); return; }
    unsigned long long e, p;
    char name[128];
    while (fscanf(f, "%llx %llx %127s", &e, &p, name) == 3) g_cubin_to_fatbin[e] = p;
    fclose(f);
    say("cubin_fatbin_map.txt: %zu pairs", g_cubin_to_fatbin.size());
}

// The module and function for a kernel whose machine code the library passed:
// the fatbins before it in the library, nearest first, until one has the kernel.
bool function_from_preceding_fatbin(const void *blob, const char *name, void **module_out, void **function_out) {
    scan_library_fatbins();
    const uint8_t *at = (const uint8_t *)blob;
    {
        const uint8_t *lib = (const uint8_t *)GetModuleHandleW(L"nvngx_dlssg.dll");
        auto hit = g_cubin_to_fatbin.find((uint64_t)(at - lib));
        if (hit != g_cubin_to_fatbin.end()) {
            const uint8_t *fat = lib + hit->second;
            const size_t fat_size = *(const uint16_t *)(fat + 6) + *(const uint64_t *)(fat + 8);
            void *module = nullptr;
            auto it = g_fatbin_modules.find(fat);
            if (it != g_fatbin_modules.end()) module = it->second;
            else {
                std::vector<char> copy((const char *)fat, (const char *)fat + fat_size);
                copy.push_back(0);
                if (cu.cuModuleLoadData(&module, copy.data()) != CUDA_SUCCESS) module = nullptr;
                g_fatbin_modules[fat] = module;
            }
            void *function = nullptr;
            if (module && cu.cuModuleGetFunction(&function, module, name) == CUDA_SUCCESS) {
                *module_out = module;
                *function_out = function;
                g_last_variant = hit->second;
                say("  %s: paired fatbin at +0x%llx", name, (unsigned long long)hit->second);
                return true;
            }
        }
    }
    // Nearest first, on either side: the PTX is next to the machine code, before or after it.
    std::vector<size_t> order(g_fatbin_starts.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t x, size_t y) {
        const auto d = [&](size_t i) { return g_fatbin_starts[i] > at ? g_fatbin_starts[i] - at : at - g_fatbin_starts[i]; };
        return d(x) < d(y);
    });
    int tried = 0;
    for (size_t k = 0; k < order.size() && tried < 80; ++k) {
        const size_t i = order[k];
        ++tried;
        void *module = nullptr;
        auto it = g_fatbin_modules.find(g_fatbin_starts[i]);
        if (it != g_fatbin_modules.end()) {
            module = it->second;
        } else {
            std::vector<char> copy((const char *)g_fatbin_starts[i], (const char *)g_fatbin_starts[i] + g_fatbin_sizes[i]);
            copy.push_back(0);
            if (cu.cuModuleLoadData(&module, copy.data()) != CUDA_SUCCESS) module = nullptr;
            g_fatbin_modules[g_fatbin_starts[i]] = module;
        }
        if (!module) continue;
        void *function = nullptr;
        if (cu.cuModuleGetFunction(&function, module, name) == CUDA_SUCCESS) {
            *module_out = module;
            *function_out = function;
            say("  %s: fatbin #%zu of the library (at +0x%llx), %zu candidates tried", name, i,
                (unsigned long long)(g_fatbin_starts[i] - (const uint8_t *)GetModuleHandleW(L"nvngx_dlssg.dll")), (size_t)tried);
            return true;
        }
    }
    return false;
}


// A kernel with a native replacement is not translated: its shader gets a token instead of a function, and
// the PTX is only loaded (and so only translated) if the native launch is ever refused.

bool try_native_shader(const void *blob, const char *name, void **token) {
    if (!g_native_enabled || !hip_ready()) return false;
    scan_library_fatbins();
    uint64_t variant = 0;
    const uint8_t *lib = (const uint8_t *)GetModuleHandleW(L"nvngx_dlssg.dll");
    auto hit = g_cubin_to_fatbin.find((uint64_t)((const uint8_t *)blob - lib));
    if (hit != g_cubin_to_fatbin.end()) variant = hit->second;
    void *t = new char[1];
    register_native(t, name, variant);
    if (!g_native_by_function.count(t) && !g_custom_by_function.count(t)) { delete[] (char *)t; return false; }
    g_last_variant = variant;
    g_lazy[t] = {blob, name};
    *token = t;
    say("  %s: native, not translated", name);
    return true;
}

NvAPI_Status create_shader(const void *blob, uint32_t size, uint32_t bx, uint32_t by, uint32_t bz, uint32_t dyn,
                           const char *name, void **handle, const char *how) {
    if (!cuda_ready() || !blob || !handle) return NVAPI_INVALID_ARGUMENT;
    // Whichever thread the host creates the feature on needs the context for the modules it loads
    if (!attach_context()) return NVAPI_ERROR;
    if (size >= 4 && *(const uint32_t *)blob == 0x464c457fu && name && *name) { // machine code
        void *m = nullptr, *f = nullptr;
        if (try_native_shader(blob, name, &f)) {
            Shader *sh = new Shader{nullptr, f, {bx, by, bz}, dyn, name};
            { char tagged[160]; snprintf(tagged, sizeof tagged, "%s@%llx", name, (unsigned long long)g_last_variant); g_function_name[f] = tagged; }
            *handle = sh;
            ++g_shaders;
            return NVAPI_OK;
        }
        if (!function_from_preceding_fatbin(blob, name, &m, &f)) {
            say("%s: machine code of %u bytes for %s, and no fatbin before it has the kernel", how, size, name);
            return NVAPI_ERROR;
        }
        Shader *sh = new Shader{m, f, {bx, by, bz}, dyn, name};
        { char tagged[160]; snprintf(tagged, sizeof tagged, "%s@%llx", name, (unsigned long long)g_last_variant); g_function_name[f] = tagged; }
        register_native(f, name, g_last_variant);
        *handle = sh;
        ++g_shaders;
        say("%s -> shader %d: %s (from the PTX beside its machine code), block %ux%ux%u, shared %u", how, g_shaders, name, bx, by, bz, dyn);
        return NVAPI_OK;
    }
    void *module = nullptr;
    const auto key = std::make_pair(fnv1a(blob, size), size);
    {
        std::lock_guard<std::mutex> lock(g_lock);
        auto it = g_blob_modules.find(key);
        if (it != g_blob_modules.end()) module = it->second;
    }
    if (!module) {
        const void *image = blob;
        size_t image_size = size;
        std::vector<char> copy((const char *)image, (const char *)image + image_size);
        copy.push_back(0);
        auto t0 = std::chrono::steady_clock::now();
        const CUresult r = cu.cuModuleLoadData(&module, copy.data());
        say("%s: module of %u bytes -> CUDA %d, %.0f ms", how, size, r,
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        if (r != CUDA_SUCCESS) return NVAPI_ERROR;
        std::lock_guard<std::mutex> lock(g_lock);
        g_blob_modules[key] = module;
    }
    void *function = nullptr;
    if (name && *name) {
        const CUresult r = cu.cuModuleGetFunction(&function, module, name);
        if (r != CUDA_SUCCESS) {
            say("%s: no kernel named %s (CUDA %d)", how, name, r);
            return NVAPI_ERROR;
        }
    } else {
        say("%s: no kernel name given", how);
        return NVAPI_ERROR;
    }
    Shader *sh = new Shader{module, function, {bx, by, bz}, dyn, name};
    g_function_name[function] = name;
    *handle = sh;
    ++g_shaders;
    say("%s -> shader %d: %s, block %ux%ux%u, shared %u", how, g_shaders, name, bx, by, bz, dyn);
    return NVAPI_OK;
}

NvAPI_Status __cdecl my_CreateCubin(void *device, const void *cubin, uint32_t size, uint32_t bx, uint32_t by,
                                    uint32_t bz, void **handle) {
    return create_shader(cubin, size, bx, by, bz, 0, nullptr, handle, "CreateCubinComputeShader");
}
NvAPI_Status __cdecl my_CreateCubinEx(void *device, const void *cubin, uint32_t size, uint32_t bx, uint32_t by,
                                      uint32_t bz, uint32_t dyn, const char *name, void **handle) {
    return create_shader(cubin, size, bx, by, bz, dyn, name, handle, "CreateCubinComputeShaderEx");
}
NvAPI_Status __cdecl my_CreateCubinWithName(void *device, const void *cubin, uint32_t size, uint32_t bx, uint32_t by,
                                            uint32_t bz, const char *name, void **handle) {
    return create_shader(cubin, size, bx, by, bz, 0, name, handle, "CreateCubinComputeShaderWithName");
}
struct CreateCubinParams {
    size_t in, out;
    void *device;
    const void *cubin;
    uint32_t size, bx, by, bz, dyn;
    const char *name;
    uint32_t flags;
    void *handle;
};
NvAPI_Status __cdecl my_CreateCubinExV2(CreateCubinParams *p) {
    if (!p) return NVAPI_INVALID_ARGUMENT;
    return create_shader(p->cubin, p->size, p->bx, p->by, p->bz, p->dyn, p->name, &p->handle, "CreateCubinComputeShaderExV2");
}

NvAPI_Status __cdecl my_DestroyCubin(void *device, void *handle) { return NVAPI_OK; }

NvAPI_Status __cdecl my_LaunchCubin(void *cmd, void *handle, uint32_t gx, uint32_t gy, uint32_t gz,
                                    const void *params, uint32_t size) {
    if (!cuda_ready() || !handle) return NVAPI_INVALID_ARGUMENT;
    Shader *sh = (Shader *)handle;
    LaunchParams one{sh->function, {gx, gy, gz}, {sh->block[0], sh->block[1], sh->block[2]}, sh->dynamic_shared,
                     params, size};
    return my_LaunchCuKernelChain(cmd, &one, 1);
}

NvAPI_Status __cdecl my_GetCudaTextureObject(void *device, size_t tex, size_t sampler, uint32_t *handle) {
    if (!handle) return NVAPI_INVALID_ARGUMENT;
    *handle = object_for_descriptor(tex, false);
    return *handle ? NVAPI_OK : NVAPI_ERROR;
}

NvAPI_Status __cdecl my_GetCudaSurfaceObject(void *device, size_t uav, uint32_t *handle) {
    if (!handle) return NVAPI_INVALID_ARGUMENT;
    *handle = object_for_descriptor(uav, true);
    return *handle ? NVAPI_OK : NVAPI_ERROR;
}

struct MergedParams {
    size_t in, out;
    void *device;
    size_t tex, sampler;
    uint64_t handle;
};
NvAPI_Status __cdecl my_GetCudaMergedTextureSamplerObject(MergedParams *p) {
    if (!p) return NVAPI_INVALID_ARGUMENT;
    p->handle = object_for_descriptor(p->tex, false);
    return p->handle ? NVAPI_OK : NVAPI_ERROR;
}

struct IndependentParams {
    size_t in, out;
    void *device;
    int type; // 0 surface, 1 texture, 2 sampler
    size_t desc;
    uint64_t handle;
};
NvAPI_Status __cdecl my_GetCudaIndependentDescriptorObject(IndependentParams *p) {
    if (!p) return NVAPI_INVALID_ARGUMENT;
    if (p->type == 2) {
        p->handle = 1; // a sampler: the texture objects are point sampled and carry their own
        return NVAPI_OK;
    }
    p->handle = object_for_descriptor(p->desc, p->type == 0);
    return p->handle ? NVAPI_OK : NVAPI_ERROR;
}

NvAPI_Status __cdecl my_IsFatbinPTXSupported(void *device, bool *supported) {
    if (supported) *supported = true;
    return NVAPI_OK;
}

struct GraphicsCaps {
    uint32_t version;       // NV_D3D12_GRAPHICS_CAPS_V1 has no version field of its own; see below
};
// NV_D3D12_GRAPHICS_CAPS_V1: 4 bytes of flags, four 16 bit SM versions, 5 reserved words.
NvAPI_Status __cdecl my_GetGraphicsCapabilities(void *device, uint32_t structVersion, void *caps) {
    if (!caps) return NVAPI_INVALID_ARGUMENT;
    uint8_t *b = (uint8_t *)caps;
    memset(b, 0, 32);
    uint16_t sm[4] = {8, 9, 8, 9}; // Ada, as the architecture identity says
    memcpy(b + 4, sm, sizeof sm);
    say("GetGraphicsCapabilities(version 0x%08x)", structVersion);
    return NVAPI_OK;
}

struct Override {
    uint32_t id;
    void *function;
};
const Override kOverrides[] = {
    {0xad1a677d, (void *)my_CreateCuModule},
    {0xe2436e22, (void *)my_CreateCuFunction},
    {0x24973538, (void *)my_LaunchCuKernelChain},
    {0x41c65285, (void *)my_DestroyCuModule},
    {0xdf295ea6, (void *)my_DestroyCuFunction},
    {0x80403fc9, (void *)my_GetCudaTextureObject},
    {0x48f5b2ee, (void *)my_GetCudaSurfaceObject},
    {0x329fe6e0, (void *)my_GetCudaMergedTextureSamplerObject},
    {0x0ddac234, (void *)my_GetCudaIndependentDescriptorObject},
    {0x70c07832, (void *)my_IsFatbinPTXSupported},
    {0x2a2c79e8, (void *)my_CreateCubin},
    {0x3151211b, (void *)my_CreateCubinEx},
    {0x1dc7261f, (void *)my_CreateCubinWithName},
    {0x5c52bb86, (void *)my_LaunchCubin},
    {0x7fb785ba, (void *)my_DestroyCubin},
    {0x01e87354, (void *)my_GetGraphicsCapabilities},
};

} // namespace

extern "C" __declspec(dllexport) void *nvapi_QueryInterface(uint32_t id) {
    for (const Override &o : kOverrides)
        if (o.id == id) { say("QueryInterface(0x%08x): answered here", id); return o.function; }
    void *real = g_real ? g_real(id) : nullptr;
    if (real) say("QueryInterface(0x%08x): ZLUDA", id);
    // Whatever the library asks that nobody answers is worth knowing.
    if (!real) say("QueryInterface(0x%08x): not implemented", id);
    return real;
}

// ---- the game's textures and the library's -----------------------------------------------------------------------
//
// The library's kernels can only reach memory that is shared with HIP, and the game's textures are not, in formats the
// library may not take. So each frame the textures it names are copied, by the converter the upscalers use, into
// shared textures of this module's (mirrors), in formats the library does take; the library is handed the mirrors
// instead; and what it generates is copied from its mirror into the game's output texture. The copies run on the
// game's queue at splits of the game's command list: one before the library's work, one after.

using dlss5nr::Converter;
using dlss5nr::Variant;

typedef uint32_t (*StateFn)(ID3D12GraphicsCommandList *list, ID3D12Resource *resource, uint32_t fallback);
StateFn g_state = nullptr;

enum Kind { kColour, kDepth, kVectors };
struct Role {
    const char *key;
    bool output;
    Kind kind;
};
const Role kRoles[] = {
    {"DLSSG.Backbuffer", false, kColour},
    {"DLSSG.HUDLess", false, kColour},
    {"DLSSG.UI", false, kColour},
    {"DLSSG.Depth", false, kDepth},
    {"DLSSG.MVecs", false, kVectors},
    {"DLSSG.BidirectionalDistortionField", false, kVectors},
    {"DLSSG.OutputInterpolated", true, kColour},
    // The real frame, with the interface the library composes onto it: written by the library in the format it was told.
    {"DLSSG.OutputReal", true, kColour},
};

struct Mirror {
    ComPtr<ID3D12Resource> resource;
    uint32_t width = 0, height = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
};
std::map<std::string, Mirror> g_mirrors;
std::vector<ComPtr<ID3D12Resource>> g_retired[3];   // replaced mirrors, kept until the GPU is done with them
unsigned g_retire_turn = 0;

Converter g_converter;
ID3D12Device *g_converter_device = nullptr;
std::mutex g_session_lock;

// The format the library gets a colour texture in: the game's own where it is one the library and HIP share, and
// 8 bit RGBA for anything else (10 bit and HDR formats are brought down to it).
DXGI_FORMAT colour_format(DXGI_FORMAT game) {
    switch (dlss5nr::view_format(game)) {
    case DXGI_FORMAT_B8G8R8A8_UNORM: return DXGI_FORMAT_B8G8R8A8_UNORM;
    default: return DXGI_FORMAT_R8G8B8A8_UNORM;
    }
}

Variant variant_of(Kind kind) { return kind == kColour ? dlss5nr::ToFloat4 : kind == kDepth ? dlss5nr::ToFloat : dlss5nr::ToFloat2; }

DXGI_FORMAT mirror_format(Kind kind, DXGI_FORMAT game) {
    return kind == kColour ? colour_format(game) : kind == kDepth ? DXGI_FORMAT_R32_FLOAT : DXGI_FORMAT_R16G16_FLOAT;
}

bool typed_store(ID3D12Device *device, DXGI_FORMAT format) {
    D3D12_FEATURE_DATA_FORMAT_SUPPORT support{format, D3D12_FORMAT_SUPPORT1_NONE, D3D12_FORMAT_SUPPORT2_NONE};
    return SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof support)) &&
           (support.Support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE);
}

Mirror *mirror_for(ID3D12Device *device, const char *key, Kind kind, const D3D12_RESOURCE_DESC &game) {
    DXGI_FORMAT want = mirror_format(kind, game.Format);
    if (!typed_store(device, want)) want = kind == kColour ? DXGI_FORMAT_R8G8B8A8_UNORM : want;
    Mirror &m = g_mirrors[key];
    if (m.resource && m.width == game.Width && m.height == game.Height && m.format == want) return &m;
    if (m.resource) g_retired[g_retire_turn % 3].push_back(std::move(m.resource));
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = game.Width;
    desc.Height = game.Height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = want;
    desc.SampleDesc.Count = 1;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ComPtr<ID3D12Resource> created;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_SHARED, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                               IID_PPV_ARGS(created.GetAddressOf())))) {
        say("could not make the mirror of %s (%llux%u, format %d)", key, (unsigned long long)game.Width, game.Height, (int)want);
        return nullptr;
    }
    m.resource = created;
    m.width = (uint32_t)game.Width;
    m.height = game.Height;
    m.format = want;
    say("mirror of %s: %ux%u, format %d (the game's: %d)", key, m.width, m.height, (int)want, (int)game.Format);
    return &m;
}

// One copy, game to mirror or back.
struct Pass {
    ComPtr<ID3D12Resource> game, mirror;
    uint32_t state;
    Variant variant;
};
struct Conversion {
    std::vector<Pass> passes;
    bool to_game = false;
};

constexpr int kSlots = 4;
struct Slot {
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    uint64_t value = 0;
};
// A list can only run on a queue of its own type, and Streamline records on a compute queue: one set per type.
Slot g_slots[2][kSlots];
ComPtr<ID3D12Fence> g_slot_fence;
uint64_t g_slot_value = 0;
unsigned g_slot_next[2] = {0, 0};

constexpr uint32_t kRead = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, kWrite = D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                   kCommon = D3D12_RESOURCE_STATE_COMMON;

bool session_ready(ID3D12Device *device) {
    if (g_converter_device == device && g_slot_fence) return true;
    std::string error;
    g_converter = Converter();
    if (!g_converter.init(device, error)) { say("the converter could not start: %s", error.c_str()); return false; }
    g_converter_device = device;
    for (auto &set : g_slots)
        for (Slot &slot : set) slot = Slot();
    g_slot_next[0] = g_slot_next[1] = 0;
    g_slot_fence.Reset();
    g_slot_value = 0;
    return SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(g_slot_fence.GetAddressOf())));
}

#include "fg_capture.inc"

// Runs when the game's list reaches the queue: the copies, on lists of this module's, on the game's queue.
void run_conversion(void *user, ID3D12CommandQueue *queue) {
    std::unique_ptr<Conversion> conversion(static_cast<Conversion *>(user));
    TRACE("run_conversion: %s, %zu passes, queue type %d", conversion->to_game ? "out" : "in", conversion->passes.size(), (int)queue->GetDesc().Type);
    std::lock_guard<std::mutex> lock(g_session_lock);
    const D3D12_COMMAND_LIST_TYPE type = queue->GetDesc().Type;
    const int set = type == D3D12_COMMAND_LIST_TYPE_COMPUTE ? 1 : 0;
    Slot &slot = g_slots[set][g_slot_next[set]++ % kSlots];
    if (!slot.list) {
        ComPtr<ID3D12Device> device;
        queue->GetDevice(IID_PPV_ARGS(device.GetAddressOf()));
        if (!device || FAILED(device->CreateCommandAllocator(type, IID_PPV_ARGS(slot.allocator.GetAddressOf()))) ||
            FAILED(device->CreateCommandList(0, type, slot.allocator.Get(), nullptr, IID_PPV_ARGS(slot.list.GetAddressOf())))) {
            say("a list of type %d for the copies could not be made", (int)type);
            slot = Slot();
            return;
        }
        slot.list->Close();
    }
    if (slot.value && g_slot_fence->GetCompletedValue() < slot.value) {
        static HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        g_slot_fence->SetEventOnCompletion(slot.value, event);
        WaitForSingleObject(event, 20000);
    }
    slot.allocator->Reset();
    slot.list->Reset(slot.allocator.Get(), nullptr);
    ID3D12GraphicsCommandList *cmd = slot.list.Get();
    for (Pass &p : conversion->passes) {
        std::string error;
        ID3D12Resource *source = conversion->to_game ? p.mirror.Get() : p.game.Get();
        ID3D12Resource *target = conversion->to_game ? p.game.Get() : p.mirror.Get();
        const D3D12_RESOURCE_DESC sd = source->GetDesc(), td = target->GetDesc();
        if (conversion->to_game) {
            dlss5nr::transition(cmd, source, kCommon, kRead);
            dlss5nr::transition(cmd, target, p.state, kWrite);
        } else {
            dlss5nr::transition(cmd, source, p.state, kRead);
            dlss5nr::transition(cmd, target, kCommon, kWrite);
        }
        const uint32_t w = (uint32_t)std::min(sd.Width, td.Width), h = std::min(sd.Height, td.Height);
        if (!g_converter.dispatch(cmd, p.variant, source, target, {0, 0}, {0, 0}, w, h, error))
            say("a copy of %ux%u could not be recorded: %s", w, h, error.c_str());
        if (conversion->to_game) {
            dlss5nr::transition(cmd, source, kRead, kCommon);
            dlss5nr::transition(cmd, target, kWrite, p.state);
        } else {
            dlss5nr::transition(cmd, source, kRead, p.state);
            dlss5nr::transition(cmd, target, kWrite, kCommon);
        }
    }
    cmd->Close();
    ID3D12CommandList *lists[] = {cmd};
    queue->ExecuteCommandLists(1, lists);
    queue->Signal(g_slot_fence.Get(), ++g_slot_value);
    slot.value = g_slot_value;
    TRACE("run_conversion: done");

    if (!conversion->to_game) capture_inputs(queue, conversion->passes);

}

extern "C" __declspec(dllexport) void fgnv_scope(int enter);

constexpr int32_t kInvalidParameter = (int32_t)0xBAD00005;

typedef int32_t (*EvaluateFn)(ID3D12GraphicsCommandList *list, const void *handle, NVSDK_NGX_Parameter *parameters, void *progress);

// The evaluation of a frame: the game's textures in, the library, the generated frame out.
extern "C" __declspec(dllexport) int32_t fgnv_evaluate(ID3D12GraphicsCommandList *list, const void *handle,
                                                       NVSDK_NGX_Parameter *parameters, void *progress, EvaluateFn evaluate) {
    if (!list || !parameters || !evaluate || !g_record_boundary) return kInvalidParameter;
    ComPtr<ID3D12Device> device;
    if (FAILED(list->GetDevice(IID_PPV_ARGS(device.GetAddressOf())))) return NVSDK_NGX_Result_Fail;

    auto in = std::make_unique<Conversion>();
    auto out = std::make_unique<Conversion>();
    struct Swap { const char *key; ID3D12Resource *original; };
    std::vector<Swap> swapped;
    DXGI_FORMAT colour = DXGI_FORMAT_UNKNOWN;
    {
        std::lock_guard<std::mutex> lock(g_session_lock);
        if (!session_ready(device.Get())) return NVSDK_NGX_Result_Fail;
        ++g_retire_turn;
        g_retired[g_retire_turn % 3].clear();   // replaced three frames ago: the GPU is past them
        for (const Role &role : kRoles) {
            ID3D12Resource *game = nullptr;
            parameters->Get(role.key, &game);
            if (!game) continue;
            const D3D12_RESOURCE_DESC desc = game->GetDesc();
            Mirror *mirror = mirror_for(device.Get(), role.key, role.kind, desc);
            if (!mirror) return NVSDK_NGX_Result_Fail;
            const uint32_t state = g_state ? g_state(list, game, kRead) : kRead;
            Pass pass{ComPtr<ID3D12Resource>(game), mirror->resource, state, variant_of(role.kind)};
            (role.output ? *out : *in).passes.push_back(std::move(pass));
            swapped.push_back({role.key, game});
            if (role.kind == kColour && colour == DXGI_FORMAT_UNKNOWN) colour = mirror->format;
        }
    }
    if (in->passes.empty()) return evaluate(list, handle, parameters, progress);

    TRACE("fgnv_evaluate: %zu passes in, %zu out", in->passes.size(), out->passes.size());
    // The copies in, ahead of whatever the library records.
    in->to_game = false;
    out->to_game = true;
    if (!g_record_boundary(list, run_conversion, in.get())) { say("the command list cannot be split for the copies"); return NVSDK_NGX_Result_Fail; }
    in.release();

    // The library sees the mirrors, and the format they are in.
    unsigned original_format = 0;
    parameters->Get("DLSSG.BackbufferFormat", &original_format);
    {
        std::lock_guard<std::mutex> lock(g_session_lock);
        for (const Swap &s : swapped) parameters->Set(s.key, g_mirrors[s.key].resource.Get());
    }
    if (colour != DXGI_FORMAT_UNKNOWN) parameters->Set("DLSSG.BackbufferFormat", (unsigned)colour);

    TRACE("fgnv_evaluate: list %p type %d, entering the library", (void *)list, (int)list->GetType());
    fgnv_scope(1);
    const int32_t result = evaluate(list, handle, parameters, progress);
    fgnv_scope(0);
    TRACE("fgnv_evaluate: the library returned %#x", (unsigned)result);

    for (const Swap &s : swapped) parameters->Set(s.key, s.original);
    if (colour != DXGI_FORMAT_UNKNOWN) parameters->Set("DLSSG.BackbufferFormat", original_format);

    // The generated frame back into the game's texture, after the library's work.
    if (!out->passes.empty() && NVSDK_NGX_SUCCEED(result)) {
        if (!g_record_boundary(list, run_conversion, out.get())) { say("the command list cannot be split for the copy out"); return NVSDK_NGX_Result_Fail; }
        out.release();
    }
    return result;
}

// ---- what the host calls ---------------------------------------------------------------------------------------------

// Attaches a device: the hooks that see what the library creates, a fence the queue and HIP share, and the host's way
// of splitting a command list. Safe to call again for the same device.
extern "C" __declspec(dllexport) int fgnv_attach(ID3D12Device *device, RecordBoundaryFn record, MarkFn mark, StateFn state) {
    if (!device || !record) return 0;
    g_record_boundary = record;
    g_mark = mark;
    g_state = state;
    if (!attach_context()) return 0;

    vtable_patch(device, kSlotCreateCommitted, (void *)hk_CreateCommitted, (void **)&o_CreateCommitted);
    vtable_patch(device, kSlotCreatePlaced, (void *)hk_CreatePlaced, (void **)&o_CreatePlaced);
    vtable_patch(device, kSlotCreateSrv, (void *)hk_CreateSrv, (void **)&o_CreateSrv);
    vtable_patch(device, kSlotCreateUav, (void *)hk_CreateUav, (void **)&o_CreateUav);
    vtable_patch(device, kSlotCopySimple, (void *)hk_CopySimple, (void **)&o_CopySimple);
    vtable_patch(device, kSlotCopyDescriptors, (void *)hk_Copy, (void **)&o_Copy);

    std::lock_guard<std::mutex> lock(g_run);
    if (!g_fence) {
        if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(g_fence.GetAddressOf())))) {
            say("could not create the fence");
            return 0;
        }
        g_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        HANDLE handle = nullptr;
        g_gpu_sync = false;
        // The GPU waiting for the GPU is a wait HIP does on a stream, and anything else queued on that stream, such as
        // the surface objects this module creates, stays behind it. The frame generation's thread and the upscaler's
        // wait for each other that way: both wait on the CPU, which holds no stream.
        // FG_NVAPI_GPU_SYNC=1 turns the GPU wait back on, for the harness, where nothing else uses HIP.
        const bool gpu_wanted = _wgetenv(L"FG_NVAPI_GPU_SYNC") != nullptr;
        if (!gpu_wanted) SetEnvironmentVariableA("DLSS5NR_UPSCALER_CPU_WAIT", "1");
        if (gpu_wanted && cu.cuImportExternalSemaphore && cu.cuWaitExternalSemaphoresAsync && cu.cuSignalExternalSemaphoresAsync &&
            SUCCEEDED(device->CreateSharedHandle(g_fence.Get(), nullptr, GENERIC_ALL, nullptr, &handle))) {
            CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC desc{};
            desc.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE;
            desc.handle.win32.handle = handle;
            g_gpu_sync = cu.cuImportExternalSemaphore(&g_semaphore, &desc) == CUDA_SUCCESS;
            CloseHandle(handle);
        }
        say("attached: the queue and HIP wait for each other on the %s", g_gpu_sync ? "GPU" : "CPU");
    }
    return 1;
}

// Brackets every call into the library on a thread: what it creates meanwhile is seen.
extern "C" __declspec(dllexport) void fgnv_scope(int enter) {
    // The library runs in this module's context, and the thread's own comes back when it is done
    static thread_local CUcontext previous = nullptr;
    if (enter && t_scope == 0) {
        previous = nullptr;
        if (cuda_ready()) cu.cuCtxGetCurrent(&previous);
        attach_context();
    }
    t_scope += enter ? 1 : -1;
    if (t_scope < 0) t_scope = 0;
    if (!enter && t_scope == 0 && previous && previous != g_ctx && cuda_ready()) cu.cuCtxSetCurrent(previous);
}

// The library has released its feature: whatever it made goes.
extern "C" __declspec(dllexport) void fgnv_purge() {
    if (!attach_context()) return;
    std::lock_guard<std::mutex> lock(g_world);
    for (auto &entry : g_buffers) destroy_buffer(entry.second);
    for (auto &entry : g_images) destroy_image(entry.second);
    g_buffers.clear();
    g_views.clear();
    g_images.clear();
    g_handles.clear();
    g_tag_of.clear();
}

extern "C" __declspec(dllexport) void fgnv_report() {
    std::lock_guard<std::mutex> lock(g_world);
    say("batches %d, kernels launched %d, failures %d; buffers %zu, views %zu, images %zu", g_batches, g_launched,
        g_failed, g_buffers.size(), g_views.size(), g_images.size());
}

BOOL APIENTRY DllMain(HMODULE self, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        wchar_t path[MAX_PATH];
        GetModuleFileNameW(self, path, MAX_PATH);
        std::wstring dir(path);
        dir = dir.substr(0, dir.find_last_of(L'\\') + 1);
        g_dir = dir;
        g_log = _wfsopen((dir + L"fg_nvapi.log").c_str(), L"w", _SH_DENYNO);
        // Where this module lives is the runtime's zluda folder; the shipped cache and the logs are in its parent. These
        // are what the runtime sets when it brings ZLUDA up, for the process where only frame generation does.
        {
            const std::wstring parent = dir.substr(0, dir.find_last_of(L'\\', dir.size() - 2) + 1);
            wchar_t existing[8];
            if (!GetEnvironmentVariableW(L"ZLUDA_CACHE_DIR", existing, 8))
                SetEnvironmentVariableW(L"ZLUDA_CACHE_DIR", (parent + L"cache\\ComputeCache").c_str());
            if (!GetEnvironmentVariableW(L"__NGX_ENABLE_OVERRIDE_LOG_PATH", existing, 8)) {
                SetEnvironmentVariableW(L"__NGX_ENABLE_OVERRIDE_LOG_PATH", L"1");
                SetEnvironmentVariableW(L"__NGX_LOG_PATH_OVERRIDE", parent.c_str());
                if (!GetEnvironmentVariableW(L"__NGX_LOG_LEVEL", existing, 8)) SetEnvironmentVariableW(L"__NGX_LOG_LEVEL", L"1");
            }
        }
        g_native_enabled = _wgetenv(L"FG_NVAPI_NO_NATIVE") == nullptr;
        if (const wchar_t *m = _wgetenv(L"FG_NVAPI_NATIVE_MASK")) g_native_mask = _wtoi(m);
        HMODULE real = LoadLibraryExW((dir + L"zluda_nvapi.dll").c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (real) g_real = (QueryInterface_t)GetProcAddress(real, "nvapi_QueryInterface");
        say("stand-in loaded; ZLUDA's NVAPI %s", g_real ? "found" : "NOT found");
        load_cubin_map();
    }
    return TRUE;
}
