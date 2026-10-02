// A proxy nvcuda.dll: forwards every export to the real ZLUDA build and logs
// the calls that decide whether an evaluation works.
//
// Why it exists: the snippet's failures surface as a single opaque code, and
// its call sites are reached through Control Flow Guard dispatch, so neither
// the arguments nor the driver's answers are visible from outside. Sitting
// between the two shows the whole conversation.
//
// Two details make it work at all:
//
//  - Everything uninteresting is an export forwarder, generated into
//    nvcuda_forwards.h. The loader binds those straight to zluda_real.dll, so
//    they cost nothing.
//
//  - The snippet does not resolve its entry points with GetProcAddress. It asks
//    the driver, through cuGetProcAddress, and keeps the pointers in a table of
//    its own; those go directly into the real driver and skip every wrapper
//    here. Intercepting that lookup is what puts the proxy in the path, and it
//    also gives us the exact pointer the driver would have returned -- which is
//    the one each wrapper then calls, so a versioned name like cuMemAlloc
//    cannot end up dispatched to the wrong implementation.
//
// Log destination: NVCUDA_PROXY_LOG, or nvcuda_proxy.log in the temporary
// directory. A game has no console, so stderr is not an option.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Every export we do not implement, forwarded straight to the real driver.
#include "nvcuda_forwards.h"

namespace {

using CUresult = int;
using CUdeviceptr = unsigned long long;
using CUarray = void *;
using CUmipmappedArray = void *;
using CUmodule = void *;
using CUfunction = void *;
using CUstream = void *;
using CUtexObject = unsigned long long;
using CUsurfObject = unsigned long long;

// Only the leading fields are needed, and they are the same in every version of
// the driver header: the resource type followed by the union whose first member
// is the array handle.
struct ResourceDesc {
    unsigned int resType;
    unsigned int _pad;
    void *handle;
};

const char *res_type_name(unsigned t) {
    switch (t) {
    case 0: return "ARRAY";
    case 1: return "MIPMAPPED_ARRAY";
    case 2: return "LINEAR";
    case 3: return "PITCH2D";
    default: return "?";
    }
}

CRITICAL_SECTION g_lock;
FILE *g_log = nullptr;

void open_log() {
    wchar_t path[MAX_PATH];
    if (!GetEnvironmentVariableW(L"NVCUDA_PROXY_LOG", path, MAX_PATH)) {
        wchar_t dir[MAX_PATH];
        GetTempPathW(MAX_PATH, dir);
        swprintf(path, MAX_PATH, L"%snvcuda_proxy.log", dir);
    }
    g_log = _wfopen(path, L"w");
}

void logf(const char *fmt, ...) {
    char line[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(line, sizeof line, fmt, ap);
    va_end(ap);
    EnterCriticalSection(&g_lock);
    if (!g_log) open_log();
    if (g_log) {
        fputs(line, g_log);
        fflush(g_log);
    }
    LeaveCriticalSection(&g_lock);
    OutputDebugStringA(line);
}

// Replacing single kernels with hand-written HIP, when one matches exactly.
// Kept outside this file because it is a separate experiment with its own
// lifetime: with NVCUDA_PROXY_NATIVE unset it costs one comparison per launch
// and changes nothing. Included here rather than at the top because it logs
// through the logf above.
#include "../../runtime/gate/native_hook.h"

// One launch's buffers, read back before and after it runs. Off unless
// NVCUDA_PROXY_DUMP names a kernel; it is how a native kernel gets checked
// against real data instead of against a whole frame.
#include "dump_buffers.h"

// The pointers cuGetProcAddress handed back, before substitution. Each wrapper
// calls its own slot, so it reaches exactly the implementation the driver chose
// for that name and version.
struct Original {
    void *tex_object_create = nullptr;
    void *tex_object_get_resource_desc = nullptr;
    void *tex_object_destroy = nullptr;
    void *surf_object_create = nullptr;
    void *surf_object_get_resource_desc = nullptr;
    void *surf_object_destroy = nullptr;
    void *array_get_descriptor = nullptr;
    void *array_create = nullptr;
    void *mipmapped_array_get_level = nullptr;
    void *module_load_data = nullptr;
    void *module_get_function = nullptr;
    void *func_set_attribute = nullptr;
    void *launch_kernel = nullptr;
    void *mem_alloc = nullptr;
    void *memcpy_htod_async = nullptr;
    void *device_get_attribute = nullptr;
};

Original g_orig;

// Falls back to the module's export when the snippet never looked a name up
// through the driver -- our own layer calls some of these directly.
void *fallback(void *&slot, const char *name) {
    if (!slot) {
        HMODULE real = GetModuleHandleW(L"zluda_real.dll");
        if (!real) real = LoadLibraryW(L"zluda_real.dll");
        slot = (void *)GetProcAddress(real, name);
    }
    return slot;
}

template <typename T>
T as(void *&slot, const char *name) {
    return reinterpret_cast<T>(fallback(slot, name));
}

} // namespace

extern "C" {

// Every surface object made so far, with the array it covers.
//
// Why it is kept: a texture object's page holds a sampler in its last four
// dwords; a surface object's page holds a host pointer there, because a surface
// has no sampler. ZLUDA's tex helper reads the sampler from that fixed offset
// whatever the handle is, so a tex fetch through a surface object samples with
// four words of heap address -- which differs from one process to the next and
// holds still within one. NVCUDA_PROXY_FIX_SAMPLER=1 copies the sampler from
// the texture object over the same array onto the surface object's page, which
// is the smallest possible test of whether that is what decides this run.
struct KnownSurface {
    void *array;
    unsigned long long object;
};
KnownSurface g_surfaces[64];
unsigned g_surface_count = 0;

constexpr size_t SAMPLER_OFFSET = 48; // twelve dwords of image descriptor

void copy_sampler_onto_surfaces(void *array, unsigned long long texture) {
    static int on = -1;
    if (on < 0) {
        char buffer[8];
        on = GetEnvironmentVariableA("NVCUDA_PROXY_FIX_SAMPLER", buffer, sizeof buffer) > 0 &&
             buffer[0] != '0';
    }
    if (!on || !texture) return;
    static CUresult (*down)(void *, CUdeviceptr, size_t) = nullptr;
    static CUresult (*up)(CUdeviceptr, const void *, size_t) = nullptr;
    if (!down) {
        HMODULE real = GetModuleHandleW(L"zluda_real.dll");
        if (!real) real = LoadLibraryW(L"zluda_real.dll");
        down = (decltype(down))GetProcAddress(real, "cuMemcpyDtoH_v2");
        up = (decltype(up))GetProcAddress(real, "cuMemcpyHtoD_v2");
    }
    unsigned char sampler[16];
    if (!down || !up || down(sampler, texture + SAMPLER_OFFSET, sizeof sampler) != 0) return;
    for (unsigned i = 0; i < g_surface_count; ++i) {
        if (g_surfaces[i].array != array) continue;
        const CUresult r = up(g_surfaces[i].object + SAMPLER_OFFSET, sampler, sizeof sampler);
        logf("[cuda] sampler copied from 0x%llX to 0x%llX -> %d\n", texture,
             g_surfaces[i].object, r);
    }
}

// The 64 bytes an image or sampler handle points at, as the kernel will read
// them. ZLUDA's tex helper takes the image descriptor from the handle itself
// and the sampler from a fixed offset into the same page, so what is actually
// in that page decides what a fetch returns -- and a surface object and a
// texture object over the same array need not hold the same thing.
void log_descriptor_page(const char *what, unsigned long long handle) {
    static CUresult (*copy)(void *, CUdeviceptr, size_t) = nullptr;
    if (!copy) {
        HMODULE real = GetModuleHandleW(L"zluda_real.dll");
        if (!real) real = LoadLibraryW(L"zluda_real.dll");
        copy = (decltype(copy))GetProcAddress(real, "cuMemcpyDtoH_v2");
    }
    unsigned char bytes[64];
    if (!copy || !handle || copy(bytes, handle, sizeof bytes) != 0) return;
    char text[2 * sizeof bytes + 16];
    int at = 0;
    for (size_t i = 0; i < sizeof bytes; i += 4)
        at += snprintf(text + at, sizeof text - at, "%02X%02X%02X%02X ", bytes[i + 3],
                       bytes[i + 2], bytes[i + 1], bytes[i]);
    logf("[cuda] page %s 0x%llX: %s\n", what, handle, text);
}

// Defined with the array table, far below.
void record_array(CUarray array);

__declspec(dllexport) CUresult cuTexObjectCreate(CUtexObject *out, const ResourceDesc *desc,
                                                 const void *tex_desc, const void *view_desc) {
    auto fn = as<CUresult (*)(CUtexObject *, const ResourceDesc *, const void *, const void *)>(
        g_orig.tex_object_create, "cuTexObjectCreate");
    CUresult r = fn(out, desc, tex_desc, view_desc);
    logf("[cuda] cuTexObjectCreate(%s, handle=%p) -> %d, object=0x%llX\n",
         desc ? res_type_name(desc->resType) : "null", desc ? desc->handle : nullptr, r,
         out ? *out : 0ull);
    // The sampler is an input to the kernel like any other, and the only one
    // that lives nowhere this proxy can read afterwards: how a fetch outside
    // the picture behaves -- clamped, wrapped, or a border colour -- is decided
    // here and then fixed for the life of the feature. The kernel that decides
    // this run reads through five of these. If the bytes differ between two
    // runs that end differently, that is the input nothing had compared.
    if (tex_desc) {
        // CUDA_TEXTURE_DESC: three address modes, filter, flags, anisotropy,
        // mipmap filter and clamps, a border colour, then reserved words.
        const unsigned char *bytes = (const unsigned char *)tex_desc;
        char text[2 * 104 + 1];
        int at = 0;
        for (size_t i = 0; i < 104; ++i)
            at += snprintf(text + at, sizeof text - at, "%02X", bytes[i]);
        logf("               sampler: %s\n", text);
    }
    // Only one of the arrays in play comes from cuArrayCreate; the rest arrive
    // as imports and were invisible to the state readback, which is how the
    // network's own pictures went unchecked while everything else was compared
    // byte for byte. A texture is a reason to track the array behind it.
    if (r == 0 && desc && desc->resType == 0) record_array((CUarray)desc->handle);
    if (r == 0 && out) log_descriptor_page("texture", *out);
    if (r == 0 && out && desc && desc->resType == 0)
        copy_sampler_onto_surfaces(desc->handle, *out);
    return r;
}

__declspec(dllexport) CUresult cuTexObjectGetResourceDesc(ResourceDesc *desc, CUtexObject obj) {
    auto fn = as<CUresult (*)(ResourceDesc *, CUtexObject)>(g_orig.tex_object_get_resource_desc,
                                                            "cuTexObjectGetResourceDesc");
    CUresult r = fn(desc, obj);
    logf("[cuda] cuTexObjectGetResourceDesc(0x%llX) -> %d%s%s\n", obj, r, r == 0 ? ", type=" : "",
         r == 0 && desc ? res_type_name(desc->resType) : "");
    return r;
}

__declspec(dllexport) CUresult cuTexObjectDestroy(CUtexObject obj) {
    auto fn = as<CUresult (*)(CUtexObject)>(g_orig.tex_object_destroy, "cuTexObjectDestroy");
    return fn(obj);
}

__declspec(dllexport) CUresult cuSurfObjectCreate(CUsurfObject *out, const ResourceDesc *desc) {
    auto fn = as<CUresult (*)(CUsurfObject *, const ResourceDesc *)>(g_orig.surf_object_create,
                                                                     "cuSurfObjectCreate");
    CUresult r = fn(out, desc);
    logf("[cuda] cuSurfObjectCreate(%s, handle=%p) -> %d, object=0x%llX\n",
         desc ? res_type_name(desc->resType) : "null", desc ? desc->handle : nullptr, r,
         out ? *out : 0ull);
    if (r == 0 && desc && desc->resType == 0) record_array((CUarray)desc->handle);
    if (r == 0 && out) log_descriptor_page("surface", *out);
    if (r == 0 && out && desc && desc->resType == 0 &&
        g_surface_count < 64)
        g_surfaces[g_surface_count++] = {desc->handle, *out};
    return r;
}

__declspec(dllexport) CUresult cuSurfObjectGetResourceDesc(ResourceDesc *desc, CUsurfObject obj) {
    auto fn = as<CUresult (*)(ResourceDesc *, CUsurfObject)>(g_orig.surf_object_get_resource_desc,
                                                             "cuSurfObjectGetResourceDesc");
    CUresult r = fn(desc, obj);
    logf("[cuda] cuSurfObjectGetResourceDesc(0x%llX) -> %d%s%s\n", obj, r, r == 0 ? ", type=" : "",
         r == 0 && desc ? res_type_name(desc->resType) : "");
    return r;
}

__declspec(dllexport) CUresult cuSurfObjectDestroy(CUsurfObject obj) {
    auto fn = as<CUresult (*)(CUsurfObject)>(g_orig.surf_object_destroy, "cuSurfObjectDestroy");
    return fn(obj);
}

__declspec(dllexport) CUresult cuArrayGetDescriptor_v2(void *out, CUarray array) {
    auto fn = as<CUresult (*)(void *, CUarray)>(g_orig.array_get_descriptor,
                                                "cuArrayGetDescriptor_v2");
    CUresult r = fn(out, array);
    if (r) logf("[cuda] cuArrayGetDescriptor(%p) -> %d\n", array, r);
    return r;
}

// Defined further down, with the rest of the state-reading machinery.
__declspec(dllexport) CUresult cuArrayCreate_v2(CUarray *out, const void *desc) {
    auto fn = as<CUresult (*)(CUarray *, const void *)>(g_orig.array_create, "cuArrayCreate_v2");
    CUresult r = fn(out, desc);
    if (r) logf("[cuda] cuArrayCreate -> %d\n", r);
    if (!r && out) record_array(*out);
    return r;
}

__declspec(dllexport) CUresult cuMipmappedArrayGetLevel(CUarray *out, CUmipmappedArray mip,
                                                        unsigned int level) {
    auto fn = as<CUresult (*)(CUarray *, CUmipmappedArray, unsigned int)>(
        g_orig.mipmapped_array_get_level, "cuMipmappedArrayGetLevel");
    CUresult r = fn(out, mip, level);
    if (r) logf("[cuda] cuMipmappedArrayGetLevel(%p, %u) -> %d\n", mip, level, r);
    return r;
}

// The dispatch path. This is where the evaluation now stops, with the snippet
// reporting "operation not supported"; these four are the calls that can say it.

// With NVCUDA_PROXY_DUMP set to a directory, every module the snippet loads is
// written there. The snippet ships PTX, so this yields the actual source of the
// kernel that faults -- which no amount of looking at the caller can give.
void dump_module(const void *image, unsigned index) {
    char dir[MAX_PATH];
    if (!GetEnvironmentVariableA("NVCUDA_PROXY_DUMP", dir, sizeof dir)) return;
    CreateDirectoryA(dir, nullptr);
    // PTX arrives as NUL-terminated text; anything else is left alone rather
    // than guessed at, since a fatbin's length is not knowable from here.
    const char *text = (const char *)image;
    if (strncmp(text, "//", 2) != 0 && strstr(text, ".version") == nullptr) return;
    char path[MAX_PATH];
    snprintf(path, sizeof path, "%s\\module_%03u.ptx", dir, index);
    if (FILE *f = fopen(path, "wb")) {
        fwrite(text, 1, strlen(text), f);
        fclose(f);
    }
}

__declspec(dllexport) CUresult cuModuleLoadData(CUmodule *out, const void *image) {
    auto fn = as<CUresult (*)(CUmodule *, const void *)>(g_orig.module_load_data,
                                                         "cuModuleLoadData");
    static unsigned index = 0;
    unsigned mine = index++;
    if (image) dump_module(image, mine);
    LARGE_INTEGER freq, start, end;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&start);
    CUresult r = fn(out, image);
    QueryPerformanceCounter(&end);
    // Every load, not only the failures. How long one takes says whether the
    // cache answered or the module was translated from scratch, and a module
    // that never loads at all is invisible in a log that only reports errors.
    logf("[cuda] cuModuleLoadData(module %u) -> %d in %.1f s\n", mine, r,
         (double)(end.QuadPart - start.QuadPart) / (double)freq.QuadPart);
    return r;
}

// A launch reports a bare pointer, which says nothing. Remembering what each
// function was called turns the trace into something readable, and the names
// are the snippet's own -- they say which stage of the network is running.
struct KnownFunction {
    CUfunction handle;
    char name[96];
};
KnownFunction g_functions[512];
unsigned g_function_count = 0;

// The variant of a swapped kernel, held back for one launch: see
// NVCUDA_PROXY_SWAP_LAUNCH in cuModuleGetFunction.
CUfunction g_swap_one = nullptr;
char g_swap_name[128] = {};

int swap_one_launch() {
    static int n = -1;
    if (n < 0) {
        char buffer[16] = {};
        n = GetEnvironmentVariableA("NVCUDA_PROXY_SWAP_LAUNCH", buffer, sizeof buffer) > 0
                ? atoi(buffer)
                : 0;
    }
    return n;
}

const char *function_name(CUfunction f) {
    for (unsigned i = 0; i < g_function_count; ++i)
        if (g_functions[i].handle == f) return g_functions[i].name;
    return "?";
}

__declspec(dllexport) CUresult cuModuleGetFunction(CUfunction *out, CUmodule mod,
                                                   const char *name) {
    auto fn = as<CUresult (*)(CUfunction *, CUmodule, const char *)>(g_orig.module_get_function,
                                                                     "cuModuleGetFunction");
    // NVCUDA_PROXY_SWAP_PTX=<file> with NVCUDA_PROXY_SWAP_KERNEL=<name> hands
    // back that kernel out of a module of the proxy's own instead of the one
    // the program asked.
    //
    // Why: the kernel that decides this run lives in 9.3 MB of PTX shared with
    // thirty-seven others, and translating that costs half an hour -- far too
    // slow to try one idea after another. On its own it is 254 KB and
    // translates in thirteen seconds, and it calls no functions, so lifting it
    // out changes nothing about what it does. With the swap in place the
    // program runs exactly as it always did, on its real inputs, while the one
    // kernel under study can be edited and retranslated between one run and
    // the next.
    bool swapped_in = false;
    {
        static char want[128];
        static int have = -1;
        static CUmodule mine = nullptr;
        if (have < 0) {
            have = GetEnvironmentVariableA("NVCUDA_PROXY_SWAP_KERNEL", want, sizeof want) > 0;
        }
        if (have && name && strcmp(name, want) == 0) {
            if (!mine) {
                char path[MAX_PATH] = {};
                GetEnvironmentVariableA("NVCUDA_PROXY_SWAP_PTX", path, sizeof path);
                FILE *file = path[0] ? fopen(path, "rb") : nullptr;
                if (file) {
                    fseek(file, 0, SEEK_END);
                    const long size = ftell(file);
                    fseek(file, 0, SEEK_SET);
                    // One trailing zero: a module held as text is read as a C
                    // string and the file has no terminator of its own.
                    char *image = (char *)calloc((size_t)size + 1, 1);
                    const bool whole = image && fread(image, 1, (size_t)size, file) == (size_t)size;
                    fclose(file);
                    auto load = as<CUresult (*)(CUmodule *, const void *)>(
                        g_orig.module_load_data, "cuModuleLoadData");
                    const CUresult loaded = whole ? load(&mine, image) : (CUresult)-1;
                    logf("[cuda] replacement module %s: %ld bytes -> %d\n", path, size, loaded);
                    free(image);
                    if (loaded != 0) mine = nullptr;
                }
            }
            if (mine) {
                const CUresult swapped = fn(out, mine, name);
                logf("[cuda] %s taken from the replacement module -> %d\n", name, swapped);
                swapped_in = swapped == 0;
                // NVCUDA_PROXY_SWAP_LAUNCH=<n>: the variant stands in for the
                // n-th launch only, and the original runs every other time. A
                // variant that writes something other than the kernel's real
                // output -- an intermediate, for comparison -- would otherwise
                // hand every later launch inputs no kernel was meant to read,
                // and by the launch under study the network holds nothing but
                // NaN.
                if (swapped_in && swap_one_launch() > 0) {
                    g_swap_one = *out;
                    strncpy(g_swap_name, name, sizeof g_swap_name - 1);
                    swapped_in = false;
                    logf("[cuda] %s: variant kept for launch %d only\n", name, swap_one_launch());
                }
            }
        }
    }

    CUresult r = swapped_in ? (CUresult)0 : fn(out, mod, name);
    if (r) {
        logf("[cuda] cuModuleGetFunction(\"%s\") -> %d\n", name ? name : "?", r);
        return r;
    }
    if (out && name && g_function_count < 512) {
        EnterCriticalSection(&g_lock);
        KnownFunction &k = g_functions[g_function_count++];
        k.handle = *out;
        strncpy(k.name, name, sizeof k.name - 1);
        k.name[sizeof k.name - 1] = 0;
        LeaveCriticalSection(&g_lock);
    }
    // What the kernel costs. For a kernel that faults with valid arguments the
    // interesting number is the private (local) size: a large one means the
    // compiler spilled, and scratch that the queue cannot satisfy shows up as a
    // memory violation rather than as an allocation failure.
    {
        static char want[96];
        static int have = -1;
        if (have < 0) have = GetEnvironmentVariableA("NVCUDA_PROXY_ARGS", want, sizeof want) > 0;
        if (have && name && strcmp(name, want) == 0) {
            static CUresult (*attr)(int *, int, CUfunction) = nullptr;
            if (!attr) {
                HMODULE real = GetModuleHandleW(L"zluda_real.dll");
                if (!real) real = LoadLibraryW(L"zluda_real.dll");
                attr = (decltype(attr))GetProcAddress(real, "cuFuncGetAttribute");
            }
            static const struct {
                int id;
                const char *label;
            } kAttrs[] = {{0, "max threads per block"},
                          {1, "shared memory (bytes)"},
                          {3, "private memory per thread (bytes)"},
                          {4, "registers per thread"}};
            for (const auto &a : kAttrs) {
                int value = -1;
                if (attr && attr(&value, a.id, *out) == 0)
                    logf("[cuda] %s: %-28s %d\n", name, a.label, value);
            }
        }
    }
    return r;
}

__declspec(dllexport) CUresult cuFuncSetAttribute(CUfunction f, int attrib, int value) {
    auto fn = as<CUresult (*)(CUfunction, int, int)>(g_orig.func_set_attribute,
                                                     "cuFuncSetAttribute");
    CUresult r = fn(f, attrib, value);
    if (r) logf("[cuda] cuFuncSetAttribute(func=%p, attr=%d, val=%d) -> %d\n", f, attrib, value, r);
    return r;
}

// Defined with the allocation tracking, below.
void describe_pointer(unsigned long long p);

// Defined there too: whether an address is in one of them.
bool inside_known_allocation(unsigned long long p);

// Defined below, next to the array table it reads.
void checksum_after_launch(unsigned index, const char *kernel);

    // Defined with them too: one digest of a whole allocation, read back now.
unsigned long long digest_allocation(unsigned which);

// And saving and putting back an allocation whole, for the experiment that
// needs the memory identical and the hardware not.
bool save_allocation(unsigned which);
bool restore_allocation(unsigned which);

// And one for a bare device address, whose extent nobody told us.
unsigned long long digest_address(unsigned long long base, size_t *read_bytes);

__declspec(dllexport) CUresult cuLaunchKernel(CUfunction f, unsigned gx, unsigned gy, unsigned gz,
                                              unsigned bx, unsigned by, unsigned bz,
                                              unsigned shared, CUstream stream, void **params,
                                              void **extra) {
    auto fn = as<CUresult (*)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned,
                              unsigned, unsigned, CUstream, void **, void **)>(g_orig.launch_kernel,
                                                                              "cuLaunchKernel");
    // With NVCUDA_PROXY_ARGS set to a kernel name, that kernel's packed
    // argument buffer is printed before it runs. A kernel that faults does so
    // on an address it was handed, and this is where those addresses are.
    {
        static char want[96];
        static int have = -1;
        if (have < 0) have = GetEnvironmentVariableA("NVCUDA_PROXY_ARGS", want, sizeof want) > 0;
        if (have && extra && strcmp(function_name(f), want) == 0) {
            const void *buffer = nullptr;
            size_t size = 0;
            for (int i = 0; extra[i] && i < 16; i += 2) {
                if ((size_t)extra[i] == 1) buffer = extra[i + 1];
                if ((size_t)extra[i] == 2) size = *(const size_t *)extra[i + 1];
            }
            logf("[cuda] arguments of %s: %zu bytes\n", want, size);
            // With NVCUDA_PROXY_ARGHASH set, an argument that looks like a
            // device address but falls outside every allocation the proxy
            // knows about is read back and folded to a digest. Those are the
            // buffers nothing has ever compared: the tracking follows
            // cuMemAlloc, and memory that arrives another way -- imported from
            // D3D12, or handed over as an array -- is invisible to it. The
            // kernel that decides this run is handed one of them.
            static int arg_hash = -1;
            if (arg_hash < 0) {
                char on[8];
                arg_hash = GetEnvironmentVariableA("NVCUDA_PROXY_ARGHASH", on, sizeof on) > 0;
            }
            const unsigned long long *w = (const unsigned long long *)buffer;
            for (size_t i = 0; buffer && i * 8 < size; ++i) {
                logf("        [+0x%02zX] 0x%016llX\n", i * 8, w[i]);
                describe_pointer(w[i]);
                // Page-aligned and in the range the driver hands out: worth
                // reading. Anything else is a float or a packed pair of
                // dimensions and reading it would only fault.
                const bool plausible = w[i] >= 0x100000000ull && w[i] < 0x1000000000ull &&
                                       (w[i] & 0xFFFull) == 0;
                if (arg_hash && plausible && !inside_known_allocation(w[i])) {
                    size_t read_bytes = 0;
                    const unsigned long long d = digest_address(w[i], &read_bytes);
                    logf("               read %zu bytes -> %016llX\n", read_bytes, d);
                }
            }
        }
    }

    // Kept because NVCUDA_PROXY_GRID1 is about to overwrite them, and the
    // experiment below needs the real grid to dirty the whole card with.
    const unsigned full_gx = gx, full_gy = gy, full_gz = gz;

    // Counted here rather than at the log line further down: the experiment
    // below has to save memory *before* the launch it is about to measure, and
    // by the time that line runs the kernel has already written.
    static unsigned seq = 0;
    ++seq;

    static int restore_at = -1;
    if (restore_at < 0) {
        char buffer[16] = {};
        restore_at = GetEnvironmentVariableA("NVCUDA_PROXY_RESTORE", buffer, sizeof buffer) > 0
                         ? atoi(buffer)
                         : 0;
    }
    const bool measuring = restore_at > 0 && (int)seq == restore_at;
    const bool saved = measuring && save_allocation(2);

    // With NVCUDA_PROXY_GRID1, the named kernel runs as a single block. The
    // result is meaningless, but whether it still faults is not: a fault with
    // one block is a fault in the kernel body, while a fault that only appears
    // with the full grid points at the tiles on the edge of the image.
    {
        static int one_block = -1;
        if (one_block < 0) {
            char buf[8];
            one_block = GetEnvironmentVariableA("NVCUDA_PROXY_GRID1", buf, sizeof buf) > 0;
        }
        static char want[96];
        static int have = -1;
        if (have < 0) have = GetEnvironmentVariableA("NVCUDA_PROXY_ARGS", want, sizeof want) > 0;
        if (one_block && have && strcmp(function_name(f), want) == 0) {
            gx = gy = gz = 1;
            logf("[cuda] %s: grid reduced to 1x1x1\n", want);
        }
    }

    // If a native kernel of ours matches this launch exactly -- same name, same
    // block shape, same argument buffer size -- it is queued on the caller's own
    // stream instead of the translated one. Anything short of an exact match
    // returns false and the translation runs untouched, which is the property
    // the whole approach rests on: the worst case is the speed we already have.
    // The packed buffer this launch carries, found the way CUDA describes it:
    // marker 1 is the pointer, marker 2 the size. Read here rather than in the
    // argument log further up because a dump has to straddle the launch.
    const void *packed = nullptr;
    size_t packed_size = 0;
    if (extra)
        for (int i = 0; extra[i] && i < 16; i += 2) {
            if ((size_t)extra[i] == 1) packed = extra[i + 1];
            if ((size_t)extra[i] == 2) packed_size = *(const size_t *)extra[i + 1];
        }
    const unsigned dump_index = dump::note(function_name(f));
    if (dump_index) dump::capture(function_name(f), dump_index, "pre", packed, packed_size);

    // The one launch a held-back variant stands in for.
    CUfunction launch_f = f;
    if (g_swap_one && strcmp(function_name(f), g_swap_name) == 0) {
        static int seen = 0;
        if (++seen == swap_one_launch()) {
            launch_f = g_swap_one;
            logf("[cuda] %s launch %d: variant\n", g_swap_name, seen);
        }
    }

    const bool native_taken =
        native::try_launch(function_name(f), gx, gy, gz, bx, by, bz, shared, stream, extra);
    CUresult r = native_taken ? (CUresult)0
                              : fn(launch_f, gx, gy, gz, bx, by, bz, shared, stream, params, extra);

    // The same pointers again, now that the kernel has run: the difference
    // between the two is this kernel's output and nothing else.
    if (dump_index) dump::capture(function_name(f), dump_index, "post", packed, packed_size);

    // Launches are asynchronous, so a fault is reported by whichever call comes
    // after it, not by the launch that caused it. With NVCUDA_PROXY_SYNC set,
    // each launch is followed by a synchronise, which pins the error on the
    // kernel that actually faulted. It makes the run slower and is a diagnostic
    // only.
    int sync_result = 0;
    double sync_ms = 0.0;
    static int sync_each = -1;
    if (sync_each < 0) {
        char buf[8];
        sync_each = GetEnvironmentVariableA("NVCUDA_PROXY_SYNC", buf, sizeof buf) > 0;
    }
    if (sync_each && r == 0) {
        static CUresult (*sync)(void) = nullptr;
        if (!sync) {
            HMODULE real = GetModuleHandleW(L"zluda_real.dll");
            if (!real) real = LoadLibraryW(L"zluda_real.dll");
            sync = (decltype(sync))GetProcAddress(real, "cuCtxSynchronize");
        }
        // How long the synchronise takes separates two very different failures
        // that report the same code: an invalid access comes back at once,
        // while a kernel that never finishes is killed by the driver's timeout
        // after a couple of seconds.
        LARGE_INTEGER freq, start, end;
        QueryPerformanceFrequency(&freq);
        QueryPerformanceCounter(&start);
        if (sync) sync_result = sync();
        QueryPerformanceCounter(&end);
        sync_ms = 1000.0 * (double)(end.QuadPart - start.QuadPart) / (double)freq.QuadPart;
    }

    // Every launch, not just the failing ones: a fault poisons the context, so
    // each launch after the first failure fails too and only the first one is
    // the real report. The sequence number is what makes that readable.
    logf("[cuda] launch %-3u %-46s grid=%ux%ux%u block=%ux%ux%u shared=%u -> %d%s%s%s\n", seq,
         function_name(f), gx, gy, gz, bx, by, bz, shared, r, r ? "  <-- FAILED" : "",
         sync_result ? "  <-- FAULT AT SYNCHRONISATION" : "",
         native_taken ? "  <-- NATIVE" : "");
    // With the synchronise on, every launch is timed, not only the failing ones:
    // the question "why is this slow" is answered by where the time actually
    // goes, and a log that reports only errors cannot answer it.
    if (sync_each)
        logf("        %8.2f ms  %s%s\n", sync_ms, function_name(f),
             sync_result ? "  <-- FAULT" : "");
    // NVCUDA_PROXY_REPLAY=<launch> runs that one launch a second time, right
    // after the first, with the same arguments, and reads the pool after each.
    // Everything the kernel is handed has been shown identical across the three
    // outcomes -- machine code, arguments, the 148 MB of weights, the working
    // pool read whole -- and the outcome is settled when the network is built
    // and never moves afterwards. What is left to ask is whether the kernel is
    // a function of its inputs at all, and two runs inside one process answer
    // that without comparing one process to another.
    static int replay_at = -1;
    if (replay_at < 0) {
        char buffer[16] = {};
        replay_at =
            GetEnvironmentVariableA("NVCUDA_PROXY_REPLAY", buffer, sizeof buffer) > 0
                ? atoi(buffer)
                : 0;
    }
    if (replay_at > 0 && (int)seq == replay_at && r == 0) {
        const unsigned long long first = digest_allocation(2);
        const CUresult again = fn(f, gx, gy, gz, bx, by, bz, shared, stream, params, extra);
        const unsigned long long second = digest_allocation(2);
        logf("[replay] %s: first %016llX, second %016llX -> %s (relaunch %d)\n",
             function_name(f), first, second, first == second ? "same" : "DIFFERENT",
             again);
    }

    // NVCUDA_PROXY_RESTORE=<launch> asks the one question a plain replay
    // cannot answer.
    //
    // Everything measurable says this kernel is a function of memory, and
    // everything measurable says the memory going into it is identical between
    // two runs that end differently. Both cannot be true. The gap a replay
    // leaves is that it starts from the state the first launch produced, on a
    // machine the first launch just warmed, so neither the memory nor the
    // hardware is the same as at the launch being explained.
    //
    // This closes it: the pool is saved *before* the launch, the launch runs
    // and is measured, the pool is put back exactly as it was, the whole card
    // is dirtied by running the same kernel over its real grid, the pool is put
    // back once more, and the launch is repeated. Two runs, the same memory to
    // the bit, a machine that has been through something else in between. If
    // the answers differ, the kernel is not a function of memory after all.
    if (saved && r == 0) {
        const unsigned long long first = digest_allocation(2);
        restore_allocation(2);
        const CUresult dirty =
            fn(f, full_gx, full_gy, full_gz, bx, by, bz, shared, stream, params, extra);
        digest_allocation(2); // synchronises
        restore_allocation(2);
        const CUresult again = fn(f, gx, gy, gz, bx, by, bz, shared, stream, params, extra);
        const unsigned long long second = digest_allocation(2);
        logf("[restore] %s: first %016llX, after dirtying the card %016llX -> %s"
             " (dirty grid %ux%ux%u, relaunches %d/%d)\n",
             function_name(f), first, second, first == second ? "same" : "DIFFERENT", full_gx,
             full_gy, full_gz, dirty, again);
    } else if (measuring) {
        logf("[restore] not run: copying the pool failed\n");
    }

    if (r == 0) checksum_after_launch(seq, function_name(f));
    return r;
}

// Every device allocation, with its extent. A kernel that faults does so on an
// address, and the only way to tell a wrong address from a correct one is to
// know which ranges are actually mapped.
struct Allocation {
    CUdeviceptr base;
    size_t size;
};
Allocation g_allocations[256];
unsigned g_allocation_count = 0;

__declspec(dllexport) CUresult cuMemAlloc_v2(CUdeviceptr *out, size_t bytes) {
    auto fn = as<CUresult (*)(CUdeviceptr *, size_t)>(g_orig.mem_alloc, "cuMemAlloc_v2");
    CUresult r = fn(out, bytes);
    if (r) {
        logf("[cuda] cuMemAlloc(%zu bytes) -> %d\n", bytes, r);
        return r;
    }
    if (out && g_allocation_count < 256) {
        EnterCriticalSection(&g_lock);
        g_allocations[g_allocation_count++] = {*out, bytes};
        LeaveCriticalSection(&g_lock);
        // The size matters when reading these back: the network's working
        // memory is a few large pools, and a window at the start of one says
        // almost nothing about the rest of it.
        logf("[cuda] cuMemAlloc(%zu bytes) -> allocation %u at 0x%llX\n", bytes,
             g_allocation_count - 1, *out);
    }
    return r;
}

// Names the allocation an address belongs to, and how far into it the address
// sits. "outside every known allocation" is the interesting answer.
// Whether an address falls in something the proxy is already reading, so that
// the expensive blind read is only done for the memory nobody tracks.
bool inside_known_allocation(unsigned long long p) {
    for (unsigned i = 0; i < g_allocation_count; ++i) {
        const Allocation &a = g_allocations[i];
        if (p >= a.base && p < a.base + a.size) return true;
    }
    return false;
}

void describe_pointer(unsigned long long p) {
    if (!p) return;
    for (unsigned i = 0; i < g_allocation_count; ++i) {
        const Allocation &a = g_allocations[i];
        if (p >= a.base && p < a.base + a.size) {
            logf("               inside allocation %u: base 0x%llX, %zu bytes, offset %llu\n", i,
                 a.base, a.size, p - a.base);
            return;
        }
    }
    logf("               outside every known allocation\n");
}

// Reading the network's state back after every launch, so that a run that comes
// out right and one that comes out flat can be compared launch by launch and
// the first one that differs named.
//
// It exists because guessing was not converging. The evaluation gives the right
// frame about three times in eight and a featureless rectangle otherwise, with
// each outcome reproducible bit for bit, and the driver call sequence of the
// two is identical line for line. Several plausible causes were tried and
// eliminated one at a time. Finding the first kernel whose output differs says
// where to look instead.
//
// NVCUDA_PROXY_HASH is how many rows of each array to read. Reading all of them
// would move tens of gigabytes over the bus across a run; the top few rows are
// enough to date a divergence, and the number can be raised if a failure turns
// out to spare them.

struct ArrayInfo {
    CUarray handle;
    size_t width, height, element_bytes;
};
ArrayInfo g_arrays[256];
unsigned g_array_count = 0;

struct ArrayDescriptor {
    size_t width, height;
    unsigned format, channels;
};

// The four bytes at +6 of a fatbin header are not more obscure than this: the
// format is an enum where the low nibble counts integer widths and 0x10 and
// 0x20 are the two float types.
size_t element_bytes_of(unsigned format, unsigned channels) {
    size_t per_channel = 0;
    switch (format) {
        case 0x01: case 0x08: per_channel = 1; break; // 8 bit, unsigned or signed
        case 0x02: case 0x09: case 0x10: per_channel = 2; break; // 16 bit, or half
        case 0x03: case 0x0a: case 0x20: per_channel = 4; break; // 32 bit, or float
        default: return 0;
    }
    return per_channel * channels;
}

// Every array the snippet creates, with enough of its shape to read it back.
// The descriptor has to be asked for separately: what cuArrayCreate is handed
// is a pointer whose layout this proxy has no reason to depend on.
void record_array(CUarray array) {
    if (g_array_count >= 256) return;
    for (unsigned i = 0; i < g_array_count; ++i)
        if (g_arrays[i].handle == array) return;
    static CUresult (*get_descriptor)(ArrayDescriptor *, CUarray) = nullptr;
    if (!get_descriptor) {
        HMODULE real = GetModuleHandleW(L"zluda_real.dll");
        if (!real) real = LoadLibraryW(L"zluda_real.dll");
        get_descriptor = (decltype(get_descriptor))GetProcAddress(real, "cuArrayGetDescriptor_v2");
    }
    ArrayDescriptor d{};
    if (!get_descriptor || get_descriptor(&d, array) != 0) return;
    EnterCriticalSection(&g_lock);
    g_arrays[g_array_count++] = {array, d.width, d.height, element_bytes_of(d.format, d.channels)};
    LeaveCriticalSection(&g_lock);
    logf("[cuda] array %p: %zux%zu, format %u, %u channels -> %zu bytes per element\n",
         array, d.width, d.height, d.format, d.channels,
         element_bytes_of(d.format, d.channels));
    EnterCriticalSection(&g_lock);
    LeaveCriticalSection(&g_lock);
}

struct Memcpy2D {
    size_t src_x_in_bytes, src_y;
    unsigned src_memory_type;
    const void *src_host;
    CUdeviceptr src_device;
    CUarray src_array;
    size_t src_pitch;
    size_t dst_x_in_bytes, dst_y;
    unsigned dst_memory_type;
    void *dst_host;
    CUdeviceptr dst_device;
    CUarray dst_array;
    size_t dst_pitch;
    size_t width_in_bytes, height;
};

unsigned long long fold(unsigned long long digest, const unsigned char *bytes, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        digest ^= bytes[i];
        digest *= 1099511628211ull;
    }
    return digest;
}

// One allocation, read whole in 4 KB windows and folded to a single number.
// Sampling has already sent this hunt the wrong way once, so this reads all of
// it: the point of a replay is that nothing is left to chance.
unsigned long long digest_allocation(unsigned which) {
    static CUresult (*sync)(void) = nullptr;
    static CUresult (*copy)(void *, CUdeviceptr, size_t) = nullptr;
    if (!sync) {
        HMODULE real = GetModuleHandleW(L"zluda_real.dll");
        if (!real) real = LoadLibraryW(L"zluda_real.dll");
        sync = (decltype(sync))GetProcAddress(real, "cuCtxSynchronize");
        copy = (decltype(copy))GetProcAddress(real, "cuMemcpyDtoH_v2");
    }
    if (!sync || !copy || sync() != 0 || which >= g_allocation_count) return 0;
    const Allocation &m = g_allocations[which];
    constexpr size_t window = 4096;
    static unsigned char buffer[window];
    unsigned long long digest = 1469598103934665603ull;
    for (size_t at = 0; at + window <= m.size; at += window) {
        if (copy(buffer, m.base + at, window) != 0) break;
        digest = fold(digest, buffer, window);
    }
    return digest;
}

// A device address with no declared extent: read 4 KB at a time until the copy
// refuses, which is where the mapping ends. The byte count is logged next to
// the digest, because two digests only mean something if they cover the same
// bytes.
unsigned long long digest_address(unsigned long long base, size_t *read_bytes) {
    static CUresult (*sync)(void) = nullptr;
    static CUresult (*copy)(void *, CUdeviceptr, size_t) = nullptr;
    if (!sync) {
        HMODULE real = GetModuleHandleW(L"zluda_real.dll");
        if (!real) real = LoadLibraryW(L"zluda_real.dll");
        sync = (decltype(sync))GetProcAddress(real, "cuCtxSynchronize");
        copy = (decltype(copy))GetProcAddress(real, "cuMemcpyDtoH_v2");
    }
    *read_bytes = 0;
    if (!sync || !copy || sync() != 0) return 0;
    // A surface object's handle is a device address too, but what sits there
    // is the image descriptor the runtime built, not an allocation the program
    // ever asked for: 64 bytes, and a larger read comes back refused. That
    // descriptor is the one input to the deciding kernel that has never been
    // compared between runs, so when the read is that small it is printed in
    // full rather than folded away into a digest.
    {
        static const size_t probes[] = {4096, 1024, 256, 64, 16};
        static unsigned char probe[4096];
        for (size_t n : probes) {
            if (copy(probe, base, n) != 0) continue;
            if (n <= 256) {
                char text[3 * 256 + 1];
                int at = 0;
                for (size_t i = 0; i < n; ++i)
                    at += snprintf(text + at, sizeof text - at, "%02X", probe[i]);
                logf("               descriptor of %zu bytes: %s\n", n, text);
                *read_bytes = n;
                return fold(1469598103934665603ull, probe, n);
            }
            break;
        }
    }
    constexpr size_t window = 4096;
    constexpr size_t ceiling = (size_t)512 * 1024 * 1024;
    static unsigned char buffer[window];
    unsigned long long digest = 1469598103934665603ull;
    for (size_t at = 0; at < ceiling; at += window) {
        if (copy(buffer, base + at, window) != 0) break;
        digest = fold(digest, buffer, window);
        *read_bytes += window;
    }
    return digest;
}

// The whole allocation held on the host, so it can be put back byte for byte.
unsigned char *g_saved = nullptr;
size_t g_saved_size = 0;

static bool copy_helpers(CUresult (**sync)(void), CUresult (**down)(void *, CUdeviceptr, size_t),
                         CUresult (**up)(CUdeviceptr, const void *, size_t)) {
    static CUresult (*s)(void) = nullptr;
    static CUresult (*d)(void *, CUdeviceptr, size_t) = nullptr;
    static CUresult (*u)(CUdeviceptr, const void *, size_t) = nullptr;
    if (!s) {
        HMODULE real = GetModuleHandleW(L"zluda_real.dll");
        if (!real) real = LoadLibraryW(L"zluda_real.dll");
        s = (decltype(s))GetProcAddress(real, "cuCtxSynchronize");
        d = (decltype(d))GetProcAddress(real, "cuMemcpyDtoH_v2");
        u = (decltype(u))GetProcAddress(real, "cuMemcpyHtoD_v2");
    }
    *sync = s;
    *down = d;
    *up = u;
    return s && d && u;
}

bool save_allocation(unsigned which) {
    CUresult (*sync)(void);
    CUresult (*down)(void *, CUdeviceptr, size_t);
    CUresult (*up)(CUdeviceptr, const void *, size_t);
    if (!copy_helpers(&sync, &down, &up) || which >= g_allocation_count) return false;
    const Allocation &m = g_allocations[which];
    if (m.size > g_saved_size) {
        free(g_saved);
        g_saved = (unsigned char *)malloc(m.size);
        g_saved_size = g_saved ? m.size : 0;
    }
    if (!g_saved || sync() != 0) return false;
    return down(g_saved, m.base, m.size) == 0;
}

bool restore_allocation(unsigned which) {
    CUresult (*sync)(void);
    CUresult (*down)(void *, CUdeviceptr, size_t);
    CUresult (*up)(CUdeviceptr, const void *, size_t);
    if (!copy_helpers(&sync, &down, &up) || which >= g_allocation_count || !g_saved) return false;
    const Allocation &m = g_allocations[which];
    if (sync() != 0) return false;
    return up(m.base, g_saved, m.size) == 0;
}

void checksum_after_launch(unsigned index, const char *kernel) {
    static int rows = -1;
    if (rows < 0) {
        char buf[16] = {};
        rows = GetEnvironmentVariableA("NVCUDA_PROXY_HASH", buf, sizeof buf) > 0 ? atoi(buf) : 0;
        if (rows < 0) rows = 0;
    }
    if (!rows) return;

    static CUresult (*sync)(void) = nullptr;
    static CUresult (*memcpy2d)(const Memcpy2D *) = nullptr;
    static CUresult (*memcpy_dtoh)(void *, CUdeviceptr, size_t) = nullptr;
    if (!sync) {
        HMODULE real = GetModuleHandleW(L"zluda_real.dll");
        if (!real) real = LoadLibraryW(L"zluda_real.dll");
        sync = (decltype(sync))GetProcAddress(real, "cuCtxSynchronize");
        memcpy2d = (decltype(memcpy2d))GetProcAddress(real, "cuMemcpy2D_v2");
        memcpy_dtoh = (decltype(memcpy_dtoh))GetProcAddress(real, "cuMemcpyDtoH_v2");
    }
    if (!sync || sync() != 0) return;

    static unsigned char *staging = nullptr;
    static size_t staging_size = 0;

    char line[2048];
    int at = snprintf(line, sizeof line, "[hash] %-3u %-44s ", index, kernel);

    for (unsigned i = 0; i < g_array_count && at < (int)sizeof line - 24; ++i) {
        const ArrayInfo &a = g_arrays[i];
        if (!a.element_bytes || !a.width) continue;
        const size_t want_rows = a.height < (size_t)rows ? a.height : (size_t)rows;
        const size_t pitch = a.width * a.element_bytes;
        const size_t bytes = pitch * want_rows;
        if (bytes > staging_size) {
            free(staging);
            staging = (unsigned char *)malloc(bytes);
            staging_size = staging ? bytes : 0;
        }
        if (!staging) break;
        Memcpy2D copy{};
        copy.src_memory_type = 3; // array
        copy.src_array = a.handle;
        copy.dst_memory_type = 1; // host
        copy.dst_host = staging;
        copy.dst_pitch = pitch;
        copy.width_in_bytes = pitch;
        copy.height = want_rows;
        if (!memcpy2d || memcpy2d(&copy) != 0) continue;
        const unsigned long long digest = fold(1469598103934665603ull, staging, bytes);
        at += snprintf(line + at, sizeof line - at, "a%u=%016llX ", i, digest);
    }

    // The allocations are the network's pools and run to hundreds of megabytes,
    // so they are sampled at intervals rather than from the start: a window at
    // the beginning would miss a divergence anywhere else in the pool.
    // NVCUDA_PROXY_HASH_FULL names one launch at which to read the pools whole
    // rather than sampling them. Sampling dates a divergence cheaply, but it
    // cannot see what a single block wrote: reduce the grid to one block and
    // its output falls between the windows, so every run looks alike for the
    // wrong reason. Reading 340 MB back is slow, which is why it is asked for
    // one launch at a time.
    static int full_at = -1;
    if (full_at < 0) {
        char buffer[16] = {};
        full_at = GetEnvironmentVariableA("NVCUDA_PROXY_HASH_FULL", buffer, sizeof buffer) > 0
                      ? atoi(buffer)
                      : 0;
    }
    // Every launch up to that number, not just that one: to say that a kernel
    // is what makes two runs differ, the state going into it has to be read
    // whole as well, and in the same run.
    const bool read_whole = full_at > 0 && (int)index <= full_at;
    constexpr size_t window = 4096;
    const unsigned windows = read_whole ? 0u : 64u;
    for (unsigned i = 0; i < g_allocation_count && at < (int)sizeof line - 24; ++i) {
        const Allocation &m = g_allocations[i];
        if (!m.size) continue;
        if (window > staging_size) {
            free(staging);
            staging = (unsigned char *)malloc(window);
            staging_size = staging ? window : 0;
        }
        if (!staging || !memcpy_dtoh) break;
        unsigned long long digest = 1469598103934665603ull;
        // Zero windows means read every one of them, back to back.
        const size_t stride = (windows && m.size > windows * window) ? m.size / windows : window;
        for (size_t at_offset = 0; at_offset + window <= m.size; at_offset += stride) {
            if (memcpy_dtoh(staging, m.base + at_offset, window) != 0) break;
            digest = fold(digest, staging, window);
        }
        at += snprintf(line + at, sizeof line - at, "m%u=%016llX ", i, digest);
    }

    logf("%s\n", line);

    // NVCUDA_PROXY_MAP_AT=<launch> with NVCUDA_PROXY_MAP_ALLOC=<n> writes one
    // digest per 4 KB window of that allocation, after that launch, to
    // "<log>.map". Comparing the maps of two runs that ended differently says
    // which parts of the pool they disagree about, which is a far smaller
    // question than which of 340 MB differs -- and small enough to point at the
    // threads that wrote them.
    static int map_at = -1;
    static int map_alloc = 0;
    if (map_at < 0) {
        char buffer[16] = {};
        map_at = GetEnvironmentVariableA("NVCUDA_PROXY_MAP_AT", buffer, sizeof buffer) > 0
                     ? atoi(buffer)
                     : 0;
        char which[16] = {};
        map_alloc = GetEnvironmentVariableA("NVCUDA_PROXY_MAP_ALLOC", which, sizeof which) > 0
                        ? atoi(which)
                        : 0;
    }
    // NVCUDA_PROXY_SLICE_OFF and _LEN write those bytes of the mapped
    // allocation verbatim to "<log>.slice", at the same launch. The digests
    // above say which parts of the pool two runs disagree about; this says what
    // the disagreement is made of, which is the difference between "the output
    // is wrong" and "the output is a constant" or "the output is NaN".
    if (map_at > 0 && (int)index == map_at && (unsigned)map_alloc < g_allocation_count) {
        char off_text[32] = {}, len_text[32] = {};
        const size_t slice_off =
            GetEnvironmentVariableA("NVCUDA_PROXY_SLICE_OFF", off_text, sizeof off_text) > 0
                ? (size_t)_strtoui64(off_text, nullptr, 0)
                : 0;
        const size_t slice_len =
            GetEnvironmentVariableA("NVCUDA_PROXY_SLICE_LEN", len_text, sizeof len_text) > 0
                ? (size_t)_strtoui64(len_text, nullptr, 0)
                : 0;
        const Allocation &m = g_allocations[map_alloc];
        if (slice_len && slice_off + slice_len <= m.size) {
            wchar_t log_path[MAX_PATH] = L"";
            GetEnvironmentVariableW(L"NVCUDA_PROXY_LOG", log_path, MAX_PATH);
            wchar_t slice_path[MAX_PATH];
            _snwprintf_s(slice_path, MAX_PATH, _TRUNCATE, L"%s.slice",
                         log_path[0] ? log_path : L"nvcuda_proxy.log");
            unsigned char *buffer = (unsigned char *)malloc(slice_len);
            if (buffer) {
                if (memcpy_dtoh(buffer, m.base + slice_off, slice_len) == 0) {
                    FILE *slice = _wfopen(slice_path, L"wb");
                    if (slice) {
                        fwrite(buffer, 1, slice_len, slice);
                        fclose(slice);
                    }
                }
                free(buffer);
            }
        }
    }

    if (map_at > 0 && (int)index == map_at && (unsigned)map_alloc < g_allocation_count) {
        const Allocation &m = g_allocations[map_alloc];
        wchar_t log_path[MAX_PATH] = L"";
        GetEnvironmentVariableW(L"NVCUDA_PROXY_LOG", log_path, MAX_PATH);
        wchar_t map_path[MAX_PATH];
        _snwprintf_s(map_path, MAX_PATH, _TRUNCATE, L"%s.map",
                     log_path[0] ? log_path : L"nvcuda_proxy.log");
        FILE *map = _wfopen(map_path, L"wb");
        if (map) {
            constexpr size_t map_window = 4096;
            if (map_window > staging_size) {
                free(staging);
                staging = (unsigned char *)malloc(map_window);
                staging_size = staging ? map_window : 0;
            }
            // NVCUDA_PROXY_MAP_HALO=<byte> widens the map past both ends of
            // the allocation.
            //
            // Why it is worth reading memory nobody allocated to this: the
            // kernel that decides this run is a shifted-window transformer, and
            // the first window of a shifted grid starts outside the tensor. A
            // read at a negative offset lands in whatever the allocator left
            // there -- different from one process to the next, fixed within
            // one -- and it is the one class of input that every check so far
            // has stepped over, because they all stopped at the allocation's
            // own bounds. Offsets are written signed, relative to the base.
            static long long halo = -1;
            if (halo < 0) {
                char buffer[24] = {};
                halo = GetEnvironmentVariableA("NVCUDA_PROXY_MAP_HALO", buffer, sizeof buffer) > 0
                           ? _strtoi64(buffer, nullptr, 0)
                           : 0;
                halo -= halo % (long long)map_window;
            }
            const long long first = -halo;
            const long long last = (long long)m.size + halo;
            for (long long at_offset = first; staging && at_offset + (long long)map_window <= last;
                 at_offset += (long long)map_window) {
                // Outside the allocation the read is allowed to fail: that only
                // says the page is not mapped, which is itself worth recording
                // rather than treating as the end of the map.
                if (memcpy_dtoh(staging, (CUdeviceptr)((long long)m.base + at_offset),
                                map_window) != 0) {
                    // Only a window wholly inside the allocation failing is
                    // the end of the road. The one straddling the far edge
                    // fails by construction, and stopping there used to throw
                    // away the whole halo past the end.
                    if (at_offset >= 0 &&
                        at_offset + (long long)map_window <= (long long)m.size)
                        break;
                    fprintf(map, "%lld unmapped\n", at_offset);
                    continue;
                }
                const unsigned long long d = fold(1469598103934665603ull, staging, map_window);
                fprintf(map, "%lld %016llX\n", at_offset, d);
            }
            fclose(map);
        }
    }
}

__declspec(dllexport) CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dst, const void *src, size_t bytes,
                                                    CUstream stream) {
    auto fn = as<CUresult (*)(CUdeviceptr, const void *, size_t, CUstream)>(
        g_orig.memcpy_htod_async, "cuMemcpyHtoDAsync_v2");
    CUresult r = fn(dst, src, bytes, stream);
    if (r) logf("[cuda] cuMemcpyHtoDAsync(%zu bytes) -> %d\n", bytes, r);
    return r;
}

__declspec(dllexport) CUresult cuDeviceGetAttribute(int *out, int attrib, int dev) {
    auto fn = as<CUresult (*)(int *, int, int)>(g_orig.device_get_attribute,
                                                "cuDeviceGetAttribute");
    CUresult r = fn(out, attrib, dev);
    if (r) logf("[cuda] cuDeviceGetAttribute(attr=%d) -> %d\n", attrib, r);
    return r;
}

} // extern "C"

namespace {

struct Substitution {
    const char *name;
    void *wrapper;
    void **original;
};

const Substitution kSubstitutions[] = {
    {"cuTexObjectCreate", (void *)&cuTexObjectCreate, &g_orig.tex_object_create},
    {"cuTexObjectGetResourceDesc", (void *)&cuTexObjectGetResourceDesc,
     &g_orig.tex_object_get_resource_desc},
    {"cuTexObjectDestroy", (void *)&cuTexObjectDestroy, &g_orig.tex_object_destroy},
    {"cuSurfObjectCreate", (void *)&cuSurfObjectCreate, &g_orig.surf_object_create},
    {"cuSurfObjectGetResourceDesc", (void *)&cuSurfObjectGetResourceDesc,
     &g_orig.surf_object_get_resource_desc},
    {"cuSurfObjectDestroy", (void *)&cuSurfObjectDestroy, &g_orig.surf_object_destroy},
    {"cuArrayGetDescriptor", (void *)&cuArrayGetDescriptor_v2, &g_orig.array_get_descriptor},
    {"cuArrayCreate", (void *)&cuArrayCreate_v2, &g_orig.array_create},
    {"cuMipmappedArrayGetLevel", (void *)&cuMipmappedArrayGetLevel,
     &g_orig.mipmapped_array_get_level},
    {"cuModuleLoadData", (void *)&cuModuleLoadData, &g_orig.module_load_data},
    {"cuModuleGetFunction", (void *)&cuModuleGetFunction, &g_orig.module_get_function},
    {"cuFuncSetAttribute", (void *)&cuFuncSetAttribute, &g_orig.func_set_attribute},
    {"cuLaunchKernel", (void *)&cuLaunchKernel, &g_orig.launch_kernel},
    {"cuMemAlloc", (void *)&cuMemAlloc_v2, &g_orig.mem_alloc},
    {"cuMemcpyHtoDAsync", (void *)&cuMemcpyHtoDAsync_v2, &g_orig.memcpy_htod_async},
    {"cuDeviceGetAttribute", (void *)&cuDeviceGetAttribute, &g_orig.device_get_attribute},
};

// The driver answers a versioned name with the implementation behind it, so
// both "cuMemAlloc" and "cuMemAlloc_v2" arrive here for the same entry.
bool same_symbol(const char *symbol, const char *name) {
    if (strcmp(symbol, name) == 0) return true;
    size_t n = strlen(name);
    return strncmp(symbol, name, n) == 0 && strcmp(symbol + n, "_v2") == 0;
}

void substitute(const char *symbol, void **pfn) {
    if (!symbol || !pfn || !*pfn) return;
    for (const auto &s : kSubstitutions) {
        if (!same_symbol(symbol, s.name)) continue;
        // Keep what the driver chose for this exact name and version: that is
        // what the wrapper has to call.
        *s.original = *pfn;
        *pfn = s.wrapper;
        logf("[cuda] resolved \"%s\" -> wrapper\n", symbol);
        return;
    }
    // Everything else passes through untouched, but the name is still worth
    // recording: the symbols the snippet resolves are the only driver calls it
    // can make.
    logf("[cuda] resolved \"%s\"\n", symbol);
}

} // namespace

extern "C" {

__declspec(dllexport) CUresult cuGetProcAddress(const char *symbol, void **pfn, int version,
                                                unsigned long long flags) {
    static CUresult (*fn)(const char *, void **, int, unsigned long long) = nullptr;
    if (!fn) {
        HMODULE real = GetModuleHandleW(L"zluda_real.dll");
        if (!real) real = LoadLibraryW(L"zluda_real.dll");
        fn = (decltype(fn))GetProcAddress(real, "cuGetProcAddress");
    }
    CUresult r = fn(symbol, pfn, version, flags);
    if (r == 0) substitute(symbol, pfn);
    return r;
}

__declspec(dllexport) CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int version,
                                                   unsigned long long flags, int *status) {
    static CUresult (*fn)(const char *, void **, int, unsigned long long, int *) = nullptr;
    if (!fn) {
        HMODULE real = GetModuleHandleW(L"zluda_real.dll");
        if (!real) real = LoadLibraryW(L"zluda_real.dll");
        fn = (decltype(fn))GetProcAddress(real, "cuGetProcAddress_v2");
    }
    CUresult r = fn(symbol, pfn, version, flags, status);
    if (r == 0) substitute(symbol, pfn);
    return r;
}

} // extern "C"

BOOL APIENTRY DllMain(HMODULE, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) InitializeCriticalSection(&g_lock);
    return TRUE;
}
