// The bridge: the native side of the HIP trampoline (../trampoline/). On Linux it
// is a shared library, preloaded into the process Wine runs the Windows program
// in; it publishes a table of functions (generated/thunks.c) that the
// trampoline calls through Wine's native-call door, and loads libamdhip64.so
// behind them the first time one is called.
//
// Built for Windows (DLSSNR_WIN_TEST) it is a plain DLL with one exported
// function, and calls AMD's own amdhip64_7.dll, renamed: that lets the whole
// chain be tested on a machine that runs Windows.

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core.h"
#include "../generated/hip_forwarders.h"

static const char *const kNames[HIPF_COUNT] = {HIPF_NAMES};
static void *g_real[HIPF_COUNT];
static int g_loaded;   // 0 not tried, 1 loaded, -1 failed

void dlssnr_log(const char *format, ...) {
    static int enabled = -1;
    if (enabled < 0) enabled = getenv("DLSSNR_BRIDGE_LOG") != NULL;
    if (!enabled) return;
    va_list ap;
    va_start(ap, format);
    fputs("[dlss5nr bridge] ", stderr);
    vfprintf(stderr, format, ap);
    fputc('\n', stderr);
    va_end(ap);
}

#ifdef DLSSNR_WIN_TEST
// ------------------------------------------------------------------ Windows

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

static void load_library(void) {
    char path[MAX_PATH];
    if (!GetEnvironmentVariableA("DLSSNR_HIP_LIB", path, sizeof path)) strcpy(path, "amdhip64_real.dll");
    HMODULE lib = LoadLibraryA(path);
    if (!lib) {
        dlssnr_log("could not load %s (Windows error %lu)", path, GetLastError());
        g_loaded = -1;
        return;
    }
    for (int i = 0; i < HIPF_COUNT; ++i) g_real[i] = (void *)GetProcAddress(lib, kNames[i]);
    g_loaded = 1;
}

__declspec(dllexport) LONG dlssnr_bridge_call(unsigned int code, void *args) {
    if (code >= HIPF_COUNT) return -1;
    return dlssnr_funcs[code](args);
}

#else
// -------------------------------------------------------------------- Linux

#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

static void load_library(void) {
    const char *named = getenv("DLSSNR_HIP_LIB");
    void *lib = NULL;
    if (named) lib = dlopen(named, RTLD_NOW | RTLD_LOCAL);
    if (!lib) lib = dlopen("libamdhip64.so.7", RTLD_NOW | RTLD_LOCAL);
    if (!lib) lib = dlopen("libamdhip64.so", RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        dlssnr_log("could not load libamdhip64.so.7: %s", dlerror());
        g_loaded = -1;
        return;
    }
    int missing = 0;
    for (int i = 0; i < HIPF_COUNT; ++i) {
        g_real[i] = dlsym(lib, kNames[i]);
        if (!g_real[i]) {
            dlssnr_log("libamdhip64 has no %s", kNames[i]);
            ++missing;
        }
    }
    dlssnr_log("loaded; %d of %d functions missing", missing, (int)HIPF_COUNT);
    g_loaded = 1;
}

// Where the trampoline finds the table: a file named after this process, which
// holds the address of this tag. The trampoline cannot tell which process it is
// by the number (what it reads from /proc is the wineserver's, not its own), so
// it tries every such file, and the tag is the proof: an address that holds it,
// in the trampoline's own memory, can only be this library's.
#define DLSSNR_TAG 0x31444252534c4444ull
static const struct {
    uint64_t magic;
    const dlssnr_entry *funcs;
} g_tag = {DLSSNR_TAG, dlssnr_funcs};
static char g_handle_path[96];

__attribute__((constructor)) static void publish(void) {
    // HIP is loaded now, before anything else in the process is. It brings its
    // own LLVM (through comgr), and so does the Vulkan driver (Mesa's RADV);
    // two of them in one process share their unique symbols, and whichever came
    // second finds the first one's state not its own. Measured with ROCm 7.15
    // and Mesa's LLVM 23.1: with the driver first, hipStreamCreate fails for
    // "out of memory" (ROCm cannot build its blit kernels); with HIP first both
    // work. A game creates its Direct3D device, so the driver, long before the
    // network is loaded. Loading is all it takes; nothing is initialised here.
    load_library();
    snprintf(g_handle_path, sizeof g_handle_path, "/tmp/dlss5nr-bridge-%d.handle", (int)getpid());
    FILE *f = fopen(g_handle_path, "w");
    if (!f) {
        g_handle_path[0] = 0;
        return;
    }
    fprintf(f, "0x%llx\n", (unsigned long long)(uintptr_t)&g_tag);
    fclose(f);
}

__attribute__((destructor)) static void withdraw(void) {
    if (g_handle_path[0]) unlink(g_handle_path);
}

// ---- the one function whose arguments differ between the two systems
//
// A shared D3D12 resource arrives as a Windows handle. HIP on Linux imports
// memory from a file descriptor, so the handle is turned into one by Wine's own
// ntdll, which keeps a native descriptor for every handle that stands for a
// native object.

typedef int32_t (*wine_handle_to_fd_t)(void *handle, unsigned int access, int *fd, unsigned int *options);

static wine_handle_to_fd_t find_handle_to_fd(void) {
    static wine_handle_to_fd_t found;
    static int tried;
    if (tried) return found;
    tried = 1;
    found = (wine_handle_to_fd_t)dlsym(RTLD_DEFAULT, "wine_server_handle_to_fd");
    if (found) return found;
    void *ntdll = dlopen("ntdll.so", RTLD_NOW | RTLD_NOLOAD);
    if (!ntdll) {
        // Wine's library is loaded by its full path: find it in the maps.
        FILE *maps = fopen("/proc/self/maps", "r");
        char line[1024];
        while (maps && fgets(line, sizeof line, maps)) {
            char *slash = strrchr(line, '/');
            size_t length = strlen(line);
            while (length && (line[length - 1] == '\n' || line[length - 1] == ' ')) line[--length] = 0;
            char *path = strchr(line, '/');
            if (path && slash && strcmp(slash, "/ntdll.so") == 0) {
                ntdll = dlopen(path, RTLD_NOW | RTLD_NOLOAD);
                if (ntdll) break;
            }
        }
        if (maps) fclose(maps);
    }
    if (ntdll) found = (wine_handle_to_fd_t)dlsym(ntdll, "wine_server_handle_to_fd");
    dlssnr_log("wine_server_handle_to_fd %s", found ? "found" : "NOT found");
    return found;
}

#endif /* DLSSNR_WIN_TEST */


// DLSSNR_BRIDGE_TRACE: every call, by name and in order, with the arguments of
// the few that say what a program does with HIP: what it launches and on how
// big a grid, the kernels it looks up, the arrays and objects it makes.
void dlssnr_trace_args(int index, const void *args) {
    static int trace = -1;
    if (trace < 0) trace = getenv("DLSSNR_BRIDGE_TRACE") != NULL;
    if (!trace || index < 0 || index >= HIPF_COUNT) return;
    const char *name = kNames[index];
    const uint64_t *a = (const uint64_t *)args;
    // hipModuleGetFunction hands the function back through a pointer, so the
    // pointers are kept and read later, when a launch names the function.
    static struct { uint64_t out; char name[100]; } functions[512];
    static int function_count;
    if (!strcmp(name, "hipModuleGetFunction") && function_count < 512) {
        functions[function_count].out = a[0];
        strncpy(functions[function_count].name, (const char *)(uintptr_t)a[2], 99);
        ++function_count;
    }
    const char *launched = "?";
    if (!strcmp(name, "hipModuleLaunchKernel"))
        for (int i = function_count - 1; i >= 0; --i)
            if (*(const uint64_t *)(uintptr_t)functions[i].out == a[0]) { launched = functions[i].name; break; }
    if (!strcmp(name, "hipModuleLaunchKernel") && a[4] == 16 && a[5] == 16 && a[6] == 1 && a[10]) {
        // 16 x 16 blocks are the plugin's own kernels (copy, post process): show what they are given.
        const void *const *extra = (const void *const *)(uintptr_t)a[10];
        const unsigned char *buffer = (const unsigned char *)extra[1];
        const size_t size = *(const size_t *)extra[3];
        char line[600] = "";
        for (size_t i = 0; i < size && i < 120; i += 4) {
            uint32_t w;
            memcpy(&w, buffer + i, 4);
            float f;
            memcpy(&f, &w, 4);
            char one[48];
            snprintf(one, sizeof one, "%s+%zu:%08x(%g)", i % 8 ? "," : " |", i, w, (double)f);
            strncat(line, one, sizeof line - strlen(line) - 1);
        }
        dlssnr_log("call %s %s grid %u x %u, %zu bytes of parameters:%s", name, launched, (unsigned)a[1], (unsigned)a[2], size, line);
    } else if (!strcmp(name, "hipModuleLaunchKernel"))
        dlssnr_log("call %s %s grid %u x %u x %u block %u x %u x %u", name, launched, (unsigned)a[1],
                   (unsigned)a[2], (unsigned)a[3], (unsigned)a[4], (unsigned)a[5], (unsigned)a[6]);
    else if (!strcmp(name, "hipModuleGetFunction"))
        dlssnr_log("call %s %#llx = \"%s\"", name, (unsigned long long)a[0], (const char *)(uintptr_t)a[2]);
    else if (!strcmp(name, "hipArray3DCreate")) {
        const uint64_t *d = (const uint64_t *)(uintptr_t)a[1];   // Width, Height, Depth, then format, channels, flags
        dlssnr_log("call %s %llu x %llu x %llu", name, (unsigned long long)d[0], (unsigned long long)d[1], (unsigned long long)d[2]);
    } else
        dlssnr_log("call %s", name);
}

void *dlssnr_real(int index) {
    if (g_loaded == 0) load_library();
    return g_loaded > 0 && index >= 0 && index < HIPF_COUNT ? g_real[index] : NULL;
}

// ---- hipImportExternalMemory

// ROCm's own layout (hipExternalMemoryHandleDesc), which is the one the caller
// uses as well: the same on both sides.
typedef struct {
    unsigned int type;
    union {
        int fd;
        struct {
            void *handle;
            const void *name;
        } win32;
        const void *nv_sci_buf;
    } handle;
    unsigned long long size;
    unsigned int flags;
    unsigned int reserved[16];
} external_memory_desc;

enum {
    kOpaqueFd = 1,
    kOpaqueWin32 = 2,
    kOpaqueWin32Kmt = 3,
    kD3D12Heap = 4,
    kD3D12Resource = 5,
    kD3D11Resource = 6,
    kD3D11ResourceKmt = 7,
};

int32_t thunk_special_hipImportExternalMemory(void *args) {
    struct args_hipImportExternalMemory *a = (struct args_hipImportExternalMemory *)args;
    void *fn = dlssnr_real(HIPF_hipImportExternalMemory);
    if (!fn) {
        a->ret = DLSSNR_HIP_NOT_SUPPORTED;
        return 0;
    }
    external_memory_desc desc = *(const external_memory_desc *)(uintptr_t)a->a1;
    dlssnr_log("hipImportExternalMemory: type %u, handle %p, size %llu", desc.type, desc.handle.win32.handle, desc.size);
#ifndef DLSSNR_WIN_TEST
    if (desc.type != kOpaqueFd) {
        wine_handle_to_fd_t convert = find_handle_to_fd();
        int fd = -1;
        // Read and write: the network writes some of the images it imports.
        int32_t status = convert ? convert(desc.handle.win32.handle, 0x3, &fd, NULL) : (int32_t)0xC0000002;
        if (status != 0) {
            dlssnr_log("no descriptor for the handle (status %#x)", (unsigned)status);
            a->ret = DLSSNR_HIP_NOT_SUPPORTED;
            return 0;
        }
        char link[64], target[256];
        snprintf(link, sizeof link, "/proc/self/fd/%d", fd);
        ssize_t n = readlink(link, target, sizeof target - 1);
        target[n > 0 ? n : 0] = 0;
        dlssnr_log("descriptor %d -> %s", fd, target);
        desc.type = kOpaqueFd;
        desc.handle.fd = fd;
    }
#endif
    a->ret = ((int32_t (*)(uint64_t, const void *))fn)(a->a0, &desc);
    dlssnr_log("hipImportExternalMemory returned %d", (int)a->ret);
    return 0;
}
