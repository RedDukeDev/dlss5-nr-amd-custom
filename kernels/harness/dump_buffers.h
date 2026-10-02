// The real inputs and the real output of one launch, written to disk.
//
// Include this in nvcuda_proxy.cpp after logf() is defined, next to
// native_hook.h. With NVCUDA_PROXY_DUMP unset it does nothing at all.
//
// Why it exists: a native kernel is correct or not against the data the
// original was handed, and comparing whole frames answers that question far too
// late and far too coarsely. One launch's buffers, captured before and after,
// turn "the image looks right" into a number per element, computed offline
// without running the network again. Reading device memory back is the same
// operation the proxy already performs for its digests -- cuCtxSynchronize then
// cuMemcpyDtoH_v2, taken from the real driver by name.
//
//   NVCUDA_PROXY_DUMP        the kernel to capture, by name
//   NVCUDA_PROXY_DUMP_DIR    where the files go (default: the working directory)
//   NVCUDA_PROXY_DUMP_LAUNCH which launch of it, counting from 1 (default 1)
//   NVCUDA_PROXY_DUMP_BYTES  how much of each buffer (default 262144)
//
// Files are named <kernel>_l<N>_<pre|post>_+<offset>.bin, plus one
// <kernel>_l<N>_args.bin holding the packed parameter itself, so the offsets in
// the dump line up with the offsets in the struct.
#pragma once

#include <stdio.h>
#include <string.h>
#include <windows.h>

namespace dump {

inline bool reader(CUresult (**sync)(void), CUresult (**down)(void *, CUdeviceptr, size_t)) {
    static CUresult (*s)(void) = nullptr;
    static CUresult (*d)(void *, CUdeviceptr, size_t) = nullptr;
    if (!s) {
        HMODULE real = GetModuleHandleW(L"zluda_real.dll");
        if (!real) real = LoadLibraryW(L"zluda_real.dll");
        s = (decltype(s))GetProcAddress(real, "cuCtxSynchronize");
        d = (decltype(d))GetProcAddress(real, "cuMemcpyDtoH_v2");
    }
    *sync = s;
    *down = d;
    return s && d;
}

inline const char *wanted() {
    static char name[96];
    static int asked = -1;
    if (asked < 0) asked = GetEnvironmentVariableA("NVCUDA_PROXY_DUMP", name, sizeof name) > 0;
    return asked ? name : nullptr;
}

inline unsigned which_launch() {
    static int n = -1;
    if (n < 0) {
        char buffer[16] = {};
        n = GetEnvironmentVariableA("NVCUDA_PROXY_DUMP_LAUNCH", buffer, sizeof buffer) > 0
                ? atoi(buffer)
                : 1;
        if (n < 1) n = 1;
    }
    return (unsigned)n;
}

inline size_t how_much() {
    static long long n = -1;
    if (n < 0) {
        char buffer[24] = {};
        n = GetEnvironmentVariableA("NVCUDA_PROXY_DUMP_BYTES", buffer, sizeof buffer) > 0
                ? atoll(buffer)
                : 256 * 1024;
        if (n < 16) n = 16;
    }
    return (size_t)n;
}

inline void path_for(char *out, size_t cap, const char *kernel, unsigned launch, const char *phase,
                     int offset) {
    char dir[MAX_PATH] = {};
    GetEnvironmentVariableA("NVCUDA_PROXY_DUMP_DIR", dir, sizeof dir);
    const char *sep = (dir[0] && dir[strlen(dir) - 1] != '\\') ? "\\" : "";
    if (offset < 0)
        snprintf(out, cap, "%s%s%s_l%u_args.bin", dir, sep, kernel, launch);
    else
        snprintf(out, cap, "%s%s%s_l%u_%s_+%02X.bin", dir, sep, kernel, launch, phase, offset);
}

inline void write_file(const char *path, const void *data, size_t bytes) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        logf("[dump] %s not writable\n", path);
        return;
    }
    fwrite(data, 1, bytes, f);
    fclose(f);
}

// Reads what it can, shrinking the window until the driver accepts it: a buffer
// smaller than the request would otherwise be dumped as nothing at all.
inline size_t read_device(unsigned long long base, unsigned char *into, size_t want) {
    CUresult (*sync)(void);
    CUresult (*down)(void *, CUdeviceptr, size_t);
    if (!reader(&sync, &down) || sync() != 0) return 0;
    for (size_t n = want; n >= 16; n /= 2)
        if (down(into, (CUdeviceptr)base, n) == 0) return n;
    return 0;
}

// Which launch of the kernel being dumped this is, counting from 1. Returns 0
// for every other kernel, so the caller skips the work without knowing anything
// about which kernel was asked for.
inline unsigned note(const char *kernel) {
    const char *want = wanted();
    if (!want || !kernel || strcmp(kernel, want) != 0) return 0;
    static unsigned seen = 0;
    return ++seen;
}

// Called on both sides of the launch. `phase` is "pre" or "post": the same
// pointers, before the kernel runs and after it has, which is what separates
// this kernel's output from what was already there.
inline void capture(const char *kernel, unsigned launch, const char *phase, const void *args,
                    size_t args_size) {
    const char *want = wanted();
    if (!want || !kernel || strcmp(kernel, want) != 0 || launch != which_launch() || !args) return;

    char path[MAX_PATH];
    if (strcmp(phase, "pre") == 0) {
        path_for(path, sizeof path, kernel, launch, phase, -1);
        write_file(path, args, args_size);
        logf("[dump] %s launch %u: parameter of %zu bytes\n", kernel, launch, args_size);
    }

    const size_t window = how_much();
    static unsigned char *buffer = nullptr;
    static size_t buffer_size = 0;
    if (window > buffer_size) {
        free(buffer);
        buffer = (unsigned char *)malloc(window);
        buffer_size = buffer ? window : 0;
    }
    if (!buffer) return;

    const unsigned long long *word = (const unsigned long long *)args;
    for (size_t i = 0; i * 8 < args_size; ++i) {
        const unsigned long long v = word[i];
        // The same test the argument log uses: in the range the driver hands
        // out, and aligned. Anything else is a dimension or a float, and
        // reading it would only fault.
        const bool plausible = v >= 0x100000000ull && v < 0x100000000000ull && (v & 0xFFull) == 0;
        if (!plausible) continue;
        const size_t got = read_device(v, buffer, window);
        if (!got) continue;
        path_for(path, sizeof path, kernel, launch, phase, (int)(i * 8));
        write_file(path, buffer, got);
        logf("[dump] %s %s +0x%02zX: %zu bytes from 0x%llX\n", kernel, phase, i * 8, got, v);
    }
}

}  // namespace dump
