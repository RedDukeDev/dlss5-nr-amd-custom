// Launching our own kernel in place of a translated one, or not at all.
//
// Include this in nvcuda_proxy.cpp after logf() is defined. cuLaunchKernel
// calls native_try_launch() before forwarding: a true return means the work is
// already queued on the caller's stream and the original launch must be
// skipped, a false return means nothing happened and the translated kernel runs
// exactly as before.
//
// The safety of the whole approach rests on that false. Every way this can go
// wrong -- no manifest, an unreadable code object, a kernel whose geometry does
// not match, a launch the runtime refuses -- ends in a false and a log line,
// never in a wrong image. A failing entry is marked dead and never tried again,
// so a broken kernel costs one failed launch per run rather than one per frame.
//
// Why launching HIP from here is legitimate: under ZLUDA a CUstream *is* a
// hipStream_t (zluda_common/src/lib.rs), so handing the caller's stream to
// hipModuleLaunchKernel keeps our kernel in the same queue, in order, between
// the same neighbours. No second context, no synchronisation of our own.
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

// Include this AFTER the proxy's logf() is defined. It is a member of an
// anonymous namespace there, so declaring it here would not resolve the calls
// below -- it would make every call in the proxy ambiguous instead. Ordinary
// lookup finds it from inside namespace native once it is already in scope.

namespace native {

// The HIP handles are all opaque pointers and the error is an int, so the
// runtime can be bound by name without including any ROCm header.
typedef void *hipModule_t;
typedef void *hipFunction_t;
typedef void *hipStream_t;
typedef int hipError_t;

struct Runtime {
    hipError_t (*module_load_data)(hipModule_t *, const void *) = nullptr;
    hipError_t (*module_get_function)(hipFunction_t *, hipModule_t, const char *) = nullptr;
    hipError_t (*module_launch)(hipFunction_t, unsigned, unsigned, unsigned, unsigned, unsigned,
                                unsigned, unsigned, hipStream_t, void **, void **) = nullptr;
    bool tried = false;
    bool ready = false;
};

inline Runtime &runtime() {
    static Runtime r;
    if (r.tried) return r;
    r.tried = true;
    // Already resident: ZLUDA loaded it to do its own work. Asking for the
    // module rather than loading one avoids a second copy of the runtime.
    HMODULE hip = GetModuleHandleW(L"amdhip64_7.dll");
    if (!hip) hip = LoadLibraryW(L"amdhip64_7.dll");
    if (!hip) {
        logf("[native] amdhip64_7.dll not available: no native kernels\n");
        return r;
    }
    r.module_load_data = (decltype(r.module_load_data))GetProcAddress(hip, "hipModuleLoadData");
    r.module_get_function =
        (decltype(r.module_get_function))GetProcAddress(hip, "hipModuleGetFunction");
    r.module_launch = (decltype(r.module_launch))GetProcAddress(hip, "hipModuleLaunchKernel");
    r.ready = r.module_load_data && r.module_get_function && r.module_launch;
    if (!r.ready) logf("[native] amdhip64_7.dll does not export the three expected functions\n");
    return r;
}

struct Entry {
    char kernel[96] = {};   // the PTX kernel this stands in for
    char object[MAX_PATH] = {};  // the .hsaco built by build_kernel.py
    char entry[96] = {};    // the symbol inside it, if not the same name
    unsigned bx = 0, by = 1, bz = 1;  // the block shape it was written for
    unsigned args = 0;      // the packed argument buffer size it expects
    hipFunction_t fn = nullptr;
    bool dead = false;      // tried and failed: never again this run
    unsigned launches = 0;
};

// Raised from 32 when the manifest outgrew it: lines past the limit used to be
// dropped without a word, and their kernels stayed translated.
const unsigned kMaxEntries = 256;
inline Entry g_entries[kMaxEntries];
inline unsigned g_entry_count = 0;

// The manifest is one kernel per line:
//
//   <ptx kernel name>  <code object>  block=256,1,1  args=248  [entry=<symbol>]
//
// Blank lines and lines starting with # are ignored. A line the parser does not
// understand is skipped with a complaint rather than guessed at.
inline void load_manifest() {
    static bool done = false;
    if (done) return;
    done = true;

    char path[MAX_PATH] = {};
    if (!GetEnvironmentVariableA("NVCUDA_PROXY_NATIVE", path, sizeof path)) return;
    FILE *file = fopen(path, "r");
    if (!file) {
        logf("[native] manifest %s unreadable: staying on the translation\n", path);
        return;
    }
    // Relative code object paths are resolved against the manifest, so a folder
    // of kernels can be moved without editing every line in it.
    char base[MAX_PATH] = {};
    strncpy(base, path, sizeof base - 1);
    char *cut = strrchr(base, '\\');
    if (!cut) cut = strrchr(base, '/');
    if (cut) cut[1] = 0; else base[0] = 0;

    char line[512];
    while (fgets(line, sizeof line, file)) {
        if (g_entry_count == kMaxEntries) {
            logf("[native] manifest has more than %u kernels: the rest stay translated\n", kMaxEntries);
            break;
        }
        char *hash = strchr(line, '#');
        if (hash) *hash = 0;
        Entry e;
        char object[MAX_PATH] = {};
        char *token = strtok(line, " \t\r\n");
        if (!token) continue;
        strncpy(e.kernel, token, sizeof e.kernel - 1);
        token = strtok(nullptr, " \t\r\n");
        if (!token) {
            logf("[native] line without a code object for %s: ignored\n", e.kernel);
            continue;
        }
        strncpy(object, token, sizeof object - 1);
        bool sane = true;
        while ((token = strtok(nullptr, " \t\r\n")) != nullptr) {
            if (sscanf(token, "block=%u,%u,%u", &e.bx, &e.by, &e.bz) >= 1) continue;
            if (sscanf(token, "args=%u", &e.args) == 1) continue;
            if (strncmp(token, "entry=", 6) == 0) {
                strncpy(e.entry, token + 6, sizeof e.entry - 1);
                continue;
            }
            logf("[native] unrecognised field \"%s\" for %s: line ignored\n", token, e.kernel);
            sane = false;
        }
        if (!sane || !e.bx || !e.args) {
            logf("[native] %s without block= or args=: not hooking blind\n", e.kernel);
            continue;
        }
        if (object[0] == '\\' || object[0] == '/' || (object[0] && object[1] == ':'))
            strncpy(e.object, object, sizeof e.object - 1);
        else
            snprintf(e.object, sizeof e.object, "%s%s", base, object);
        if (!e.entry[0]) strncpy(e.entry, e.kernel, sizeof e.entry - 1);
        g_entries[g_entry_count++] = e;
        logf("[native] %s <- %s (block %ux%ux%u, %u bytes of arguments)\n", e.kernel, e.object,
             e.bx, e.by, e.bz, e.args);
    }
    fclose(file);
}

// The code object is loaded the first time its kernel is launched, not at
// startup: a manifest listing kernels this run never reaches costs nothing.
inline bool bind(Entry &e) {
    if (e.fn) return true;
    if (e.dead) return false;
    Runtime &r = runtime();
    if (!r.ready) {
        e.dead = true;
        return false;
    }
    FILE *file = fopen(e.object, "rb");
    if (!file) {
        logf("[native] %s: code object %s missing, staying on the translation\n", e.kernel, e.object);
        e.dead = true;
        return false;
    }
    fseek(file, 0, SEEK_END);
    const long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    void *image = size > 0 ? malloc((size_t)size) : nullptr;
    const bool whole = image && fread(image, 1, (size_t)size, file) == (size_t)size;
    fclose(file);
    hipModule_t module = nullptr;
    const hipError_t loaded = whole ? r.module_load_data(&module, image) : 1;
    free(image);
    if (loaded != 0) {
        logf("[native] %s: hipModuleLoadData -> %d, staying on the translation\n", e.kernel, loaded);
        e.dead = true;
        return false;
    }
    const hipError_t got = r.module_get_function(&e.fn, module, e.entry);
    if (got != 0 || !e.fn) {
        logf("[native] %s: symbol \"%s\" not found (%d), staying on the translation\n", e.kernel,
             e.entry, got);
        e.fn = nullptr;
        e.dead = true;
        return false;
    }
    logf("[native] %s hooked by %s\n", e.kernel, e.object);
    return true;
}

// The gate. Everything it checks is something a native kernel would be wrong
// about if it differed: the name says which layer, the block shape is compiled
// into the kernel, and the argument buffer size is the ABI itself. The grid is
// deliberately not checked -- it follows the render resolution and is passed
// through untouched.
inline bool try_launch(const char *name, unsigned gx, unsigned gy, unsigned gz, unsigned bx,
                       unsigned by, unsigned bz, unsigned shared, void *stream, void **extra) {
    load_manifest();
    if (!g_entry_count || !name || !extra) return false;

    for (unsigned i = 0; i < g_entry_count; ++i) {
        Entry &e = g_entries[i];
        if (e.dead || strcmp(e.kernel, name) != 0) continue;

        // The packed buffer, found the way CUDA describes it: marker 1 is the
        // pointer, marker 2 the size.
        void *buffer = nullptr;
        size_t size = 0;
        for (int k = 0; extra[k] && k < 16; k += 2) {
            if ((size_t)extra[k] == 1) buffer = extra[k + 1];
            if ((size_t)extra[k] == 2) size = *(const size_t *)extra[k + 1];
        }
        if (!buffer || size != e.args) {
            logf("[native] %s: %zu bytes of arguments, the kernel wants %u -> translation\n", name,
                 size, e.args);
            e.dead = true;
            return false;
        }
        if (bx != e.bx || by != e.by || bz != e.bz) {
            logf("[native] %s: block %ux%ux%u against %ux%ux%u expected -> translation\n", name, bx,
                 by, bz, e.bx, e.by, e.bz);
            e.dead = true;
            return false;
        }
        if (!bind(e)) return false;

        // CUDA and HIP agree that marker 1 is the buffer and 2 its size, but
        // not on how the list ends: CUDA uses a null pointer, HIP the value 3.
        // Forwarding the caller's list unchanged would leave HIP reading past
        // the end of it (zluda/src/impl/function.rs says the same).
        void *rebuilt[5] = {(void *)1, buffer, (void *)2, &size, (void *)3};
        const hipError_t launched = runtime().module_launch(
            e.fn, gx, gy, gz, bx, by, bz, shared, (hipStream_t)stream, nullptr, rebuilt);
        if (launched != 0) {
            logf("[native] %s: hipModuleLaunchKernel -> %d, translation from here on\n", name,
                 launched);
            e.dead = true;
            return false;
        }
        ++e.launches;
        return true;
    }
    return false;
}

// Printed at the end of a run: how much of the network actually went native.
// A kernel that fell back says so on the line it fell back on; this is the
// summary that says whether the experiment measured what it meant to.
inline void report() {
    for (unsigned i = 0; i < g_entry_count; ++i) {
        const Entry &e = g_entries[i];
        logf("[native] %-56s %s, %u launches\n", e.kernel, e.dead ? "fell back to the translation" : "native",
             e.launches);
    }
}

}  // namespace native
