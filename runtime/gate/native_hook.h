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
    // Optional: read a u32 global out of a code object (see Entry::prep).
    hipError_t (*module_get_global)(void **, size_t *, hipModule_t, const char *) = nullptr;
    hipError_t (*memcpy_dtoh)(void *, void *, size_t) = nullptr;
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
    r.module_get_global = (decltype(r.module_get_global))GetProcAddress(hip, "hipModuleGetGlobal");
    r.memcpy_dtoh = (decltype(r.memcpy_dtoh))GetProcAddress(hip, "hipMemcpyDtoH");
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
    unsigned waves = 0;     // waves per block to launch with, when not the PTX's (the grid is the same)
    hipFunction_t fn = nullptr;
    bool dead = false;      // tried and failed: never again this run
    unsigned launches = 0;

    // Kernels that carry their own weight preparation (the super resolution layers) say so through
    // u32 globals in the code object. All optional; a code object with none of them is launched
    // exactly as it was before these existed.
    //   <entry>_prep        kernel launched first, on the same stream and with the same arguments
    //   d4r_prep_blocks     its grid size (128 threads per block)
    //   d4r_prep_key_at     byte offset + 1 of the u64 argument (the weights pointer) that decides
    //                       what the prep produces: it is skipped while that value repeats
    //   d4r_prep_key_slots  N > 0: the kernel keeps one prepared image per key for N keys, so the
    //                       prep runs once per key; keys beyond N are prepared on every launch
    //   d4r_block_z         block z the native kernel is built for (more waves per window)
    //   d4r_grid_x          persistent kernels: this many workgroups in x
    hipFunction_t prep = nullptr;
    unsigned prep_blocks = 0, block_z = 0, grid_x = 0, prep_slots = 0;
    long prep_key_offset = -1;
    unsigned long long prep_last = 0;
    unsigned long long prep_seen[16] = {};
    unsigned prep_seen_count = 0;
};

// Raised from 32 when the manifest outgrew it: lines past the limit used to be
// dropped without a word, and their kernels stayed translated.
const unsigned kMaxEntries = 256;
inline Entry g_entries[kMaxEntries];
inline volatile unsigned g_entry_count = 0;

// The manifest is one kernel per line:
//
//   <ptx kernel name>  <code object>  block=256,1,1  args=248  [entry=<symbol>]
//   waves=N launches the native kernel with N waves per block (block z), where the PTX uses fewer: the
//   kernel is written for any count
//
// Blank lines and lines starting with # are ignored. A line the parser does not
// understand is skipped with a complaint rather than guessed at.
inline void load_one_manifest(const char *path) {
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
            if (sscanf(token, "waves=%u", &e.waves) == 1) continue;
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
        // Written whole before it is counted: a launch on another thread reads the count and then the
        // entries, while a later network's manifest may still be coming in.
        g_entries[g_entry_count] = e;
        MemoryBarrier();
        ++g_entry_count;
        logf("[native] %s <- %s (block %ux%ux%u, %u bytes of arguments)\n", e.kernel, e.object,
             e.bx, e.by, e.bz, e.args);
    }
    fclose(file);
}

// NVCUDA_PROXY_NATIVE names one manifest, or several joined by ';' (the neural rendering kernels
// and the super resolution ones live in separate files).
inline void load_manifest() {
    static bool done = false;
    if (done) return;
    done = true;

    char paths[4 * MAX_PATH] = {};
    if (!GetEnvironmentVariableA("NVCUDA_PROXY_NATIVE", paths, sizeof paths)) return;
    // Split by hand: load_one_manifest uses strtok itself, which keeps one global position.
    for (char *start = paths; start && *start;) {
        char *end = strchr(start, ';');
        if (end) *end = 0;
        if (*start) load_one_manifest(start);
        start = end ? end + 1 : nullptr;
    }
}


// The optional description of a kernel's own preparation step (see Entry::prep).
inline bool read_u32_global(hipModule_t module, const char *symbol, unsigned &value) {
    Runtime &r = runtime();
    if (!r.module_get_global || !r.memcpy_dtoh) return false;
    void *where = nullptr;
    size_t size = 0;
    if (r.module_get_global(&where, &size, module, symbol) != 0 || size != 4) return false;
    return r.memcpy_dtoh(&value, where, 4) == 0;
}

inline void read_prep_description(Entry &e, hipModule_t module) {
    Runtime &r = runtime();
    char name[128];
    snprintf(name, sizeof name, "%s_prep", e.entry);
    hipFunction_t prep = nullptr;
    if (r.module_get_function(&prep, module, name) == 0 && prep) {
        unsigned blocks = 0;
        if (!read_u32_global(module, "d4r_prep_blocks", blocks) || !blocks) {
            logf("[native] %s has a prep kernel but no d4r_prep_blocks: prep ignored\n", e.kernel);
        } else {
            e.prep = prep;
            e.prep_blocks = blocks;
        }
    }
    unsigned at = 0;
    if (read_u32_global(module, "d4r_prep_key_at", at) && at > 0) {
        e.prep_key_offset = (long)at - 1;
    } else if (read_u32_global(module, "d4r_prep_key_offset", at) && at > 0) {
        e.prep_key_offset = (long)at;   // the older form: 0 means "no key"
    }
    read_u32_global(module, "d4r_prep_key_slots", e.prep_slots);
    read_u32_global(module, "d4r_block_z", e.block_z);
    read_u32_global(module, "d4r_grid_x", e.grid_x);
    if (e.prep_slots > 16) e.prep_slots = 16;
    if (e.prep || e.block_z || e.grid_x)
        logf("[native] %s: prep %u blocks, key at %ld, %u slots, block z %u, grid x %u\n", e.kernel,
             e.prep_blocks, e.prep_key_offset, e.prep_slots, e.block_z, e.grid_x);
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
    read_prep_description(e, module);
    logf("[native] %s hooked by %s\n", e.kernel, e.object);
    return true;
}

// The gate. Everything it checks is something a native kernel would be wrong
// about if it differed: the name says which layer, the block shape is compiled
// into the kernel, and the argument buffer size is the ABI itself. The grid is
// deliberately not checked -- it follows the render resolution and is passed
// through untouched.
// ZLUDA_PER_THREAD_STREAM=1 turns the null stream into the calling thread's own default stream (see
// zluda_common): a native kernel launched here has to go where the translated ones go, or the two
// kinds of kernel of one network would no longer follow each other in order.
inline hipStream_t effective_stream(hipStream_t stream) {
    static const bool per_thread = [] {
        char value[8] = {};
        return GetEnvironmentVariableA("ZLUDA_PER_THREAD_STREAM", value, sizeof value) > 0 && value[0] == '1';
    }();
    if (stream || !per_thread) return stream;
    // The stream ZLUDA itself would give this thread: a plain per-thread one, or a high priority one
    // for a thread that asked for it (zluda_set_thread_high_priority).
    typedef void *(*thread_stream_fn)();
    static const thread_stream_fn thread_stream = [] {
        HMODULE zluda = GetModuleHandleW(L"zluda_real.dll");
        return zluda ? (thread_stream_fn)GetProcAddress(zluda, "zluda_thread_default_stream") : nullptr;
    }();
    return thread_stream ? (hipStream_t)thread_stream() : (hipStream_t)(uintptr_t)2;
}

inline bool try_launch(const char *name, unsigned gx, unsigned gy, unsigned gz, unsigned bx,
                       unsigned by, unsigned bz, unsigned shared, void *stream, void **extra) {
    load_manifest();
    // DLSS5NR_GATE_DUMP=<kernel name>: the argument buffer of its first three launches, in hex, to
    // the log. What a new version of the library passes, when a kernel's parameter size
    // changes, is what the native kernel's layout has to be made from.
    {
        static const char *dump = getenv("DLSS5NR_GATE_DUMP");
        static int dumped = 0;
        if (dump && name && extra && dumped < 3 && strcmp(dump, name) == 0) {
            const unsigned char *buffer = nullptr;
            size_t size = 0;
            for (int k = 0; extra[k] && k < 16; k += 2) {
                if ((size_t)extra[k] == 1) buffer = (const unsigned char *)extra[k + 1];
                if ((size_t)extra[k] == 2) size = *(const size_t *)extra[k + 1];
            }
            if (buffer) {
                ++dumped;
                logf("[native] %s launch %d: grid %u,%u,%u block %u,%u,%u, %zu bytes of arguments\n", name, dumped, gx, gy,
                     gz, bx, by, bz, size);
                for (size_t at = 0; at < size; at += 16) {
                    char line[160];
                    int n = snprintf(line, sizeof line, "[native]   +0x%03zx", at);
                    for (size_t i = at; i < at + 16 && i < size; ++i) n += snprintf(line + n, sizeof line - n, " %02x", buffer[i]);
                    logf("%s\n", line);
                }
                // DLSS5NR_GATE_DUMP_WEIGHTS=<offset of the pointer in the arguments>: and the
                // 64 KB it points at, to <kernel name>_weights<n>.bin in the working directory.
                if (const char *where = getenv("DLSS5NR_GATE_DUMP_WEIGHTS")) {
                    const size_t at = (size_t)atoi(where);
                    typedef int (*copy_fn)(void *, const void *, size_t);
                    typedef int (*sync_fn)();
                    HMODULE hip = GetModuleHandleW(L"amdhip64_7.dll");
                    copy_fn copy = hip ? (copy_fn)GetProcAddress(hip, "hipMemcpyDtoH") : nullptr;
                    sync_fn sync = hip ? (sync_fn)GetProcAddress(hip, "hipDeviceSynchronize") : nullptr;
                    if (copy && at + 8 <= size) {
                        if (sync) sync();
                        const void *device = *(const void *const *)(buffer + at);
                        static unsigned char blob[1 << 16];
                        const int status = copy(blob, device, sizeof blob);
                        char file[256];
                        snprintf(file, sizeof file, "%s_weights%d.bin", name, dumped);
                        if (FILE *out = fopen(file, "wb")) {
                            fwrite(blob, 1, sizeof blob, out);
                            fclose(out);
                        }
                        logf("[native] weights at %p -> %s (%d)\n", device, file, status);
                    }
                }
            }
        }
    }
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
            continue;   // another line of the manifest may be for this version's size
        }
        if (bx != e.bx || by != e.by || bz != e.bz) {
            logf("[native] %s: block %ux%ux%u against %ux%ux%u expected -> translation\n", name, bx,
                 by, bz, e.bx, e.by, e.bz);
            e.dead = true;
            continue;
        }
        if (!bind(e)) return false;

        // CUDA and HIP agree that marker 1 is the buffer and 2 its size, but
        // not on how the list ends: CUDA uses a null pointer, HIP the value 3.
        // Forwarding the caller's list unchanged would leave HIP reading past
        // the end of it (zluda/src/impl/function.rs says the same).
        void *rebuilt[5] = {(void *)1, buffer, (void *)2, &size, (void *)3};
        if (e.prep) {
            unsigned long long key = 0;
            if (e.prep_key_offset >= 0 && (size_t)e.prep_key_offset + 8 <= size)
                memcpy(&key, (const char *)buffer + e.prep_key_offset, 8);
            bool run = true;
            if (key) {
                if (e.prep_slots == 0) {
                    run = e.prep_last != key;
                } else {
                    run = true;
                    for (unsigned k = 0; k < e.prep_seen_count; ++k)
                        if (e.prep_seen[k] == key) run = false;
                    if (run && e.prep_seen_count < e.prep_slots) e.prep_seen[e.prep_seen_count++] = key;
                }
            }
            if (run) {
                const hipError_t prepared = runtime().module_launch(
                    e.prep, e.prep_blocks, 1, 1, 128, 1, 1, 0, effective_stream((hipStream_t)stream), nullptr, rebuilt);
                if (prepared != 0) {
                    logf("[native] %s: prep launch -> %d, translation from here on\n", name, prepared);
                    e.dead = true;
                    return false;
                }
                e.prep_last = key;
                static const bool trace = GetEnvironmentVariableA("DLSS5NR_GATE_TRACE", nullptr, 0) > 0;
                if (trace) {
                    typedef int (*sync_fn)();
                    static HMODULE hip = GetModuleHandleW(L"amdhip64_7.dll");
                    static sync_fn sync = hip ? (sync_fn)GetProcAddress(hip, "hipDeviceSynchronize") : nullptr;
                    logf("[native] %s prep (%u blocks) launched, waiting\n", name, e.prep_blocks);
                    const int status = sync ? sync() : 0;
                    logf("[native] %s prep done -> %d\n", name, status);
                }
            }
        }
        if (e.waves) bz = e.waves;
        if (e.block_z) bz = e.block_z;
        if (e.grid_x) { gx = e.grid_x; gy = 1; gz = 1; }
        const hipError_t launched = runtime().module_launch(
            e.fn, gx, gy, gz, bx, by, bz, shared, effective_stream((hipStream_t)stream), nullptr, rebuilt);
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
