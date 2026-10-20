// The gate: a thin nvcuda.dll in front of ZLUDA that launches the native
// kernels (kernels/) in place of the translated ones they reproduce.
//
// Everything is forwarded to ZLUDA, renamed zluda_real.dll, except the four
// calls the gate needs:
//
//   cuGetProcAddress(_v2)  the network resolves its entry points through the
//                          driver rather than through the export table, so the
//                          lookup is where the other two get put in its path
//   cuModuleGetFunction    remembers which name each function handle has
//   cuLaunchKernel         asks native_hook.h whether a native kernel takes
//                          the launch; if not, ZLUDA runs the translation
//
// The native side is strictly opt-in and strictly fallible: without a manifest
// (NVCUDA_PROXY_NATIVE) the gate changes nothing, and any kernel whose
// geometry, argument size or code object does not match falls back to the
// translation for good. See native_hook.h.
//
// zluda_real.dll must already be in the process when this DLL is loaded: the
// forwarders name it by base name, and the loader only finds it by that name
// if it sits beside the game or was loaded before. The runtime loads it by
// full path first for exactly that reason.
//
// Log: DLSS5NR_GATE_LOG names a file; without it the gate is silent.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cstdarg>
#include <cstdio>
#include <cstring>

#include "gate_forwards.h"

namespace {

using CUresult = int;
using CUfunction = void *;
using CUmodule = void *;
using CUstream = void *;

CRITICAL_SECTION g_lock;
FILE *g_log = nullptr;
bool g_log_tried = false;

void logf(const char *fmt, ...) {
    EnterCriticalSection(&g_lock);
    if (!g_log_tried) {
        g_log_tried = true;
        wchar_t path[MAX_PATH];
        if (GetEnvironmentVariableW(L"DLSS5NR_GATE_LOG", path, MAX_PATH)) g_log = _wfopen(path, L"w");
    }
    if (g_log) {
        va_list ap;
        va_start(ap, fmt);
        vfprintf(g_log, fmt, ap);
        va_end(ap);
        fflush(g_log);
    }
    LeaveCriticalSection(&g_lock);
}

} // namespace

// After logf, which it calls: see the note at the top of native_hook.h.
#include "native_hook.h"

namespace {

HMODULE zluda() {
    static HMODULE module = nullptr;
    if (!module) module = GetModuleHandleW(L"zluda_real.dll");
    if (!module) module = LoadLibraryW(L"zluda_real.dll");
    return module;
}

void *real(const char *name) { return zluda() ? (void *)GetProcAddress(zluda(), name) : nullptr; }

// The pointers the driver handed back for the names the gate wraps, so each
// wrapper calls exactly the implementation ZLUDA chose for that name and
// version. Filled by cuGetProcAddress, or from the export table when the
// caller went through it instead.
void *g_get_function = nullptr;
void *g_launch = nullptr;

// Function handle -> kernel name. Grows with the modules the network loads,
// a few hundred entries; read on every launch, so a flat array under a lock
// is plenty.
struct Named {
    CUfunction handle;
    char name[112];
};
Named g_names[1024];
unsigned g_name_count = 0;

const char *name_of(CUfunction f) {
    for (unsigned i = 0; i < g_name_count; ++i)
        if (g_names[i].handle == f) return g_names[i].name;
    return "";
}

} // namespace

// DLSS5NR_GATE_PROFILE=1: the GPU time of every kernel, native or translated, from a pair
// of events around each launch, waited for at once (which serialises the network, so
// the times are the kernels' own, not the frame's), summed per kernel and logged every
// 2000 launches. What to make native next is the top of that list.
namespace profile {

typedef int (*event_create_fn)(void **);
typedef int (*event_record_fn)(void *, void *);
typedef int (*event_sync_fn)(void *);
typedef int (*event_elapsed_fn)(float *, void *, void *);

struct Entry {
    char name[112];
    double ms;
    unsigned launches;
    bool native;
};
Entry g_entries[256];
unsigned g_count = 0, g_launches = 0;
void *g_start = nullptr, *g_stop = nullptr;
event_record_fn g_record = nullptr;
event_sync_fn g_sync = nullptr;
event_elapsed_fn g_elapsed = nullptr;

inline bool on() {
    static int state = -1;
    if (state < 0) {
        state = 0;
        char v[8];
        if (GetEnvironmentVariableA("DLSS5NR_GATE_PROFILE", v, sizeof v) > 0) {
            HMODULE hip = GetModuleHandleW(L"amdhip64_7.dll");
            auto create = hip ? (event_create_fn)GetProcAddress(hip, "hipEventCreate") : nullptr;
            g_record = hip ? (event_record_fn)GetProcAddress(hip, "hipEventRecord") : nullptr;
            g_sync = hip ? (event_sync_fn)GetProcAddress(hip, "hipEventSynchronize") : nullptr;
            g_elapsed = hip ? (event_elapsed_fn)GetProcAddress(hip, "hipEventElapsedTime") : nullptr;
            if (create && g_record && g_sync && g_elapsed && create(&g_start) == 0 && create(&g_stop) == 0) state = 1;
            else logf("[profile] the HIP event functions are not there: no profile\n");
        }
    }
    return state == 1;
}

inline void begin(CUstream stream) { g_record(g_start, stream); }

inline void end(CUstream stream, const char *name, bool native_ran) {
    g_record(g_stop, stream);
    g_sync(g_stop);
    float ms = 0.0f;
    g_elapsed(&ms, g_start, g_stop);
    unsigned i = 0;
    while (i < g_count && strcmp(g_entries[i].name, name) != 0) ++i;
    if (i == g_count) {
        if (g_count == sizeof g_entries / sizeof g_entries[0]) return;
        strncpy(g_entries[i].name, name, sizeof g_entries[i].name - 1);
        ++g_count;
    }
    g_entries[i].ms += ms;
    g_entries[i].launches += 1;
    g_entries[i].native = native_ran;
    if (++g_launches % 2000 != 0) return;
    double total = 0.0;
    for (unsigned k = 0; k < g_count; ++k) total += g_entries[k].ms;
    logf("[profile] %u launches, %.1f ms of kernels\n", g_launches, total);
    bool shown[256] = {};
    for (unsigned rank = 0; rank < g_count; ++rank) {
        unsigned best = g_count;
        for (unsigned k = 0; k < g_count; ++k)
            if (!shown[k] && (best == g_count || g_entries[k].ms > g_entries[best].ms)) best = k;
        shown[best] = true;
        const Entry &e = g_entries[best];
        logf("[profile] %6.1f%% %9.2f ms %6u  %-10s %s\n", 100.0 * e.ms / total, e.ms, e.launches,
             e.native ? "native" : "translated", e.name);
    }
}

} // namespace profile

// DLSS5NR_GATE_TRACE also times every launch (the launch and the wait for it) and prints the totals per
// kernel, largest first, when the process unloads the gate. A tracing run serialises the work, so the
// times are each kernel's own, not the frame's.
namespace trace_times {
struct Row {
    char name[112];
    double ms;
    unsigned launches;
};
Row g_rows[256];
unsigned g_count = 0;

inline double now_ms() {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return 1000.0 * (double)c.QuadPart / (double)f.QuadPart;
}

inline void report();

inline void add(const char *name, double ms) {
    static unsigned adds = 0;
    EnterCriticalSection(&g_lock);
    unsigned i = 0;
    while (i < g_count && strcmp(g_rows[i].name, name) != 0) ++i;
    if (i == g_count && g_count < sizeof g_rows / sizeof g_rows[0]) {
        strncpy(g_rows[i].name, name, sizeof g_rows[i].name - 1);
        ++g_count;
    }
    if (i < g_count) {
        g_rows[i].ms += ms;
        g_rows[i].launches += 1;
    }
    const bool print = ++adds % 100 == 0;
    LeaveCriticalSection(&g_lock);
    // A tracing run ends by process termination, which no destructor sees: the table is rewritten
    // as the run goes, and the last copy in the log is the result.
    if (print) report();
}

inline void report() {
    if (!g_count) return;
    double total = 0.0;
    for (unsigned i = 0; i < g_count; ++i) total += g_rows[i].ms;
    logf("[times] %.1f ms in %u kernels\n", total, g_count);
    bool shown[256] = {};
    for (unsigned rank = 0; rank < g_count; ++rank) {
        unsigned best = g_count;
        for (unsigned k = 0; k < g_count; ++k)
            if (!shown[k] && (best == g_count || g_rows[k].ms > g_rows[best].ms)) best = k;
        shown[best] = true;
        logf("[times] %5.1f%% %10.2f ms %6u launches  %s\n", 100.0 * g_rows[best].ms / total, g_rows[best].ms,
             g_rows[best].launches, g_rows[best].name);
    }
}
}  // namespace trace_times

extern "C" {

// The runtime hands the gate the native kernels of each network as that network loads, after the first
// launch may have happened: the manifests of the networks loaded together are all the gate knows.
// Returns how many kernels it now has.
__declspec(dllexport) int dlss5nr_gate_add_manifest(const char *path) {
    static CRITICAL_SECTION lock;
    static bool made = false;
    if (!made) {
        InitializeCriticalSection(&lock);
        made = true;
    }
    EnterCriticalSection(&lock);
    native::load_manifest();   // the one the environment names, once
    if (path && *path) native::load_one_manifest(path);
    const int count = (int)native::g_entry_count;
    LeaveCriticalSection(&lock);
    return count;
}

__declspec(dllexport) CUresult cuModuleGetFunction(CUfunction *out, CUmodule module, const char *name) {
    if (!g_get_function) g_get_function = real("cuModuleGetFunction");
    auto fn = (CUresult (*)(CUfunction *, CUmodule, const char *))g_get_function;
    const CUresult r = fn(out, module, name);
    // A kernel the translation does not have is otherwise reported only as the caller's own generic error.
    if (r != 0) logf("[module] cuModuleGetFunction(%s) -> %d\n", name ? name : "?", r);
    if (r == 0 && out && name) {
        EnterCriticalSection(&g_lock);
        if (g_name_count < sizeof g_names / sizeof g_names[0]) {
            Named &n = g_names[g_name_count++];
            n.handle = *out;
            strncpy(n.name, name, sizeof n.name - 1);
            n.name[sizeof n.name - 1] = 0;
        }
        LeaveCriticalSection(&g_lock);
    }
    return r;
}

__declspec(dllexport) CUresult cuLaunchKernel(CUfunction f, unsigned gx, unsigned gy, unsigned gz,
                                              unsigned bx, unsigned by, unsigned bz, unsigned shared,
                                              CUstream stream, void **params, void **extra) {
    if (!g_launch) g_launch = real("cuLaunchKernel");
    auto fn = (CUresult (*)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                            unsigned, CUstream, void **, void **))g_launch;
    // DLSS5NR_GATE_SKIP=<kernel name>: do not launch that kernel (isolates which one faults)
    {
        static char skip[128];
        static const bool have = GetEnvironmentVariableA("DLSS5NR_GATE_SKIP", skip, sizeof skip) > 0;
        if (have && strcmp(skip, name_of(f)) == 0) return 0;
    }
    if (!profile::on()) {
        const double started = trace_times::now_ms();
        if (native::try_launch(name_of(f), gx, gy, gz, bx, by, bz, shared, stream, extra)) {
            static const bool trace_native = GetEnvironmentVariableA("DLSS5NR_GATE_TRACE", nullptr, 0) > 0;
            if (trace_native) {
                static HMODULE hip = GetModuleHandleW(L"amdhip64_7.dll");
                static void *sync = hip ? (void *)GetProcAddress(hip, "hipDeviceSynchronize") : nullptr;
                const int status = sync ? ((int (*)())sync)() : 0;
                trace_times::add(name_of(f), trace_times::now_ms() - started);
                logf("[trace] native %s grid %u,%u,%u block %u,%u,%u -> %d\n", name_of(f), gx, gy, gz, bx, by, bz, status);
            }
            return 0;
        }
        CUresult r = fn(f, gx, gy, gz, bx, by, bz, shared, stream, params, extra);
        // DLSS5NR_GATE_TRACE=1: wait for every launch, so a fault is blamed on the kernel that caused it
        // rather than on whichever one is launched next (a launch failure is reported late).
        static const bool trace = GetEnvironmentVariableA("DLSS5NR_GATE_TRACE", nullptr, 0) > 0;
        if (trace && r == 0) {
            static HMODULE hip = GetModuleHandleW(L"amdhip64_7.dll");
            static void *sync = hip ? (void *)GetProcAddress(hip, "hipDeviceSynchronize") : nullptr;
            if (sync) r = ((CUresult(*)())sync)() ? 719 : 0;
            trace_times::add(name_of(f), trace_times::now_ms() - started);
            logf("[trace] %s grid %u,%u,%u block %u,%u,%u -> %d\n", name_of(f), gx, gy, gz, bx, by, bz, r);
        }
        // A refused launch is otherwise silent: the caller only sees its own generic error.
        if (r != 0)
            logf("[launch] %s grid %u,%u,%u block %u,%u,%u shared %u -> error %d\n", name_of(f), gx, gy, gz, bx, by, bz,
                 shared, r);
        return r;
    }
    profile::begin(stream);
    const bool native_ran = native::try_launch(name_of(f), gx, gy, gz, bx, by, bz, shared, stream, extra);
    const CUresult r = native_ran ? 0 : fn(f, gx, gy, gz, bx, by, bz, shared, stream, params, extra);
    profile::end(stream, name_of(f), native_ran);
    return r;
}

} // extern "C"

namespace {

// The driver answers a versioned name with the implementation behind it, so
// "cuLaunchKernel" and "cuLaunchKernel_v2" (should one ever appear) both land
// here for the same entry.
bool same_symbol(const char *symbol, const char *name) {
    if (strcmp(symbol, name) == 0) return true;
    const size_t n = strlen(name);
    return strncmp(symbol, name, n) == 0 && strcmp(symbol + n, "_v2") == 0;
}

void substitute(const char *symbol, void **pfn) {
    if (!symbol || !pfn || !*pfn) return;
    if (GetEnvironmentVariableA("DLSS5NR_GATE_TRACE", nullptr, 0) > 0) logf("[symbol] %s\n", symbol);
    if (same_symbol(symbol, "cuModuleGetFunction")) {
        g_get_function = *pfn;
        *pfn = (void *)&cuModuleGetFunction;
    } else if (same_symbol(symbol, "cuLaunchKernel")) {
        g_launch = *pfn;
        *pfn = (void *)&cuLaunchKernel;
    }
}

} // namespace

extern "C" {

__declspec(dllexport) CUresult cuGetProcAddress(const char *symbol, void **pfn, int version,
                                                unsigned long long flags) {
    static auto fn = (CUresult (*)(const char *, void **, int, unsigned long long))real("cuGetProcAddress");
    const CUresult r = fn(symbol, pfn, version, flags);
    if (r == 0) substitute(symbol, pfn);
    return r;
}

__declspec(dllexport) CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int version,
                                                   unsigned long long flags, int *status) {
    static auto fn =
        (CUresult (*)(const char *, void **, int, unsigned long long, int *))real("cuGetProcAddress_v2");
    const CUresult r = fn(symbol, pfn, version, flags, status);
    if (r == 0) substitute(symbol, pfn);
    return r;
}

} // extern "C"

BOOL APIENTRY DllMain(HMODULE, DWORD reason, LPVOID reserved) {
    if (reason == DLL_PROCESS_ATTACH) InitializeCriticalSection(&g_lock);
    // At a normal unload only: during process termination (reserved != null)
    // other threads are already gone and the log may be half torn down.
    // The kernel totals are written even at process termination (reserved != null), where the
    // other reports are skipped: a run under DLSS5NR_GATE_TRACE ends that way and these are its result.
    if (reason == DLL_PROCESS_DETACH) trace_times::report();
    if (reason == DLL_PROCESS_DETACH && !reserved) native::report();
    return TRUE;
}
