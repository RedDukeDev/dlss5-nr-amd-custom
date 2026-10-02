// The missing NGX runtime.
//
// nvngx_dlss.dll is a "snippet": it refuses to be driven by an application
// directly without using the NGX runtime
//
// This file is that runtime. It is built as nvngx.dll purely so the calls into
// the snippet originate from a module with the right name; nothing here fakes a
// GPU, a driver or a license -- the check is an internal loader contract, and
// this supplies the missing half of it.
//
// The address the snippet inspects is the return address of the call, so the
// NVSDK_NGX_CUDA_* calls have to be made from inside this DLL. That is the whole
// reason the layer is split in two.
//
// It also means no forward here may become a tail call: `return f(args);` lets
// the compiler emit a jmp, which leaves the *caller's* return address on the
// stack and defeats the whole arrangement. Every forward therefore stores the
// result and touches a volatile before returning, which keeps the call in
// non-tail position.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cstdio>
#include <cstring>

#include "../common/ngx_cuda.h"

// Only ever passed through, never used, so the D3D headers stay out of here.
struct ID3D12Device;
struct ID3D12GraphicsCommandList;

namespace {

HMODULE g_snippet = nullptr;

// Touched after each forwarded call so the call cannot sit in tail position.
volatile int g_no_tail_call = 0;

// The fourth argument of Init_Ext is not a plain struct: the snippet reads it as
// an NVSDK_NGX_Parameter and calls Get through the vtable. Verified at
// 0x18003EFC0, which does
//
//     mov rax,[rdx]            ; vtable
//     mov rax,[rax+58h]        ; Get(const char*, unsigned*)
//     call rax                 ; ("Minimum.Logging.Level", &out)
//
// then [vt+0x40] Get(const char*, unsigned long long*) for "Log.Callback" and
// [vt+0x58] again for "Disable.Other.Logging.Sinks". Passing a plain struct made
// it read a null vtable and fault reading [0x58]; passing null skips the block
// entirely, which is why null merely gets no logging.
//
// So the runtime has to hand the snippet a real parameter object. This is the
// smallest one that answers what init asks for; every other key reports "not
// found" so the snippet falls back to its defaults.
// A real key/value store, because NVSDK_NGX_CUDA_PopulateParameters_Impl does
// not allocate a parameter block: it fills one the runtime provides. Values are
// kept in the widest slot of their kind and converted on the way out, which is
// what lets the snippet Set an int and the caller Get it as an unsigned.
bool g_trace_params = false;

// Records every Get the snippet performs. Evaluation reads its inputs from the
// parameter block first, so a missing or wrongly typed entry shows up here as a
// query that returns Fail -- which is far quicker to find than the matching
// branch in a 3000 instruction function.
void trace_get(const char *name, const char *kind, bool found) {
    if (!g_trace_params) return;
    char line[256];
    snprintf(line, sizeof line, "  [param] Get<%s>(\"%s\") -> %s\n",
             kind, name ? name : "(null)", found ? "ok" : "MISSING");
    OutputDebugStringA(line);
    fputs(line, stderr);
    fflush(stderr);
}

class RuntimeParameters : public NVSDK_NGX_Parameter {
public:
    void reset_all() { count_ = 0; }

    void Set(const char *n, unsigned long long v) override { trace_set(n, "u64", (unsigned long long)v); store(n, Kind::U64).u64 = v; }
    void Set(const char *n, float v) override { store(n, Kind::F64).f64 = v; }
    void Set(const char *n, double v) override { store(n, Kind::F64).f64 = v; }
    void Set(const char *n, unsigned int v) override { trace_set(n, "u32", (unsigned long long)v); store(n, Kind::U64).u64 = v; }
    void Set(const char *n, int v) override { trace_set(n, "int", (unsigned long long)(long long)v); store(n, Kind::I64).i64 = v; }
    void Set(const char *n, ID3D11Resource *v) override { store(n, Kind::Ptr).ptr = v; }
    void Set(const char *n, ID3D12Resource *v) override { store(n, Kind::Ptr).ptr = v; }
    void Set(const char *n, void *v) override { trace_set(n, "ptr", (unsigned long long)v); store(n, Kind::Ptr).ptr = v; }

    NVSDK_NGX_Result Get(const char *n, unsigned long long *o) const override {
        const Entry *e = find(n); trace_get(n, "u64", e != nullptr); if (!e || !o) return NVSDK_NGX_Result_Fail;
        // Bits, not value: only a float read converts.
        *o = e->kind == Kind::Ptr ? (unsigned long long)e->v.ptr : e->v.u64;
        return NVSDK_NGX_Result_Success;
    }
    NVSDK_NGX_Result Get(const char *n, float *o) const override {
        const Entry *e = find(n); trace_get(n, "float", e != nullptr); if (!e || !o) return NVSDK_NGX_Result_Fail;
        *o = (float)(e->kind == Kind::F64 ? e->v.f64
                   : e->kind == Kind::I64 ? (double)e->v.i64 : (double)e->v.u64);
        return NVSDK_NGX_Result_Success;
    }
    NVSDK_NGX_Result Get(const char *n, double *o) const override {
        const Entry *e = find(n); trace_get(n, "double", e != nullptr); if (!e || !o) return NVSDK_NGX_Result_Fail;
        *o = e->kind == Kind::F64 ? e->v.f64 : e->kind == Kind::I64 ? (double)e->v.i64 : (double)e->v.u64;
        return NVSDK_NGX_Result_Success;
    }
    NVSDK_NGX_Result Get(const char *n, unsigned int *o) const override {
        const Entry *e = find(n); trace_get(n, "u32", e != nullptr); if (!e || !o) return NVSDK_NGX_Result_Fail;
        *o = (unsigned)(e->kind == Kind::F64 ? (unsigned long long)e->v.f64 : e->v.u64);
        return NVSDK_NGX_Result_Success;
    }
    NVSDK_NGX_Result Get(const char *n, int *o) const override {
        const Entry *e = find(n); trace_get(n, "int", e != nullptr); if (!e || !o) return NVSDK_NGX_Result_Fail;
        *o = (int)(e->kind == Kind::F64 ? (long long)e->v.f64 : (long long)e->v.i64);
        return NVSDK_NGX_Result_Success;
    }
    NVSDK_NGX_Result Get(const char *n, ID3D11Resource **o) const override { return get_ptr(n, (void **)o); }
    NVSDK_NGX_Result Get(const char *n, ID3D12Resource **o) const override { return get_ptr(n, (void **)o); }
    NVSDK_NGX_Result Get(const char *n, void **o) const override { return get_ptr(n, o); }
    void Reset() override { reset_all(); }

private:
    enum class Kind { U64, I64, F64, Ptr };
    struct Entry {
        char name[96];
        Kind kind;
        union { unsigned long long u64; long long i64; double f64; void *ptr; } v;
    };
    // NGX populates on the order of a hundred entries; overflow drops the write
    // rather than corrupting memory, and shows up as a failed Get.
    static constexpr int kCapacity = 512;
    mutable Entry entries_[kCapacity]{};
    mutable int count_ = 0;

    const Entry *find(const char *n) const {
        if (!n) return nullptr;
        for (int i = 0; i < count_; i++)
            if (!strcmp(entries_[i].name, n)) return &entries_[i];
        return nullptr;
    }
    NVSDK_NGX_Result get_ptr(const char *n, void **o) const {
        const Entry *e = find(n); trace_get(n, "ptr", e != nullptr);
        if (!e || !o) return NVSDK_NGX_Result_Fail;
        *o = e->kind == Kind::Ptr ? e->v.ptr : (void *)e->v.u64;
        return NVSDK_NGX_Result_Success;
    }

    // Records every Set as well, so a value that changes shape between being
    // written and read is visible rather than inferred.
    void trace_set(const char *n, const char *kind, unsigned long long bits) const {
        if (!g_trace_params) return;
        char line[256];
        snprintf(line, sizeof line, "  [param] Set<%s>(\"%s\") = 0x%llX\n",
                 kind, n ? n : "(null)", bits);
        OutputDebugStringA(line);
        fputs(line, stderr);
        fflush(stderr);
    }
    decltype(Entry::v) &store(const char *n, Kind k) {
        static decltype(Entry::v) discard{};
        if (!n) return discard;
        for (int i = 0; i < count_; i++)
            if (!strcmp(entries_[i].name, n)) { entries_[i].kind = k; return entries_[i].v; }
        if (count_ >= kCapacity) return discard;
        Entry &e = entries_[count_++];
        strncpy(e.name, n, sizeof e.name - 1);
        e.name[sizeof e.name - 1] = 0;
        e.kind = k;
        e.v.u64 = 0;
        return e.v;
    }
};

RuntimeParameters g_init_params;   // carries the logging settings into Init_Ext
RuntimeParameters g_feature_params; // the block the snippet populates and the caller drives

PFN_NVSDK_NGX_CUDA_Init_Ext s_init_ext = nullptr;
PFN_NVSDK_NGX_CUDA_Init_Ext1 s_init_ext1 = nullptr;
PFN_NVSDK_NGX_CUDA_CreateFeature s_create = nullptr;
PFN_NVSDK_NGX_CUDA_CreateFeature1 s_create1 = nullptr;
void *s_cuda_context = nullptr;

// [RE] What the snippet means by a CUDA device.
//
// CreateFeature1 does two things with its first argument (+0x26696 and
// +0x2673B): it reads two qwords out of it, and it looks the pointer itself up
// in the map of registered devices. So the value Init_Ext1 registers has to be
// a pointer to a two field object that stays alive, not a bare CUDA context --
// handing it the context makes CreateFeature1 dereference the context and the
// process dies.
//
// Registering the context works only as long as nothing else in the process
// registers a device, because only then can CreateFeature pick the single one
// without being told which. A game that brings up its own DLSS breaks that, and
// the refusal is 0xBAD00002.
struct NgxCudaDevice {
    void *field0; // [RE] handed to a method on the device data, which may refuse
    void *field1; // [RE] stored at device data +0x3F8
};
NgxCudaDevice s_device{};

// Which branch ngxrt_create_feature took last: 0 plain, 1 input params, 2 named
// device. Reported so a log says what actually ran rather than what should have.
int s_create_path = -1;

// Set through ngxrt_set_real_driver; see the note there.
bool g_real_driver = false;
unsigned long long g_input1 = 0, g_input2 = 0;
bool g_use_input_params = false;
PFN_NVSDK_NGX_CUDA_EvaluateFeature s_evaluate = nullptr;
PFN_NVSDK_NGX_CUDA_ReleaseFeature s_release = nullptr;
PFN_NVSDK_NGX_CUDA_Shutdown s_shutdown = nullptr;
PFN_NVSDK_NGX_CUDA_PopulateParameters_Impl s_populate = nullptr;
PFN_NVSDK_NGX_CUDA_GetScratchBufferSize s_scratch_size = nullptr;

// The snippet's own D3D12 entry points, for a real NVIDIA driver.
//
// Init_Ext takes the version before the parameter block, unlike the public
// D3D12_Init, and reads its logging settings from that block the same way the
// CUDA Init_Ext does -- which is why g_init_params serves both.
using PFN_D3D12_InitExt = NVSDK_NGX_Result (*)(unsigned long long, const wchar_t *, ID3D12Device *,
                                               NVSDK_NGX_Version, NVSDK_NGX_Parameter *);
using PFN_D3D12_Create = NVSDK_NGX_Result (*)(ID3D12GraphicsCommandList *, NVSDK_NGX_Feature,
                                              NVSDK_NGX_Parameter *, NVSDK_NGX_Handle **);
using PFN_D3D12_Evaluate = NVSDK_NGX_Result (*)(ID3D12GraphicsCommandList *, const NVSDK_NGX_Handle *,
                                                const NVSDK_NGX_Parameter *, void *);
using PFN_D3D12_Release = NVSDK_NGX_Result (*)(NVSDK_NGX_Handle *);
using PFN_D3D12_Populate = NVSDK_NGX_Result (*)(NVSDK_NGX_Parameter *);
using PFN_D3D12_Shutdown1 = NVSDK_NGX_Result (*)(ID3D12Device *);

PFN_D3D12_InitExt s_d3d12_init_ext = nullptr;
PFN_D3D12_Create s_d3d12_create = nullptr;
PFN_D3D12_Evaluate s_d3d12_evaluate = nullptr;
PFN_D3D12_Release s_d3d12_release = nullptr;
PFN_D3D12_Populate s_d3d12_populate = nullptr;
PFN_D3D12_Shutdown1 s_d3d12_shutdown1 = nullptr;

template <typename T>
bool resolve(T &slot, const char *name) {
    slot = reinterpret_cast<T>(GetProcAddress(g_snippet, name));
    return slot != nullptr;
}

} // namespace

extern "C" {

// Loads the snippet and resolves its CUDA entry points. Returns the name of the
// first missing export, or nullptr on success.
__declspec(dllexport) const char *ngxrt_load(const wchar_t *snippet_path) {
    if (g_snippet) return nullptr;
    // Backslashes only. The GUI hands over what Qt's file dialog produced,
    // which uses forward slashes, and the loader documents backslashes as the
    // only separator it supports -- with LOAD_WITH_ALTERED_SEARCH_PATH forward
    // slashes are explicitly undefined behaviour.
    static wchar_t normalised[MAX_PATH * 4];
    wcsncpy_s(normalised, snippet_path ? snippet_path : L"nvngx_dlss.dll", _TRUNCATE);
    for (wchar_t *c = normalised; *c; ++c)
        if (*c == L'/') *c = L'\\';
    const wchar_t *path = normalised;
    // With a full path, resolve the snippet's own imports from its directory
    // rather than from the executable's. A plain LoadLibraryW searches beside
    // the program, so a snippet that brings a dependency along in its own
    // folder fails with "module not found" -- naming the snippet, which is
    // there, and not the file that is missing.
    const bool full_path = wcschr(path, L'\\') != nullptr;
    g_snippet = full_path ? LoadLibraryExW(path, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH)
                          : LoadLibraryW(path);
    if (!g_snippet) {
        // Name the file that was actually asked for, and why Windows refused
        // it. Without the error code a missing dependency, a damaged file, a
        // snippet whose DllMain declined and a code-integrity refusal all read
        // the same, and a tester's screenshot cannot tell them apart.
        const DWORD error = GetLastError();
        const char *why = "";
        switch (error) {
        case ERROR_MOD_NOT_FOUND:
            // LoadLibraryExW answers 126 for the file itself being absent too,
            // not only for a missing dependency -- measured: a path that does
            // not exist comes back as 126, not 2. Only a look at the file
            // separates the two.
            why = GetFileAttributesW(path) == INVALID_FILE_ATTRIBUTES
                      ? " -- the file does not exist at that path"
                      : " -- the file is there, but a DLL it depends on was not found";
            break;
        case ERROR_BAD_EXE_FORMAT: why = " -- not a valid 64-bit DLL (damaged or wrong file)"; break;
        case ERROR_DLL_INIT_FAILED: why = " -- the snippet's own initialisation refused to load"; break;
        // A patched snippet no longer carries a valid signature. On a stock
        // Windows machine that loads fine -- measured, from a path with spaces
        // and forward slashes alike -- but Windows 11 Smart App Control, a WDAC
        // policy or an antivirus can refuse exactly that, and only on the
        // machine where it is enabled. These are the codes that says so.
        case ERROR_INVALID_IMAGE_HASH:
        case 4551: // ERROR_SYSTEM_INTEGRITY_POLICY_VIOLATION
            why = " -- blocked by a code integrity policy because its signature is not valid "
                  "(a patched snippet; check Smart App Control / WDAC, or use the signed original)";
            break;
        case ERROR_ACCESS_DISABLED_BY_POLICY:
            why = " -- blocked by a system policy (Smart App Control, WDAC or software restriction)";
            break;
        case ERROR_VIRUS_INFECTED:
        case ERROR_VIRUS_DELETED:
            why = " -- the antivirus blocked or removed it";
            break;
        case ERROR_ACCESS_DENIED: why = " -- access denied (antivirus or file permissions)"; break;
        // Measured: while another process holds the file open for writing, or
        // without sharing it, the load fails with this -- and the signature
        // check just before it reports the file as unverifiable. A copy still
        // in progress, an antivirus scan or a cloud sync all look like this.
        case ERROR_SHARING_VIOLATION:
            why = " -- another program has the file open (a copy still in progress, an antivirus "
                  "scan or a cloud sync such as OneDrive); wait for it or move the file elsewhere";
            break;
        case ERROR_FILE_NOT_FOUND:
        case ERROR_PATH_NOT_FOUND: why = " -- the file does not exist at that path"; break;
        }
        static char asked[768];
        char name[512];
        WideCharToMultiByte(CP_UTF8, 0, path, -1, name, sizeof name, nullptr, nullptr);
        snprintf(asked, sizeof asked, "%s (Windows error %lu%s)", name, error, why);
        return asked;
    }

    // NVSDK_NGX_CUDA_Init is deliberately not resolved: that export is a stub
    // returning 0xBAD00001, shared with D3D11_Init and D3D12_Init.
    if (!resolve(s_init_ext, "NVSDK_NGX_CUDA_Init_Ext")) return "NVSDK_NGX_CUDA_Init_Ext";
    // Optional: only the _Ext1 form registers the feature under a usable key,
    // but keep working if a snippet build lacks it.
    resolve(s_init_ext1, "NVSDK_NGX_CUDA_Init_Ext1");
    if (!resolve(s_create, "NVSDK_NGX_CUDA_CreateFeature")) return "NVSDK_NGX_CUDA_CreateFeature";
    resolve(s_create1, "NVSDK_NGX_CUDA_CreateFeature1");
    if (!resolve(s_evaluate, "NVSDK_NGX_CUDA_EvaluateFeature")) return "NVSDK_NGX_CUDA_EvaluateFeature";
    if (!resolve(s_release, "NVSDK_NGX_CUDA_ReleaseFeature")) return "NVSDK_NGX_CUDA_ReleaseFeature";
    if (!resolve(s_shutdown, "NVSDK_NGX_CUDA_Shutdown")) return "NVSDK_NGX_CUDA_Shutdown";
    if (!resolve(s_populate, "NVSDK_NGX_CUDA_PopulateParameters_Impl"))
        return "NVSDK_NGX_CUDA_PopulateParameters_Impl";
    // Optional: not every snippet build exports it, and nothing else depends on
    // it -- it is only used to identify which feature id a snippet implements.
    resolve(s_scratch_size, "NVSDK_NGX_CUDA_GetScratchBufferSize");
    // Optional here, required by the direct D3D12 path: that path asks
    // ngxrt_d3d12_available and says what is missing, while the stand-in path
    // never touches these at all.
    resolve(s_d3d12_init_ext, "NVSDK_NGX_D3D12_Init_Ext");
    resolve(s_d3d12_create, "NVSDK_NGX_D3D12_CreateFeature");
    resolve(s_d3d12_evaluate, "NVSDK_NGX_D3D12_EvaluateFeature");
    resolve(s_d3d12_release, "NVSDK_NGX_D3D12_ReleaseFeature");
    resolve(s_d3d12_populate, "NVSDK_NGX_D3D12_PopulateParameters_Impl");
    resolve(s_d3d12_shutdown1, "NVSDK_NGX_D3D12_Shutdown1");
    return nullptr;
}

// Each of these is a one line forward, and that is the point: the call
// instruction has to sit in this module so the snippet's caller check sees
// nvngx.dll.

// `log_callback` is void(*)(const char *message, int level, unsigned feature),
// passed as an integer because it travels through the parameter object. Null
// turns logging off, which is also what passing no parameter object at all does.
__declspec(dllexport) NVSDK_NGX_Result ngxrt_init(unsigned long long application_id,
                                                  const wchar_t *data_path,
                                                  NVSDK_NGX_Version sdk_version,
                                                  unsigned long long log_callback,
                                                  unsigned logging_level,
                                                  void *cuda_context) {
    if (!s_init_ext) return NVSDK_NGX_Result_Fail;
    g_init_params.reset_all();
    g_init_params.Set("Log.Callback", log_callback);
    g_init_params.Set("Minimum.Logging.Level", logging_level);
    g_init_params.Set("Disable.Other.Logging.Sinks", 0u);

    // Prefer Init_Ext1: it registers the runtime under the CUDA context, which
    // is the key EvaluateFeature looks the feature up by. Init_Ext registers
    // under null and evaluation then fails with PlatformError.
    NVSDK_NGX_Result r;
    // Init_Ext1 with the CUDA context is what gives the snippet a real device:
    // the log goes from "m_pDevice: 0x0 -> 0x0" to a live pointer. CreateFeature
    // (not the _1 form) is kept alongside it -- the _1 form dereferences its
    // first argument as a two field struct, which a raw context is not.
    s_cuda_context = cuda_context;
    // Both fields empty. Measured: the snippet accepts this and also accepts the
    // context in the first field, with no difference in outcome, so the object
    // exists purely to be an identity the caller can name later.
    s_device.field0 = nullptr;
    s_device.field1 = nullptr;
    if (s_init_ext1 && cuda_context) {
        r = s_init_ext1(application_id, data_path, &s_device, sdk_version, &g_init_params);
    } else {
        r = s_init_ext(application_id, data_path, sdk_version, &g_init_params);
    }
    g_no_tail_call = 1;
    return r;
}


// PopulateParameters_Impl fills a block the runtime owns rather than allocating
// one, so the object is ours and only a pointer to it goes back to the caller.
__declspec(dllexport) NVSDK_NGX_Result ngxrt_populate_parameters(NVSDK_NGX_Parameter **out_params) {
    if (!s_populate || !out_params) return NVSDK_NGX_Result_Fail;
    g_feature_params.reset_all();
    NVSDK_NGX_Result r = s_populate(&g_feature_params);
    g_no_tail_call = 1;
    *out_params = &g_feature_params;
    return r;
}

__declspec(dllexport) NVSDK_NGX_Result ngxrt_create_feature(NVSDK_NGX_Feature feature_id,
                                                            NVSDK_NGX_Parameter *params,
                                                            NVSDK_NGX_Handle **out_handle) {
    if (!s_create) return NVSDK_NGX_Result_Fail;
    // Pair with whichever init form ran: the feature has to be registered under
    // the same key the evaluation will look it up by.
    //
    // This runtime normally stands in for NGX's core where there is no core to
    // call -- the stand-in CUDA driver. It was once tried on NVIDIA's own driver
    // too and failed with 0xBAD00002, which is why the program moved to NVIDIA's
    // public interface there; but that attempt took the path below
    // unconditionally, and naming our own device is precisely what a real driver
    // rejects. With ngxrt_set_real_driver that path is now skipped on a real
    // driver, so the old result is not evidence about what happens now.
    NVSDK_NGX_Result r;
    if (s_create1 && g_use_input_params) {
        // CreateFeature1 with a null first argument reads two values from the
        // parameter block: "Input1" is passed to a method on the device data
        // that can itself fail with PlatformError, and "Input2" is stored at
        // +0x3F8. Both are set by the caller through ngxrt_set_inputs.
        params->Set("Input1", g_input1);
        params->Set("Input2", g_input2);
        s_create_path = 1;
        r = s_create1(nullptr, feature_id, params, out_handle);
    } else if (s_create1 && s_cuda_context && !g_real_driver) {
        // Name the device instead of letting the snippet choose. It chooses on
        // its own only while exactly one is registered; a second user of the
        // same snippet in the process -- a game bringing up its own DLSS, which
        // our NVAPI stand-in invites it to do -- makes it refuse with
        // 0xBAD00002 instead. [RE] The key it stores is the CUDA context handed
        // to Init_Ext1: read out of the snippet's own device map, the leftmost
        // entry's field at +0x90 holds exactly that pointer.
        s_create_path = 2;
        r = s_create1(&s_device, feature_id, params, out_handle);
    } else {
        s_create_path = 0;
        r = s_create(feature_id, params, out_handle);
    }
    g_no_tail_call = 1;
    return r;
}

__declspec(dllexport) NVSDK_NGX_Result ngxrt_evaluate(NVSDK_NGX_Handle *handle,
                                                      const NVSDK_NGX_Parameter *params) {
    if (!s_evaluate) return NVSDK_NGX_Result_Fail;
    NVSDK_NGX_Result r = s_evaluate(handle, params, nullptr);
    g_no_tail_call = 1;
    return r;
}

__declspec(dllexport) NVSDK_NGX_Result ngxrt_release_feature(NVSDK_NGX_Handle *handle) {
    if (!s_release) return NVSDK_NGX_Result_Fail;
    NVSDK_NGX_Result r = s_release(handle);
    g_no_tail_call = 1;
    return r;
}

__declspec(dllexport) NVSDK_NGX_Result ngxrt_shutdown(void) {
    if (!s_shutdown) return NVSDK_NGX_Result_Fail;
    NVSDK_NGX_Result r = s_shutdown();
    g_no_tail_call = 1;
    return r;
}

// ---- The snippet's D3D12 interface, for a real NVIDIA driver ----------------
//
// Same one-line forwards as the CUDA ones above, for the same reason: the call
// has to leave from this module. See the note above s_d3d12_init_ext for where
// the shapes come from.

__declspec(dllexport) int ngxrt_d3d12_available(void) {
    return s_d3d12_init_ext && s_d3d12_create && s_d3d12_evaluate && s_d3d12_release ? 1 : 0;
}

__declspec(dllexport) NVSDK_NGX_Result ngxrt_d3d12_init(unsigned long long application_id,
                                                        const wchar_t *data_path,
                                                        ID3D12Device *device,
                                                        NVSDK_NGX_Version sdk_version,
                                                        unsigned long long log_callback,
                                                        unsigned logging_level) {
    if (!s_d3d12_init_ext) return NVSDK_NGX_Result_Fail;
    g_init_params.reset_all();
    g_init_params.Set("Log.Callback", log_callback);
    g_init_params.Set("Minimum.Logging.Level", logging_level);
    g_init_params.Set("Disable.Other.Logging.Sinks", 0u);
    NVSDK_NGX_Result r = s_d3d12_init_ext(application_id, data_path, device, sdk_version, &g_init_params);
    g_no_tail_call = 1;
    return r;
}

// The block is this runtime's, as on the CUDA path. Without the export it is
// handed back empty and reported as success: the reference this path follows
// creates the feature without ever populating first.
__declspec(dllexport) NVSDK_NGX_Result ngxrt_d3d12_populate(NVSDK_NGX_Parameter **out_params) {
    if (!out_params) return NVSDK_NGX_Result_Fail;
    g_feature_params.reset_all();
    NVSDK_NGX_Result r = NVSDK_NGX_Result_Success;
    if (s_d3d12_populate) r = s_d3d12_populate(&g_feature_params);
    g_no_tail_call = 1;
    *out_params = &g_feature_params;
    return r;
}

// Records initialisation work into the command list, so the list has to be open
// and then actually executed -- which the caller's open/submit pair does.
__declspec(dllexport) NVSDK_NGX_Result ngxrt_d3d12_create(ID3D12GraphicsCommandList *cmd,
                                                          NVSDK_NGX_Feature feature_id,
                                                          NVSDK_NGX_Parameter *params,
                                                          NVSDK_NGX_Handle **out_handle) {
    if (!s_d3d12_create) return NVSDK_NGX_Result_Fail;
    NVSDK_NGX_Result r = s_d3d12_create(cmd, feature_id, params, out_handle);
    g_no_tail_call = 1;
    return r;
}

__declspec(dllexport) NVSDK_NGX_Result ngxrt_d3d12_evaluate(ID3D12GraphicsCommandList *cmd,
                                                            const NVSDK_NGX_Handle *handle,
                                                            const NVSDK_NGX_Parameter *params,
                                                            void *progress_callback) {
    if (!s_d3d12_evaluate) return NVSDK_NGX_Result_Fail;
    NVSDK_NGX_Result r = s_d3d12_evaluate(cmd, handle, params, progress_callback);
    g_no_tail_call = 1;
    return r;
}

__declspec(dllexport) NVSDK_NGX_Result ngxrt_d3d12_release(NVSDK_NGX_Handle *handle) {
    if (!s_d3d12_release) return NVSDK_NGX_Result_Fail;
    NVSDK_NGX_Result r = s_d3d12_release(handle);
    g_no_tail_call = 1;
    return r;
}

__declspec(dllexport) NVSDK_NGX_Result ngxrt_d3d12_shutdown(ID3D12Device *device) {
    if (!s_d3d12_shutdown1) return NVSDK_NGX_Result_Fail;
    NVSDK_NGX_Result r = s_d3d12_shutdown1(device);
    g_no_tail_call = 1;
    return r;
}

// Selects the CreateFeature1 path and the two values it reads from the
// parameter block. Their meaning is not yet pinned down, so they stay settable.
// Turns the parameter trace on. Off by default: it is a diagnostic, and in a
// game it would be one line per query per frame.
// Asks the snippet how much scratch memory a feature id would need. A snippet
// implements exactly one feature and rejects every other id, so sweeping this
// over the id space identifies the one it answers to -- without paying for the
// network build that CreateFeature would start.
__declspec(dllexport) NVSDK_NGX_Result ngxrt_scratch_size(NVSDK_NGX_Feature feature_id,
                                                          NVSDK_NGX_Parameter *params,
                                                          unsigned long long *out_size) {
    if (!s_scratch_size) return NVSDK_NGX_Result_Fail;
    size_t size = 0;
    NVSDK_NGX_Result r = s_scratch_size(feature_id, params ? params : &g_feature_params, &size);
    g_no_tail_call = 1;
    if (out_size) *out_size = size;
    return r;
}

// Diagnostic only. Registers a second device under a key of its own, the way
// another user of the same snippet in the same process does -- a game bringing
// up its own DLSS. It is what lets the bench reproduce the condition that makes
// CreateFeature refuse to choose a device, and so what makes naming ours a
// verifiable fix rather than a plausible one.
__declspec(dllexport) NVSDK_NGX_Result ngxrt_register_foreign_device(unsigned long long app_id,
                                                                     const wchar_t *data_path,
                                                                     NVSDK_NGX_Version version) {
    static NgxCudaDevice foreign{};
    if (!s_init_ext1) return NVSDK_NGX_Result_Fail;
    return s_init_ext1(app_id, data_path, &foreign, version, &g_init_params);
}

// When this runtime was built, and which create path it took. A fix that lives
// in this DLL is invisible if an older copy of it is the one sitting beside the
// addon, and no error code says so.
__declspec(dllexport) const char *ngxrt_build_id(void) { return __DATE__ " " __TIME__; }

__declspec(dllexport) int ngxrt_create_path(void) { return s_create_path; }

// Whether a real NVIDIA driver is underneath rather than the stand-in.
//
// It changes one decision, in ngxrt_create_feature: naming our own device to
// CreateFeature1 is a workaround for a condition the stand-in creates -- our
// NVAPI stand-in invites a second user of the snippet into the process, and the
// snippet then refuses to pick a device on its own. On a real driver there is
// no second user, the workaround is not needed, and it is actively wrong: it
// was what made a real driver answer 0xBAD00002.
//
// Default stays false so the AMD path behaves exactly as before; only the
// NVIDIA path sets it.
__declspec(dllexport) void ngxrt_set_real_driver(int real) { g_real_driver = real != 0; }

__declspec(dllexport) void ngxrt_trace_params(int enable) { g_trace_params = enable != 0; }


__declspec(dllexport) void ngxrt_set_inputs(int enable, unsigned long long input1,
                                            unsigned long long input2) {
    g_use_input_params = enable != 0;
    g_input1 = input1;
    g_input2 = input2;
}

// Diagnostic: reports the module path the snippet's check would resolve from an
// address inside this DLL, so a mismatch is visible instead of inferred.
__declspec(dllexport) int ngxrt_self_path(wchar_t *out, int count) {
    HMODULE self = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(&ngxrt_load), &self))
        return 0;
    return (int)GetModuleFileNameW(self, out, (DWORD)count);
}

} // extern "C"

BOOL APIENTRY DllMain(HMODULE, DWORD, LPVOID) { return TRUE; }
