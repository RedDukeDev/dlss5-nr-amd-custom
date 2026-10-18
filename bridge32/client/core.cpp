#include "core.h"
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>

HMODULE g_self = nullptr;
std::wstring g_dir;
Config g_config;
Bridge g_bridge;

namespace {
FILE *g_log = nullptr;
CRITICAL_SECTION g_log_lock;
}

void log(const char *format, ...) {
    if (!g_log) return;
    EnterCriticalSection(&g_log_lock);
    SYSTEMTIME t;
    GetLocalTime(&t);
    fprintf(g_log, "[%02d:%02d:%02d.%03d] ", t.wHour, t.wMinute, t.wSecond, t.wMilliseconds);
    va_list args;
    va_start(args, format);
    vfprintf(g_log, format, args);
    va_end(args);
    fputc('\n', g_log);
    fflush(g_log);
    LeaveCriticalSection(&g_log_lock);
}

// For the parts of the client that do not see the log above.
void bridge_log(const char *format, ...) {
    if (!g_log) return;
    char text[512];
    va_list args;
    va_start(args, format);
    vsnprintf(text, sizeof text, format, args);
    va_end(args);
    log("%s", text);
}

void core_init(HMODULE self, const wchar_t *log_name) {
    g_self = self;
    // This DLL is never unloaded. It patches the method tables of objects of the real DLL it loads (and of
    // DirectInput's), and those stay patched: a game component that loads d3d9.dll, makes an object, and
    // frees the library again would leave them pointing at code that is gone, and the next call would crash.
    HMODULE pinned = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_PIN | GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, (LPCWSTR)&core_init, &pinned);
    wchar_t path[MAX_PATH];
    GetModuleFileNameW(g_self, path, MAX_PATH);
    g_dir = path;
    g_dir = g_dir.substr(0, g_dir.find_last_of(L'\\') + 1);
    InitializeCriticalSection(&g_log_lock);
    g_log = _wfsopen((g_dir + log_name).c_str(), L"w", _SH_DENYNO);
    g_config.ini = g_dir + L"OptiScaler.ini";
    read_config();
    log("client loaded in %lu, enabled=%d", GetCurrentProcessId(), (int)g_config.enabled);
}

// ------------------------------------------------------------------ settings

float read_float(const wchar_t *key, float otherwise) {
    wchar_t text[64] = {};
    GetPrivateProfileStringW(L"NeuralRendering", key, L"auto", text, 64, g_config.ini.c_str());
    if (!_wcsicmp(text, L"auto") || !text[0]) return otherwise;
    return (float)_wtof(text);
}

int32_t read_int(const wchar_t *key, int32_t otherwise) {
    wchar_t text[64] = {};
    GetPrivateProfileStringW(L"NeuralRendering", key, L"auto", text, 64, g_config.ini.c_str());
    if (!_wcsicmp(text, L"auto") || !text[0]) return otherwise;
    if (!_wcsicmp(text, L"true")) return 1;
    if (!_wcsicmp(text, L"false")) return 0;
    return _wtoi(text);
}

bool read_config() {
    WIN32_FILE_ATTRIBUTE_DATA data;
    if (!GetFileAttributesExW(g_config.ini.c_str(), GetFileExInfoStandard, &data)) return false;
    if (CompareFileTime(&data.ftLastWriteTime, &g_config.stamp) == 0) return false;
    g_config.stamp = data.ftLastWriteTime;
    bridge::Settings &s = g_config.settings;
    s.enabled = read_int(L"Enabled", 0);
    s.resolutionScale = read_float(L"ResolutionScale", bridge::kAutoFloat);
    s.style = read_int(L"Style", bridge::kAutoInt);
    s.intensity = read_float(L"Intensity", bridge::kAutoFloat);
    s.localTone = read_float(L"LocalTone", bridge::kAutoFloat);
    s.localStructure = read_float(L"LocalStructure", bridge::kAutoFloat);
    s.skinStructure = read_float(L"SkinStructure", bridge::kAutoFloat);
    s.detailStrength = read_float(L"DetailStrength", bridge::kAutoFloat);
    s.colourStrength = read_float(L"ColourStrength", bridge::kAutoFloat);
    s.maxRatio = read_float(L"MaxRatio", bridge::kAutoFloat);
    s.passes = read_int(L"Passes", bridge::kAutoInt);
    s.keepLocalTone = read_int(L"KeepLocalTone", bridge::kAutoInt);
    s.preset = read_int(L"Preset", bridge::kAutoInt);
    s.smoothing = read_float(L"BridgeSmoothing", bridge::kAutoFloat);
    g_config.enabled = s.enabled != 0;
    wchar_t host[MAX_PATH] = {};
    GetPrivateProfileStringW(L"NeuralRendering", L"BridgeHost", L"", host, MAX_PATH, g_config.ini.c_str());
    g_config.host = host;
    g_config.dump_frame = read_int(L"BridgeDebugDumpFrame", 0);
    g_config.dump_series = read_int(L"BridgeDebugDumpSeries", 0);
    g_config.open_menu = read_int(L"BridgeDebugOpenMenu", 0) != 0;
    g_config.options.reduceTrailing = read_int(L"BridgeReduceTrailing", 1) != 0;
    g_config.options.tolerance = read_float(L"BridgeTrailingTolerance", 0.025f);
    g_config.options.wait = read_int(L"BridgeWaitForNetwork", 1) != 0;
    g_config.options.follow = read_int(L"BridgeFollowMotion", 0) != 0;
    return true;
}

// What the menu changes goes to the host at once and, when it is final, to OptiScaler.ini.
bridge::Settings &bridge_user_settings() { return g_config.settings; }
BridgeOptions &bridge_options() { return g_config.options; }

void bridge_setting_changed(const wchar_t *key, const wchar_t *text, bool save) {
    g_config.enabled = g_config.settings.enabled != 0;
    if (g_bridge.header) {
        g_bridge.header->settings = g_config.settings;
        InterlockedIncrement(&g_bridge.header->settingsSeq);
    }
    if (save) WritePrivateProfileStringW(L"NeuralRendering", key, text, g_config.ini.c_str());
}

// ------------------------------------------------------------------ the bridge

bool Bridge::start() {
    if (started) return !failed;
    started = true;
    const DWORD pid = GetCurrentProcessId();
    wchar_t name[96];
    swprintf(name, 96, L"Local\\dlss5nr_bridge_%lu", pid);
    const uint64_t bytes = bridge::max_total_bytes();
    mapping = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, (DWORD)(bytes >> 32), (DWORD)bytes, name);
    swprintf(name, 96, L"Local\\dlss5nr_bridge_capture_%lu", pid);
    capture_event = CreateEventW(nullptr, FALSE, FALSE, name);
    swprintf(name, 96, L"Local\\dlss5nr_bridge_result_%lu", pid);
    result_event = CreateEventW(nullptr, FALSE, FALSE, name);
    if (!mapping || !capture_event || !result_event) return fail("could not create the shared memory");
    header = (bridge::Header *)MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, bridge::kHeaderBytes);
    if (!header) return fail("could not map the header");
    memset(header, 0, sizeof *header);
    header->magic = bridge::kMagic;
    header->version = bridge::kVersion;
    header->clientPid = pid;
    header->settings = g_config.settings;
    header->settingsSeq = 1;
    return spawn();
}

bool Bridge::fail(const char *what) {
    failed = true;
    log("bridge: %s (error %lu)", what, GetLastError());
    return false;
}

bool Bridge::spawn() {
    std::wstring host_path = g_config.host;
    if (host_path.empty()) host_path = g_dir + L"OptiScaler\\dlss5nr\\dlss5nr_host.exe";
    if (GetFileAttributesW(host_path.c_str()) == INVALID_FILE_ATTRIBUTES) {
        log("bridge: the host is not there: %ls", host_path.c_str());
        return fail("no host");
    }
    // The host goes where the game goes: when the game ends, the job takes it with it.
    job = CreateJobObjectW(nullptr, nullptr);
    if (job) {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits = {};
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof limits);
    }
    wchar_t command[2 * MAX_PATH];
    swprintf(command, 2 * MAX_PATH, L"\"%ls\" --client %lu", host_path.c_str(), GetCurrentProcessId());
    std::wstring work = host_path.substr(0, host_path.find_last_of(L'\\'));
    STARTUPINFOW si = {sizeof si};
    PROCESS_INFORMATION pi = {};
    if (!CreateProcessW(nullptr, command, nullptr, nullptr, FALSE, CREATE_NO_WINDOW | CREATE_SUSPENDED, nullptr, work.c_str(), &si, &pi))
        return fail("could not start the host");
    if (job) AssignProcessToJobObject(job, pi.hProcess);
    ResumeThread(pi.hThread);
    CloseHandle(pi.hThread);
    process = pi.hProcess;
    log("bridge: host started (%ls)", host_path.c_str());
    return true;
}

bool Bridge::map(uint32_t w, uint32_t h) {
    if (view && view_w == w && view_h == h) return true;
    if (view) UnmapViewOfFile(view);
    view = nullptr;
    if (w > bridge::kMaxWidth || h > bridge::kMaxHeight) return false;
    view = (uint8_t *)MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, (SIZE_T)bridge::total_bytes(w, h));
    view_w = w;
    view_h = h;
    if (!view) log("bridge: could not map %ux%u (error %lu)", w, h, GetLastError());
    return view != nullptr;
}

void Bridge::stop() {
    if (header) InterlockedExchange(&header->shutdown, 1);
    if (capture_event) SetEvent(capture_event);
}

void network_size(UINT w, UINT h, UINT &cw, UINT &ch) {
    float scale = g_config.settings.resolutionScale > bridge::kAutoFloat / 2 ? g_config.settings.resolutionScale : bridge::kDefaultScale;
    scale = scale < 0.25f ? 0.25f : scale > 1.0f ? 1.0f : scale;
    if (scale >= 0.999f) {
        cw = w;
        ch = h;
        return;
    }
    cw = std::max(64u, (UINT)(w * scale + 0.5f) & ~7u);
    ch = std::max(64u, (UINT)(h * scale + 0.5f) & ~7u);
    cw = std::min(cw, w);
    ch = std::min(ch, h);
}
