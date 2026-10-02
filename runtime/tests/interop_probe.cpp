// dlss5nr_interop_probe.exe: can a D3D12 resource's memory be handed to ROCm?
//
// Run it inside the Wine or Proton prefix of a game (any Windows program
// runs there; it needs no game). It asks D3D12 -- vkd3d-proton under Wine --
// for a shared handle to a buffer and to a texture, and then, under Wine,
// whether Wine can turn that handle into the Unix file descriptor ROCm
// imports external memory from (hipExternalMemoryHandleTypeOpaqueFd). It
// prints what it finds to the console and to dlss5nr_interop_probe.log
// beside it. On Windows it only reports the handles.
//
//   dlss5nr_interop_probe.exe [--hold seconds]
//
// --hold keeps the process, and so its descriptors, alive, to look at them
// from a terminal:  ls -l /proc/$(pgrep -f interop_probe | head -1)/fd
//
// Nothing here talks to HIP: the question it answers is only whether the
// descriptor can be had, and what it is.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <string>
#include <vector>

namespace {

FILE *g_log = nullptr;

void say(const char *format, ...) {
    char line[1024];
    va_list args;
    va_start(args, format);
    vsnprintf(line, sizeof line, format, args);
    va_end(args);
    printf("%s\n", line);
    fflush(stdout);
    if (g_log) {
        fprintf(g_log, "%s\n", line);
        fflush(g_log);
    }
}

typedef LONG NT_STATUS;

struct ObjectTypeName {
    USHORT length, maximum;
    PWSTR buffer;
};

typedef NT_STATUS(NTAPI *NtQueryObjectFn)(HANDLE, int, void *, ULONG, ULONG *);
typedef NT_STATUS(__cdecl *HandleToFdFn)(HANDLE, unsigned int, int *, unsigned int *);
typedef const char *(__cdecl *WineVersionFn)();

std::string narrow(const wchar_t *text, size_t count) {
    std::string out;
    for (size_t i = 0; i < count; ++i) out += text[i] < 128 ? (char)text[i] : '?';
    return out;
}

// What kind of object the handle names, as the system calls it.
std::string type_name(HANDLE handle) {
    static NtQueryObjectFn query = (NtQueryObjectFn)GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtQueryObject");
    if (!query) return "(NtQueryObject missing)";
    std::vector<char> buffer(1024);
    ULONG needed = 0;
    const NT_STATUS status = query(handle, 2 /* ObjectTypeInformation */, buffer.data(), (ULONG)buffer.size(), &needed);
    if (status < 0) {
        char text[64];
        snprintf(text, sizeof text, "(status %#lx)", (unsigned long)status);
        return text;
    }
    const auto *name = (const ObjectTypeName *)buffer.data();
    return narrow(name->buffer, name->length / sizeof(wchar_t));
}

// Where a Unix descriptor leads, asked through the Windows view of /proc.
void describe_fd(int fd) {
    wchar_t path[96];
    swprintf(path, 96, L"\\\\?\\Z:\\proc\\self\\fd\\%d", fd);
    HANDLE opened = CreateFileW(path, FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    if (opened == INVALID_HANDLE_VALUE) {
        say("    /proc/self/fd/%d cannot be opened from here (error %lu): use --hold and look at it from a terminal",
            fd, GetLastError());
        return;
    }
    wchar_t final_path[1024] = {};
    const DWORD length = GetFinalPathNameByHandleW(opened, final_path, 1024, FILE_NAME_NORMALIZED);
    if (length && length < 1024) say("    /proc/self/fd/%d leads to %s", fd, narrow(final_path, wcslen(final_path)).c_str());
    else say("    /proc/self/fd/%d opens, but its target cannot be named (error %lu)", fd, GetLastError());
    CloseHandle(opened);
}

bool under_wine() {
    return GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "wine_get_version") != nullptr;
}

// One shared handle, and what can be made of it.
void probe(ID3D12Device *device, const char *label, ID3D12Resource *resource) {
    say("");
    say("== %s", label);
    HANDLE handle = nullptr;
    const HRESULT made = device->CreateSharedHandle(resource, nullptr, GENERIC_ALL, nullptr, &handle);
    if (FAILED(made) || !handle) {
        say("  CreateSharedHandle failed: %#lx", (unsigned long)made);
        return;
    }
    say("  CreateSharedHandle: handle %p, object type \"%s\"", handle, type_name(handle).c_str());

    ID3D12Resource *again = nullptr;
    const HRESULT opened = device->OpenSharedHandle(handle, IID_PPV_ARGS(&again));
    say("  OpenSharedHandle on it: %s (%#lx)", SUCCEEDED(opened) ? "works" : "fails", (unsigned long)opened);
    if (again) again->Release();

    if (!under_wine()) {
        say("  Not under Wine: no Unix descriptor to ask for.");
        CloseHandle(handle);
        return;
    }

    static HandleToFdFn handle_to_fd = (HandleToFdFn)GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "wine_server_handle_to_fd");
    if (!handle_to_fd) {
        say("  ntdll does not export wine_server_handle_to_fd: no way found from here to the descriptor.");
        CloseHandle(handle);
        return;
    }

    // The access the server wants for a descriptor depends on the object.
    const unsigned int accesses[] = {0, FILE_READ_DATA, FILE_READ_DATA | FILE_WRITE_DATA, SYNCHRONIZE, GENERIC_ALL};
    bool got = false;
    for (unsigned int access : accesses) {
        int fd = -1;
        unsigned int options = 0;
        const NT_STATUS status = handle_to_fd(handle, access, &fd, &options);
        if (status >= 0) {
            say("  wine_server_handle_to_fd(access %#x): descriptor %d (options %#x)", access, fd, options);
            if (!got) describe_fd(fd);
            got = true;
        } else {
            say("  wine_server_handle_to_fd(access %#x): status %#lx", access, (unsigned long)status);
        }
    }
    say("  RESULT %s: %s", label, got ? "a Unix descriptor can be had" : "NO descriptor from this handle");
    // The descriptors stay open: they are the thing to look at.
    CloseHandle(handle);
}

} // namespace

int main(int argc, char **argv) {
    int hold = 0;
    for (int i = 1; i + 1 < argc; ++i)
        if (!strcmp(argv[i], "--hold")) hold = atoi(argv[i + 1]);

    wchar_t path[MAX_PATH];
    GetModuleFileNameW(nullptr, path, MAX_PATH);
    std::wstring log_path = path;
    log_path = log_path.substr(0, log_path.find_last_of(L"\\/") + 1) + L"dlss5nr_interop_probe.log";
    g_log = _wfopen(log_path.c_str(), L"w");

    say("dlss5nr interop probe");
    if (WineVersionFn version = (WineVersionFn)GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "wine_get_version"))
        say("Wine %s", version());
    else
        say("Not Wine (native Windows)");
    for (const char *name : {"STEAM_COMPAT_DATA_PATH", "STEAM_COMPAT_TOOL_PATHS", "PROTON_VERSION", "WINEPREFIX", "WINEDLLOVERRIDES",
                             "VKD3D_CONFIG", "VKD3D_FILTER_DEVICE_NAME"}) {
        char value[1024];
        if (GetEnvironmentVariableA(name, value, sizeof value)) say("%s=%s", name, value);
    }

    ID3D12Device *device = nullptr;
    const HRESULT created = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device));
    if (FAILED(created)) {
        say("D3D12CreateDevice failed: %#lx", (unsigned long)created);
        return 1;
    }
    {
        LUID luid = device->GetAdapterLuid();
        IDXGIFactory4 *factory = nullptr;
        IDXGIAdapter1 *adapter = nullptr;
        if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) &&
            SUCCEEDED(factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter)))) {
            DXGI_ADAPTER_DESC1 description{};
            adapter->GetDesc1(&description);
            say("Adapter: %s", narrow(description.Description, wcslen(description.Description)).c_str());
        }
        if (adapter) adapter->Release();
        if (factory) factory->Release();
    }

    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;

    // A buffer, which is what ROCm imports.
    D3D12_RESOURCE_DESC buffer{};
    buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer.Width = 4u << 20;
    buffer.Height = buffer.DepthOrArraySize = buffer.MipLevels = 1;
    buffer.SampleDesc.Count = 1;
    buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    buffer.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ID3D12Resource *shared_buffer = nullptr;
    HRESULT hr = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_SHARED, &buffer, D3D12_RESOURCE_STATE_COMMON,
                                                 nullptr, IID_PPV_ARGS(&shared_buffer));
    if (FAILED(hr)) say("Creating a shared buffer failed: %#lx", (unsigned long)hr);
    else probe(device, "shared buffer, 4 MB", shared_buffer);

    // A texture, for comparison: what the runtime shares today.
    D3D12_RESOURCE_DESC texture{};
    texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texture.Width = 256;
    texture.Height = 256;
    texture.DepthOrArraySize = texture.MipLevels = 1;
    texture.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    texture.SampleDesc.Count = 1;
    texture.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    texture.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ID3D12Resource *shared_texture = nullptr;
    hr = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_SHARED, &texture, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                         IID_PPV_ARGS(&shared_texture));
    if (FAILED(hr)) say("Creating a shared texture failed: %#lx", (unsigned long)hr);
    else probe(device, "shared texture, 256 x 256 RGBA16F", shared_texture);

    if (hold > 0) {
        say("");
        say("Holding for %d s. From a terminal:  ls -l /proc/$(pgrep -f interop_probe | head -1)/fd", hold);
        Sleep(hold * 1000);
    }
    say("");
    say("Done. The same text is in dlss5nr_interop_probe.log beside the program.");
    if (shared_buffer) shared_buffer->Release();
    if (shared_texture) shared_texture->Release();
    device->Release();
    if (g_log) fclose(g_log);
    return 0;
}
