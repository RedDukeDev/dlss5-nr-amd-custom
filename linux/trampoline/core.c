// The trampoline: amdhip64_7.dll for a Windows program running under Wine or
// Proton, which is not AMD's Windows runtime. Every HIP function ZLUDA uses is
// exported from here (generated/exports.c), and does nothing but pack its
// arguments and pass them to dlssnr_call, which hands them to the native
// library -- the bridge -- loaded into the same process.
//
// How the call crosses. Wine runs a Windows program and the native code under
// it in one process, and keeps a door between the two: ntdll.dll exports
// __wine_unix_call_dispatcher, which takes a handle -- the address of a table of
// native functions -- a number in that table, and a pointer, switches to the
// native stack and calls the function with the pointer. The bridge is a native
// library, preloaded into the process, whose table this is. All that is left to
// find is where the bridge keeps the address of its table: it writes it to a
// file named after its own process, and this DLL reads that file.
//
// For testing without Wine, DLSSNR_BRIDGE_DLL names a Windows build of the
// bridge, which is called directly.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core.h"

typedef LONG (WINAPI *dispatcher_t)(UINT64 handle, unsigned int code, void *args);
typedef LONG (*direct_t)(unsigned int code, void *args);

#define DLSSNR_TAG 0x31444252534c4444ull   // the bridge's, bridge/core.c

static INIT_ONCE g_once = INIT_ONCE_STATIC_INIT;
static dispatcher_t g_dispatch;
static UINT64 g_handle;
static direct_t g_direct;
static int g_ready;

static void say(const char *what) {
    char line[512];
    snprintf(line, sizeof line, "[dlss5nr trampoline] %s\n", what);
    OutputDebugStringA(line);
    fputs(line, stderr);
}

// A native file read through the drive Wine gives the whole filesystem.
static int read_unix_file(const char *path, char *into, size_t size) {
    static const char *const kRoots[] = {"Z:", "\\\\?\\unix"};
    for (int i = 0; i < 2; ++i) {
        char name[512];
        snprintf(name, sizeof name, "%s%s", kRoots[i], path);
        for (char *c = name; *c; ++c)
            if (*c == '/') *c = '\\';
        HANDLE f = CreateFileA(name, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING,
                               FILE_ATTRIBUTE_NORMAL, NULL);
        if (f == INVALID_HANDLE_VALUE) continue;
        DWORD got = 0;
        BOOL ok = ReadFile(f, into, (DWORD)size - 1, &got, NULL);
        CloseHandle(f);
        if (ok && got) {
            into[got] = 0;
            return 1;
        }
    }
    return 0;
}

static BOOL CALLBACK initialise(PINIT_ONCE once, PVOID parameter, PVOID *context) {
    (void)once;
    (void)parameter;
    (void)context;

    char test[MAX_PATH];
    if (GetEnvironmentVariableA("DLSSNR_BRIDGE_DLL", test, sizeof test)) {
        HMODULE bridge = LoadLibraryA(test);
        g_direct = bridge ? (direct_t)GetProcAddress(bridge, "dlssnr_bridge_call") : NULL;
        if (!g_direct) say("DLSSNR_BRIDGE_DLL could not be loaded or has no dlssnr_bridge_call");
        g_ready = g_direct != NULL;
        return TRUE;
    }

    HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    void **door = ntdll ? (void **)GetProcAddress(ntdll, "__wine_unix_call_dispatcher") : NULL;
    if (!door || !*door) {
        say("this is not Wine: ntdll.dll has no __wine_unix_call_dispatcher");
        return TRUE;
    }
    g_dispatch = (dispatcher_t)*door;

    // The address of the bridge's table. Every process of the session has
    // written a file with the address of its own tag, and which one is this
    // process's is not told by the number in its name: /proc/self, read from
    // here, is the wineserver's. So each is tried, and the one whose address
    // holds the tag in this process's memory is the one.
    char text[256];
    unsigned long long handle = 0;
    WIN32_FIND_DATAA found;
    HANDLE search = FindFirstFileA("Z:\\tmp\\dlss5nr-bridge-*.handle", &found);
    if (search != INVALID_HANDLE_VALUE) {
        do {
            char path[MAX_PATH];
            snprintf(path, sizeof path, "/tmp/%s", found.cFileName);
            if (!read_unix_file(path, text, sizeof text)) continue;
            const UINT64 tag = _strtoui64(text, NULL, 0);
            if (!tag) continue;
            // Most of these addresses belong to other processes and are not mapped
            // here. They are read with ReadProcessMemory, which fails with an error
            // inside the system call: IsBadReadPtr takes an access violation and
            // catches it, and a game with a vectored exception handler (Cyberpunk
            // 2077's) sees the exception first, takes it for a crash and freezes
            // the process writing a dump.
            UINT64 words[2] = {0, 0};
            SIZE_T got = 0;
            if (!ReadProcessMemory(GetCurrentProcess(), (LPCVOID)(uintptr_t)tag, words, sizeof words, &got) ||
                got != sizeof words || words[0] != DLSSNR_TAG)
                continue;
            handle = words[1];
            break;
        } while (FindNextFileA(search, &found));
        FindClose(search);
    }
    if (!handle && GetEnvironmentVariableA("DLSSNR_UNIX_HANDLE", text, sizeof text))
        handle = _strtoui64(text, NULL, 0);
    if (!handle) {
        say("the native bridge is not loaded: no handle file holds this process's tag (is LD_PRELOAD set?)");
        return TRUE;
    }
    g_handle = handle;
    g_ready = 1;
    return TRUE;
}

int dlssnr_call(int code, void *args) {
    InitOnceExecuteOnce(&g_once, initialise, NULL, NULL);
    if (!g_ready) return 0;
    if (g_direct) return g_direct((unsigned int)code, args) == 0;
    return g_dispatch(g_handle, (unsigned int)code, args) == 0;
}

BOOL WINAPI DllMain(HINSTANCE module, DWORD reason, LPVOID reserved) {
    (void)module;
    (void)reason;
    (void)reserved;
    return TRUE;
}
