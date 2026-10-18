// What a game that reads its mouse through DirectInput does to a menu drawn inside it: the cursor
// of Windows is hidden and held still, and no mouse message ever arrives, so clicks and movements
// cannot be had from the window. They can from DirectInput itself. This hooks DirectInput8Create,
// and through it the devices the game creates: while the menu is open their mouse data goes to the
// menu, and what the game reads is empty, so that it does not turn the camera under the menu.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#define DIRECTINPUT_VERSION 0x0800
#include <windows.h>
#include <dinput.h>
#include <string.h>
#include <vector>
#include "input.h"
#include "menu.h"

void bridge_log(const char *format, ...);

namespace input {
namespace {

const GUID kSysMouse = {0x6F1D2B60, 0xD5A0, 0x11CF, {0xBF, 0xC7, 0x44, 0x45, 0x53, 0x54, 0x00, 0x00}};
const GUID kSysKeyboard = {0x6F1D2B61, 0xD5A0, 0x11CF, {0xBF, 0xC7, 0x44, 0x45, 0x53, 0x54, 0x00, 0x00}};

enum Kind { kOther, kMouse, kKeyboard };

struct Device {
    void *object;
    Kind kind;
};
std::vector<Device> g_devices;

Kind kind_of(void *object) {
    for (const Device &d : g_devices)
        if (d.object == object) return d.kind;
    return kOther;
}

typedef HRESULT(WINAPI *Create_t)(HINSTANCE, DWORD, REFIID, LPVOID *, LPUNKNOWN);
typedef HRESULT(STDMETHODCALLTYPE *CreateDevice_t)(void *, REFGUID, void **, IUnknown *);
typedef HRESULT(STDMETHODCALLTYPE *GetState_t)(void *, DWORD, LPVOID);
typedef HRESULT(STDMETHODCALLTYPE *GetData_t)(void *, DWORD, DIDEVICEOBJECTDATA *, LPDWORD, DWORD);

Create_t g_create = nullptr;   // the original, past the hot-patch slot
struct DiHooks {
    void **vtable;
    CreateDevice_t create_device;
};
struct DeviceHooks {
    void **vtable;
    GetState_t get_state;
    GetData_t get_data;
};
std::vector<DiHooks> g_di;
std::vector<DeviceHooks> g_dev;

bool patch(void **vtable, int index, void *hook, void **original) {
    DWORD old;
    if (!VirtualProtect(&vtable[index], sizeof(void *), PAGE_EXECUTE_READWRITE, &old)) return false;
    if (vtable[index] != hook) {
        *original = vtable[index];
        vtable[index] = hook;
    }
    VirtualProtect(&vtable[index], sizeof(void *), old, &old);
    return true;
}

DeviceHooks *device_hooks(void *device) {
    void **vtable = *(void ***)device;
    for (DeviceHooks &h : g_dev)
        if (h.vtable == vtable) return &h;
    return nullptr;
}

// A mouse state read by the game (DIMOUSESTATE or DIMOUSESTATE2): the menu takes its movement and
// buttons, the game is left with none.
void take_mouse_state(void *data, DWORD size) {
    if (size < 16) return;
    LONG *axes = (LONG *)data;
    uint8_t *buttons = (uint8_t *)data + 12;
    unsigned mask = 0;
    for (int i = 0; i < 4; ++i)
        if (buttons[i] & 0x80) mask |= 1u << i;
    menu::feed_mouse(axes[0], axes[1], axes[2], mask, true);
    memset(data, 0, size);
}

HRESULT STDMETHODCALLTYPE hk_get_state(void *device, DWORD size, LPVOID data) {
    DeviceHooks *h = device_hooks(device);
    const HRESULT hr = h->get_state(device, size, data);
    if (SUCCEEDED(hr) && data && menu::visible()) {
        const Kind kind = kind_of(device);
        if (kind == kMouse) take_mouse_state(data, size);
        else if (kind == kKeyboard) memset(data, 0, size);
    }
    return hr;
}

HRESULT STDMETHODCALLTYPE hk_get_data(void *device, DWORD size, DIDEVICEOBJECTDATA *events, LPDWORD count, DWORD flags) {
    DeviceHooks *h = device_hooks(device);
    const HRESULT hr = h->get_data(device, size, events, count, flags);
    if (SUCCEEDED(hr) && events && count && menu::visible()) {
        const Kind kind = kind_of(device);
        if (kind == kMouse) {
            LONG dx = 0, dy = 0, dz = 0;
            static unsigned mask = 0;
            for (DWORD i = 0; i < *count; ++i) {
                const DIDEVICEOBJECTDATA &e = *(const DIDEVICEOBJECTDATA *)((const uint8_t *)events + (size_t)i * size);
                if (e.dwOfs == 0) dx += (LONG)e.dwData;
                else if (e.dwOfs == 4) dy += (LONG)e.dwData;
                else if (e.dwOfs == 8) dz += (LONG)e.dwData;
                else if (e.dwOfs >= 12 && e.dwOfs < 20) {
                    if (e.dwData & 0x80) mask |= 1u << (e.dwOfs - 12);
                    else mask &= ~(1u << (e.dwOfs - 12));
                }
            }
            menu::feed_mouse(dx, dy, dz, mask, true);
        }
        if (kind != kOther && !(flags & DIGDD_PEEK)) *count = 0;
    }
    return hr;
}

HRESULT STDMETHODCALLTYPE hk_create_device(void *di, REFGUID guid, void **device, IUnknown *outer) {
    CreateDevice_t original = nullptr;
    for (DiHooks &h : g_di)
        if (h.vtable == *(void ***)di) original = h.create_device;
    const HRESULT hr = original(di, guid, device, outer);
    if (SUCCEEDED(hr) && device && *device) {
        Kind kind = kOther;
        if (IsEqualGUID(guid, kSysMouse)) kind = kMouse;
        else if (IsEqualGUID(guid, kSysKeyboard)) kind = kKeyboard;
        g_devices.push_back({*device, kind});
        if (kind != kOther && !device_hooks(*device)) {
            DeviceHooks h = {};
            h.vtable = *(void ***)*device;
            patch(h.vtable, 9, (void *)hk_get_state, (void **)&h.get_state);
            patch(h.vtable, 10, (void *)hk_get_data, (void **)&h.get_data);
            g_dev.push_back(h);
        }
        bridge_log("input: a %s device was created", kind == kMouse ? "mouse" : kind == kKeyboard ? "keyboard" : "other");
    }
    return hr;
}

HRESULT WINAPI hk_create(HINSTANCE instance, DWORD version, REFIID riid, LPVOID *out, LPUNKNOWN outer) {
    const HRESULT hr = g_create(instance, version, riid, out, outer);
    if (SUCCEEDED(hr) && out && *out) {
        void **vtable = *(void ***)*out;
        bool known = false;
        for (DiHooks &h : g_di) known |= h.vtable == vtable;
        if (!known) {
            DiHooks h = {};
            h.vtable = vtable;
            patch(vtable, 3, (void *)hk_create_device, (void **)&h.create_device);
            g_di.push_back(h);
        }
    }
    return hr;
}

} // namespace

// Windows' own system DLLs leave a two-byte no-op in front of every function and five bytes of
// padding before it, for exactly this: a short jump back into the padding, which holds a jump to the
// hook, and the function itself runs again from past the no-op.
bool install() {
    HMODULE dll = LoadLibraryW(L"dinput8.dll");
    if (!dll) return false;
    uint8_t *f = (uint8_t *)GetProcAddress(dll, "DirectInput8Create");
    if (!f) return false;
    if (f[0] != 0x8B || f[1] != 0xFF) {
        bridge_log("input: DirectInput8Create has no hot-patch slot (%02x %02x), the menu will not get the mouse", f[0], f[1]);
        return false;
    }
    for (int i = 1; i <= 5; ++i)
        if (f[-i] != 0x90 && f[-i] != 0xCC) {
            bridge_log("input: DirectInput8Create has no padding before it");
            return false;
        }
    DWORD old;
    if (!VirtualProtect(f - 5, 7, PAGE_EXECUTE_READWRITE, &old)) return false;
    g_create = (Create_t)(f + 2);
    f[-5] = 0xE9;
    *(int32_t *)(f - 4) = (int32_t)((uint8_t *)hk_create - f);
    f[0] = 0xEB;
    f[1] = 0xF9;
    VirtualProtect(f - 5, 7, old, &old);
    FlushInstructionCache(GetCurrentProcess(), f - 5, 7);
    bridge_log("input: DirectInput8Create hooked");
    return true;
}

} // namespace input
