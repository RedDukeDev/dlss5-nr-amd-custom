// Smoke test of the DirectInput hook: our d3d9.dll is loaded, a mouse device is created and read.
#define DIRECTINPUT_VERSION 0x0800
#include <windows.h>
#include <dinput.h>
#include <stdio.h>
#pragma comment(lib, "dinput8.lib")
#pragma comment(lib, "dxguid.lib")
int main() {
    LoadLibraryW(L"d3d9.dll");   // the bridge's: its DllMain installs the hook
    WNDCLASSA wc = {};
    wc.lpfnWndProc = DefWindowProcA;
    wc.hInstance = GetModuleHandleA(nullptr);
    wc.lpszClassName = "di_smoke";
    RegisterClassA(&wc);
    HWND w = CreateWindowA("di_smoke", "di_smoke", WS_OVERLAPPEDWINDOW, 0, 0, 100, 100, nullptr, nullptr, wc.hInstance, nullptr);
    IDirectInput8A *di = nullptr;
    HRESULT hr = DirectInput8Create(wc.hInstance, DIRECTINPUT_VERSION, IID_IDirectInput8A, (void **)&di, nullptr);
    printf("DirectInput8Create: %08lx\n", hr);
    IDirectInputDevice8A *mouse = nullptr;
    hr = di->CreateDevice(GUID_SysMouse, &mouse, nullptr);
    printf("CreateDevice: %08lx\n", hr);
    mouse->SetDataFormat(&c_dfDIMouse2);
    mouse->SetCooperativeLevel(w, DISCL_BACKGROUND | DISCL_NONEXCLUSIVE);
    mouse->Acquire();
    for (int i = 0; i < 3; ++i) {
        DIMOUSESTATE2 s = {};
        hr = mouse->GetDeviceState(sizeof s, &s);
        printf("GetDeviceState: %08lx x=%ld y=%ld z=%ld b0=%d\n", hr, s.lX, s.lY, s.lZ, s.rgbButtons[0]);
        Sleep(50);
    }
    return 0;
}
