// The menu of the 32-bit bridge: the neural rendering part of OptiScaler's menu, drawn with the same
// Dear ImGui in the game's own Direct3D 9 device, opened with Insert. OptiScaler itself is 64-bit
// and cannot be loaded into a 32-bit game, so this is what stands in for its menu there.
#pragma once
#include <windows.h>
#include "../common/protocol.h"

struct IDirect3DDevice9;
struct ID3D11Device;
struct ID3D11DeviceContext;
struct ID3D11RenderTargetView;

// Implemented by the client: a setting the menu changed. `key` is the [NeuralRendering] key and
// `text` its value; `save` is false while a slider is still being dragged (the value goes to the
// host at once, the file is written when the drag ends).
// Options of the client itself, not of the network: how a result is applied.
struct BridgeOptions {
    bool reduceTrailing;   // fade the gain where the scene has changed since the network saw it
    float tolerance;       // how much a pixel may change before that starts, 0..1
    bool wait;             // the game waits for the network, so a result is never old
    bool follow;           // carry the result along the motion estimated from the pictures
};
BridgeOptions &bridge_options();
void bridge_setting_changed(const wchar_t *key, const wchar_t *text, bool save);
bridge::Settings &bridge_user_settings();

namespace menu {

// What the menu shows besides the settings.
struct Status {
    bool enabled;
    const bridge::Header *header;   // null until the host is started
    unsigned frames, sent, applied;
    const char *problem;            // why the bridge cannot run, or null
    bool mask_available;            // Reduce trailing and Follow motion exist in this client
};

// What does not depend on the graphics API (menu.cpp): the window (Insert, the mouse messages), the
// input, the menu itself. A client's own part (menu_dx9.cpp, menu_dx11.cpp) makes ImGui's renderer and
// calls these around it.
void attach(HWND window);
// Starts the frame if the menu is open: false when there is nothing to draw. `new_frame` is the
// renderer's own, called first.
bool begin(unsigned width, unsigned height, const Status &status, void (*new_frame)());
// Ends it; the draw data (ImDrawData *) for the renderer.
void *end();

// Direct3D 9 (menu_dx9.cpp).
void init(IDirect3DDevice9 *device, HWND window);
// Draws the menu over the backbuffer, if it is open. `width` and `height` are the backbuffer's.
void frame(IDirect3DDevice9 *device, unsigned width, unsigned height, const Status &status);
// The device is about to be reset: what the menu holds in video memory has to go.
void device_lost();
void device_restored();

// Direct3D 11 (menu_dx11.cpp).
void init(ID3D11Device *device, ID3D11DeviceContext *context, HWND window);
void frame(ID3D11DeviceContext *context, ID3D11RenderTargetView *target, unsigned width, unsigned height, const Status &status);

bool visible();
void open();   // for testing: as if Insert had been pressed
// The mouse, from DirectInput (relative movement, the wheel, the buttons as a bit mask: 1 left, 2 right,
// 4 middle): while the menu is open it is the menu's cursor that moves.
void feed_mouse(long dx, long dy, long wheel, unsigned buttons, bool from_directinput);

} // namespace menu
