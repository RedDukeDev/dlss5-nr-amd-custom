// The menu of the 32-bit bridge: the neural rendering part of OptiScaler's menu, drawn with the same
// Dear ImGui in the game's own Direct3D 9 device, opened with Insert. OptiScaler itself is 64-bit
// and cannot be loaded into a 32-bit game, so this is what stands in for its menu there.
#pragma once
#include <windows.h>
#include <d3d9.h>
#include "../common/protocol.h"

// Implemented by the client: a setting the menu changed. `key` is the [NeuralRendering] key and
// `text` its value; `save` is false while a slider is still being dragged (the value goes to the
// host at once, the file is written when the drag ends).
void bridge_setting_changed(const wchar_t *key, const wchar_t *text, bool save);
bridge::Settings &bridge_user_settings();

namespace menu {

// What the menu shows besides the settings.
struct Status {
    bool enabled;
    const bridge::Header *header;   // null until the host is started
    unsigned frames, sent, applied;
    const char *problem;            // why the bridge cannot run, or null
};

void init(IDirect3DDevice9 *device, HWND window);
bool visible();
void open();   // for testing: as if Insert had been pressed
// Draws the menu over the backbuffer, if it is open. `width` and `height` are the backbuffer's.
void frame(IDirect3DDevice9 *device, unsigned width, unsigned height, const Status &status);
// The device is about to be reset: what the menu holds in video memory has to go.
void device_lost();
void device_restored();

} // namespace menu
