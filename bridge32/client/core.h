// What the Direct3D 9 client and the DXGI (Direct3D 10 and 11) client have in common: the log, the
// settings read from OptiScaler.ini, the connection to the 64-bit host (shared memory, events, the
// process itself) and the size the network works at.
#pragma once
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <stdint.h>
#include <string>
#include "../common/protocol.h"
#include "menu.h"

extern HMODULE g_self;
extern std::wstring g_dir;   // the folder of the DLL: the game's own

void log(const char *format, ...);

struct Config {
    bool enabled = false;
    std::wstring host;       // the host's path
    BridgeOptions options = {true, 0.025f, true, false};
    bool open_menu = false;  // BridgeDebugOpenMenu: the menu starts open
    int dump_frame = 0;      // BridgeDebugDumpFrame: write the frame before and after the gain to bmp files
    int dump_series = 0;     // BridgeDebugDumpSeries: this many frames in a row from dump_frame, as dump_s_NNN.bmp
    bridge::Settings settings{};
    FILETIME stamp{};
    std::wstring ini;
};
extern Config g_config;

float read_float(const wchar_t *key, float otherwise);
int32_t read_int(const wchar_t *key, int32_t otherwise);
// Reads the [NeuralRendering] keys OptiScaler.ini has, the same ones the 64-bit games use.
// Returns true when something changed.
bool read_config();

struct Bridge {
    HANDLE mapping = nullptr;
    HANDLE capture_event = nullptr;
    HANDLE result_event = nullptr;
    HANDLE job = nullptr;
    HANDLE process = nullptr;
    bridge::Header *header = nullptr;
    uint8_t *view = nullptr;
    uint32_t view_w = 0, view_h = 0;
    bool failed = false;
    bool started = false;
    int32_t seen_settings = 0;
    int32_t seen_result = 0;

    bool start();
    bool fail(const char *what);
    bool spawn();
    // The view has to cover the buffers of the current size, and no more: this is a 32-bit
    // process, and a game with a large address space has little to spare.
    bool map(uint32_t w, uint32_t h);
    void stop();
};
extern Bridge g_bridge;

// Opens the log (named by the client), finds the game's folder and reads the settings.
void core_init(HMODULE self, const wchar_t *log_name);

// The size the network works at: the user's resolution scale of the frame, in multiples of 8.
void network_size(UINT w, UINT h, UINT &cw, UINT &ch);
