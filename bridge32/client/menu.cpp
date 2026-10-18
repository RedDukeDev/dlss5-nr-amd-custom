#include "menu.h"
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include "imgui.h"
#include <windowsx.h>

namespace menu {
namespace {

bool g_inited = false;
bool g_open = false;
HWND g_window = nullptr;
WNDPROC g_original = nullptr;
// The menu's own mouse. Games that read the mouse through DirectInput send no mouse messages and
// keep the cursor of Windows still, so the cursor here is a virtual one: DirectInput moves it
// (feed_mouse), or else, for a game that does send messages, the window does.
float g_scale = 0.0f;
float g_mx = 0.0f, g_my = 0.0f;
unsigned g_buttons = 0, g_buttons_seen = 0;
float g_wheel = 0.0f;
bool g_directinput = false;
bool g_message_mouse = false;
float g_message_x = 0.0f, g_message_y = 0.0f;
unsigned g_width = 0, g_height = 0;
LARGE_INTEGER g_last_time = {};

void toggle() {
    g_open = !g_open;
    g_buttons = 0;
    g_wheel = 0.0f;
    if (g_open && g_width) {
        g_mx = g_width / 2.0f;
        g_my = g_height / 2.0f;
    }
}

// Insert opens and closes the menu; while it is open it keeps the mouse and keyboard to itself.
LRESULT CALLBACK window_proc(HWND window, UINT message, WPARAM wparam, LPARAM lparam) {
    if (message == WM_KEYDOWN && wparam == VK_INSERT) {
        if (!(lparam & (1 << 30))) toggle();
        return 0;
    }
    if (message == WM_KEYUP && wparam == VK_INSERT) return 0;
    if (g_open) {
        switch (message) {
        case WM_SETCURSOR:
            SetCursor(nullptr);
            return TRUE;
        case WM_INPUT:
            return DefWindowProcW(window, message, wparam, lparam);
        case WM_MOUSEMOVE:
            g_message_mouse = true;
            g_message_x = (float)GET_X_LPARAM(lparam);
            g_message_y = (float)GET_Y_LPARAM(lparam);
            return 0;
        case WM_LBUTTONDOWN: case WM_LBUTTONDBLCLK: g_buttons |= 1; return 0;
        case WM_LBUTTONUP: g_buttons &= ~1u; return 0;
        case WM_RBUTTONDOWN: case WM_RBUTTONDBLCLK: g_buttons |= 2; return 0;
        case WM_RBUTTONUP: g_buttons &= ~2u; return 0;
        case WM_MBUTTONDOWN: case WM_MBUTTONDBLCLK: g_buttons |= 4; return 0;
        case WM_MBUTTONUP: g_buttons &= ~4u; return 0;
        case WM_MOUSEWHEEL: g_wheel += (float)GET_WHEEL_DELTA_WPARAM(wparam) / 120.0f; return 0;
        case WM_KEYDOWN: case WM_KEYUP: return 0;
        case WM_CHAR:
            if (wparam > 0 && wparam < 0x10000) ImGui::GetIO().AddInputCharacter((unsigned int)wparam);
            return 0;
        case WM_XBUTTONDOWN: case WM_XBUTTONUP: case WM_XBUTTONDBLCLK: case WM_MOUSEHWHEEL:
            return 0;
        }
    }
    return CallWindowProcW(g_original, window, message, wparam, lparam);
}

void tooltip(const char *text) {
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) ImGui::SetTooltip("%s", text);
}

void put(const wchar_t *key, const wchar_t *text, bool save) { bridge_setting_changed(key, text, save); }

void put_float(const wchar_t *key, float value, bool save) {
    wchar_t text[32];
    swprintf(text, 32, L"%.3f", value);
    put(key, text, save);
}

void put_int(const wchar_t *key, int value, bool save) {
    wchar_t text[32];
    swprintf(text, 32, L"%d", value);
    put(key, text, save);
}

// A slider over a user value that may be "auto": it shows the host's effective value then, and
// the first change makes the value the user's.
bool slider(const char *label, const wchar_t *key, float &user, float effective, float low, float high) {
    float value = user > bridge::kAutoFloat / 2 ? user : effective;
    const bool changed = ImGui::SliderFloat(label, &value, low, high, "%.2f");
    const bool done = ImGui::IsItemDeactivatedAfterEdit();
    if (changed) user = value;
    if (changed || done) put_float(key, user > bridge::kAutoFloat / 2 ? user : value, done);
    return changed;
}

void draw(const Status &status, float scale) {
    ImGui::SetNextWindowPos(ImVec2(20.0f * scale, 20.0f * scale), ImGuiCond_FirstUseEver);
    ImGui::Begin("DLSS 5 Neural Rendering (32-bit game)", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    bridge::Settings &user = bridge_user_settings();
    const bridge::Header *h = status.header;
    const bool running = h && h->hostState == bridge::kRunning;
    const bridge::Settings &eff = h ? h->effective : user;

    ImGui::SeparatorText("Neural Rendering (DLSS 5)");

    bool enabled = status.enabled;
    if (ImGui::Checkbox("Enable##nr", &enabled)) {
        user.enabled = enabled ? 1 : 0;
        put(L"Enabled", enabled ? L"true" : L"false", true);
    }
    tooltip("DLSS 5 Neural Rendering on this GPU. This game is 32-bit, so the network runs in a\n"
            "separate 64-bit process and its result is applied to the frame the game is showing:\n"
            "a slow network lowers how often the image is enhanced, not the frame rate");

    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
    ImGui::TextWrapped("The network runs on the finished image, without motion vectors: its result "
                       "trails behind whatever moves.");
    ImGui::PopStyleColor();

    if (!status.enabled) {
        ImGui::TextDisabled("Off");
    } else if (status.problem) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
        ImGui::TextWrapped("%s", status.problem);
        ImGui::PopStyleColor();
    } else if (!h) {
        ImGui::TextDisabled("Starting the host...");
    } else if (h->hostState == bridge::kStarting) {
        ImGui::TextDisabled("Loading the network: %s", h->message);
    } else if (h->hostState == bridge::kFailed) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
        ImGui::TextWrapped("Failed: %s", h->message);
        ImGui::PopStyleColor();
    } else {
        ImGui::TextWrapped("Network %ux%u, %.1f ms, %.1f per second, %u native kernels", h->networkWidth, h->networkHeight,
                           h->evaluationMs, h->evaluationsPerSecond, h->nativeKernels);
        ImGui::TextDisabled("%u frames, %u sent to the network, %u results applied", status.frames, status.sent, status.applied);
    }

    ImGui::BeginDisabled(!status.enabled || !running);

    {
        BridgeOptions &options = bridge_options();
        if (status.mask_available) {
        if (ImGui::Checkbox("Reduce trailing##nr", &options.reduceTrailing))
            put(L"BridgeReduceTrailing", options.reduceTrailing ? L"true" : L"false", true);
        tooltip("A result is a few frames old when it arrives. Applied to a scene that has moved since,\n"
                "it shows as a trail behind whatever moves. This takes it away wherever the picture has\n"
                "changed since the network saw it: what moves is left as the game drew it");
        if (ImGui::Checkbox("Follow motion (experimental)##nr", &options.follow))
            put(L"BridgeFollowMotion", options.follow ? L"true" : L"false", true);
        tooltip("There are no motion vectors in this game, so they are estimated from the pictures: where each\n"
                "block of the picture now was in the one the network saw. The result is carried along them,\n"
                "which keeps the effect on what moves, as long as it moves by less than a few tens of pixels\n"
                "between the network seeing the picture and the result arriving. Needs Reduce trailing");
        ImGui::BeginDisabled(!options.reduceTrailing);
        {
            float tolerance = options.tolerance;
            const bool changed = ImGui::SliderFloat("Tolerance##nr", &tolerance, 0.01f, 0.2f, "%.3f");
            const bool done = ImGui::IsItemDeactivatedAfterEdit();
            if (changed) options.tolerance = tolerance;
            if (changed || done) put_float(L"BridgeTrailingTolerance", tolerance, done);
            tooltip("How much a pixel may change before the effect fades there. Lower removes more of the\n"
                    "trail, but also the effect on slow movement; higher keeps more of both");
        }
        ImGui::EndDisabled();
        }
        slider("Temporal smoothing##nr", L"BridgeSmoothing", user.smoothing, eff.smoothing, 0.0f, 1.0f);
        tooltip("The network decides some of its detail and tone anew with every picture, which shows as a\n"
                "flicker. Where the game's picture has not changed since the last one, this holds back that\n"
                "much of the result's change from the one before; where it has, nothing is held back.\n"
                "Higher is steadier, and slower to follow a slow fade");
        if (ImGui::Checkbox("Wait for the network##nr", &options.wait))
            put(L"BridgeWaitForNetwork", options.wait ? L"true" : L"false", true);
        tooltip("The game waits for the network on every frame and its answer is applied to that very\n"
                "frame: no trail at all, but the frame rate becomes the network's");
    }

    slider("Network resolution", L"ResolutionScale", user.resolutionScale, eff.resolutionScale, 0.25f, 1.0f);
    tooltip("The network's resolution as a fraction of the game's.\nIts cost follows its pixels");

    // The presets are numbered in the order the library carries its networks.
    const char *presets[] = {"Library default", "Preset 1 (larger)", "Preset 2 (smaller, faster)"};
    int preset = std::clamp(user.preset != bridge::kAutoInt ? user.preset : eff.preset, 0, 2);
    if (ImGui::Combo("Network##nr", &preset, presets, 3)) {
        user.preset = preset;
        put_int(L"Preset", preset, true);
    }
    tooltip("Which network the library runs. Library default is the one NVIDIA's library picks.\n"
            "Preset 1 is the larger network of the earlier versions of the library: slower.\n"
            "Preset 2 is the smaller one. A change restarts the network");

    const char *styles[] = {"Default", "Natural", "Cinematic"};
    int style = std::clamp(user.style != bridge::kAutoInt ? user.style : eff.style, 0, 2);
    if (ImGui::Combo("Style##nr", &style, styles, 3)) {
        user.style = style;
        put_int(L"Style", style, true);
    }

    // The network caps Intensity at 1 whatever it is given; tone and structure go on responding up to 2.
    slider("Intensity##nr", L"Intensity", user.intensity, eff.intensity, 0.0f, 1.0f);
    slider("Local tone##nr", L"LocalTone", user.localTone, eff.localTone, 0.0f, 2.0f);
    slider("Local structure##nr", L"LocalStructure", user.localStructure, eff.localStructure, 0.0f, 2.0f);

    {
        const float skin = user.skinStructure > bridge::kAutoFloat / 2 ? user.skinStructure : eff.skinStructure;
        bool automatic = skin < 0.0f;
        if (ImGui::Checkbox("Skin structure: auto##nr", &automatic)) {
            user.skinStructure = automatic ? -1.0f : 1.0f;
            put_float(L"SkinStructure", user.skinStructure, true);
        }
        tooltip("The detail the network adds to faces and skin: pores, shading, texture.\n"
                "Auto follows Local structure; untick to set it yourself, where 0 leaves them as the game drew them");
        ImGui::BeginDisabled(automatic);
        float value = automatic ? 1.0f : skin;
        const bool changed = ImGui::SliderFloat("Skin structure##nr", &value, 0.0f, 2.0f, "%.2f");
        const bool done = ImGui::IsItemDeactivatedAfterEdit();
        if (changed) user.skinStructure = value;
        if (changed || done) put_float(L"SkinStructure", value, done);
        ImGui::EndDisabled();
    }

    {
        int passes = user.passes != bridge::kAutoInt ? user.passes : eff.passes;
        const bool changed = ImGui::SliderInt("Passes##nr", &passes, 1, 8);
        const bool done = ImGui::IsItemDeactivatedAfterEdit();
        if (changed) user.passes = passes;
        if (changed || done) put_int(L"Passes", passes, done);
        tooltip("How many times the effect is applied, one on top of the other: each pass works on the\n"
                "result of the one before. More passes give a stronger effect, but too many look artificial\n"
                "and shift the colours. Each pass takes as long as the first");

        ImGui::BeginDisabled(passes < 2);
        bool keep = (user.keepLocalTone != bridge::kAutoInt ? user.keepLocalTone : eff.keepLocalTone) != 0;
        if (ImGui::Checkbox("Keep local tone##nr", &keep)) {
            user.keepLocalTone = keep ? 1 : 0;
            put_int(L"KeepLocalTone", keep ? 1 : 0, true);
        }
        tooltip("Applies Local tone on every pass. Because the passes stack, the lighting is pushed further\n"
                "each time: untick it to apply the local tone only on the first pass");
        ImGui::EndDisabled();
    }

    if (ImGui::CollapsingHeader("Composition##nr")) {
        slider("Detail##nr", L"DetailStrength", user.detailStrength, eff.detailStrength, 0.0f, 2.0f);
        tooltip("How much of the network's light and detail is applied:\n0 the game's image, 1 the network's, above 1 amplified");
        slider("Colour##nr", L"ColourStrength", user.colourStrength, eff.colourStrength, 0.0f, 4.0f);
        tooltip("0 keeps the game's hue, 1 takes the network's, above 1 its colour is pushed further");
        slider("Max ratio##nr", L"MaxRatio", user.maxRatio, eff.maxRatio, 1.0f, 2.0f);
        tooltip("The most a pixel may be brightened or darkened by the network. In this bridge the\n"
                "result travels as a gain between 0 and 2, so this stops there");
    }

    ImGui::EndDisabled();

    ImGui::Separator();
    ImGui::TextDisabled("Insert closes the menu. Changes are saved to OptiScaler.ini.");
    ImGui::End();
}

} // namespace

void attach(HWND window) {
    if (g_inited) return;
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.LogFilename = nullptr;
    ImGui::StyleColorsDark();
    g_window = window;
    g_original = (WNDPROC)SetWindowLongPtrW(window, GWLP_WNDPROC, (LONG_PTR)window_proc);
    g_inited = true;
}

bool visible() { return g_open; }

void feed_mouse(long dx, long dy, long wheel, unsigned buttons, bool from_directinput) {
    if (!g_open) return;
    if (from_directinput) g_directinput = true;
    g_mx += (float)dx;
    g_my += (float)dy;
    g_wheel += (float)wheel / 120.0f;
    g_buttons = buttons;
}

void open() {
    if (!g_open) toggle();
}

bool begin(unsigned width, unsigned height, const Status &status, void (*new_frame)()) {
    if (!g_inited || !g_open) return false;
    new_frame();
    ImGuiIO &io = ImGui::GetIO();
    g_width = width;
    g_height = height;
    io.DisplaySize = ImVec2((float)width, (float)height);
    LARGE_INTEGER now, frequency;
    QueryPerformanceCounter(&now);
    QueryPerformanceFrequency(&frequency);
    io.DeltaTime = g_last_time.QuadPart ? std::max(1.0f / 1000.0f, (float)((double)(now.QuadPart - g_last_time.QuadPart) / (double)frequency.QuadPart)) : 1.0f / 60.0f;
    g_last_time = now;

    // Messages, for a game that sends them: the window's size and the backbuffer's need not be the same.
    if (!g_directinput && g_message_mouse) {
        RECT client = {};
        GetClientRect(g_window, &client);
        g_mx = g_message_x * (float)width / (float)std::max<LONG>(client.right, 1);
        g_my = g_message_y * (float)height / (float)std::max<LONG>(client.bottom, 1);
    }
    g_mx = std::clamp(g_mx, 0.0f, (float)width - 1.0f);
    g_my = std::clamp(g_my, 0.0f, (float)height - 1.0f);
    io.MouseDrawCursor = true;
    io.AddMousePosEvent(g_mx, g_my);
    for (int i = 0; i < 3; ++i)
        if (((g_buttons ^ g_buttons_seen) >> i) & 1) io.AddMouseButtonEvent(i, (g_buttons >> i) & 1);
    g_buttons_seen = g_buttons;
    if (g_wheel != 0.0f) {
        io.AddMouseWheelEvent(0.0f, g_wheel);
        g_wheel = 0.0f;
    }
    const float ch = (float)height;
    const float scale = std::clamp(ch / 1080.0f, 1.0f, 3.0f);
    if (scale != g_scale) {
        ImGuiStyle fresh;
        ImGui::GetStyle() = fresh;
        ImGui::StyleColorsDark();
        ImGui::GetStyle().ScaleAllSizes(scale);
        ImGui::GetStyle().FontScaleMain = scale;
        g_scale = scale;
    }

    ImGui::NewFrame();
    draw(status, scale);
    return true;
}

void *end() {
    ImGui::Render();
    return ImGui::GetDrawData();
}

} // namespace menu
