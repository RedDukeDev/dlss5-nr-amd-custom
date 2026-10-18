// The protocol between the 32-bit client (a d3d9.dll in a 32-bit game) and the 64-bit host
// (dlss5nr_host.exe, which runs the network): one file mapping and one event, both named after
// the client's process id.
//
//   Header   | capture: the game's frame, BGRA8 | result: the network's answer, as a gain map
//
// The client writes a capture when the host says it is ready for one, and wakes the host. The
// host runs the network on it and writes the result, a gain per channel (what the answer did to
// each pixel of the capture), then marks itself ready again. The client applies the gain to the
// frame it has by then, not to the one that was captured, so the picture the player sees is never
// older than the game's own.
#pragma once
#include <stdint.h>

namespace bridge {

constexpr uint32_t kMagic = 0x42524E44; // 'DNRB'
constexpr uint32_t kVersion = 4;
constexpr uint32_t kHeaderBytes = 4096;
constexpr uint32_t kMaxWidth = 4096, kMaxHeight = 2304;
constexpr uint32_t kAlign = 65536;

// Gains are stored as value / kGainRange of the channel's range: 0.5 is a gain of 1.
constexpr float kGainRange = 2.0f;

// A setting the user left at "auto", for the host to leave at the runtime's default.
constexpr float kAutoFloat = -1.0e9f;
constexpr int32_t kAutoInt = INT32_MIN;

enum HostState : int32_t { kStarting = 0, kRunning = 1, kFailed = 2 };

struct Settings {
    int32_t enabled;
    float resolutionScale;
    int32_t style;
    float intensity, localTone, localStructure, skinStructure;
    float detailStrength, colourStrength, maxRatio;
    int32_t passes, keepLocalTone, preset;
};

struct Header {
    uint32_t magic, version;
    uint32_t clientPid;

    // The geometry of the capture, set by the client before it signals.
    uint32_t width, height;
    // The geometry of the result, set by the host when it writes one.
    uint32_t resultWidth, resultHeight;

    volatile long hostState;   // HostState
    volatile long hostReady;   // 1: the host is idle and wants a capture
    volatile long captureSeq;  // the client's count of captures
    volatile long resultSeq;   // the host's count of results
    volatile long settingsSeq; // the client's count of settings changes
    volatile long shutdown;    // the client is going away

    float evaluationMs;           // the network's last run, from the host
    float evaluationsPerSecond;
    uint32_t networkWidth, networkHeight;
    uint32_t nativeKernels;
    Settings settings;            // what the user set: kAuto where nothing was
    Settings effective;           // what the host runs with, the defaults filled in
    char message[256];
};

inline uint64_t capture_bytes(uint32_t w, uint32_t h) { return (uint64_t)w * h * 4; }
inline uint64_t result_bytes(uint32_t w, uint32_t h) { return (uint64_t)w * h * 8; }
inline uint64_t round_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }
inline uint64_t capture_offset() { return kHeaderBytes; }
inline uint64_t result_offset(uint32_t w, uint32_t h) {
    return kHeaderBytes + round_up(capture_bytes(w, h), kAlign);
}
inline uint64_t total_bytes(uint32_t w, uint32_t h) {
    return result_offset(w, h) + round_up(result_bytes(w, h), kAlign);
}
inline uint64_t max_total_bytes() { return total_bytes(kMaxWidth, kMaxHeight); }

} // namespace bridge
