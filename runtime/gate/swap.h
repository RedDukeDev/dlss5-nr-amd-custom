// Running a modified copy of one kernel's PTX inside the real process, on the real inputs, and reading back what
// it leaves in a debug buffer. This is how the inside of a kernel that reads textures is observed: the PTX is
// patched (replay_gpu.py's instrument_mma, or anything that stores a register to the extra pointer), the gate
// swaps it in for one launch, and the buffer comes out beside the launch.
//
//   DLSS5NR_GATE_SWAP_KERNEL=<name>   which kernel
//   DLSS5NR_GATE_SWAP_PTX=<file>      the replacement: a standalone module with that one entry
//   DLSS5NR_GATE_SWAP_DEBUG=<bytes>   size of the debug buffer; the replacement has one more 8-byte parameter,
//                                     right after the original's bytes, holding its address
//   DLSS5NR_GATE_SWAP_LAUNCH=<n>      the launch it stands in for (default 1); the original runs every other
//   DLSS5NR_GATE_SWAP_BLOCK=<x,y>     patched PTX reads these two numbers from the first 8 bytes of the buffer
//                                     (the window to log), written before the launch
//   DLSS5NR_GATE_SWAP_OUT=<file>      where the buffer goes after the launch (default <ptx>.debug.bin)
//
// Nothing here is derived from the network; the replacement is whatever the user made of their own copy.
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <windows.h>

namespace swap {

struct State {
    bool wanted = false;
    char name[128] = {};
    char ptx[MAX_PATH] = {};
    char out[MAX_PATH + 16] = {};
    size_t debug_bytes = 0;
    int launch = 1, seen = 0;
    unsigned block[2] = {0, 0};
    void *original = nullptr;      // the handle the caller was given
    void *replacement = nullptr;   // the function of the replacement module
    void *debug = nullptr;         // device buffer
};

inline State &state() {
    static State s = [] {
        State s;
        s.wanted = GetEnvironmentVariableA("DLSS5NR_GATE_SWAP_KERNEL", s.name, sizeof s.name) > 0 &&
                   GetEnvironmentVariableA("DLSS5NR_GATE_SWAP_PTX", s.ptx, sizeof s.ptx) > 0;
        char v[32] = {};
        if (GetEnvironmentVariableA("DLSS5NR_GATE_SWAP_DEBUG", v, sizeof v)) s.debug_bytes = (size_t)_strtoui64(v, nullptr, 10);
        if (GetEnvironmentVariableA("DLSS5NR_GATE_SWAP_LAUNCH", v, sizeof v)) s.launch = atoi(v);
        if (GetEnvironmentVariableA("DLSS5NR_GATE_SWAP_BLOCK", v, sizeof v)) sscanf(v, "%u,%u", &s.block[0], &s.block[1]);
        if (!GetEnvironmentVariableA("DLSS5NR_GATE_SWAP_OUT", s.out, sizeof s.out)) snprintf(s.out, sizeof s.out, "%s.debug.bin", s.ptx);
        return s;
    }();
    return s;
}

} // namespace swap
