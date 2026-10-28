// A/B in the running process: a native kernel against the translated one, on the very memory and descriptors
// the network handed them -- which a capture cannot do for kernels that read textures or write surfaces.
//
//   DLSS5NR_GATE_AB=<kernel name>        which kernel
//   DLSS5NR_GATE_AB_LAUNCHES=<n>         how many of its launches to check (default 1)
//
// For each checked launch the gate takes a snapshot of every allocation the arguments point into, runs the
// translated kernel, takes a second snapshot, puts the first one back, runs the native kernel and compares its
// result with the translated one's. The outcome is in the gate log, per allocation and per pointer of the
// argument buffer. Only memory the arguments point to is compared: what a kernel writes through a surface
// has to be checked through the buffers it fills beside it, or by a later kernel that reads it.
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <windows.h>

namespace ab {

struct Region {
    unsigned long long base = 0;
    size_t size = 0;
    std::vector<unsigned char> data;
    std::vector<std::pair<size_t, size_t>> pointers;   // (offset in the arguments, offset in the region)
};

inline bool enabled(const char *name) {
    static char wanted[128];
    static const bool have = GetEnvironmentVariableA("DLSS5NR_GATE_AB", wanted, sizeof wanted) > 0;
    static int left = [] {
        char v[16] = {};
        return GetEnvironmentVariableA("DLSS5NR_GATE_AB_LAUNCHES", v, sizeof v) ? atoi(v) : 1;
    }();
    if (!have || !name || strcmp(wanted, name) != 0 || left <= 0) return false;
    --left;
    return true;
}

// The allocations the argument buffer points into.
inline std::vector<Region> regions_of(const unsigned char *args, size_t size) {
    std::vector<Region> out;
    for (size_t at = 0; at + 8 <= size; at += 8) {
        unsigned long long value;
        memcpy(&value, args + at, 8);
        if (value < 0x10000) continue;
        void *base = nullptr;
        size_t length = 0;
        if (capture::hip().range((void **)&base, &length, (void *)value) != 0 || !base) continue;
        Region *found = nullptr;
        for (Region &r : out)
            if (r.base == (unsigned long long)base) found = &r;
        if (!found) {
            out.push_back({});
            found = &out.back();
            found->base = (unsigned long long)base;
            found->size = length < (512u << 20) ? length : (512u << 20);
        }
        found->pointers.push_back({at, (size_t)(value - found->base)});
    }
    return out;
}

inline void take(std::vector<Region> &regions) {
    capture::hip().sync();
    for (Region &r : regions) {
        r.data.resize(r.size);
        capture::hip().to_host(r.data.data(), (void *)r.base, r.size);
    }
}

inline void restore(const std::vector<Region> &regions) {
    static int (*to_device)(void *, const void *, size_t) = [] {
        HMODULE hip = GetModuleHandleW(L"amdhip64_7.dll");
        return hip ? (int (*)(void *, const void *, size_t))GetProcAddress(hip, "hipMemcpyHtoD") : nullptr;
    }();
    for (const Region &r : regions) to_device((void *)r.base, r.data.data(), r.size);
    capture::hip().sync();
}

} // namespace ab
