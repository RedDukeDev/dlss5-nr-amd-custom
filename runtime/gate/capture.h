// Capturing one launch: its argument buffer and the device memory it points into, before and after it runs.
//
// What a native kernel is checked against: the translated kernel's own output on the exact bytes a real launch
// was given. Include this in gate.cpp after logf(); cuLaunchKernel makes a capture::Scope around the launch.
//
//   DLSS5NR_GATE_CAPTURE=<kernel name>   which kernel
//   DLSS5NR_GATE_CAPTURE_LAUNCH=<n>      which of its launches, counted from 1 (default 1)
//   DLSS5NR_GATE_CAPTURE_DIR=<dir>       where the files go (must exist)
//   DLSS5NR_GATE_CAPTURE_BYTES=<n>       at most this much of each allocation (default 512 MB)
//
// Files: args.bin (the argument buffer), info.txt (geometry, then one line per allocation the arguments point
// into and one per pointer: offset in the arguments, allocation base, offset of the pointer in it), and
// pre_<base>.bin / post_<base>.bin for each allocation, from its first byte.
//
// Nothing derived from NVIDIA's network belongs in the repository: the files are the user's, on the user's disk.
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <windows.h>

namespace capture {

struct Region {
    unsigned long long base = 0;
    size_t size = 0;
};

struct Hip {
    int (*range)(void **, size_t *, void *) = nullptr;
    int (*to_host)(void *, void *, size_t) = nullptr;
    int (*sync)() = nullptr;
    bool ready = false;
};

inline Hip &hip() {
    static Hip h = [] {
        Hip h;
        HMODULE module = GetModuleHandleW(L"amdhip64_7.dll");
        if (!module) return h;
        h.range = (decltype(h.range))GetProcAddress(module, "hipMemGetAddressRange");
        h.to_host = (decltype(h.to_host))GetProcAddress(module, "hipMemcpyDtoH");
        h.sync = (decltype(h.sync))GetProcAddress(module, "hipDeviceSynchronize");
        h.ready = h.range && h.to_host && h.sync;
        return h;
    }();
    return h;
}

inline void dump(const char *file, unsigned long long base, size_t size) {
    std::vector<unsigned char> block;
    FILE *out = fopen(file, "wb");
    if (!out) return;
    const size_t step = 64u << 20;
    for (size_t at = 0; at < size; at += step) {
        const size_t n = size - at < step ? size - at : step;
        block.resize(n);
        if (hip().to_host(block.data(), (void *)(base + at), n) != 0) break;
        fwrite(block.data(), 1, n, out);
    }
    fclose(out);
}

class Scope {
public:
    Scope(const char *name, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz,
          void **extra) {
        static char wanted[128];
        static const bool have = GetEnvironmentVariableA("DLSS5NR_GATE_CAPTURE", wanted, sizeof wanted) > 0;
        if (!have || !name || !extra || strcmp(wanted, name) != 0 || !hip().ready) return;
        static int seen = 0;
        static const int launch = [] {
            char value[16] = {};
            return GetEnvironmentVariableA("DLSS5NR_GATE_CAPTURE_LAUNCH", value, sizeof value) ? atoi(value) : 1;
        }();
        if (++seen != launch) return;
        char dir[MAX_PATH] = ".";
        GetEnvironmentVariableA("DLSS5NR_GATE_CAPTURE_DIR", dir, sizeof dir);
        char cap[32] = {};
        if (GetEnvironmentVariableA("DLSS5NR_GATE_CAPTURE_BYTES", cap, sizeof cap)) limit_ = (size_t)_strtoui64(cap, nullptr, 10);
        dir_ = dir;

        const unsigned char *buffer = nullptr;
        size_t size = 0;
        for (int k = 0; extra[k] && k < 16; k += 2) {
            if ((size_t)extra[k] == 1) buffer = (const unsigned char *)extra[k + 1];
            if ((size_t)extra[k] == 2) size = *(const size_t *)extra[k + 1];
        }
        if (!buffer) return;
        hip().sync();
        char path[MAX_PATH + 64];
        snprintf(path, sizeof path, "%s\\args.bin", dir);
        if (FILE *out = fopen(path, "wb")) {
            fwrite(buffer, 1, size, out);
            fclose(out);
        }
        snprintf(path, sizeof path, "%s\\info.txt", dir);
        FILE *info = fopen(path, "w");
        if (info) fprintf(info, "kernel %s\ngrid %u %u %u\nblock %u %u %u\nargs %zu\n", name, gx, gy, gz, bx, by, bz, size);
        for (size_t at = 0; at + 8 <= size; at += 8) {
            unsigned long long value;
            memcpy(&value, buffer + at, 8);
            if (value < 0x10000) continue;
            void *base = nullptr;
            size_t length = 0;
            if (hip().range((void **)&base, &length, (void *)value) != 0 || !base) continue;
            bool known = false;
            for (const Region &r : regions_) known = known || r.base == (unsigned long long)base;
            if (!known) {
                regions_.push_back({(unsigned long long)base, length});
                if (info) fprintf(info, "region %llx %zu\n", (unsigned long long)base, length);
                const size_t n = length < limit_ ? length : limit_;
                snprintf(path, sizeof path, "%s\\pre_%llx.bin", dir, (unsigned long long)base);
                dump(path, (unsigned long long)base, n);
            }
            if (info) fprintf(info, "pointer %zu %llx %llu\n", at, (unsigned long long)base, value - (unsigned long long)base);
        }
        if (info) fclose(info);
        active_ = true;
    }

    // After the launch (translated or native) has been queued: wait for it and dump what it left behind.
    ~Scope() {
        if (!active_) return;
        hip().sync();
        for (const Region &r : regions_) {
            char path[MAX_PATH + 64];
            snprintf(path, sizeof path, "%s\\post_%llx.bin", dir_.c_str(), r.base);
            dump(path, r.base, r.size < limit_ ? r.size : limit_);
        }
    }

private:
    bool active_ = false;
    std::vector<Region> regions_;
    std::string dir_;
    size_t limit_ = 512u << 20;
};

} // namespace capture
