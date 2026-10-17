// Every pair of f16 bit patterns through round_e4m3x2 and round_e4m3x2_bits
// (dlssnr_hip.h): the count of words on which they disagree, and the first one.

#include "dlssnr_hip.h"

extern "C" __global__ void round_check(unsigned int *mismatches, unsigned int *first) {
    const uint32_t base = (blockIdx.x * blockDim.x + threadIdx.x) * 256u;
    unsigned int bad = 0;
    for (uint32_t i = 0; i < 256u; ++i) {
        const uint32_t w = base + i;
        if (dlssnr::round_e4m3x2(w) != dlssnr::round_e4m3x2_bits(w)) {
            ++bad;
            atomicMin(first, w);
        }
    }
    if (bad) atomicAdd(mismatches, bad);
}
