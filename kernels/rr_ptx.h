// What code made by ptx2hip.py needs: the PTX instructions that are not plain C, with the semantics ZLUDA gives
// them on this hardware (its zluda_ptx_impl.cpp is where each of these was taken from, and the reason for each
// quirk: texel fetches through a normalised sampler, texture object 0 reading as zero, denormal-safe
// approximations, shuffles with their range bit).
//
// Compile kernels that include this with -fgpu-flush-denormals-to-zero: PTX's .ftz forms are what the network's
// f32 code uses, and ZLUDA runs them with f32 denormals flushed. Half precision keeps its denormals.
#pragma once

#include "dlssnr_hip.h"

using namespace dlssnr;

typedef _Float16 f16_t;

struct Special {
    uint32_t tid_x, tid_y, tid_z, ntid_x, ntid_y, ntid_z, ctaid_x, ctaid_y, ctaid_z, nctaid_x, nctaid_y, nctaid_z, lane, warp;
};

__device__ inline Special special_registers() {
    Special s;
    s.tid_x = threadIdx.x; s.tid_y = threadIdx.y; s.tid_z = threadIdx.z;
    s.ntid_x = blockDim.x; s.ntid_y = blockDim.y; s.ntid_z = blockDim.z;
    s.ctaid_x = blockIdx.x; s.ctaid_y = blockIdx.y; s.ctaid_z = blockIdx.z;
    s.nctaid_x = gridDim.x; s.nctaid_y = gridDim.y; s.nctaid_z = gridDim.z;
    s.lane = __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0));
    s.warp = (threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) >> 5;
    return s;
}

// ---------------------------------------------------------------- registers are bits
__device__ inline uint32_t f2u(float v) { return __builtin_bit_cast(uint32_t, v); }
__device__ inline float u2f(uint32_t v) { return __builtin_bit_cast(float, v); }
__device__ inline uint64_t d2u(double v) { return __builtin_bit_cast(uint64_t, v); }
__device__ inline double u2d(uint64_t v) { return __builtin_bit_cast(double, v); }
__device__ inline uint16_t h2u(f16_t v) { return __builtin_bit_cast(uint16_t, v); }
__device__ inline f16_t u2h(uint16_t v) { return __builtin_bit_cast(f16_t, v); }
__device__ inline dlssnr::half2 u2h2(uint32_t v) { return dlssnr::as_half2(v); }
__device__ inline uint32_t h22u(dlssnr::half2 v) { return dlssnr::as_u32(v); }

__device__ inline float ftz(float v) { return v; }   // the compile flag flushes
__device__ inline float fmin_ptx(float a, float b) { return __builtin_fminf(a, b); }
__device__ inline float fmax_ptx(float a, float b) { return __builtin_fmaxf(a, b); }

// ---------------------------------------------------------------- approximate math (ZLUDA's)
constexpr float kDenormalToNormal = 16777216.0f;
constexpr float kRcpDenormalOutput = 8.50706018714406320806444272332455743547934627837873057975602739772164e37f;
constexpr float kReverseDenormalToNormal = 0.029387360490963111877208252592662410455594571842846914442095471744599661631813495980086003637902577995683214210345151992265999035207077609582844f;

__device__ inline float rcp_approx(float x) {
    float factor = 1.0f;
    if (__builtin_isfpclass(x, __FPCLASS_NEGSUBNORMAL | __FPCLASS_POSSUBNORMAL)) factor = kDenormalToNormal;
    if (__builtin_fabsf(x) >= kRcpDenormalOutput) factor = kReverseDenormalToNormal;
    return __builtin_amdgcn_rcpf(x * factor) * factor;
}
// div.approx is an fdiv with the reciprocal and approximate-function flags: a times the hardware reciprocal.
__device__ inline float div_approx(float a, float b) { return a * __builtin_amdgcn_rcpf(b); }
__device__ inline float sqrt_approx(float x) {
    const bool sub = __builtin_isfpclass(x, __FPCLASS_NEGSUBNORMAL | __FPCLASS_POSSUBNORMAL);
    const float v = __builtin_amdgcn_sqrtf(sub ? x * kDenormalToNormal : x);
    return sub ? v * 0.000244140625f : v;
}
__device__ inline float ex2_approx(float x) {
    const bool special = x < -126.0f;
    const float r = __builtin_amdgcn_exp2f(special ? x * 0.5f : x);
    return special ? r * r : r;
}
__device__ inline float lg2_approx(float x) {
    const bool sub = __builtin_isfpclass(x, __FPCLASS_NEGSUBNORMAL | __FPCLASS_POSSUBNORMAL);
    const float v = __builtin_amdgcn_logf(sub ? x * kDenormalToNormal : x);
    return sub ? v - 24.0f : v;
}

// ---------------------------------------------------------------- bits
__device__ inline uint32_t bfi32(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    const uint32_t pos = c & 255u, len = d & 255u;
    if (len == 0 || pos >= 32) return b;
    uint32_t mask = len >= 32 ? 0xFFFFFFFFu : ((1u << len) - 1u);
    mask <<= pos;
    return (b & ~mask) | ((a << pos) & mask);
}

__device__ inline uint32_t prmt32(uint32_t a, uint32_t b, uint32_t c) {
    const uint64_t bytes = (uint64_t)a | ((uint64_t)b << 32);
    uint32_t out = 0;
    for (int i = 0; i < 4; ++i) {
        const uint32_t sel = (c >> (4 * i)) & 0xF;
        uint32_t byte = (uint32_t)(bytes >> (8 * (sel & 7))) & 0xFF;
        if (sel & 8) byte = (byte & 0x80) ? 0xFF : 0;
        out |= byte << (8 * i);
    }
    return out;
}

__device__ inline float clamp_f16(float v) { return v != v ? v : __builtin_fminf(__builtin_fmaxf(v, -65504.0f), 65504.0f); }

// cvt.rn.satfinite.e4m3x2.f16x2: two codes in the low two bytes
__device__ inline uint16_t cvt_e4m3x2(uint32_t pair) {
    const uint32_t c = dlssnr::codes_e4m3x2(dlssnr::round_e4m3x2(pair));
    return (uint16_t)((c & 0xFF) | (((c >> 16) & 0xFF) << 8));
}

// ---------------------------------------------------------------- shuffles (ZLUDA's, with the range bit)
struct ShflResult {
    uint32_t value;
    bool valid;
};
__device__ inline ShflResult shfl_generic(uint32_t input, int32_t delta, uint32_t opts, int mode) {
    const int32_t section_mask = (opts >> 8) & 31, warp_end = opts & 31;
    delta &= 31;
    const int32_t self = (int32_t)__builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0));
    const int32_t subsection = section_mask & self;
    const int32_t subsection_end = subsection | (~section_mask & warp_end);
    int32_t idx;
    bool out_of_bounds;
    switch (mode) {
    case 0: idx = self - delta; out_of_bounds = idx < subsection_end; break;                       // up
    case 1: idx = self + delta; out_of_bounds = idx > subsection_end; break;                       // down
    case 2: idx = self ^ delta; out_of_bounds = idx > subsection_end; break;                       // bfly
    default: idx = (delta & ~section_mask) | subsection; out_of_bounds = idx > subsection_end; break;   // idx
    }
    if (out_of_bounds) idx = self;
    return {(uint32_t)__builtin_amdgcn_ds_bpermute(idx << 2, (int32_t)input), !out_of_bounds};
}
__device__ inline ShflResult shfl_up(uint32_t i, int32_t d, uint32_t o) { return shfl_generic(i, d, o, 0); }
__device__ inline ShflResult shfl_down(uint32_t i, int32_t d, uint32_t o) { return shfl_generic(i, d, o, 1); }
__device__ inline ShflResult shfl_bfly(uint32_t i, int32_t d, uint32_t o) { return shfl_generic(i, d, o, 2); }
__device__ inline ShflResult shfl_idx(uint32_t i, int32_t d, uint32_t o) { return shfl_generic(i, d, o, 3); }

// ---------------------------------------------------------------- textures and surfaces
// A texture object is a pointer to the image descriptor, the sampler 12 dwords in; 0 reads as zeros. A surface
// object carries the size of its pixel in the three low bits of the handle.
__device__ float4v llvm_image_sample_l_2d(unsigned dmask, float x, float y, float lod, int8v rsrc, int4v samp, bool unorm,
                                          int tfe, int cache) __asm("llvm.amdgcn.image.sample.l.2d.v4f32.f32");

__device__ unsigned int rr_null_texture[32];

__device__ inline const int *texture_descriptor(uint64_t handle) {
    return handle ? (const int *)(handle & ~(uint64_t)7) : (const int *)rr_null_texture;
}

__device__ inline float4v tex2d(uint64_t handle, float x, float y) {
    const int *image = texture_descriptor(handle);
    return dlssnr::llvm_image_sample_lz_2d(0xf, x, y, *(const int8v *)image, *(const int4v *)(image + 12), false, 0, 0);
}

__device__ inline float4v tex2d_lod(uint64_t handle, float x, float y, float lod) {
    const int *image = texture_descriptor(handle);
    return llvm_image_sample_l_2d(0xf, x, y, lod, *(const int8v *)image, *(const int4v *)(image + 12), false, 0, 0);
}

// An integer-coordinate fetch is a texel read, unnormalised and unfiltered, whatever the sampler says
__device__ inline float4v tex2d_texel(uint64_t handle, int32_t x, int32_t y) {
    const int *image = texture_descriptor(handle);
    int4v own = *(const int4v *)(image + 12);
    float fx = (float)x, fy = (float)y;
    if ((own.x & 0x8000) == 0) {
        own.x |= 0x8000;
        own.z &= ~(0x3F << 20);
        fx += 0.5f;
        fy += 0.5f;
    }
    return dlssnr::llvm_image_sample_lz_2d(0xf, fx, fy, *(const int8v *)image, own, false, 0, 0);
}

// sust.b with 16-bit channels: x is in bytes. A surface whose pixel is as wide as the store takes it as the
// pixel's channels converted from half to the format; anything else is not something this has seen.
__device__ inline void sust_b16(uint64_t handle, int32_t x_bytes, int32_t y, int count, uint16_t v0, uint16_t v1 = 0, uint16_t v2 = 0,
                                uint16_t v3 = 0) {
    const uint64_t image = handle & ~(uint64_t)7;
    const int pixel_bytes = 1 << (handle & 7);
    if (pixel_bytes != 2 * count || x_bytes % pixel_bytes != 0) __builtin_trap();
    float4v data = {(float)u2h(v0), (float)u2h(v1), (float)u2h(v2), (float)u2h(v3)};
    dlssnr::store2d(image, x_bytes / pixel_bytes, y, data);
}
