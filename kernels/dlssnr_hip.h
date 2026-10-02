// Building blocks shared by the native DLSSNR kernels.
//
// Every one of these kernels streams e4m3 activations and weights stored in
// NVIDIA's mma fragment order, multiplies them with an f16 accumulator that the
// PTX rounds after every 32-deep slice, and narrows the results back to e4m3
// between layers. What follows is that vocabulary, written once for RDNA3:
// widening four codes at a time, narrowing two values at a time without
// branches, the fragment permutation, the partner exchange that turns a WMMA
// result into the next WMMA's operand, and the tile flags.
//
// Numerics are those of ZLUDA on the same hardware, on purpose: a 32-deep slice
// is two f32-accumulating 16-deep WMMAs started from the f16 accumulator and
// rounded back to f16; e4m3 narrowing is round-to-nearest-even saturating at
// 448. A kernel built from these pieces reproduces the translated kernel's
// bytes, which is what makes it checkable against a capture.
//
// Layout vocabulary used throughout:
//
//   fragment column  k of an m16n8k32 operand, 0..31; the channel it carries is
//                    PERM32[k] (see ffwd_ref.py), because results are repacked
//                    into operands two n-tiles at a time
//   row              a token (activations) or an output channel (weights),
//                    stored as 32 f16 in fragment-column order, so that one
//                    16-deep WMMA operand is 16 consecutive halves
//   result tile      a 16x16 f32 WMMA result, [output row][token]: lane l holds
//                    token l%16 and rows 2v + l/16, v = 0..7

#pragma once

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace dlssnr {

typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef _Float16 half2 __attribute__((ext_vector_type(2)));
typedef int16_t short2 __attribute__((ext_vector_type(2)));
typedef float float8 __attribute__((ext_vector_type(8)));
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x8 __attribute__((ext_vector_type(8)));

// RDNA3 and RDNA3.5 share the wave32 f16 WMMA and its operand layout; RDNA4
// has its own (see "RDNA4" below).
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) ||     defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__)
#define DLSSNR_GFX11 1
#elif defined(__gfx1200__) || defined(__gfx1201__)
#define DLSSNR_GFX12 1
#elif defined(__HIP_DEVICE_COMPILE__)
// Anything else would compile to a kernel that silently skips its multiplies.
#error "dlssnr_hip.h: no WMMA path for this target"
#endif

// DLSSNR_EMULATE_GFX12, defined on an RDNA3 build, runs the RDNA4 code path on
// RDNA3: every operand and accumulator goes through the RDNA4 layout, and only
// the RDNA4 WMMA instruction itself is emulated, from AMD's published layout.
// Kernel code is then exactly what an RDNA4 card runs, and a capture checks it
// byte for byte on hardware that is at hand (see README, "RDNA4").
#if defined(DLSSNR_GFX11) && defined(DLSSNR_EMULATE_GFX12)
#define DLSSNR_EMULATING 1
#endif
#if defined(DLSSNR_GFX12) || defined(DLSSNR_EMULATING)
#define DLSSNR_GFX12_LAYOUT 1
#endif

__device__ inline half2 as_half2(uint32_t v) { return __builtin_bit_cast(half2, v); }
__device__ inline uint32_t as_u32(half2 v) { return __builtin_bit_cast(uint32_t, v); }

// Per 16-bit half: all ones where the magnitude bits are at least `limit`.
__device__ inline uint32_t halves_at_least(uint32_t magnitudes, uint32_t limit) {
    // Setting bit 15 first makes each half's subtraction non-negative, so no
    // borrow crosses into the other half; bit 15 survives exactly when the
    // magnitude was not below the limit. v_perm's selectors 8 and 9 replicate
    // the sign bit of bytes 1 and 3 across a byte, which spreads it over each
    // half in one instruction (a packed arithmetic shift compiles to a compare
    // and a select per half).
    const uint32_t t = (magnitudes | 0x80008000u) - (limit | (limit << 16));
    return __builtin_amdgcn_perm(0u, t, 0x09090808u);
}

// ------------------------------------------------------------ e4m3 -> f16
//
// Four e4m3 codes to four halves, as two packed words. Shifting the seven
// magnitude bits up by seven lands each value on the f16 whose exponent is
// biased by 15 instead of 7 -- the true value divided by 256 -- and the sign
// moves up by eight; one packed multiply by 256 restores the value, denormals
// included. The same arithmetic as ZLUDA's widening, so the WMMAs see the same
// inputs. (The e4m3 NaN, 0x7F, is not special-cased: activations never carry
// it, and weights are what the network shipped with.)
// A kernel whose device runs with f16 denormals flushed loses the e4m3
// subnormals here: divided by 256 they are f16 denormals, and the multiply
// below returns zero for them. DLSSNR_FLUSH_E4M3 follows that, for kernels
// whose translation is built that way.
#ifndef DLSSNR_FLUSH_E4M3
#define DLSSNR_FLUSH_E4M3 0
#endif

__device__ inline uint32_t flush_halves(uint32_t v) {
    const uint32_t lo = (v & 0x7C00u) ? (v & 0xFFFFu) : (v & 0x8000u);
    const uint32_t hi = (v & 0x7C000000u) ? (v & 0xFFFF0000u) : (v & 0x80000000u);
    return lo | hi;
}

__device__ inline u32x2 widen4(uint32_t v) {
    uint32_t lo = __builtin_amdgcn_perm(0u, v, 0x0c010c00u);   // codes 0, 1 into halves
    uint32_t hi = __builtin_amdgcn_perm(0u, v, 0x0c030c02u);   // codes 2, 3
    lo = (lo + (lo & 0x00800080u)) << 7;
    hi = (hi + (hi & 0x00800080u)) << 7;
#if DLSSNR_FLUSH_E4M3
    lo = flush_halves(lo);
    hi = flush_halves(hi);
#endif
    const half2 k256 = {256.0f16, 256.0f16};
    return (u32x2){as_u32(as_half2(lo) * k256), as_u32(as_half2(hi) * k256)};
}

// Sixteen codes -- one operand's worth -- to sixteen halves.
__device__ inline half16 widen16(u32x4 v) {
    u32x8 out;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const u32x2 pair = widen4(v[i]);
        out[2 * i] = pair[0];
        out[2 * i + 1] = pair[1];
    }
    return __builtin_bit_cast(half16, out);
}

// ------------------------------------------------------------ f16 -> e4m3
//
// Two f16 values rounded to the nearest e4m3 value, ties to even, saturating at
// 448, returned as f16 (every e4m3 value is one). No branches: both roundings
// are computed and the right one kept per half.
//
//   from 2^-6 up   drop seven mantissa bits with round-to-nearest-even on the
//                  bit pattern; a carry into the exponent is correct
//   below 2^-6     e4m3 denormals are multiples of 2^-9: adding and removing 2
//                  in f16, whose spacing at 2 is exactly 2^-9, rounds to them
//
// Neither half can carry into the other in the 32-bit add: both magnitudes are
// at most 0x5F00 after saturation. Infinity and NaN, which ZLUDA turns into the
// e4m3 NaN, saturate here instead; this network's f16 accumulators do not
// reach them.
__device__ inline uint32_t round_e4m3x2(uint32_t h2) {
    const uint32_t sign = h2 & 0x80008000u;
    const half2 k448 = {448.0f16, 448.0f16};
    const uint32_t a = as_u32(__builtin_elementwise_min(as_half2(h2 & 0x7FFF7FFFu), k448));
    const uint32_t normal = (a + 0x003F003Fu + ((a >> 7) & 0x00010001u)) & 0xFF80FF80u;
    const half2 k2 = {2.0f16, 2.0f16};
    const uint32_t denormal = as_u32((as_half2(a) + k2) - k2);
    const uint32_t big = halves_at_least(a, 0x2400u);
    return ((normal & big) | (denormal & ~big)) | sign;
}

// The e4m3 codes of two values round_e4m3x2 produced, one in the low byte of
// each half. Above 2^-6 the code is the f16 pattern shifted down by seven, less
// the difference of the exponent biases; below, it is the value in units of
// 2^-9, read off as the low bits of 1024 + 512 v (f16 spacing at 1024 is 1).
__device__ inline uint32_t codes_e4m3x2(uint32_t q2) {
    const uint32_t a = q2 & 0x7FFF7FFFu;
    // "- 64" as "+ 192, mod 256" per half: a half below 2^-6 would otherwise
    // go negative and borrow from its neighbour.
    const uint32_t normal = (((a >> 7) & 0x00FF00FFu) + 0x00C000C0u) & 0x00FF00FFu;
    const half2 k512 = {512.0f16, 512.0f16}, k1024 = {1024.0f16, 1024.0f16};
    const uint32_t denormal = as_u32(__builtin_elementwise_fma(as_half2(a), k512, k1024)) - 0x64006400u;
    const uint32_t big = halves_at_least(a, 0x2400u);
    return ((normal & big) | (denormal & ~big)) | ((q2 >> 8) & 0x00800080u);
}

// Sixteen halves -- one operand's worth -- narrowed to sixteen e4m3 codes,
// packed as four words in fragment-column order. What a result tile becomes
// when it crosses shared memory, or a surface, as codes rather than halves.
__device__ inline u32x4 narrow16(half16 v) {
    const u32x8 w = __builtin_bit_cast(u32x8, v);
    u32x4 out;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t a = codes_e4m3x2(round_e4m3x2(w[2 * j]));
        const uint32_t b = codes_e4m3x2(round_e4m3x2(w[2 * j + 1]));
        out[j] = __builtin_amdgcn_perm(b, a, 0x06040200u);
    }
    return out;
}

// ------------------------------------------------------ fragment permutation
//
// INV32[channel] = the fragment column carrying that channel. A weight row is
// placed at the position of its output channel's fragment column, so that row
// r of a result tile is directly operand position r of the next layer.
// PERM32[k] = the channel fragment column k carries -- inv32's inverse. It
// permutes within blocks of 16, so it holds for any k, not only k < 32.
__device__ inline int perm32(int k) {
    return 16 * (k >> 4) + 8 * ((k & 3) >> 1) + 2 * ((k & 15) >> 2) + (k & 1);
}

__device__ inline int inv32(int ch) {
    return (ch & 16) | (((ch >> 1) & 3) << 2) | (((ch >> 3) & 1) << 1) | (ch & 1);
}

// Where 4 fragment bytes land when an NVIDIA operand fragment is read as rows:
// the lane (g, t) that loaded 16 bytes holds, in word w, the four consecutive
// fragment columns 16(w>>1) + 4t .. +3 (A operand: of token g + 8(w&1)) or
// 16(w&1) + 4t .. +3 (B operand: of output 8(w>>1) + g).
struct FragWord {
    int row;     // token, or output within the 16 of a piece
    int column;  // first of the four fragment columns
};
__device__ inline FragWord a_fragment_word(int lane, int w) {
    return {(lane >> 2) + 8 * (w & 1), 16 * (w >> 1) + 4 * (lane & 3)};
}
__device__ inline FragWord b_fragment_word(int lane, int w) {
    return {8 * (w >> 1) + (lane >> 2), 16 * (w & 1) + 4 * (lane & 3)};
}

// --------------------------------------------------------- WMMA and slices

// ------------------------------------------------------------------ RDNA4
//
// The layouts, from AMD's Matrix Instruction Calculator (which reproduces the
// RDNA3 layout these kernels were verified against), wave32, lane = 16u + i:
//
//   RDNA3 f16   A/B  dword e holds k = 2e, 2e+1 of row i (all 16 k, both u)
//               C/D  register v holds row 2v + u
//   RDNA4 f16   A/B  dword d holds k = 8(d/2) + 4u + 2(d%2), +1   (8 of 16 k)
//   RDNA4 fp8   A/B  dword d holds k = 8u + 4d .. +3                (8 of 16 k)
//               C/D  register v holds row 8u + v
//
// The kernels speak RDNA3: results as rows 2v + u, operands with all sixteen k
// in every lane. The RDNA4 path keeps that vocabulary by feeding the A operand
// with its rows permuted -- physical row m carries row sigma(m) = 2(m%8) + m/8
// -- so that register v of lane 16u + i comes out holding row sigma(8u + v) =
// 2v + u, which is where every epilogue already looks for it. The permutation
// is one v_permlane16 per dword with uniform selectors; the k half a lane
// needs is already in its registers. Nothing past the multiply changes.

__device__ inline uint32_t swap_rows(uint32_t v);

#ifdef DLSSNR_GFX12_LAYOUT
typedef _Float16 half8 __attribute__((ext_vector_type(8)));

// Lane 16u + i takes the value of lane 16u + sigma(i).
//
// The result is pinned where the kernel asks for it. Without the empty asm
// the compiler may sink the exchange, and the loads feeding it, under a guard
// that follows -- the 1H block's halved output stores under one that depends
// on the lane's token -- and a lane then reads a neighbour that is inactive
// and never loaded its operand. RDNA3 never notices: each lane feeds the WMMA
// only its own operand. Found by the emulated build on the tokens at the
// image's edge; an RDNA4 card would have shown the same.
__device__ inline uint32_t rows_sigma(uint32_t v) {
    uint32_t r = __builtin_amdgcn_permlane16(v, v, 0xECA86420u, 0xFDB97531u, false, false);
    asm volatile("" : "+v"(r));
    return r;
}

// swap_rows, pinned the same way (rows_sigma): the partner may be kept by the
// same guards as the lane, but nothing here relies on it.
__device__ inline uint32_t exchange(uint32_t v) {
    uint32_t r = swap_rows(v);
    asm volatile("" : "+v"(r));
    return r;
}
__device__ inline float exchange(float v) {
    return __builtin_bit_cast(float, exchange(__builtin_bit_cast(uint32_t, v)));
}

// The value a lane of half u wants from register g0 of half h0 (u = 0) or
// register g1 of half h1 (u = 1). Registers are named at compile time, so the
// exchange runs on a uniform register and each lane keeps the side it wants.
template <int G0, int H0, int G1, int H1, class V>
__device__ inline auto pick(const V &src, bool u) {
    const auto a = H0 == 0 ? src[G0] : exchange(src[G0]);
    const auto b = H1 == 1 ? src[G1] : exchange(src[G1]);
    return u ? b : a;
}

#ifdef DLSSNR_EMULATING
// The RDNA4 instructions, on RDNA3, written from the layouts above and nothing
// else: operands and accumulator are rebuilt in RDNA3 order, the RDNA3 WMMA
// runs, and the result goes back to RDNA4 order. Test builds only.
template <int V>
__device__ inline float c12_to_c11(const float8 &c, bool u) {
    // row 2V + u sits in half (2V + u) / 8, register (2V + u) % 8
    return pick<(2 * V) % 8, (2 * V) / 8, (2 * V + 1) % 8, (2 * V + 1) / 8>(c, u);
}
template <int V>
__device__ inline float d11_to_d12(const float8 &d, bool u) {
    // row 8u + V sits in half (8u + V) % 2, register (8u + V) / 2
    return pick<V / 2, V % 2, (8 + V) / 2, (8 + V) % 2>(d, u);
}
__device__ inline float8 emulate_mma(half16 a11, half16 b11, float8 c12) {
    const bool u = (threadIdx.x & 16) != 0;
    float8 c11, d12;
    c11[0] = c12_to_c11<0>(c12, u); c11[1] = c12_to_c11<1>(c12, u);
    c11[2] = c12_to_c11<2>(c12, u); c11[3] = c12_to_c11<3>(c12, u);
    c11[4] = c12_to_c11<4>(c12, u); c11[5] = c12_to_c11<5>(c12, u);
    c11[6] = c12_to_c11<6>(c12, u); c11[7] = c12_to_c11<7>(c12, u);
    const float8 d11 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a11, b11, c11);
    d12[0] = d11_to_d12<0>(d11, u); d12[1] = d11_to_d12<1>(d11, u);
    d12[2] = d11_to_d12<2>(d11, u); d12[3] = d11_to_d12<3>(d11, u);
    d12[4] = d11_to_d12<4>(d11, u); d12[5] = d11_to_d12<5>(d11, u);
    d12[6] = d11_to_d12<6>(d11, u); d12[7] = d11_to_d12<7>(d11, u);
    return d12;
}
// RDNA3 dword e (k = 2e, 2e+1) is RDNA4 f16 dword 2(e/4) + e%2 of half (e/2)%2.
template <int E>
__device__ inline uint32_t f16_operand_to_11(const u32x4 &v, bool u) {
    constexpr int D = 2 * (E / 4) + E % 2, H = (E / 2) % 2;
    return pick<D, H, D, H>(v, u);
}
__device__ inline half16 f16_operand_11(u32x4 v) {
    const bool u = (threadIdx.x & 16) != 0;
    u32x8 w;
    w[0] = f16_operand_to_11<0>(v, u); w[1] = f16_operand_to_11<1>(v, u);
    w[2] = f16_operand_to_11<2>(v, u); w[3] = f16_operand_to_11<3>(v, u);
    w[4] = f16_operand_to_11<4>(v, u); w[5] = f16_operand_to_11<5>(v, u);
    w[6] = f16_operand_to_11<6>(v, u); w[7] = f16_operand_to_11<7>(v, u);
    return __builtin_bit_cast(half16, w);
}
#endif

// v_wmma_f32_16x16x16_f16 on RDNA4 operands.
__device__ inline float8 wmma12_f16(u32x4 a, u32x4 b, float8 c) {
#ifdef DLSSNR_EMULATING
    return emulate_mma(f16_operand_11(a), f16_operand_11(b), c);
#else
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(__builtin_bit_cast(half8, a),
                                                            __builtin_bit_cast(half8, b), c);
#endif
}

// The RDNA4 f16 operand a lane of half u holds, out of the sixteen k it has.
__device__ inline u32x4 f16_operand_12(half16 v) {
    const u32x8 w = __builtin_bit_cast(u32x8, v);
    const bool u = (threadIdx.x & 16) != 0;
    return (u32x4){u ? w[2] : w[0], u ? w[3] : w[1], u ? w[6] : w[4], u ? w[7] : w[5]};
}
#endif  // DLSSNR_GFX12_LAYOUT

__device__ inline float8 wmma(half16 a, half16 b, float8 c) {
#ifdef DLSSNR_GFX12_LAYOUT
    u32x4 a12 = f16_operand_12(a);
#pragma unroll
    for (int d = 0; d < 4; ++d) a12[d] = rows_sigma(a12[d]);
    return wmma12_f16(a12, f16_operand_12(b), c);
#else
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#endif
}

// A result tile as four packed f16 words: word j holds rows 4j + u and
// 4j + 2 + u of the lane's token, u = lane / 16. Conversion rounds to nearest
// even, as the PTX's f16 accumulator does.
__device__ inline u32x4 pack_tile(float8 d) {
    u32x4 r;
#pragma unroll
    for (int j = 0; j < 4; ++j) r[j] = as_u32((half2){(_Float16)d[2 * j], (_Float16)d[2 * j + 1]});
    return r;
}

__device__ inline float8 unpack_tile(u32x4 p) {
    float8 d;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const half2 h = as_half2(p[j]);
        d[2 * j] = (float)h[0];
        d[2 * j + 1] = (float)h[1];
    }
    return d;
}

// One 32-deep slice of an f16-accumulated product: the f16 accumulator widened
// to f32, two 16-deep WMMAs, rounded back to f16. Accumulators live packed
// between slices -- the conversions are needed either way, and packed they
// take half the registers.
__device__ inline u32x4 slice(u32x4 acc, half16 a0, half16 b0, half16 a1, half16 b1) {
    float8 c = unpack_tile(acc);
    c = wmma(a0, b0, c);
    c = wmma(a1, b1, c);
    return pack_tile(c);
}

// Every lane receives the value of the lane 16 away.
__device__ inline uint32_t swap_rows(uint32_t v) {
    return __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xfedcba98u, false, false);
}

// ------------------------------------------------------------ fp8 operands
//
// An operand whose values are e4m3. RDNA3 multiplies it as f16, widened once:
// op8 is half16 and everything below is the code above under another name.
// RDNA4 multiplies the codes themselves (v_wmma_f32_16x16x16_fp8_fp8, twice
// the f16 rate): op8 is the eight codes of the lane's k half, already where
// the instruction wants them -- for an A operand, from row sigma(i) (see
// "RDNA4" above). LDS rows keep their stride and hold codes in their first
// half, so every offset a kernel computes stays valid.
//
// The products and the f32 accumulation are those of the widened f16 path,
// so an RDNA4 result can differ from RDNA3 only by how the card rounds inside
// one accumulation. A kernel built with DLSSNR_FLUSH_E4M3 would also need the
// codes flushed; none of the native kernels is, and this path refuses it.
#if defined(DLSSNR_GFX12_LAYOUT) && DLSSNR_FLUSH_E4M3
#error "dlssnr_hip.h: the RDNA4 fp8 path does not flush e4m3 subnormals"
#endif

#ifdef DLSSNR_GFX12_LAYOUT
typedef u32x2 op8;
typedef int int2x __attribute__((ext_vector_type(2)));

__device__ inline int sigma16(int i) { return 2 * (i & 7) + (i >> 3); }

#ifdef DLSSNR_EMULATING
// RDNA3 dword e (k = 4e .. 4e+3) is RDNA4 fp8 dword e % 2 of half e / 2.
template <int E>
__device__ inline uint32_t fp8_operand_to_11(const u32x2 &v, bool u) {
    return pick<E % 2, E / 2, E % 2, E / 2>(v, u);
}
__device__ inline half16 fp8_operand_11(u32x2 v) {
    const bool u = (threadIdx.x & 16) != 0;
    return widen16((u32x4){fp8_operand_to_11<0>(v, u), fp8_operand_to_11<1>(v, u),
                           fp8_operand_to_11<2>(v, u), fp8_operand_to_11<3>(v, u)});
}
#endif

// v_wmma_f32_16x16x16_fp8_fp8 on RDNA4 operands.
__device__ inline float8 wmma12_fp8(op8 a, op8 b, float8 c) {
#ifdef DLSSNR_EMULATING
    return emulate_mma(fp8_operand_11(a), fp8_operand_11(b), c);
#else
    return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(__builtin_bit_cast(int2x, a),
                                                                __builtin_bit_cast(int2x, b), c);
#endif
}

// The lane's k half of sixteen codes held whole (a u32x4 in fragment-column
// order, as read from memory), as a B operand, or as an A operand of row i.
__device__ inline op8 codes_b8(u32x4 v) {
    const bool u = (threadIdx.x & 16) != 0;
    return (u32x2){u ? v[2] : v[0], u ? v[3] : v[1]};
}
__device__ inline op8 codes_a8(u32x4 v) {
    const op8 b = codes_b8(v);
    return (u32x2){rows_sigma(b[0]), rows_sigma(b[1])};
}

// A packed result tile whose values are e4m3 (rounded with round_e4m3x2), as
// the next product's B operand: the lane's token, rows 8u .. 8u+7 as codes.
// The lane holds rows 2v + u; the other parity comes from the lane 16 away.
__device__ inline op8 tile_to_b8(u32x4 q) {
    const bool u = (threadIdx.x & 16) != 0;
    const uint32_t got0 = exchange(u ? q[0] : q[2]);
    const uint32_t got1 = exchange(u ? q[1] : q[3]);
    // rows (8u, 8u+2), (8u+1, 8u+3), (8u+4, 8u+6), (8u+5, 8u+7)
    const uint32_t e0 = u ? got0 : q[0], o0 = u ? q[2] : got0;
    const uint32_t e1 = u ? got1 : q[1], o1 = u ? q[3] : got1;
    const uint32_t lo = __builtin_amdgcn_perm(codes_e4m3x2(o0), codes_e4m3x2(e0), 0x06020400u);
    const uint32_t hi = __builtin_amdgcn_perm(codes_e4m3x2(o1), codes_e4m3x2(e1), 0x06020400u);
    return (u32x2){lo, hi};
}
// The same as an A operand: the token becomes the row, taken from sigma(i).
__device__ inline op8 tile_to_a8(u32x4 q) {
    const op8 b = tile_to_b8(q);
    return (u32x2){rows_sigma(b[0]), rows_sigma(b[1])};
}

// A B operand turned into the A operand of the same rows (its lane is the row).
__device__ inline op8 b8_to_a8(op8 b) { return (u32x2){rows_sigma(b[0]), rows_sigma(b[1])}; }

// Sixteen halves held whole and already rounded to e4m3 values (a 2x2 mean,
// say), as a B operand: the lane's k half, as codes.
__device__ inline op8 rounded_b8(half16 v) {
    const u32x8 w = __builtin_bit_cast(u32x8, v);
    const bool u = (threadIdx.x & 16) != 0;
    const uint32_t c0 = codes_e4m3x2(u ? w[4] : w[0]), c1 = codes_e4m3x2(u ? w[5] : w[1]);
    const uint32_t c2 = codes_e4m3x2(u ? w[6] : w[2]), c3 = codes_e4m3x2(u ? w[7] : w[3]);
    return (u32x2){__builtin_amdgcn_perm(c1, c0, 0x06040200u), __builtin_amdgcn_perm(c3, c2, 0x06040200u)};
}
#else
typedef half16 op8;
__device__ inline op8 rounded_b8(half16 v) { return v; }
__device__ inline op8 b8_to_a8(op8 b) { return b; }
__device__ inline op8 codes_b8(u32x4 v) { return widen16(v); }
__device__ inline op8 codes_a8(u32x4 v) { return widen16(v); }
__device__ inline half16 tile_to_operand(u32x4 own);
__device__ inline op8 tile_to_b8(u32x4 q) { return tile_to_operand(q); }
__device__ inline op8 tile_to_a8(u32x4 q) { return tile_to_operand(q); }
#endif

// ---------------------------------------------- operands in fragment order
//
// Products whose sixteen terms the translation adds in the order its operand
// fragments hold them rather than in k order (the P . V of the attentions): k
// position i carries term FO[i] = 2(i/4) + 8((i/2)%2) + i%2. On RDNA3 the
// order is visible in the last bit and is reproduced; on RDNA4 it is kept too,
// which costs nothing and keeps the emulated build exact.
__device__ inline half16 fragment_order16(half16 v) {
    half16 r;
#pragma unroll
    for (int i = 0; i < 16; ++i) r[i] = v[2 * (i >> 2) + 8 * ((i >> 1) & 1) + (i & 1)];
    return r;
}

#ifdef DLSSNR_GFX12_LAYOUT
// k half u is terms {0,1,8,9,2,3,10,11} + 4u: codes 4u..4u+3 then 8+4u..
__device__ inline op8 frag_order_codes_b8(u32x4 codes) {
    const bool u = (threadIdx.x & 16) != 0;
    const uint32_t lo = u ? codes[1] : codes[0], hi = u ? codes[3] : codes[2];
    return (u32x2){__builtin_amdgcn_perm(hi, lo, 0x05040100u), __builtin_amdgcn_perm(hi, lo, 0x07060302u)};
}
// Sixteen halves already rounded to e4m3 values, as codes in that order.
__device__ inline op8 frag_order_rounded_b8(half16 v) {
    const u32x8 w = __builtin_bit_cast(u32x8, v);
    const bool u = (threadIdx.x & 16) != 0;
    // terms (4u, 4u+1), (8+4u, 9+4u), (4u+2, 4u+3), (10+4u, 11+4u)
    const uint32_t c0 = codes_e4m3x2(u ? w[2] : w[0]), c1 = codes_e4m3x2(u ? w[6] : w[4]);
    const uint32_t c2 = codes_e4m3x2(u ? w[3] : w[1]), c3 = codes_e4m3x2(u ? w[7] : w[5]);
    return (u32x2){__builtin_amdgcn_perm(c1, c0, 0x06040200u), __builtin_amdgcn_perm(c3, c2, 0x06040200u)};
}
// A packed result tile of e4m3 values, its rows in that order, as a B
// operand; the lane holds rows 2v + u and the other parity comes across.
__device__ inline op8 frag_order_tile_b8(u32x4 q) {
    const bool u = (threadIdx.x & 16) != 0;
    const uint32_t got0 = exchange(u ? q[0] : q[1]);
    const uint32_t got1 = exchange(u ? q[2] : q[3]);
    // rows (4u, 4u+2), (4u+1, 4u+3), (8+4u, 10+4u), (9+4u, 11+4u)
    const uint32_t e0 = u ? got0 : q[0], o0 = u ? q[1] : got0;
    const uint32_t e1 = u ? got1 : q[2], o1 = u ? q[3] : got1;
    const uint32_t x = __builtin_amdgcn_perm(codes_e4m3x2(o0), codes_e4m3x2(e0), 0x06020400u);
    const uint32_t y = __builtin_amdgcn_perm(codes_e4m3x2(o1), codes_e4m3x2(e1), 0x06020400u);
    return (u32x2){__builtin_amdgcn_perm(y, x, 0x05040100u), __builtin_amdgcn_perm(y, x, 0x07060302u)};
}
__device__ inline op8 frag_order_tile_a8(u32x4 q) {
    const op8 b = frag_order_tile_b8(q);
    return (u32x2){rows_sigma(b[0]), rows_sigma(b[1])};
}
#else
__device__ inline op8 frag_order_codes_b8(u32x4 codes) { return fragment_order16(widen16(codes)); }
__device__ inline op8 frag_order_rounded_b8(half16 v) { return fragment_order16(v); }
__device__ inline op8 frag_order_tile_b8(u32x4 q) { return fragment_order16(tile_to_operand(q)); }
__device__ inline op8 frag_order_tile_a8(u32x4 q) { return fragment_order16(tile_to_operand(q)); }
#endif

// A 32-deep slice of e4m3 operands (slice below, on RDNA3).
__device__ inline u32x4 slice(u32x4 acc, half16 a0, half16 b0, half16 a1, half16 b1);
__device__ inline u32x4 slice8(u32x4 acc, op8 a0, op8 b0, op8 a1, op8 b1) {
#ifdef DLSSNR_GFX12_LAYOUT
    float8 c = unpack_tile(acc);
    c = wmma12_fp8(a0, b0, c);
    c = wmma12_fp8(a1, b1, c);
    return pack_tile(c);
#else
    return slice(acc, a0, b0, a1, b1);
#endif
}

// A packed result tile (pack_tile, possibly narrowed) turned into the next
// layer's B operand: all sixteen rows of the lane's token, in order, in both
// halves of the wave as gfx11 requires. Half the values come from the lane 16
// away.
__device__ inline half16 tile_to_operand(u32x4 own) {
    // Lower lanes hold the even rows and receive the odd ones, upper lanes the
    // reverse; the interleave is the same with source order swapped, which a
    // per-lane perm selector expresses without a select.
    const bool upper = (threadIdx.x & 16) != 0;
    const uint32_t sel_lo = upper ? 0x01000504u : 0x05040100u;   // (even, odd) low halves
    const uint32_t sel_hi = upper ? 0x03020706u : 0x07060302u;   // (even, odd) high halves
    u32x8 out;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint32_t other = swap_rows(own[j]);
        out[2 * j] = __builtin_amdgcn_perm(other, own[j], sel_lo);       // rows 4j, 4j+1
        out[2 * j + 1] = __builtin_amdgcn_perm(other, own[j], sel_hi);   // rows 4j+2, 4j+3
    }
    return __builtin_bit_cast(half16, out);
}

// ------------------------------------------------------------ LDS rows
//
// Rows of f16 in LDS, unpadded, with their 16-byte segments permuted by an XOR
// of the row number: sixteen lanes reading the same segment of sixteen
// consecutive rows then spread over the banks instead of queueing on the same
// four. Padding would do the same, but costs a fifth of the space, and space
// decides how many blocks fit on a WGP.
template <int ROW_BYTES>
__device__ inline int lds_at(int row, int byte_in_row) {
    constexpr int SEGMENTS = ROW_BYTES / 16;
    const int seg = (byte_in_row >> 4) ^ ((row >> 1) & (SEGMENTS - 1));
    return row * ROW_BYTES + (seg << 4) + (byte_in_row & 15);
}

// Sixteen halves (two segments) of a row, from LDS.
template <int ROW_BYTES>
__device__ inline half16 lds_row16(const uint8_t *lds, int row, int byte_in_row) {
    const u32x4 a = *(const u32x4 *)(lds + lds_at<ROW_BYTES>(row, byte_in_row));
    const u32x4 b = *(const u32x4 *)(lds + lds_at<ROW_BYTES>(row, byte_in_row + 16));
    u32x8 v;
    v[0] = a[0]; v[1] = a[1]; v[2] = a[2]; v[3] = a[3];
    v[4] = b[0]; v[5] = b[1]; v[6] = b[2]; v[7] = b[3];
    return __builtin_bit_cast(half16, v);
}

// Four codes widened into a row: fragment columns column..column+3.
template <int ROW_BYTES>
__device__ inline void lds_put4(uint8_t *lds, int row, int column, uint32_t codes) {
    *(u32x2 *)(lds + lds_at<ROW_BYTES>(row, column * 2)) = widen4(codes);
}

// The same rows for e4m3 operands (op8): widened on RDNA3, exactly as above;
// on RDNA4 the codes themselves, fragment column c at byte c of the row --
// the first half of it, through the same segment permutation. Offsets are
// passed as for f16 rows either way.
template <int ROW_BYTES>
__device__ inline void stage4(uint8_t *lds, int row, int column, uint32_t codes) {
#ifdef DLSSNR_GFX12_LAYOUT
    *(uint32_t *)(lds + lds_at<ROW_BYTES>(row, column)) = codes;
#else
    lds_put4<ROW_BYTES>(lds, row, column, codes);
#endif
}

// A 16-deep operand from such rows (byte_in_row as for f16 rows, a multiple
// of 32). As a B operand the lane reads its own row; as an A operand, row must
// be 16n + lane % 16, and RDNA4 reads row 16n + sigma(lane % 16) instead --
// the permutation of dlssnr_hip.h's RDNA4 section, free when it is an address.
template <int ROW_BYTES>
__device__ inline op8 lds_b8(const uint8_t *lds, int row, int byte_in_row) {
#ifdef DLSSNR_GFX12_LAYOUT
    const int u = (threadIdx.x >> 4) & 1;
    return *(const u32x2 *)(lds + lds_at<ROW_BYTES>(row, byte_in_row >> 1) + 8 * u);
#else
    return lds_row16<ROW_BYTES>(lds, row, byte_in_row);
#endif
}
template <int ROW_BYTES>
__device__ inline op8 lds_a8(const uint8_t *lds, int row, int byte_in_row) {
#ifdef DLSSNR_GFX12_LAYOUT
    return lds_b8<ROW_BYTES>(lds, (row & ~15) | sigma16(row & 15), byte_in_row);
#else
    return lds_row16<ROW_BYTES>(lds, row, byte_in_row);
#endif
}

// Rows that hold codes on every architecture (the 8H block's), sixteen codes
// at `at`: the B operand, or the A operand when `at` was addressed with
// a_row(row) -- the row an A operand's lane reads, see lds_a8.
__device__ inline op8 lds_codes_b8(const uint8_t *at) {
#ifdef DLSSNR_GFX12_LAYOUT
    return *(const u32x2 *)(at + 8 * ((threadIdx.x >> 4) & 1));
#else
    return widen16(*(const u32x4 *)at);
#endif
}
__device__ inline int a_row(int row) {
#ifdef DLSSNR_GFX12_LAYOUT
    return (row & ~15) | sigma16(row & 15);
#else
    return row;
#endif
}

// The same rows read and written as values rather than as operands: the fused
// blocks keep the state, the residual and the attention output in them, and
// read single channels back in the result layout. Values must be e4m3 ones.
template <int ROW_BYTES>
__device__ inline _Float16 lds_e4m3_value(const uint8_t *lds, int row, int column) {
#ifdef DLSSNR_GFX12_LAYOUT
    const uint32_t code = lds[lds_at<ROW_BYTES>(row, column)];
    return as_half2(widen4(code)[0])[0];
#else
    return *(const _Float16 *)(lds + lds_at<ROW_BYTES>(row, 2 * column));
#endif
}
template <int ROW_BYTES>
__device__ inline void lds_e4m3_put(uint8_t *lds, int row, int column, _Float16 v) {
#ifdef DLSSNR_GFX12_LAYOUT
    lds[lds_at<ROW_BYTES>(row, column)] = (uint8_t)codes_e4m3x2(as_u32((half2){v, v}));
#else
    *(_Float16 *)(lds + lds_at<ROW_BYTES>(row, 2 * column)) = v;
#endif
}
// Sixteen values of a row (byte_in_row as for f16 rows), and sixteen rounded
// ones written back.
template <int ROW_BYTES>
__device__ inline half16 lds_e4m3_row16(const uint8_t *lds, int row, int byte_in_row) {
#ifdef DLSSNR_GFX12_LAYOUT
    return widen16(*(const u32x4 *)(lds + lds_at<ROW_BYTES>(row, byte_in_row >> 1)));
#else
    return lds_row16<ROW_BYTES>(lds, row, byte_in_row);
#endif
}
template <int ROW_BYTES>
__device__ inline void lds_e4m3_put16(uint8_t *lds, int row, int byte_in_row, half16 rounded) {
#ifdef DLSSNR_GFX12_LAYOUT
    *(u32x4 *)(lds + lds_at<ROW_BYTES>(row, byte_in_row >> 1)) = narrow16(rounded);
#else
    const u32x8 w = __builtin_bit_cast(u32x8, rounded);
    *(u32x4 *)(lds + lds_at<ROW_BYTES>(row, byte_in_row)) = (u32x4){w[0], w[1], w[2], w[3]};
    *(u32x4 *)(lds + lds_at<ROW_BYTES>(row, byte_in_row + 16)) = (u32x4){w[4], w[5], w[6], w[7]};
#endif
}

// Rows addressed plainly, without the segment permutation, that hold one
// operand's sixteen e4m3 values at `at` (32 bytes of f16 on RDNA3, 16 codes
// on RDNA4): written whole by every lane, read back in fragment order as an A
// operand of row 16n + lane % 16.
__device__ inline void raw_put16(uint8_t *at, half16 rounded) {
#ifdef DLSSNR_GFX12_LAYOUT
    *(u32x4 *)at = narrow16(rounded);
#else
    *(half16 *)at = rounded;
#endif
}
__device__ inline op8 frag_order_raw_a8(const uint8_t *base, int row_bytes, int row, int byte) {
#ifdef DLSSNR_GFX12_LAYOUT
    row = (row & ~15) | sigma16(row & 15);
    return frag_order_codes_b8(*(const u32x4 *)(base + row * row_bytes + byte));
#else
    return fragment_order16(*(const half16 *)(base + row * row_bytes + byte));
#endif
}

// An operand kept in LDS by the lane that made it, in a 32-byte slot per
// row: the whole of it on RDNA3 (both halves of the wave write the same), the
// lane's own eight codes on RDNA4.
__device__ inline void stash_op8(uint8_t *slot, op8 v) {
#ifdef DLSSNR_GFX12_LAYOUT
    *(u32x2 *)(slot + 8 * ((threadIdx.x >> 4) & 1)) = v;
#else
    *(half16 *)slot = v;
#endif
}
__device__ inline op8 unstash_op8(const uint8_t *slot) {
#ifdef DLSSNR_GFX12_LAYOUT
    return *(const u32x2 *)(slot + 8 * ((threadIdx.x >> 4) & 1));
#else
    return *(const half16 *)slot;
#endif
}

// ------------------------------------------------------------ attention bias
//
// The bias of one 16x16 tile of queries by keys, as the score tile's
// accumulator: word j holds queries 4j + u and 4j + 2 + u of the lane's key.
// The PTX stores the table in its own accumulator fragment layout, 512 bytes a
// tile; gathering it straight from global memory is eight scattered 2-byte
// loads a lane per tile, which cost the fused blocks a few per cent. One
// 16-byte load a lane into 512 bytes of LDS, and the gather from there, costs
// less. stage is the wave's own, and nothing else may use it meanwhile.
__device__ inline u32x4 bias_tile(const uint8_t *piece, uint8_t *stage, int lane) {
    const u32x4 w = *(const u32x4 *)(piece + lane * 16);
    __builtin_amdgcn_wave_barrier();
    *(u32x4 *)(stage + lane * 16) = w;
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_wave_barrier();
    const int kk = lane & 15, u = lane >> 4;
    u32x4 c;
    for (int j = 0; j < 4; ++j) {
        uint32_t pair = 0;
        for (int h = 0; h < 2; ++h) {
            const int q = 4 * j + 2 * h + u;
            const int src_lane = 4 * (q & 7) + (kk & 7) / 2;
            const int word = 2 * (kk >> 3) + (q >> 3);
            pair |= (uint32_t)*(const uint16_t *)(stage + src_lane * 16 + word * 4 + (kk & 1) * 2) << (16 * h);
        }
        c[j] = pair;
    }
    return c;
}

// ------------------------------------------------------------ approximate f32
//
// The PTX .approx operations as ZLUDA translates them, for kernels that must
// reproduce the translation's scalar arithmetic bit for bit (approx_check.py
// compares each one against the translation). Inputs are assumed normal, which
// is where ZLUDA's helpers reduce to the bare hardware instruction.
__device__ inline float approx_rcp(float x) { return __builtin_amdgcn_rcpf(x); }
__device__ inline float approx_div(float x, float y) { return x * __builtin_amdgcn_rcpf(y); }
__device__ inline float approx_lg2(float x) { return __builtin_amdgcn_logf(x); }
__device__ inline float approx_sqrt(float x) { return __builtin_amdgcn_sqrtf(x); }
__device__ inline float approx_rsqrt(float x) { return __builtin_amdgcn_rsqf(x); }
// Below -126 the result would be a denormal the instruction flushes: halve,
// and square the result, as ZLUDA's helper does.
__device__ inline float approx_ex2(float x) {
    if (x < -126.0f) {
        const float r = __builtin_amdgcn_exp2f(x * 0.5f);
        return r * r;
    }
    return __builtin_amdgcn_exp2f(x);
}
// sin and cos take turns, not radians, after the range is folded by fract.
__device__ inline float approx_sin(float x) {
    return __builtin_amdgcn_sinf(__builtin_amdgcn_fractf(x * 0.15915494309189535f));
}
__device__ inline float approx_cos(float x) {
    return __builtin_amdgcn_cosf(__builtin_amdgcn_fractf(x * 0.15915494309189535f));
}

// An f32 value rounded to f16 on its own. Written as a plain cast after a
// multiply, the compiler folds the two into one mixed-precision instruction
// that rounds the exact product once -- different from the PTX's mul.f32 then
// cvt.rn.f16.f32 whenever the f32 product sits on an f16 tie. The empty asm
// makes the f32 value real first.
__device__ inline _Float16 to_f16(float x) {
    asm volatile("" : "+v"(x));
    return (_Float16)x;
}

// A pointer the compiler cannot see through. Loads from constant tables that
// a kernel only needs late -- a residual scale read in the last stage -- get
// hoisted to the top when the address is known there, and then stay live the
// whole kernel, which is what made the fused blocks spill. Passing the pointer
// through this at the point of use keeps each load where it is written.
template <class T>
__device__ inline T *opaque(T *p) {
    asm volatile("" : "+s"(p));
    return p;
}
// The same for a pointer the compiler holds in vector registers.
template <class T>
__device__ inline T *opaque_v(T *p) {
    asm volatile("" : "+v"(p));
    return p;
}

// ------------------------------------------------------------ textures
//
// A texture object, as ZLUDA hands it over in a kernel parameter, is a pointer
// to the image descriptor (8 dwords) with the sampler 12 dwords in. Sampling
// through the same intrinsic on the same descriptors is what the translation
// does, so the filtering, addressing and format conversion are the hardware's
// either way. A surface object is the image descriptor alone.
typedef float float4v __attribute__((ext_vector_type(4)));
typedef int int8v __attribute__((ext_vector_type(8)));
typedef int int4v __attribute__((ext_vector_type(4)));
typedef int int2v __attribute__((ext_vector_type(2)));

__device__ float4v llvm_image_sample_lz_2d(unsigned dmask, float x, float y, int8v rsrc, int4v samp,
                                           bool unorm, int tfe, int cache)
    __asm("llvm.amdgcn.image.sample.lz.2d.v4f32.f32");
extern "C" __device__ void __ockl_image_store_2D(const unsigned int __attribute__((address_space(4))) *image,
                                                 int2v coords, float4v data);

__device__ inline float4v sample2d(uint64_t texture, float u, float v) {
    const int *image = (const int *)texture;
    return llvm_image_sample_lz_2d(0xf, u, v, *(const int8v *)image, *(const int4v *)(image + 12), false, 0, 0);
}

__device__ inline void store2d(uint64_t surface, int x, int y, float4v value) {
    __ockl_image_store_2D((const unsigned int __attribute__((address_space(4))) *)surface, (int2v){x, y}, value);
}

// ------------------------------------------------------------ tile flags
//
// Producers raise a flag (store 0 with release) when a tile is written; a
// consumer spins with relaxed loads until it rises above -1. Under ZLUDA a
// stream runs its kernels in order, so these waits are satisfied on arrival,
// but a kernel that replaces a translated one keeps the protocol whole.
__device__ inline void wait_flag(const int *flag) {
    while (__hip_atomic_load(flag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) <= -1)
        __builtin_amdgcn_s_sleep(1);
}

// The fused blocks' view siblings (inpview_*, outview_*) take their input, or
// give their output, as a plain surface instead of patches: planes of sixteen
// channels, plane q at q * sy * sx * 16, a token sixteen bytes wide at
// (y * sx + x) * 16. A patch is 512-byte chunks of 32 channels; chunk c is
// planes 2c and 2c+1, and word w of lane L in a chunk is word L & 3 of token
// (L >> 2) + 8 * (w & 1) of the patch in the chunk's plane w >> 1 -- the
// chunk holds a fragment's two halves and the surface puts them apart.
enum View { PATCHES, VIEW_IN, VIEW_OUT };

// The surface's word for byte b of the patch whose first token is (y0, x0),
// or -1 outside the surface. An input surface one token high or wide is
// broadcast along that side.
__device__ inline long surface_word(int y0, int x0, int b, int sy, int sx, bool broadcast) {
    const int L = (b >> 4) & 31, w = (b >> 2) & 3, r = (L >> 2) + 8 * (w & 1);
    int y = y0 + (r >> 2), x = x0 + (r & 3);
    if (broadcast && sy == 1) y = 0;
    if (broadcast && sx == 1) x = 0;
    if (y < 0 || x < 0 || y >= sy || x >= sx) return -1;
    const int plane = 2 * (b >> 9) + (w >> 1);
    return (((long)plane * sy + y) * sx + x) * 4 + (L & 3);
}

// Sixteen bytes of a patch, from a surface.
__device__ inline u32x4 surface_load16(const uint8_t *s, int y0, int x0, int b, int sy, int sx) {
    u32x4 v = {0, 0, 0, 0};
    for (int w = 0; w < 4; ++w) {
        const long at = surface_word(y0, x0, b + 4 * w, sy, sx, true);
        if (at >= 0) v[w] = ((const uint32_t *)s)[at];
    }
    return v;
}

__device__ inline void raise_flag(int *flag) {
    __hip_atomic_store(flag, 0, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}

}  // namespace dlssnr
