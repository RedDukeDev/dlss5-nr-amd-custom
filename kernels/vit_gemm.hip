// tin3::Conv1d1x1Layer on the ViT stage's 1D tokens, written for the hardware:
// cc_vit_1d_ffn_contract_chained_fp8 (4096 -> 1024) and
// cc_vit_1d_projection_chained_fp8 (1024 -> 1024).
//
// Per token, with x the input, r the residual and s a per-channel f16 scale
// stored after the weights (ref_layers.py states this and checks it against
// ptxsim bit for bit):
//
//   p_0 = chain over the first quarter of k, from f16(r[n] * s[n])
//   p_z = chain over quarter z, from zero
//   out = e4m3( f16(...f16(f16(p_0 + p_1) + p_2) + p_3) )
//
// The quarters are not an implementation detail: the PTX splits k over the four
// z slices of its grid, each slice adding its partial into an f16 scratch
// buffer in turn (a flag hands the turn on), so the sum is rounded four times
// in that order. A single 128-deep chain gives different bytes. Here one block
// does all four quarters itself, in the same order, in registers.
//
// Launch: grid (8 tiles * 8 channel blocks, 1, 4), block 32x4. Block
// (x, ., q) covers the 128 tokens of tile x % tiles and 32 of the 128 output
// channels of block x / tiles -- quarter q. Wave w takes 32 of those tokens.
// Since four blocks now share what was one block's output tile, the flag that
// tells the consumer the tile is finished is raised by whichever of them
// arrives last, through the same counter the PTX advances from -1 to 3.

#include "dlssnr_hip.h"

using namespace dlssnr;

struct VitArgs {
    const uint8_t *input;      // +0x00  K-channel tokens, patches of 16
    const uint8_t *residual;   // +0x08  1024-channel tokens
    uint8_t *output;           // +0x10  1024-channel tokens
    const uint8_t *weights;    // +0x18  1024 x K B fragments, then 1024 f16 scales
    int *counters;             // +0x20  one per output tile, -1 until the four quarters land
    uint8_t *scratch;          // +0x28  the PTX's f16 partial sums; unused here
    const int *ready_in;       // +0x30  the producer's flags, 32 per token tile
    int *ready_out;            // +0x38  one per output tile
    int dim_y, dim_x;          // +0x40  tokens = dim_y * dim_x
};
static_assert(sizeof(VitArgs) == 72, "the parameter is 72 bytes, by value");

constexpr int ROW = 64;
constexpr int ACT = 0, WGT = 128 * ROW, BUF = (128 + 32) * ROW;          // 10 KB per buffer

__device__ inline u32x4 add_f16(u32x4 a, u32x4 b) {
    u32x4 r;
#pragma unroll
    for (int j = 0; j < 4; ++j) r[j] = as_u32(as_half2(a[j]) + as_half2(b[j]));
    return r;
}

template <int K>
__device__ void vit_gemm(const VitArgs &args) {
    constexpr int CHUNKS = K / 32, QUARTER = CHUNKS / 4;
    __shared__ __attribute__((aligned(16))) uint8_t lds[2 * BUF];

    const int lane = threadIdx.x, wave = threadIdx.y, tid = wave * 32 + lane;
    const int row16 = lane & 15, u = lane >> 4;
    const int tokens = args.dim_y * args.dim_x;
    const int tiles = (tokens - 1) / 128 + 1;
    const int patches = (((tokens - 1) & ~31) + 32) >> 4;
    const int ttile = blockIdx.x % tiles, nblock = blockIdx.x / tiles, quarter = blockIdx.z;
    const int ch0 = nblock * 128 + quarter * 32;
    const int chunk_out = ch0 >> 5;                                       // its 32-channel chunk

    // The producer's flags: lane 0 of waves 0 and 2, for every 128-token tile
    // the wave's tokens touch, and for all four of the producer's planes --
    // this block reads all of K, where the PTX's slice read a quarter.
    if (lane == 0 && (wave & 1) == 0 && args.ready_in) {
        const int first = ttile * 128 + wave * 32;
        const int end = min(((first + 32) >> 7) + 1, tiles);
        for (int t = first >> 7; t < end; ++t)
            for (int z = 0; z < 4; ++z) wait_flag(args.ready_in + 32 * t + 8 * z);
    }
    __syncthreads();

    // ------------------------------------------------ cooperative loading
    // Per slice of 32 input channels: 8 patches x 32 lanes of activations (two
    // slots per thread) and 2 pieces x 32 lanes of weights (the first 64
    // threads).
    const uint8_t *act_src[2];
    bool act_ok[2];
    for (int i = 0; i < 2; ++i) {
        const int slot = tid + 128 * i, patch = ttile * 8 + (slot >> 5);
        act_ok[i] = patch < patches;
        act_src[i] = args.input + (size_t)patch * (16 * K) + (slot & 31) * 16;
    }
    const uint8_t *wsrc = args.weights + (size_t)(ch0 >> 4) * 512 + lane * 16;
    const uint16_t *scales = (const uint16_t *)(args.weights + (size_t)1024 * K);

    auto load = [&](int c, u32x4 (&a)[2], u32x4 &w) {
        for (int i = 0; i < 2; ++i)
            a[i] = act_ok[i] ? *(const u32x4 *)(act_src[i] + c * 512) : (u32x4){0, 0, 0, 0};
        if (tid < 64) w = *(const u32x4 *)(wsrc + (size_t)c * (1024 * 32) + (tid >> 5) * 512);
    };
    auto store = [&](uint8_t *buf, const u32x4 (&a)[2], const u32x4 &w) {
        for (int i = 0; i < 2; ++i) {
            const int slot = tid + 128 * i, patch_of = slot >> 5, l = slot & 31;
            for (int k = 0; k < 4; ++k) {
                const FragWord f = a_fragment_word(l, k);
                stage4<ROW>(buf + ACT, patch_of * 16 + f.row, f.column, a[i][k]);
            }
        }
        if (tid < 64)
            for (int k = 0; k < 4; ++k) {
                const FragWord f = b_fragment_word(lane, k);
                stage4<ROW>(buf + WGT, inv32(16 * (tid >> 5) + f.row), f.column, w[k]);
            }
    };

    // ------------------------------------------------ the first partial's start
    u32x4 start[2][2];                                  // [channel tile][token tile]
    for (int tt = 0; tt < 2; ++tt) {
        const int patch = ttile * 8 + wave * 2 + tt;
        const bool ok = patch < patches;
        for (int h = 0; h < 2; ++h) {
            const uint8_t *chunk = args.residual + (size_t)patch * (16 * 1024) + (size_t)chunk_out * 512;
            for (int j = 0; j < 4; ++j) {
                uint32_t word = 0;
                if (ok) word = *(const uint32_t *)(chunk + (4 * (row16 & 7) + j) * 16 + (2 * h + (row16 >> 3)) * 4);
                const uint32_t pair = __builtin_amdgcn_perm(0u, word, u ? 0x0c030c01u : 0x0c020c00u);
                const half2 r = as_half2((pair + (pair & 0x00800080u)) << 7) * (half2){256.0f16, 256.0f16};
                const int c = ch0 + 16 * h + 2 * j + u;
                const half2 sc = {__builtin_bit_cast(_Float16, scales[c]),
                                  __builtin_bit_cast(_Float16, scales[c + 8])};
                start[h][tt][j] = as_u32(r * sc);
            }
        }
    }

    // ------------------------------------------------ four quarters of K
    u32x4 sum[2][2], acc[2][2];
    {
        u32x4 a[2], w;
        load(0, a, w);
        store(lds, a, w);
        __syncthreads();
        for (int c = 0; c < CHUNKS; ++c) {
            if (c % QUARTER == 0) {
                if (c) {                                 // the quarter just finished
                    if (c == QUARTER)
                        for (int h = 0; h < 2; ++h)
                            for (int tt = 0; tt < 2; ++tt) sum[h][tt] = acc[h][tt];
                    else
                        for (int h = 0; h < 2; ++h)
                            for (int tt = 0; tt < 2; ++tt) sum[h][tt] = add_f16(sum[h][tt], acc[h][tt]);
                }
                for (int h = 0; h < 2; ++h)
                    for (int tt = 0; tt < 2; ++tt)
                        acc[h][tt] = c == 0 ? start[h][tt] : (u32x4){0, 0, 0, 0};
            }
            const bool more = c + 1 < CHUNKS;
            if (more) load(c + 1, a, w);
            const uint8_t *buf = lds + (c & 1) * BUF;
            op8 wa[2][2];
            for (int h = 0; h < 2; ++h)
                for (int kh = 0; kh < 2; ++kh) wa[h][kh] = lds_a8<ROW>(buf + WGT, h * 16 + row16, kh * 32);
            for (int tt = 0; tt < 2; ++tt) {
                const int row = wave * 32 + tt * 16 + row16;
                const op8 b0 = lds_b8<ROW>(buf + ACT, row, 0);
                const op8 b1 = lds_b8<ROW>(buf + ACT, row, 32);
                for (int h = 0; h < 2; ++h) acc[h][tt] = slice8(acc[h][tt], wa[h][0], b0, wa[h][1], b1);
            }
            if (more) store(lds + ((c + 1) & 1) * BUF, a, w);
            __syncthreads();
        }
        for (int h = 0; h < 2; ++h)
            for (int tt = 0; tt < 2; ++tt) sum[h][tt] = add_f16(sum[h][tt], acc[h][tt]);
    }

    // ------------------------------------------------ out = e4m3(sum)
    const bool upper = lane >= 16;
    const uint32_t sel_lo = upper ? 0x01050004u : 0x05010400u;
    const uint32_t sel_hi = upper ? 0x03070206u : 0x07030602u;
    for (int tt = 0; tt < 2; ++tt) {
        const int patch = ttile * 8 + wave * 2 + tt;
        if (patch >= patches) continue;
        uint8_t *chunk = args.output + (size_t)patch * (16 * 1024) + (size_t)chunk_out * 512;
        uint32_t bytes[2][2];
        for (int h = 0; h < 2; ++h) {
            uint32_t codes[4];
            for (int j = 0; j < 4; ++j) codes[j] = codes_e4m3x2(round_e4m3x2(sum[h][tt][j]));
            for (int j = 0; j < 2; ++j)
                bytes[h][j] = __builtin_amdgcn_perm(codes[2 * j + 1], codes[2 * j], 0x06040200u);
        }
        const int h = upper ? 1 : 0, T = row16, word = 2 * h + (T >> 3);
        for (int j = 0; j < 2; ++j) {
            const uint32_t keep = upper ? bytes[1][j] : bytes[0][j];
            const uint32_t got = swap_rows(upper ? bytes[0][j] : bytes[1][j]);
            *(uint32_t *)(chunk + (4 * (T & 7) + 2 * j) * 16 + word * 4) = __builtin_amdgcn_perm(got, keep, sel_lo);
            *(uint32_t *)(chunk + (4 * (T & 7) + 2 * j + 1) * 16 + word * 4) = __builtin_amdgcn_perm(got, keep, sel_hi);
        }
    }

    // The four quarters of an output tile count themselves in; the last one
    // raises the tile's flag, as the PTX's fourth z slice does. The counter
    // starts at -1 and ends at 3 either way.
    __syncthreads();
    if (lane == 0 && wave == 0) {
        const int index = ttile * 8 + nblock;
        const int before = __hip_atomic_fetch_add(args.counters + index, 1, __ATOMIC_ACQ_REL,
                                                  __HIP_MEMORY_SCOPE_AGENT);
        if (before == 2 && args.ready_out) raise_flag(args.ready_out + index);
    }
}

extern "C" __global__ __launch_bounds__(128) void
cc_vit_1d_ffn_contract_chained_fp8_native(VitArgs args) { vit_gemm<4096>(args); }

extern "C" __global__ __launch_bounds__(128) void
cc_vit_1d_projection_chained_fp8_native(VitArgs args) { vit_gemm<1024>(args); }
