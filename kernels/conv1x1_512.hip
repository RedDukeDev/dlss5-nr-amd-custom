// tin3::Conv2d1x1Layer<512, 512, ...> -- cc_split_swin_16h_proj_512_chained_fp8
// and cc_split_swin_16h_ffwd_proj_512_chained_fp8 -- written for the hardware.
//
// Per token, with x the input, r the residual stream and s a per-channel f16
// scale stored after the weights (ref_layers.py states it and checks it against
// captures; it matches ptxsim bit for bit):
//
//   out[n] = e4m3( f16(r[n] * s[n]) + sum_k W[n][k] x[k] )
//
// the sum taken as a chain of 32-deep f16-accumulated slices in k order,
// starting from the scaled residual. The two kernels are the same layer with
// different launch shapes: proj uses 8 warps per block, ffwd_proj 4.
//
// Launch: grid (2 * tiles_x, tiles_y, 1). Block (bx, by) covers the 8x8 tokens
// of tile (bx % tiles_x, by) -- four 4x4 patches -- and the 256 output channels
// of half bx / tiles_x. Wave w takes the 64 channels of group w % 4 of that
// half, for all 64 tokens (4 warps) or the 32 of patch row w / 4 (8 warps).
// Weights and activations are widened into LDS once per block, as in
// ffwd_512.hip; the residual and the scale are read straight into the
// accumulator layout, one token per lane.

#include "dlssnr_hip.h"

using namespace dlssnr;

// ---------------------------------------------------------------- the ABI
struct ConvArgs {
    const uint8_t *input;      // +0x00  activations, 512 channels, patch layout
    const uint8_t *residual;   // +0x08  same layout
    uint8_t *output;           // +0x10  same layout
    const uint8_t *weights;    // +0x18  512x512 B fragments, then 512 f16 scales
    int dim_y, dim_x;          // +0x20  tokens
    int view_y, view_x;        // +0x28  the view siblings' surface; 0 = dim
    int *ready_out;            // +0x30  one flag per block
    const int *ready_in;       // +0x38  PLANES flags per tile row of the producer
    int off_x, off_y;          // +0x40  window offset
};
static_assert(sizeof(ConvArgs) == 72, "the parameter is 72 bytes, by value");

constexpr int ROW = 64;
constexpr int ACT = 0, WGT = 64 * ROW, BUF = (64 + 256) * ROW;
static_assert(2 * BUF <= 131072 / 3, "three blocks per WGP");

// PLANES: flags the producer raises per tile row, one per slice of K it
// finishes (4 for proj, whose producer is the attention; 2 for ffwd_proj, fed by
// ffwd_512 and its two z halves). Waiting on a plane nobody raises would spin
// forever, so the count is the PTX's, per kernel.
// The view siblings: cc_split_swin_16h_ffwd_proj_inpview_512_chained_fp8
// reads its residual as a plain surface -- planes of sixteen channels, see
// surface_word in dlssnr_hip.h -- and cc_split_swin_16h_proj_512_outview_wait_fp8
// writes its output as one, on four warps, raising no flag.
template <int WAVES, int PLANES, View VIEW = PATCHES>
__device__ void conv1x1(const ConvArgs &args) {
    constexpr int THREADS = WAVES * 32;
    constexpr int TT = WAVES == 8 ? 2 : 4;          // token tiles per wave
    __shared__ __attribute__((aligned(16))) uint8_t lds[2 * BUF];

    const int lane = threadIdx.x, wave = threadIdx.y, tid = wave * 32 + lane;
    const int dim_y = args.dim_y, dim_x = args.dim_x;
    const int tiles_x = dim_x == 0 ? 0 : (dim_x - 1) / 8 + 1;
    const int tile_x = blockIdx.x % tiles_x, half = blockIdx.x / tiles_x;
    const int by = blockIdx.y, bz = blockIdx.z;
    const int patch_rows = dim_y / 4, patch_cols = dim_x / 4;
    const int in_x = dim_x - min(args.off_x, 0), in_y = dim_y - min(args.off_y, 0);
    const int in_tiles_x = in_x == 0 ? 0 : (in_x - 1) / 8 + 1;
    const int in_tiles_y = in_y == 0 ? 0 : (in_y - 1) / 8 + 1;
    const int row16 = lane & 15;
    const int surf_y = args.view_y > 0 ? args.view_y : dim_y;
    const int surf_x = args.view_x > 0 ? args.view_x : dim_x;

    // The producer raises PLANES flags per tile row, one per slice of the
    // input channels; the PTX waits for each before reading its slice of K,
    // this waits for all of them up front. Lane 0 of warps 0 and 4, each for
    // the tile rows its half-block height touches.
    if (lane == 0 && (wave & 3) == 0 && args.ready_in) {
        const int row_tok = by * 8 + wave - args.off_y;
        const int r0 = max(row_tok, 0) >> 3;
        const int r1 = min(row_tok + 4 > 0 ? ((row_tok + 3) >> 3) + 1 : 0, in_tiles_y);
        const int col_tok = tile_x * 8 - args.off_x;
        const int c0 = max(col_tok, 0) >> 3;
        const int c1 = min(col_tok + 8 > 0 ? ((col_tok + 7) >> 3) + 1 : 0, in_tiles_x);
        for (int r = r0; r < r1; ++r)
            for (int p = 0; p < PLANES; ++p)
                for (int c = c0; c < c1; ++c) wait_flag(args.ready_in + c + in_tiles_x * (PLANES * (r + bz) + p));
    }
    __syncthreads();

    // Patch addressing as the PTX does it, including its special cases for
    // dimensions of 4..7; out-of-range patches read as zeros.
    auto patch_source = [&](int patch, bool &valid) -> size_t {
        const int prow = 2 * by + (patch >> 1), pcol = 2 * tile_x + (patch & 1);
        const bool row_special = (dim_y & ~3) == 4;
        const int rowbase = row_special ? 0 : prow * patch_cols;
        int col = pcol;
        if (!row_special && prow >= patch_rows) valid = false;
        else if ((dim_x & ~3) == 4) { valid = true; col = 0; }
        else valid = pcol < patch_cols;
        return (size_t)(rowbase + col) * 8192;
    };

    // ------------------------------------------------ cooperative loading
    // Per slice: 4 patches x 32 lanes of activations, 16 pieces x 32 lanes of
    // weights (256 rows). Each thread owns fixed slots.
    constexpr int ACT_SLOTS = 128, W_SLOTS = 512;
    constexpr int ACT_PER = (ACT_SLOTS + THREADS - 1) / THREADS, W_PER = W_SLOTS / THREADS;
    size_t act_off[ACT_PER];
    bool act_ok[ACT_PER];
    for (int i = 0; i < ACT_PER; ++i) {
        const int slot = tid + THREADS * i;
        act_ok[i] = false;
        act_off[i] = 0;
        if (slot < ACT_SLOTS) act_off[i] = patch_source(slot >> 5, act_ok[i]) + (slot & 31) * 16;
    }
    const uint8_t *wbase = args.weights + (size_t)bz * 262144 + half * 256 * 32;

    auto load = [&](int c, u32x4 (&a)[ACT_PER], u32x4 (&w)[W_PER]) {
        for (int i = 0; i < ACT_PER; ++i)
            a[i] = act_ok[i] ? *(const u32x4 *)(args.input + act_off[i] + c * 512) : (u32x4){0, 0, 0, 0};
        for (int i = 0; i < W_PER; ++i) {
            const int slot = tid + THREADS * i;
            w[i] = *(const u32x4 *)(wbase + c * 16384 + (slot >> 5) * 512 + (slot & 31) * 16);
        }
    };
    auto store = [&](uint8_t *buf, const u32x4 (&a)[ACT_PER], const u32x4 (&w)[W_PER]) {
        for (int i = 0; i < ACT_PER; ++i) {
            const int slot = tid + THREADS * i;
            if (slot >= ACT_SLOTS) continue;
            const int patch = slot >> 5, l = slot & 31;
            for (int k = 0; k < 4; ++k) {
                const FragWord f = a_fragment_word(l, k);
                stage4<ROW>(buf + ACT, patch * 16 + f.row, f.column, a[i][k]);
            }
        }
        for (int i = 0; i < W_PER; ++i) {
            const int slot = tid + THREADS * i;
            const int piece = slot >> 5, l = slot & 31;          // piece = 16 rows of the half
            for (int k = 0; k < 4; ++k) {
                const FragWord f = b_fragment_word(l, k);
                const int n = 16 * (piece & 3) + f.row;          // 0..63 within the group
                const int row = (piece >> 2) * 64 + 32 * (n >> 5) + inv32(n & 31);
                stage4<ROW>(buf + WGT, row, f.column, w[i][k]);
            }
        }
    };

    // ------------------------------------------------ accumulators: r * s
    //
    // Tile ct (chunk ct/2, half ct%2) row p = 16(ct%2) + 2v + u holds the
    // channel PERM32[p] of chunk ct/2: word j of the lane's packed tile is rows
    // 4j + u and 4j + 2 + u, which are fragment columns (t = j, b = u) and
    // (t = j, b = 2 + u) -- bytes u and 2 + u of the residual word at lane
    // slot 4(T%8) + j, word 2(ct%2) + T/8 -- and channels 16(ct%2) + 2j + u and
    // 16(ct%2) + 8 + 2j + u.
    const int grp = wave & 3, tbase = WAVES == 8 ? (wave >> 2) * 2 : 0;   // first token tile
    const int u = lane >> 4;
    const int ch0 = half * 256 + grp * 64;                                  // first channel of the wave
    const uint16_t *scales = (const uint16_t *)(args.weights + (size_t)bz * 262144 + 262144);
    u32x4 acc[4][TT];
    for (int tt = 0; tt < TT; ++tt) {
        bool ok;
        const size_t src = patch_source(tbase + tt, ok);
        const int patch = tbase + tt;
        const int py0 = 4 * (2 * by + (patch >> 1)), px0 = 4 * (2 * tile_x + (patch & 1));
        for (int ct = 0; ct < 4; ++ct) {
            const int cc = ct >> 1, h = ct & 1;
            const uint8_t *chunk = args.residual + src + (size_t)((ch0 >> 5) + cc) * 512;
            for (int j = 0; j < 4; ++j) {
                uint32_t word = 0;
                const int b = (4 * (row16 & 7) + j) * 16 + (2 * h + (row16 >> 3)) * 4;
                if (VIEW == VIEW_IN) {
                    const long at = surface_word(py0, px0, ((ch0 >> 5) + cc) * 512 + b, surf_y, surf_x, true);
                    if (at >= 0) word = ((const uint32_t *)args.residual)[at];
                } else if (ok) {
                    word = *(const uint32_t *)(chunk + b);
                }
                // bytes u and 2+u into the two halves, then widened
                const uint32_t pair = __builtin_amdgcn_perm(0u, word, u ? 0x0c030c01u : 0x0c020c00u);
                const uint32_t f16 = ((pair + (pair & 0x00800080u)) << 7);
                const half2 r = as_half2(f16) * (half2){256.0f16, 256.0f16};
                const int c = ch0 + 32 * cc + 16 * h + 2 * j + u;
                const half2 sc = {__builtin_bit_cast(_Float16, scales[c]), __builtin_bit_cast(_Float16, scales[c + 8])};
                acc[ct][tt][j] = as_u32(r * sc);
            }
        }
    }

    // ------------------------------------------------ out += W x, 16 slices
    {
        u32x4 a[ACT_PER], w[W_PER];
        load(0, a, w);
        store(lds, a, w);
        __syncthreads();
        for (int c = 0; c < 16; ++c) {
            const bool more = c + 1 < 16;
            if (more) load(c + 1, a, w);
            const uint8_t *buf = lds + (c & 1) * BUF;
            op8 b[TT][2];
            for (int tt = 0; tt < TT; ++tt)
                for (int kh = 0; kh < 2; ++kh) b[tt][kh] = lds_b8<ROW>(buf + ACT, (tbase + tt) * 16 + row16, kh * 32);
            for (int ct = 0; ct < 4; ++ct) {
                const int row = grp * 64 + ct * 16 + row16;
                const op8 a0 = lds_a8<ROW>(buf + WGT, row, 0);
                const op8 a1 = lds_a8<ROW>(buf + WGT, row, 32);
                for (int tt = 0; tt < TT; ++tt) acc[ct][tt] = slice8(acc[ct][tt], a0, b[tt][0], a1, b[tt][1]);
            }
            if (more) store(lds + ((c + 1) & 1) * BUF, a, w);
            __syncthreads();
        }
    }

    // ------------------------------------------------ out = e4m3(acc)
    const bool upper = lane >= 16;
    const uint32_t sel_lo = upper ? 0x01050004u : 0x05010400u;
    const uint32_t sel_hi = upper ? 0x03070206u : 0x07030602u;
    for (int tt = 0; tt < TT; ++tt) {
        const int patch = tbase + tt;
        const int prow = 2 * by + (patch >> 1), pcol = 2 * tile_x + (patch & 1);
        if (VIEW != VIEW_OUT && (prow >= patch_rows || pcol >= patch_cols)) continue;
        uint8_t *dst = args.output + (size_t)(prow * patch_cols + pcol) * 8192;
        auto put = [&](int b, uint32_t value) {        // b: byte in the patch
            if (VIEW == VIEW_OUT) {
                const long at = surface_word(4 * prow, 4 * pcol, b, surf_y, surf_x, false);
                if (at >= 0) ((uint32_t *)args.output)[at] = value;
            } else {
                *(uint32_t *)(dst + b) = value;
            }
        };
        for (int cc = 0; cc < 2; ++cc) {
            uint32_t bytes[2][2];
            for (int h = 0; h < 2; ++h) {
                const u32x4 q = acc[2 * cc + h][tt];
                uint32_t codes[4];
                for (int j = 0; j < 4; ++j) codes[j] = codes_e4m3x2(round_e4m3x2(q[j]));
                for (int j = 0; j < 2; ++j)
                    bytes[h][j] = __builtin_amdgcn_perm(codes[2 * j + 1], codes[2 * j], 0x06040200u);
            }
            const int h = upper ? 1 : 0, T = row16, word = 2 * h + (T >> 3);
            const int cbase = ((ch0 >> 5) + cc) * 512;
            for (int j = 0; j < 2; ++j) {
                const uint32_t keep = upper ? bytes[1][j] : bytes[0][j];
                const uint32_t got = swap_rows(upper ? bytes[0][j] : bytes[1][j]);
                put(cbase + (4 * (T & 7) + 2 * j) * 16 + word * 4, __builtin_amdgcn_perm(got, keep, sel_lo));
                put(cbase + (4 * (T & 7) + 2 * j + 1) * 16 + word * 4, __builtin_amdgcn_perm(got, keep, sel_hi));
            }
        }
    }

    __syncthreads();
    if (lane == 0 && wave == 0 && args.ready_out)
        raise_flag(args.ready_out + 2 * by * tiles_x + blockIdx.x);
}

extern "C" __global__ __launch_bounds__(256) void
cc_split_swin_16h_proj_512_chained_fp8_native(ConvArgs args) { conv1x1<8, 4>(args); }

extern "C" __global__ __launch_bounds__(128) void
cc_split_swin_16h_ffwd_proj_512_chained_fp8_native(ConvArgs args) { conv1x1<4, 2>(args); }

extern "C" __global__ __launch_bounds__(128) void
cc_split_swin_16h_ffwd_proj_inpview_512_chained_fp8_native(ConvArgs args) { conv1x1<4, 2, VIEW_IN>(args); }

extern "C" __global__ __launch_bounds__(128) void
cc_split_swin_16h_proj_512_outview_wait_fp8_native(ConvArgs args) { conv1x1<4, 4, VIEW_OUT>(args); }
