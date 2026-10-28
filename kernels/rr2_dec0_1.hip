// rr2_dec0_kernel_1, the last decoder kernel (preset F), native.
//
// The shared stage (rr2_stage.h) up to the end of the MLP, then the generated tail (gen_dec0_1.py, from the user's
// own copy of the library; rr2_dec0_1_gen.inc is never committed) forms y = r + m * s2 and writes the network's
// output surfaces. The tail runs on the block's first wave; the stage uses them all.
//
// Compile with -fgpu-flush-denormals-to-zero (rr2_dec0_1.flags): the PTX's f32 code is .ftz.

#include "rr_ptx.h"
#include "rr2_stage.h"
#include "rr2_dec0_1_gen.inc"

struct Dec0View {
    uint32_t width, height;
    const uint8_t *up, *skip, *weights;
    uint8_t *tokens, *merged;
    const uint16_t *guide, *guide_enc;
    uint16_t *guide_out, *half_out;
    uint16_t shift_x, shift_y;
    f16 s0, s1, s2, s3;
};

template <class T>
__device__ inline T field(const uint8_t *p, int offset) { return *(const T *)(p + offset); }

struct Dec0Args {
    uint8_t bytes[216];
};

extern "C" __global__ __launch_bounds__(256) void rr2_dec0_kernel_1_native(Dec0Args) {
    __shared__ __attribute__((aligned(32))) uint8_t lds[stage_lds<32>()];
    const uint8_t *const P = (const uint8_t *)__builtin_amdgcn_kernarg_segment_ptr();

    Dec0View v;
    v.width = field<uint32_t>(P, 0);
    v.height = field<uint32_t>(P, 4);
    v.up = field<const uint8_t *>(P, 8);
    v.skip = nullptr;
    v.weights = field<const uint8_t *>(P, 24);
    v.tokens = v.merged = nullptr;
    v.guide = v.guide_enc = nullptr;
    v.guide_out = v.half_out = nullptr;
    v.shift_x = field<uint16_t>(P, 80);
    v.shift_y = field<uint16_t>(P, 82);
    v.s0 = field<f16>(P, 92);
    v.s1 = field<f16>(P, 94);
    v.s2 = field<f16>(P, 96);
    v.s3 = field<f16>(P, 98);

    stage<32, 2, 4, 0, 0, 0, TOKENS, MLP_ONLY>(lds, v);
    if (threadIdx.z == 0) {
        const Special S = special_registers();
        Regs io;
        pack_fragments(io, (const uint16_t *)(lds + 192 * 32), (const uint16_t *)(lds + 64 * 32), S.lane);
        dec0_tail(P, S, io);
    }
}
