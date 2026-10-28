// rr2_enc0_kernel, the ray reconstruction's first encoder (preset F), native.
//
// Three stretches. The kernel gathers 43 features per pixel from the game's textures: that is the generated
// enc0_gather (ptx2hip.py, from the user's own copy of the library -- rr2_enc0_gen.inc is made by gen_enc0.py and
// never committed). The embedding and the window's attention and MLP are the stage every encoder shares
// (rr2_stage.h, input EMBED). Last it gates the guide image with the head and writes it: the generated enc0_tail.
// The two generated stretches run on the block's first wave, as the PTX's one warp did; the stage uses them all.
//
// Compile with -fgpu-flush-denormals-to-zero (rr2_enc0.flags): the PTX's f32 code is .ftz.

#include "rr_ptx.h"
#include "rr2_stage.h"
#include "rr2_enc0_gen.inc"

// The fields stage<> reads, for a kernel whose parameter block is not a StageArgs.
struct Enc0View {
    uint32_t width, height;
    const uint8_t *up, *skip, *weights;
    uint8_t *tokens, *merged;
    const uint16_t *guide, *guide_enc;
    uint16_t *guide_out, *half_out;
    uint16_t shift_x, shift_y;
    f16 s0, s1, s2, s3;
    const uint8_t *embed;
};

template <class T>
__device__ inline T field(const uint8_t *p, int offset) { return *(const T *)(p + offset); }

struct Enc0Args {
    uint8_t bytes[1088];
};

extern "C" __global__ __launch_bounds__(256) void rr2_enc0_kernel_native(Enc0Args) {
    __shared__ __attribute__((aligned(32))) uint8_t lds[stage_lds<32>() + 4096];
    const uint8_t *const P = (const uint8_t *)__builtin_amdgcn_kernarg_segment_ptr();
    const Special S = special_registers();
    uint8_t *const features = lds + stage_lds<32>();

    Enc0View v;
    v.width = field<uint32_t>(P, 0);
    v.height = field<uint32_t>(P, 4);
    v.up = v.skip = nullptr;
    v.weights = field<const uint8_t *>(P, 24);
    v.tokens = field<uint8_t *>(P, 32);
    v.merged = nullptr;
    v.guide = v.guide_enc = nullptr;
    v.guide_out = v.half_out = nullptr;
    v.shift_x = field<uint16_t>(P, 80);
    v.shift_y = field<uint16_t>(P, 82);
    v.s0 = field<f16>(P, 92);
    v.s1 = field<f16>(P, 94);
    v.s2 = field<f16>(P, 96);
    v.s3 = field<f16>(P, 98);
    v.embed = field<const uint8_t *>(P, 104);

    Regs io;
#ifndef TIME_WITHOUT_GATHER
    if (threadIdx.z == 0) {
        enc0_gather(P, S, io);
        unpack_features(io, features, S.lane);
    }
#endif
    __syncthreads();
#ifndef TIME_WITHOUT_STAGE
    stage<32, 2, 4, 0, 0, 0, EMBED, HEAD_ONLY>(lds, v);
#endif
#ifndef TIME_WITHOUT_TAIL
    if (threadIdx.z == 0) {
        pack_head(io, (const uint16_t *)(lds + 320 * 32), S.lane);
        enc0_tail(P, S, io);
    }
#endif
}
