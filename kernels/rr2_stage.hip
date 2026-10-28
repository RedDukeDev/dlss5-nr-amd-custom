// The native ray reconstruction stages (preset F): see rr2_stage.h for what one does.
#include "rr2_stage.h"

// C, heads, MLP branches, channels one level up (merge), channels one level down (expand), channels merged for
// the next encoder
#define STAGE_KERNEL(NAME, BOUNDS, C, NH, NB, CUP, CLO, CM, IN, OUT)                                        \
    extern "C" __global__ __launch_bounds__(BOUNDS) void NAME(StageArgs args) {                             \
        __shared__ __attribute__((aligned(32))) uint8_t lds[stage_lds<C>()];                                \
        stage<C, NH, NB, CUP, CLO, CM, IN, OUT>(lds, args);                                                 \
    }

STAGE_KERNEL(rr2_enc1_kernel_native, 256, 64, 2, 8, 32, 0, 96, MERGE, ENCODER)
STAGE_KERNEL(rr2_enc2_kernel_native, 256, 96, 4, 12, 0, 0, 128, TOKENS, ENCODER)
STAGE_KERNEL(rr2_enc3_kernel_native, 128, 128, 4, 16, 0, 0, 160, TOKENS, ENCODER)
STAGE_KERNEL(rr2_enc4_kernel_native, 256, 160, 8, 20, 0, 0, 160, TOKENS, ENCODER)
STAGE_KERNEL(rr2_enc5_kernel_native, 256, 160, 8, 20, 0, 0, 0, TOKENS, ENCODER)
STAGE_KERNEL(rr2_dec4_kernel_0_native, 256, 160, 8, 20, 0, 160, 0, EXPAND, TOKENS_ONLY)
STAGE_KERNEL(rr2_dec4_kernel_1_native, 256, 160, 8, 20, 0, 0, 0, TOKENS, DECODER_GUIDE)
STAGE_KERNEL(rr2_dec3_kernel_0_native, 128, 128, 4, 16, 0, 160, 0, EXPAND, TOKENS_ONLY)
STAGE_KERNEL(rr2_dec3_kernel_1_native, 128, 128, 4, 16, 0, 0, 0, TOKENS, DECODER_GUIDE)
STAGE_KERNEL(rr2_dec2_kernel_0_native, 128, 96, 4, 12, 0, 128, 0, EXPAND, TOKENS_ONLY)
STAGE_KERNEL(rr2_dec2_kernel_1_native, 128, 96, 4, 12, 0, 0, 0, TOKENS, DECODER_GUIDE)
STAGE_KERNEL(rr2_dec1_kernel_0_native, 256, 64, 2, 8, 0, 96, 0, EXPAND, TOKENS_ONLY)
STAGE_KERNEL(rr2_dec1_kernel_1_native, 256, 64, 2, 8, 0, 0, 0, TOKENS, DECODER_GUIDE)
STAGE_KERNEL(rr2_dec0_kernel_0_native, 256, 32, 2, 4, 0, 64, 0, EXPAND, TOKENS_ONLY)
