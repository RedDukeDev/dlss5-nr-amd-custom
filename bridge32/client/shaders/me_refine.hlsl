// The fine motion search: around what the coarse one found, for every 16x16 block, compares its
// 4x4 texels of the 1/4-size picture with the old picture shifted by a few texels either way.
//   s0  the picture now at 1/4, s1 the old one, s2 the coarse motion
//   c0  xy: texel size of the 1/4 pictures; z: the cost of a texel of change; w: unused
//   c1  xy: pixels in a texel of the 1/16 pictures, z: pixels in a texel of the 1/4 ones (x), w: (y)
// Output: x, y the motion in pixels, z the cost of the best candidate.
sampler2D now_tex : register(s0);
sampler2D old_tex : register(s1);
sampler2D coarse_tex : register(s2);
float4 k : register(c0);
float4 scale : register(c1);

float4 main(float2 uv : TEXCOORD0) : COLOR0 {
    // The coarse motion, in texels of the 1/16 picture, as texels of this one (four times as many).
    float2 start = tex2Dlod(coarse_tex, float4(uv, 0, 0)).xy * 4.0;
    float3 now[16];
    [unroll] for (int i = 0; i < 16; ++i)
        now[i] = tex2Dlod(now_tex, float4(uv + float2(i % 4 - 1.5, i / 4 - 1.5) * k.xy, 0, 0)).rgb;
    float best = 1.0e9;
    float2 best_d = start;
    [loop] for (int dy = -3; dy <= 3; ++dy) {
        [loop] for (int dx = -3; dx <= 3; ++dx) {
            float2 d = start + float2(dx, dy);
            float2 shifted = uv + d * k.xy;
            float cost = k.z * (abs(dx) + abs(dy));
            [unroll] for (int j = 0; j < 16; ++j) {
                float3 d = abs(now[j] - tex2Dlod(old_tex, float4(shifted + float2(j % 4 - 1.5, j / 4 - 1.5) * k.xy, 0, 0)).rgb);
                cost += d.r + d.g + d.b;
            }
            if (cost < best) {
                best = cost;
                best_d = d;
            }
        }
    }
    return float4(best_d * scale.zw, best, 1.0);
}
