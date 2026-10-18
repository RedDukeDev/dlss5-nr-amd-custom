// The coarse motion search: for every 16x16 block of the picture, where in the old picture did it
// come from? Compares a 3x3 window of the 1/16-size picture of the picture now with the old one
// shifted by every candidate, and keeps the best.
//   s0  the picture now at 1/16, s1 the old one
//   c0  xy: texel size of those; z: the cost of one texel of motion (so that a flat area, where
//       everything matches, stays at no motion)
// Output: x, y the motion in texels, z the cost of the best candidate.
sampler2D now_tex : register(s0);
sampler2D old_tex : register(s1);
float4 k : register(c0);

float4 main(float2 uv : TEXCOORD0) : COLOR0 {
    float3 now[9];
    [unroll] for (int i = 0; i < 9; ++i)
        now[i] = tex2Dlod(now_tex, float4(uv + float2(i % 3 - 1, i / 3 - 1) * k.xy, 0, 0)).rgb;
    float best = 1.0e9;
    float2 best_d = float2(0, 0);
    [loop] for (int dy = -12; dy <= 12; ++dy) {
        [loop] for (int dx = -12; dx <= 12; ++dx) {
            float2 shifted = uv + float2(dx, dy) * k.xy;
            float cost = k.z * (abs(dx) + abs(dy));
            [unroll] for (int j = 0; j < 9; ++j) {
                float3 d = abs(now[j] - tex2Dlod(old_tex, float4(shifted + float2(j % 3 - 1, j / 3 - 1) * k.xy, 0, 0)).rgb);
                cost += d.r + d.g + d.b;
            }
            if (cost < best) {
                best = cost;
                best_d = float2(dx, dy);
            }
        }
    }
    return float4(best_d, best, 1.0);
}
