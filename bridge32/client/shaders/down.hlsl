// The colour of a picture, averaged over blocks of 4x4 pixels: one level of the pyramid the
// motion search runs on.
//   s0  the picture
//   c0  xy: its texel size
sampler2D src : register(s0);
float4 k : register(c0);

float4 main(float2 uv : TEXCOORD0) : COLOR0 {
    float3 sum = float3(0, 0, 0);
    [unroll] for (int y = 0; y < 4; ++y)
        [unroll] for (int x = 0; x < 4; ++x)
            sum += tex2D(src, uv + float2(x - 1.5, y - 1.5) * k.xy).rgb;
    return float4(sum * (1.0 / 16.0), 1.0);
}
