// The game's frame times the network's gain (stored as value / 2: 0.5 is a gain of 1), over the whole
// frame: the gain is at the network's resolution and is stretched. The product is taken on the values as
// the backbuffer holds them, which is where the gain was measured; for a backbuffer that is sRGB the
// result is turned back to linear light, which the target's own conversion then undoes.
Texture2D frame : register(t0);
Texture2D gain_map : register(t1);
SamplerState point_smp : register(s0);
SamplerState linear_smp : register(s1);
cbuffer Constants : register(b0) {
    float4 k;   // x: 1 when the target is sRGB
};

float3 to_linear(float3 c) {
    return lerp(c / 12.92, pow((c + 0.055) / 1.055, 2.4), step(0.04045, c));
}

float4 main(float4 pos : SV_Position, float2 uv : TEXCOORD0) : SV_Target {
    float4 f = frame.Sample(point_smp, uv);
    float3 g = gain_map.Sample(linear_smp, uv).rgb * 2.0;
    float3 o = saturate(f.rgb * g);
    if (k.x > 0.5) o = to_linear(o);
    return float4(o, f.a);
}
