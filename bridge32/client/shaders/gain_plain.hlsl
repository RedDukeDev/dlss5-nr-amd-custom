// Applies the network's gain to the game's frame where the scene still looks as it did when the
// network saw it, and fades it out where it does not: a result that is a few frames old, laid on
// something that has moved, shows as a copy of the moving thing trailing behind it.
//
//   s0  the game's frame now
//   s4  the same, at the size of the network's picture, for the comparison
//   s1  the frame the network saw
//   s2  the gain, as value / 2 (0.5 is a gain of 1)
//   c0  xy: two texels in texture coordinates; z: the difference where the gain starts to fade;
//       w: one over the width of that fade
sampler2D frame_now : register(s0);
sampler2D frame_seen : register(s1);
sampler2D gain_map : register(s2);
sampler2D frame_small : register(s4);
float4 k : register(c0);

float diff(float2 uv) {
    float3 d = abs(tex2D(frame_small, uv).rgb - tex2D(frame_seen, uv).rgb);
    return max(max(d.r, d.g), d.b);
}

float4 main(float2 uv : TEXCOORD0) : COLOR0 {
    float3 now = tex2D(frame_now, uv).rgb;
    float3 gain = tex2D(gain_map, uv).rgb * 2.0;
    // The change is looked for around the pixel too: the edge of something that moved changes
    // its neighbours' gain as much as its own.
    float change = max(max(diff(uv), diff(uv + float2(k.x, 0))), max(diff(uv - float2(k.x, 0)), max(diff(uv + float2(0, k.y)), diff(uv - float2(0, k.y)))));
    float keep = saturate(1.0 - (change - k.z) * k.w);
    keep = keep * keep * (3.0 - 2.0 * keep);
    return float4(now * lerp(float3(1.0, 1.0, 1.0), gain, keep), 1.0);
}
