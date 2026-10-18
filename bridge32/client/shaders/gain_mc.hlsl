// Applies the network's gain to the game's frame, following the motion found since the network saw
// it, and fades the gain out where, even so, the picture is not what the network saw: a result a
// few frames old, laid on something that has moved, shows as a copy of the moving thing trailing
// behind it.
//
//   s0  the game's frame now
//   s1  the frame the network saw
//   s2  the gain, as value / 2 (0.5 is a gain of 1)
//   s3  the motion: where in the old picture each block of the new one came from, in pixels
//   c0  xy: two texels in texture coordinates; z: the difference where the gain starts to fade;
//       w: one over the width of that fade
//   c1  xy: one pixel in texture coordinates; z: the cost of a match below which its motion is trusted
//       fully; w: one over how far above that the trust falls to nothing (a block whose best match is
//       a poor one has found nothing, and is left where it is)
sampler2D frame_now : register(s0);
sampler2D frame_seen : register(s1);
sampler2D gain_map : register(s2);
sampler2D motion : register(s3);
float4 k : register(c0);
float4 pixel : register(c1);

float4 main(float2 uv : TEXCOORD0) : COLOR0 {
    float4 found = tex2D(motion, uv);
    float trust = saturate(1.0 - (found.z - pixel.z) * pixel.w);
    float2 from = uv + found.xy * trust * pixel.xy;
    float3 now = tex2D(frame_now, uv).rgb;
    float3 gain = tex2D(gain_map, from).rgb * 2.0;
    float3 a = abs(now - tex2D(frame_seen, from).rgb);
    float3 b = abs(tex2D(frame_now, uv + float2(k.x, 0)).rgb - tex2D(frame_seen, from + float2(k.x, 0)).rgb);
    float3 c = abs(tex2D(frame_now, uv - float2(k.x, 0)).rgb - tex2D(frame_seen, from - float2(k.x, 0)).rgb);
    float3 d = abs(tex2D(frame_now, uv + float2(0, k.y)).rgb - tex2D(frame_seen, from + float2(0, k.y)).rgb);
    float3 e = abs(tex2D(frame_now, uv - float2(0, k.y)).rgb - tex2D(frame_seen, from - float2(0, k.y)).rgb);
    float3 m = max(max(max(a, b), max(c, d)), e);
    float change = max(max(m.r, m.g), m.b);
    float keep = saturate(1.0 - (change - k.z) * k.w);
    keep = keep * keep * (3.0 - 2.0 * keep);
    return float4(now * lerp(float3(1.0, 1.0, 1.0), gain, keep), 1.0);
}
