// The frame at the size of the network's picture (filtered when it is smaller), into a target whose memory
// order is the one the host reads: whatever the game's backbuffer looks like, this is blue, green, red.
Texture2D frame : register(t0);
SamplerState smp : register(s0);

float4 main(float4 pos : SV_Position, float2 uv : TEXCOORD0) : SV_Target {
    return float4(frame.Sample(smp, uv).rgb, 1.0);
}
