// One triangle that covers the whole target, from the vertex number alone: no vertex buffer, no input layout.
struct Out {
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
};

Out main(uint id : SV_VertexID) {
    Out o;
    float2 uv = float2((id << 1) & 2, id & 2);
    o.pos = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    o.uv = uv;
    return o;
}
