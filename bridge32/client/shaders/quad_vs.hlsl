// A quad in clip space with its texture coordinates, for the passes that need a Shader Model 3
// pixel shader (which has to be paired with a vertex shader).
struct In {
    float4 pos : POSITION;
    float2 uv : TEXCOORD0;
};
struct Out {
    float4 pos : POSITION;
    float2 uv : TEXCOORD0;
};
Out main(In i) {
    Out o;
    o.pos = i.pos;
    o.uv = i.uv;
    return o;
}
