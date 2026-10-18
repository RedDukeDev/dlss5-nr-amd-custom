// Takes the stray vectors out of the motion field: each block gets the median, per component, of
// its own vector and its eight neighbours'. A block whose best match was an accident, in a patch of
// the picture with little to match on, takes what the blocks around it found; at the edge of
// something that moves, where the neighbours disagree, the median keeps the edge sharp.
//   s0  the field
//   c0  xy: texel size of the field
sampler2D field : register(s0);
float4 k : register(c0);

void order(inout float a, inout float b) {
    float lo = min(a, b);
    b = max(a, b);
    a = lo;
}

// The median of nine values, by a sorting network.
float median9(float v[9]) {
    order(v[0], v[1]); order(v[3], v[4]); order(v[6], v[7]);
    order(v[1], v[2]); order(v[4], v[5]); order(v[7], v[8]);
    order(v[0], v[1]); order(v[3], v[4]); order(v[6], v[7]);
    order(v[0], v[3]); order(v[5], v[8]); order(v[4], v[7]);
    order(v[3], v[6]); order(v[1], v[4]); order(v[2], v[5]);
    order(v[4], v[7]); order(v[4], v[2]); order(v[6], v[4]);
    order(v[4], v[2]);
    return v[4];
}

float4 main(float2 uv : TEXCOORD0) : COLOR0 {
    float xs[9];
    float ys[9];
    [unroll] for (int i = 0; i < 9; ++i) {
        float4 m = tex2Dlod(field, float4(uv + float2(i % 3 - 1, i / 3 - 1) * k.xy, 0, 0));
        xs[i] = m.x;
        ys[i] = m.y;
    }
    float4 own = tex2Dlod(field, float4(uv, 0, 0));
    return float4(median9(xs), median9(ys), own.z, 1.0);
}
