// The runtime's compute passes, as HLSL compiled when the runtime starts.
//
// All of them share one root signature (passes.cpp): 24 root constants at b0,
// eight shader resources at t0..t7, four unordered access views at u0..u3, a
// linear and a point sampler.
//
// Colour. The network was trained on finished SDR pictures: sRGB encoded, in
// [0, 1]. A scene-referred frame is shown to it as a proxy of that: divided by
// its white point (pre-exposure over exposure), highlights rolled off by a
// soft knee on luminance, then sRGB encoded. No tone curve beyond the knee:
// the game tone maps the picture later, and tone mapping it here as well shows
// the network a doubly compressed, flat image. A frame that is display encoded
// already is shown as it is.
//
// What the network returns is not added back as a difference. The difference
// between its answer and its input is kept in the proxy's linear space (the
// residual), carried to the frame being composed along the motion vectors,
// laid on that frame's own proxy, and the result is brought back as a
// luminance ratio against the frame itself, with the network's hue. Additive
// composition loses what the network does to highlights and lets colour run
// away; a ratio keeps both ends of every blend a real colour.
//
// The composition follows the one of RenoDX's DLSS 5 add-on by clshortfuse
// (https://github.com/clshortfuse/renodx, MIT), as described by the
// OptiScaler-DLSSNR fork that adopted it; this is an independent
// implementation. See THIRD_PARTY.md.

#pragma once

#include <cstdint>

namespace dlss5nr {

// FNV-1a of the source below. core/shaders_dxbc.h, the shaders already compiled
// (tools/gen_dxbc.cpp), records the one of the source it was made from, and is
// only used while the two agree.
inline constexpr uint64_t shader_hash(const char *s) {
    uint64_t h = 1469598103934665603ull;
    for (; *s; ++s) h = (h ^ (unsigned char)*s) * 1099511628211ull;
    return h;
}

inline constexpr char kShaderSource[] = R"HLSL(
cbuffer Constants : register(b0)
{
    uint2  size;            // the dispatch domain, in pixels
    float2 inv_size;
    float2 uv0_scale;       // region / texture size of t0
    float2 uv1_scale;       // of t1
    float2 uv2_scale;       // of t2
    float2 motion_to_px;    // raw motion vector -> pixels of the dispatch domain
    uint   encoding;        // 0 linear HDR, 1 display encoded, 2 linear LDR
    float  colour_strength; // 0 keeps the game's hue, 1 takes the network's
    float  exposure_mul;
    float  pre_exposure;
    float  strength;        // detail strength: 0 the game's image, 1 the network's
    float  max_ratio;       // the most the composition may brighten or darken, >= 1
    float  fade;
    uint   flags;           // see FLAG_ below
    uint   exposure_mode;   // 0 auto, 1 game texture, 2 fixed
    float  adaptation;      // 0..1, how far the exposure moves towards its target per frame
    float  blend;           // compose: how much of the newest result, the rest from the one before
    uint   pad;
};

static const uint FLAG_RESET       = 1;
static const uint FLAG_COPY        = 4;   // compose without a residual
static const uint FLAG_DEPTH_INV   = 8;
static const uint FLAG_REPLACE     = 16;  // the network's answer as it is
static const uint FLAG_IDENTITY    = 32;  // a displacement map that is still zero
static const uint FLAG_KEEP_DEPTH  = 64;  // track: keep this frame's depth
static const uint FLAG_DEPTH_CHECK = 128; // track: the previous frame's depth is there
static const uint FLAG_FILL        = 256; // compose: fill in what the network never saw
static const uint FLAG_SHOW_TRACKING = 512;
static const uint FLAG_PREVIOUS    = 1024; // compose: the result before this one is there
static const uint FLAG_SYNC        = 2048; // compose: the frame waited for the network; t7 says if it answered
static const uint FLAG_RAWMV       = 4096; // capture: the game's motion vectors (t2) are the way back to the capture before
static const uint FLAG_EDGES       = 8192; // compose: how far each pixel of the results was from an edge (t8, t9)

static const float3 LUMA = float3(0.2126, 0.7152, 0.0722);

Texture2D<float4> t0 : register(t0);
Texture2D<float4> t1 : register(t1);
Texture2D<float4> t2 : register(t2);
Texture2D<float4> t3 : register(t3);
Texture2D<float4> t4 : register(t4);
Texture2D<float4> t5 : register(t5);
Texture2D<float4> t6 : register(t6);
Texture2D<float4> t7 : register(t7);
Texture2D<float4> t8 : register(t8);
Texture2D<float4> t9 : register(t9);
RWTexture2D<float4> u0 : register(u0);
RWTexture2D<float4> u1 : register(u1);
RWTexture2D<float4> u2 : register(u2);
RWByteAddressBuffer u3 : register(u3);
SamplerState linear_clamp : register(s0);
SamplerState point_clamp : register(s1);

float luminance(float3 c) { return dot(c, LUMA); }

float3 srgb_encode(float3 c)
{
    c = saturate(c);
    return c <= 0.0031308 ? c * 12.92 : 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

float3 srgb_decode(float3 c)
{
    c = saturate(c);
    return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4);
}

float exposure()
{
    return exposure_mode == 2 ? exposure_mul : asfloat(u3.Load(0)) * exposure_mul;
}

// The divisor that puts the frame's paper white at 1.
float white_point()
{
    return encoding == 1 ? 1.0 : clamp(max(pre_exposure, 1e-6) / max(exposure(), 1e-6), 1e-3, 1e5);
}

// Luminance above 0.75 approaches 1 instead of passing it, and a channel
// still above 1 brings the whole colour down with it: hue kept, nothing
// clipped into a flat field whose blown pixels change from frame to frame.
float3 soft_knee(float3 c)
{
    const float knee = 0.75;
    float y = luminance(c);
    if (y > knee) c *= (knee + (1.0 - knee) * (1.0 - exp(-(y - knee) / (1.0 - knee)))) / y;
    float peak = max(c.r, max(c.g, c.b));
    return peak > 1.0 ? c / peak : c;
}

// The frame, divided by its white point -> the proxy the network is shown,
// both linear. Pure, so the composition can rebuild it for any frame.
float3 proxy_of(float3 normalised)
{
    normalised = max(normalised, 0.0);
    return saturate(encoding == 0 ? soft_knee(normalised) : normalised);
}

// A proxy -> what the network reads, and back.
float3 to_net(float3 proxy) { return encoding == 1 ? proxy : srgb_encode(proxy); }
float3 from_net(float3 n) { return encoding == 1 ? saturate(n) : srgb_decode(n); }

// ------------------------------------------------------------------- OkLab
// Bjorn Ottosson's OkLab (https://bottosson.github.io/posts/oklab/), on
// linear BT.709.
float3 to_oklab(float3 c)
{
    float3 lms = float3(dot(c, float3(0.4122214708, 0.5363325363, 0.0514459929)),
                        dot(c, float3(0.2119034982, 0.6806995451, 0.1073969566)),
                        dot(c, float3(0.0883024619, 0.2817188376, 0.6299787005)));
    lms = sign(lms) * pow(abs(lms), 1.0 / 3.0);
    return float3(dot(lms, float3(0.2104542553, 0.7936177850, -0.0040720468)),
                  dot(lms, float3(1.9779984951, -2.4285922050, 0.4505937099)),
                  dot(lms, float3(0.0259040371, 0.7827717662, -0.8086757660)));
}

float3 from_oklab(float3 lab)
{
    float3 lms = float3(dot(lab, float3(1.0, 0.3963377774, 0.2158037573)),
                        dot(lab, float3(1.0, -0.1055613458, -0.0638541728)),
                        dot(lab, float3(1.0, -0.0894841775, -1.2914855480)));
    lms = lms * lms * lms;
    return float3(dot(lms, float3(4.0767416621, -3.3077115913, 0.2309699292)),
                  dot(lms, float3(-1.2684380046, 2.6097574011, -0.3413193965)),
                  dot(lms, float3(-0.0041960863, -0.7034186147, 1.7076147010)));
}

// A colour with a negative channel is moved towards the grey of the same
// luminance until it has none: desaturated, not clipped, luminance kept.
float3 into_gamut(float3 c)
{
    float y = luminance(c);
    float3 scale = c < 0.0 ? y / max(y - c, 1e-8) : 1.0;
    float k = saturate(min(scale.r, min(scale.g, scale.b)));
    float3 result = y <= 1e-8 ? max(c, 0.0) : (k >= 1.0 ? c : y + (c - y) * k);
    return all(isfinite(result)) ? result : 0.0;
}

// The lightness and chroma of `c`, with the hue of `hue_of`.
float3 with_hue(float3 c, float3 hue_of)
{
    float3 lab = to_oklab(c);
    float2 ab = to_oklab(hue_of).yz;
    float target = length(ab);
    lab.yz = target > 1e-5 ? ab / target * length(lab.yz) : 0.0;
    return into_gamut(from_oklab(lab));
}

// ---------------------------------------------------------------- exposure
// One group of 256 threads samples a 64 x 64 grid of the image (t0), and
// moves the exposure in u3 towards the one that puts the log-average
// luminance at middle grey. Mode 1 takes the game's own texture (t1) instead.
groupshared float g_sum[256];

[numthreads(16, 16, 1)]
void exposure_main(uint3 tid : SV_GroupThreadID, uint index : SV_GroupIndex)
{
    float sum = 0.0;
    [unroll] for (uint j = 0; j < 4; ++j)
    [unroll] for (uint i = 0; i < 4; ++i) {
        float2 uv = (float2(tid.x * 4 + i, tid.y * 4 + j) + 0.5) / 64.0;
        float3 c = t0.SampleLevel(linear_clamp, uv * uv0_scale, 0).rgb;
        sum += log2(max(luminance(c / max(pre_exposure, 1e-6)), 1e-4));
    }
    g_sum[index] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (index < stride) g_sum[index] += g_sum[index + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (index != 0) return;

    float target;
    if (exposure_mode == 1) target = max(t1.Load(int3(0, 0, 0)).r, 1e-6);
    else target = 0.18 / exp2(g_sum[0] / 4096.0);
    target = clamp(target, 1e-4, 1e4);

    float current = asfloat(u3.Load(0));
    bool valid = u3.Load(4) == 1;
    float next = (!valid || (flags & FLAG_RESET)) ? target
               : exp2(lerp(log2(max(current, 1e-6)), log2(target), adaptation));
    u3.Store(0, asuint(next));
    u3.Store(4, 1);
}

// ------------------------------------------------------------------ track
// A displacement map says, for every pixel of this frame, where the same
// point of the scene was in an earlier frame: the frame a slot was captured
// in, the frame of the last capture. All at the network's size, in its
// pixels, pointing from now to then, the way motion vectors do. A pixel whose
// point was not visible then -- outside the image, or behind something else --
// holds GONE.
static const float GONE = 1e9;

bool gone(float2 d) { return d.x >= 1e8; }

// Two depths of the same surface in the same frame: within `tolerance` of each other,
// relative to the depth and also to one minus it. Depth buffers are not linear: with
// the far plane at 1 everything a few metres off is crowded under 1 and only
// 1 - depth tells it apart, with it at 0 (reversed) only the depth does; whichever
// way the game has it, the one that tells them apart decides. No floor beyond what
// a float holds: a mountain a few kilometres off has a depth of a few 1e-5 against
// the sky's 0, and is still not the sky.
bool same_depth(float a, float b, float tolerance)
{
    float apart = abs(a - b);
    return apart <= max(tolerance * max(abs(a), abs(b)), 1e-9) &&
           apart <= max(tolerance * max(abs(1.0 - a), abs(1.0 - b)), 1e-9);
}

// One over the distance, up to a constant: the depth with the far plane at 0
// (FLAG_DEPTH_INV), 1 - depth otherwise (a few metres out).
float closeness(float d) { return (flags & FLAG_DEPTH_INV) ? d : 1.0 - d; }

// The depth `before` of a point in an earlier frame and `now` in this one, of the same
// surface: the camera moves (orbiting a character, walking), so the same surface gets
// nearer or farther from frame to frame, by more than any fixed tolerance allows.
// What says it was hidden then is something nearer there: by more than `nearer`, as a
// fraction of its distance. Farther then is the camera having come closer, up to
// `farther`. Depth is taken as proportional to one over the distance, which it is with
// the far plane at 0 (FLAG_DEPTH_INV) and, a few metres out, as 1 - depth otherwise.
bool continues(float before, float now, float nearer, float farther)
{
    float wb = closeness(before);
    float wn = closeness(now);
    if (max(wb, wn) < 1e-12) return true;  // both the sky
    float ratio = wb / max(wn, 1e-12);     // the distance now over the distance then
    return ratio <= 1.0 + nearer && ratio * (1.0 + farther) >= 1.0;
}

// A depth into a half float and back, as the logarithm of its closeness: the same
// relative precision (about 2%) at any distance, a hand's breadth or kilometres off,
// where the depth itself would lose everything past a few hundred metres. The sky
// (closeness 0) comes back as 2^-100, which nothing else is near.
float pack_depth(float d) { return log2(max(closeness(d), 1e-30)); }
float unpack_depth(float p)
{
    float c = exp2(p);
    return (flags & FLAG_DEPTH_INV) ? c : 1.0 - c;
}

// The game's motion vector for the point at `uv` (texture coordinates of the
// network's size), in pixels of it: taken from the nearest surface among the 3 x 3
// depth texels around it (t1, the game's depth; t2, its vectors), as every temporal
// anti-aliasing does. At an outline the vector under a pixel can be the background's
// while the pixel shows what is in front, and that is the one that moved.
float2 nearest_motion(float2 uv)
{
    uint w, h;
    t1.GetDimensions(w, h);
    const float2 texel = 1.0 / float2(w, h);
    const float2 centre = uv * uv1_scale;
    float2 best = centre;
    float nearest = -1.0;
    [unroll] for (int j = -1; j <= 1; ++j)
    [unroll] for (int i = -1; i <= 1; ++i) {
        float2 p = centre + float2(i, j) * texel;
        float c = closeness(t1.SampleLevel(point_clamp, p, 0).r);
        if (c > nearest) {
            nearest = c;
            best = p;
        }
    }
    return t2.SampleLevel(point_clamp, best / uv1_scale * uv2_scale, 0).rg * motion_to_px;
}

// Moves a displacement map (t0) on by one frame, into u0: the game's motion
// vectors (t2) say where each pixel was in the previous frame, and the map
// there says where that was then. FLAG_IDENTITY: the earlier frame is the
// previous one, and the map starts from nothing.
//
// Occlusion: the depth of the previous frame (t3) where the pixel came from
// has to agree with this pixel's depth (t1), allowing for the camera having
// moved (see "continues"). Where something nearer was there, the pixel was
// covered then, and what the map holds there belongs to whatever covered it.
// FLAG_KEEP_DEPTH also writes this frame's depth (u1), for the next frame.
[numthreads(8, 8, 1)]
void track_main(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= size)) return;
    float2 uv = (float2(id.xy) + 0.5) * inv_size;
    float2 mv = nearest_motion(uv);
    float depth = t1.SampleLevel(point_clamp, uv * uv1_scale, 0).r;
    if (flags & FLAG_KEEP_DEPTH) u1[id.xy] = float4(depth, 0, 0, 0);

    float2 from = float2(id.xy) + 0.5 + mv;
    float2 d = GONE;
    if (all(from >= 0.0) && all(from < float2(size))) {
        int2 at = int2(from);
        float before = t3.Load(int3(at, 0)).r;
        bool seen = !(flags & FLAG_DEPTH_CHECK) || continues(before, depth, 0.10, 0.30);
        float2 earlier = (flags & FLAG_IDENTITY) ? 0.0 : t0.Load(int3(at, 0)).rg;
        if (seen && !gone(earlier)) d = mv + earlier;
    }
    u0[id.xy] = float4(d, 0, 0);
}

// ----------------------------------------------------------------- capture
// Into a slot, at the network's size: colour (t0) as the network reads it
// (u0), depth (t1, u1), and as motion vectors (u2) the displacement to the
// frame of the previous capture (t3), which is the frame the network's own
// history holds. FLAG_IDENTITY: there is no previous capture.
[numthreads(8, 8, 1)]
void capture_main(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= size)) return;
    float2 uv = (float2(id.xy) + 0.5) * inv_size;
    float3 c = t0.SampleLevel(linear_clamp, uv * uv0_scale, 0).rgb;
    u0[id.xy] = float4(to_net(proxy_of(c / white_point())), 1.0);
    u1[id.xy] = float4(t1.SampleLevel(point_clamp, uv * uv1_scale, 0).r, 0, 0, 0);
    float2 d = (flags & FLAG_IDENTITY) ? 0.0 : t3.Load(int3(id.xy, 0)).rg;
    // When the capture before was the previous frame, its way back is one frame of motion.
    if (flags & FLAG_RAWMV) d = t2.SampleLevel(point_clamp, uv * uv2_scale, 0).rg * motion_to_px;
    u2[id.xy] = float4(gone(d) ? 0.0 : d, 0, 0);
}

// ----------------------------------------------------------------- residual
// A fresh result: network output (t0) minus network input (t1), both in the
// proxy's linear space, into u0. It stays where the network put it, in the
// pixels of the frame it was captured in; the composition finds it through
// that slot's displacement map. It is never blended with older answers, which
// the network re-decides with every framing. The capture's depth (t2) goes
// with it, in alpha: it says what surface each of its pixels showed.
// What a pixel looks like, as the network reads it (`n`): its luminance, and its
// colour as red and blue less green. Compared with this frame's around the same
// point, to tell whether an answer still belongs to what is shown there: a halo the
// network put on the sky beside a dark sail can be as bright as the sky and still
// not be the sky's colour.
float3 print(float3 n) { return float3(luminance(n), n.r - n.g, n.b - n.g); }

// How far a pixel is from the outline of something in front of it, up to
// EDGE_REACH pixels of the network's size (EDGE_REACH: none that near), `uv` and
// `texel` in the texture coordinates of the depth. The network lights up or
// darkens the pixels of the background around the outline of something nearer: a
// halo that belongs to the outline, not to the pixels it is on, and that has to go
// when the outline moves away. Only a neighbour clearly nearer counts, by more than
// a fifth of the distance: a slope, or the bumps of a rock, change the depth by
// more than any small tolerance over a few pixels, and are not outlines.
static const int EDGE_STEPS = 5;
static const float EDGE_DISTANCE[EDGE_STEPS] = {1, 2, 4, 6, 8};
static const float EDGE_REACH = 10.0;
static const float2 EDGE_DIRECTIONS[8] = {float2(1, 0), float2(-1, 0), float2(0, 1), float2(0, -1),
                                          float2(0.7071, 0.7071), float2(-0.7071, 0.7071),
                                          float2(0.7071, -0.7071), float2(-0.7071, -0.7071)};

float edge_distance(Texture2D<float4> depth_of, float2 uv, float2 texel)
{
    float here = closeness(depth_of.SampleLevel(point_clamp, uv, 0).r);
    float found = EDGE_REACH;
    [unroll] for (int j = EDGE_STEPS - 1; j >= 0; --j) {
        [unroll] for (int i = 0; i < 8; ++i) {
            float2 spot = clamp(uv + EDGE_DIRECTIONS[i] * EDGE_DISTANCE[j] * texel, 0.0, 1.0);
            float there = closeness(depth_of.SampleLevel(point_clamp, spot, 0).r);
            if (there > here * 1.25 + 1e-12) found = EDGE_DISTANCE[j];
        }
    }
    return found;
}

[numthreads(8, 8, 1)]
void refresh_main(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= size)) return;
    // How far each pixel was from an edge, and what the network saw there (t1, already
    // as it reads it): its luminance and its colour (see "print"), beside the residual (u1).
    float3 seen = print(t1.Load(int3(id.xy, 0)).rgb);
    u1[id.xy] = float4(edge_distance(t2, (float2(id.xy) + 0.5) * inv_size, inv_size), seen);
    float3 r = from_net(t0.Load(int3(id.xy, 0)).rgb) - from_net(t1.Load(int3(id.xy, 0)).rgb);
    u0[id.xy] = float4(r, pack_depth(t2.Load(int3(id.xy, 0)).r));
}

// What the network did to the light of a place, without what it did to its
// detail: its output (t0) over its input (t1), each averaged over 16 x 16
// pixels, into u0, a sixteenth of their size each way (uv1_scale holds the
// inverse of their size). A gain, not a difference: the network relights, and
// the same relighting is a large difference on a bright pixel and a small one
// on a dark pixel.
[numthreads(8, 8, 1)]
void soften_main(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= size)) return;
    float3 after = 0.0, before = 0.0;
    [unroll] for (int j = 0; j < 8; ++j)
    [unroll] for (int i = 0; i < 8; ++i) {
        float2 uv = (float2(id.xy) * 16.0 + float2(i, j) * 2.0 + 1.0) * uv1_scale;
        after += from_net(t0.SampleLevel(point_clamp, uv, 0).rgb);
        before += from_net(t1.SampleLevel(point_clamp, uv, 0).rgb);
    }
    // The floor keeps the gain near 1 where there is almost no light to
    // compare.
    const float floor_sum = 64.0 * 0.02;
    u0[id.xy] = float4(clamp((after + floor_sum) / (before + floor_sum), 0.25, 4.0), 1.0);
}

// ----------------------------------------------------------------- compose

// The residual moves `proxy` in its own direction as far as it can go and
// stay inside the unit cube; clamping per channel would bend the hue.
float3 apply_residual(float3 proxy, float3 r)
{
    float travel = 1.0;
    [unroll] for (int i = 0; i < 3; ++i) {
        if (r[i] > 1e-6) travel = min(travel, (1.0 - proxy[i]) / r[i]);
        else if (r[i] < -1e-6) travel = min(travel, -proxy[i] / r[i]);
    }
    return proxy + saturate(travel) * r;
}

// `frame` divided by its white point, `r` the residual here -> the composed
// colour, in the same space.
float3 compose(float3 frame, float3 r)
{
    float3 proxy = proxy_of(frame);
    float3 answer = apply_residual(proxy, r);

    float frame_y = luminance(frame);
    float proxy_y = luminance(proxy);
    float answer_y = luminance(answer);

    // Where the frame is no brighter than its proxy, the answer is scaled to
    // the frame's luminance. Above, the frame has light the knee took away
    // and the network never saw: it goes back on top of the answer.
    float ratio = frame_y < proxy_y ? frame_y / max(proxy_y, 1e-6)
                                    : (answer_y + (frame_y - proxy_y)) / max(answer_y, 1e-5);
    float3 detailed = lerp(frame, with_hue(answer * ratio, answer), saturate(strength));

    // As a ratio against the frame, floored so a pixel with almost no light
    // is not made twice as bright by a tiny change (it boils); raised to a
    // power for a strength above 1, and bounded both ways.
    const float floor_y = 1.0 / 512.0;
    float change = (luminance(detailed) + floor_y) / (frame_y + floor_y);
    float bounded = pow(max(change, 1e-6), 1.0 + max(strength - 1.0, 0.0));
    float guard = max(max_ratio, 1.0);
    bounded = clamp(bounded, 1.0 / guard, guard);
    detailed *= bounded / max(change, 1e-6);

    // Colour strength: from the frame's own hue at the same light, to the
    // network's; above 1 the network's chroma grows further.
    float3 result = lerp(frame * bounded, detailed, min(colour_strength, 1.0));
    if (colour_strength > 1.0) {
        float3 lab = to_oklab(max(result, 0.0));
        lab.yz *= colour_strength;
        result = into_gamut(from_oklab(lab));
    }
    // An empty answer is a network that could not read its input: keep the
    // frame rather than scale it to black.
    return answer_y <= 1e-5 ? frame : result;
}

// ---- halos left behind
//
// A pixel of a result that was near an edge when the network looked (its distance
// in `edges`) and is not as near one now (`now`) carries a halo that belongs to
// that edge. It takes instead what the result did around it, on its own surface
// and away from any edge: the network's own light and detail for that surface,
// without the halo. `residual` and `edges` at `at`, in texture coordinates of the
// network's size; `texel` one pixel of it.
static const float HALO_REACH[4] = {4, 8, 12, 16};

float3 without_halo(Texture2D<float4> residual, Texture2D<float4> edges, float2 at, float now, float2 texel,
                    float3 r, inout float taken)
{
    float then = edges.SampleLevel(point_clamp, at, 0).r;
    if (then >= EDGE_REACH && now >= EDGE_REACH) return r;
    float moved = saturate((abs(then - now) - 1.0) / 2.0);
    if (moved <= 0.0) return r;
    float depth = unpack_depth(residual.SampleLevel(point_clamp, at, 0).a);
    float3 sum = 0.0;
    float count = 0.0;
    [unroll] for (int j = 0; j < 4; ++j) {
        if (count > 0.0) continue;   // found nearer already
        [unroll] for (int i = 0; i < 8; ++i) {
            float2 spot = at + EDGE_DIRECTIONS[i] * (then + HALO_REACH[j]) * texel;
            if (any(spot < 0.0) || any(spot > 1.0)) continue;
            float4 v = residual.SampleLevel(point_clamp, spot, 0);
            if (edges.SampleLevel(point_clamp, spot, 0).r < EDGE_REACH) continue;
            if (!continues(unpack_depth(v.a), depth, 0.15, 0.15)) continue;   // the same surface
            sum += v.rgb;
            count += 1.0;
        }
    }
    if (count > 0.0) taken = max(taken, moved);
    return count > 0.0 ? lerp(r, sum / count, moved) : r;
}

// ---- answers that are not this pixel's
//
// The game's motion vectors leave out some motion: whatever its vertex shaders move
// (sails turning, leaves and grass in the wind) moves on the screen with no vector to
// say so, and the tracking carries the network's answer for it along with what is
// behind. Left there, it shows as a copy of the thing where it was. What the network
// saw where the answer comes from (`seen`, the luminance as the network reads it) has
// to be something this frame shows around this pixel (`lo` to `hi`, the same for the
// pixels of this frame within a pixel of the network's size): further out than a
// little, the answer is someone else's. Held this way round, a thin line the frame
// still shows passes, and one it no longer shows does not. 0 when it fits, up to 1 when
// it clearly does not.
float mismatch(float3 seen, float3 lo, float3 hi)
{
    // A little more allowed in bright places: light that changes (exposure adapting, a
    // flame) changes the value more there.
    float3 off = max(max(lo - seen, seen - hi), 0.0);
    float light = saturate((off.x - 0.02 - 0.05 * max(seen.x, hi.x)) / 0.05);
    float colour = saturate((max(off.y, off.z) - 0.03) / 0.05);
    return max(light, colour);
}

// The range of what this frame (t0) looks like (see "print") around the output pixel
// `p`, `step` output pixels apart: one pixel of the network's size.
void around(int2 p, int step, float wp, out float3 lo, out float3 hi)
{
    lo = 1e9;
    hi = -1e9;
    [unroll] for (int j = -1; j <= 1; ++j)
    [unroll] for (int i = -1; i <= 1; ++i) {
        int2 q = clamp(p + int2(i, j) * step, int2(0, 0), int2(size) - 1);
        float3 y = print(to_net(proxy_of(max(t0.Load(int3(q, 0)).rgb, 0.0) / wp)));
        lo = min(lo, y);
        hi = max(hi, y);
    }
}

// A residual at `at` (texture coordinates of the network's size, `texel` one pixel),
// bilinearly, with each of the four taps taken only where the surface the capture
// showed there (the depth in its alpha) is this pixel's (`depth`): a read across an
// outline would mix what the network did to the thing in front with what it did
// behind, and leave a line of the one on the other. Where no tap is this surface, the
// plain read.
float3 surface_read(Texture2D<float4> residual, float2 at, float depth, float2 texel)
{
    const float2 pos = at / texel - 0.5;
    const float2 f = pos - floor(pos);
    const int2 i0 = int2(floor(pos));
    uint w, h;
    residual.GetDimensions(w, h);
    float3 sum = 0.0;
    float weight = 0.0;
    [unroll] for (int k = 0; k < 4; ++k) {
        const int2 o = int2(k & 1, k >> 1);
        const int2 t = clamp(i0 + o, int2(0, 0), int2(w, h) - 1);
        const float4 v = residual.Load(int3(t, 0));
        const float wb = (o.x ? f.x : 1.0 - f.x) * (o.y ? f.y : 1.0 - f.y);
        const float wd = continues(unpack_depth(v.a), depth, 0.10, 0.5) ? 1.0 : 0.0;
        sum += v.rgb * wb * wd;
        weight += wb * wd;
    }
    return weight > 1e-3 ? sum / weight : residual.SampleLevel(linear_clamp, at, 0).rgb;
}

// ---- pixels the network never saw
//
// A pixel whose point was behind something, or outside the image, in the
// frame of the capture has no residual of its own. Left without one, such
// pixels show as a patch the network did not touch, the shape of whatever
// moved off them, trailing behind it until the next result.
//
// Its map cannot say where the point was: a map moves on one frame at a time,
// and a point hidden in any frame on the way is lost to it. But the surface
// around it was seen, and moved the same way -- the same wall, under the same
// camera. So the nearest pixel that has a displacement, at this pixel's depth
// (t5, this frame's depth at the network's size), lends its own.

static const float2 DIRECTIONS[8] = {float2(1, 0), float2(-1, 0), float2(0, 1), float2(0, -1),
                                     float2(0.7071, 0.7071), float2(-0.7071, 0.7071),
                                     float2(0.7071, -0.7071), float2(-0.7071, -0.7071)};
static const int REACHES = 9;
static const float REACH[REACHES] = {6, 12, 24, 48, 96, 192, 384, 768, 1536};   // pixels of the network's size

bool same_surface(float a, float b) { return same_depth(a, b, 0.05); }

// The network's own answer for this point, from a result (its residual and
// its displacement map) that did see it: read where the borrowed displacement
// leads, and taken only if what that capture showed there was this pixel's
// surface -- its depth then (the residual's alpha) against the depth now.
// Nearest first.
bool borrow(Texture2D<float4> residual, Texture2D<float4> map, float2 uv, float depth, out float3 r, out float2 from)
{
    r = 0.0;
    from = uv;
    bool found = false;
    [loop] for (int j = 0; j < REACHES && !found; ++j) {
        [loop] for (int i = 0; i < 8 && !found; ++i) {
            float2 spot = uv + DIRECTIONS[i] * REACH[j] * uv1_scale;
            if (any(spot < 0.0) || any(spot > 1.0)) continue;
            float2 d = map.SampleLevel(point_clamp, spot, 0).rg;
            if (gone(d) || !same_surface(t5.SampleLevel(point_clamp, spot, 0).r, depth)) continue;
            float2 at = uv + d * uv1_scale;
            if (any(at < 0.0) || any(at > 1.0)) continue;
            if (!continues(unpack_depth(residual.SampleLevel(point_clamp, at, 0).a), depth, 0.10, 0.5)) continue;
            r = residual.SampleLevel(linear_clamp, at, 0).rgb;
            from = at;
            found = true;
        }
    }
    return found;
}

// When no result saw the point: what the network did to the light of the
// nearest pixels on the same surface, outwards in eight directions, the first
// in each that has a residual. Its gain there (t1), not its residual: the
// network's detail belongs to the pixels it was made for, and carried over
// here it shows as a faint copy of them; how much it lit or dimmed the place
// is what this spot shares with its surroundings.
// Steps half again as far each: what is still seen of a surface can be a narrow strip
// beside what was hidden (the part of a tower above a character that walked past it).
static const int FILL_REACHES = 16;
static const float FILL_REACH[FILL_REACHES] = {4, 6, 9, 13, 19, 28, 42, 63, 94, 141, 211, 316, 474, 711, 1066, 1600};

// Light found far away says less about this spot than light found next to it: the
// network does not light a whole sky or a whole wall alike. What is found is taken in
// full when near, and fades towards no change at all with its distance, over FILL_FADE
// pixels of the network's size.
static const float FILL_FADE = 768.0;

bool fill_in(float2 uv, float depth, out float3 gain)
{
    float3 sum = 0.0;
    float count = 0.0;
    [loop] for (int i = 0; i < 8; ++i) {
        [loop] for (int j = 0; j < FILL_REACHES; ++j) {
            float2 spot = uv + DIRECTIONS[i] * FILL_REACH[j] * uv1_scale;
            if (any(spot < 0.0) || any(spot > 1.0)) break;
            float2 d = t4.SampleLevel(point_clamp, spot, 0).rg;
            // The same thing, more loosely than for an answer: the light of a wall seen
            // from below, its foot and its top some way apart in depth, is the same light.
            const float there = t5.SampleLevel(point_clamp, spot, 0).r;
            if (gone(d) || !same_depth(there, depth, 0.25)) continue;
            float2 at = spot + d * uv1_scale;
            if (any(at < 0.0) || any(at > 1.0)) continue;
            const float trust = exp(-FILL_REACH[j] / FILL_FADE);
            sum += 1.0 + (t1.SampleLevel(linear_clamp, at, 0).rgb - 1.0) * trust;
            count += 1.0;
            break;
        }
    }
    gain = count > 0.0 ? sum / count : 1.0;
    return count > 0.0;
}

// The game's image (t0) with the residual (t3) composed in, into u0. The
// residual is in the pixels of the frame it was captured in, at the network's
// size (uv1_scale holds the inverse of that size): the displacement map of
// that capture (t4) says where each pixel of this frame was then, and the
// residual is sampled there, with filtering, and faded by its age. Both
// images are read and written through views of their storage as it is --
// sRGB storage as UNORM -- so display encoded values arrive encoded and leave
// encoded.
//
// A pixel that capture never saw is looked for, in this order: in the same
// result, through a displacement borrowed from around it; in the result
// before (t2, with its map t6), which saw what has since been covered and
// uncovered again, through its own displacement or a borrowed one; and
// failing both, it is given the network's gain from around it. An answer of
// the network for that very point, even an older one, beats a guess.
// FLAG_SHOW_TRACKING tints those pixels: cyan from the same result, blue from
// the one before, green where only the gain was filled in, red where nothing
// was found; yellow where a halo left behind by an edge was taken off, orange
// where an answer was not the pixel's (see "mismatch").
[numthreads(8, 8, 1)]
void compose_main(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= size)) return;
    float4 c = t0.Load(int3(id.xy, 0));
    float4 result = c;
    // A wait that ran out: the frame as the game made it.
    if ((flags & FLAG_SYNC) && t7.Load(int3(0, 0, 0)).r < 0.5) {
        u0[id.xy] = c;
        return;
    }
    if (!(flags & FLAG_COPY)) {
        float2 uv = (float2(id.xy) + 0.5) * inv_size;
        float2 d = (flags & FLAG_IDENTITY) ? 0.0 : t4.SampleLevel(point_clamp, uv, 0).rg;
        float2 at = uv + d * uv1_scale;
        float wp = white_point();
        float3 frame = max(c.rgb, 0.0) / wp;
        float3 r = 0.0;
        float3 tint = 0.0;
        if (!gone(d) && all(at >= 0.0) && all(at <= 1.0)) {
            const float depth_now = t5.SampleLevel(point_clamp, uv, 0).r;
            r = (flags & FLAG_FILL) ? surface_read(t3, at, depth_now, uv1_scale) : t3.SampleLevel(linear_clamp, at, 0).rgb;
            // Halos left behind by edges that have moved (FLAG_EDGES; t5 is this
            // frame's depth at the network's size).
            float now = (flags & FLAG_EDGES) ? edge_distance(t5, uv, uv1_scale) : EDGE_REACH;
            float halo = 0.0;
            float3 light = proxy_of(frame) * (t1.SampleLevel(linear_clamp, at, 0).rgb - 1.0);
            if (flags & FLAG_EDGES) r = without_halo(t3, t8, at, now, uv1_scale, r, halo);
            // An answer that was carried here with what is behind something that moved
            // on its own keeps only the network's light (t1, around it): see "mismatch".
            float3 lo = 0.0, hi = 1.0;
            float wrong = 0.0;
            if (flags & FLAG_EDGES) {
                around(int2(id.xy), max(1, (int)round(uv1_scale.x / inv_size.x)), wp, lo, hi);
                wrong = mismatch(t8.SampleLevel(linear_clamp, at, 0).gba, lo, hi);
                r = lerp(r, light, wrong);
            }
            // A result that has just arrived takes over from the one before over
            // a few frames, not at once: its detail differs from the old one's
            // wherever the network decided otherwise for the new framing.
            if ((flags & FLAG_PREVIOUS) && blend < 1.0) {
                float2 od = t6.SampleLevel(point_clamp, uv, 0).rg;
                float2 there = uv + od * uv1_scale;
                if (!gone(od) && all(there >= 0.0) && all(there <= 1.0)) {
                    float3 before = (flags & FLAG_FILL) ? surface_read(t2, there, depth_now, uv1_scale)
                                                        : t2.SampleLevel(linear_clamp, there, 0).rgb;
                    if (flags & FLAG_EDGES) {
                        before = without_halo(t2, t9, there, now, uv1_scale, before, halo);
                        float wrong_before = mismatch(t9.SampleLevel(linear_clamp, there, 0).gba, lo, hi);
                        before = lerp(before, light, wrong_before);
                        wrong = max(wrong, wrong_before * (1.0 - blend));
                    }
                    r = lerp(before, r, blend);
                }
            }
            tint = float3(0.3, 0.3, 0.0) * halo;
            tint = lerp(tint, float3(0.35, 0.15, 0.0), wrong);
        } else if (flags & FLAG_FILL) {
            float depth = t5.SampleLevel(point_clamp, uv, 0).r;
            float2 older = (flags & FLAG_PREVIOUS) ? t6.SampleLevel(point_clamp, uv, 0).rg : GONE;
            float2 there = uv + older * uv1_scale;
            float2 from;
            // A halo in what is filled in is taken off too (FLAG_EDGES). What is borrowed
            // was another pixel's, by an outline that is not this pixel's: off whenever
            // it was near one, as if this pixel were far from any (EDGE_REACH).
            float halo = 0.0;
            bool found = true;
            // What the capture showed where the answer comes from (see "print"), to check it
            // against this frame: an answer borrowed from a place that has since changed
            // its look -- a fade, a door opening -- is not this pixel's.
            float3 seen = 0.0;
            bool checked = false;
            if (borrow(t3, t4, uv, depth, r, from)) {
                if (flags & FLAG_EDGES) {
                    r = without_halo(t3, t8, from, EDGE_REACH, uv1_scale, r, halo);
                    seen = t8.SampleLevel(linear_clamp, from, 0).gba;
                    checked = true;
                }
                tint = float3(0.0, 0.25, 0.25);
            } else if (!gone(older) && all(there >= 0.0) && all(there <= 1.0)) {
                r = t2.SampleLevel(linear_clamp, there, 0).rgb;
                if ((flags & FLAG_EDGES) && (flags & FLAG_PREVIOUS)) {
                    r = without_halo(t2, t9, there, edge_distance(t5, uv, uv1_scale), uv1_scale, r, halo);
                    seen = t9.SampleLevel(linear_clamp, there, 0).gba;
                    checked = true;
                }
                tint = float3(0.0, 0.0, 0.3);
            } else if ((flags & FLAG_PREVIOUS) && borrow(t2, t6, uv, depth, r, from)) {
                if (flags & FLAG_EDGES) {
                    r = without_halo(t2, t9, from, EDGE_REACH, uv1_scale, r, halo);
                    seen = t9.SampleLevel(linear_clamp, from, 0).gba;
                    checked = true;
                }
                tint = float3(0.0, 0.0, 0.3);
            } else {
                found = false;
            }
            if (halo > 0.0) tint = lerp(tint, float3(0.3, 0.3, 0.0), halo);
            float wrong = 0.0;
            if (found && checked) {
                float3 lo, hi;
                around(int2(id.xy), max(1, (int)round(uv1_scale.x / inv_size.x)), wp, lo, hi);
                wrong = mismatch(seen, lo, hi);
            }
            if (!found || wrong > 0.0) {
                // Only the network's light, from around it (see "fill_in").
                float3 gain = 1.0;
                const bool lit = fill_in(uv, depth, gain);
                const float3 light = proxy_of(frame) * (gain - 1.0);
                if (found) {
                    r = lerp(r, light, wrong);
                    tint = lerp(tint, float3(0.35, 0.15, 0.0), wrong);
                } else {
                    r = light;
                    tint = lit ? float3(0.0, 0.25, 0.0) : float3(0.25, 0.0, 0.0);
                }
            }
        } else {
            tint = float3(0.25, 0.0, 0.0);
        }
        if (!(flags & FLAG_SHOW_TRACKING)) tint = 0.0;
        r *= fade;
        float3 composed = (flags & FLAG_REPLACE) ? apply_residual(proxy_of(frame), r) : compose(frame, r);
        result.rgb = (max(composed, 0.0) + tint) * wp;
    }
    u0[id.xy] = result;
}

// ------------------------------------------------------------------- sync
// Waiting for the network inside the frame. The game's command list goes on
// only once the network has answered, so the answer is composed into the very
// frame it was made from. The GPU and the worker thread talk through host
// memory (u3), a line of 128 bytes to a word: the capture's number is written
// when it is in place (by the command processor, or sync_signal when the list
// does not allow that), to the line of its own in a ring of four, which the worker
// looks for; no shader writes to those lines, since a line a shader has written
// stays in the GPU's cache and hides what is written to it after. The worker answers by
// writing the capture's number (with the top bit set if the network failed)
// into every line of a stretch of the buffer, from line exposure_mode on, and
// sync_wait reads those lines one after the other, size.y of them at most,
// until it finds its number. Every line is read once: what a GPU has read
// stays in its cache, and a word it keeps looking at is never seen to change,
// whatever the way it is read (measured), where a line it has not looked at
// before is read from memory as it is then. Ending at the last line is the
// wait running out; how many it read goes to the host (byte 512), and u0 gets 1
// if the network answered, 0 if not.
[numthreads(1, 1, 1)]
void sync_signal_main()
{
    u3.Store((size.x % 4) * 128, size.x);
    DeviceMemoryBarrier();
}

[numthreads(1, 1, 1)]
void sync_wait_main()
{
    uint spins = 0;
    uint answer = 0;
    uint idle = 1;
    [loop] while (spins < size.y) {
        answer = u3.Load((exposure_mode + spins) * 128);
        if ((answer & 0x7FFFFFFFu) == size.x) break;
        ++spins;
        // A few microseconds between two looks: the lines to read, and so the host
        // memory they take, are as many as the wait is long over this.
        [loop] for (uint i = 0; i < 3000; ++i) idle = idle * 1664525u + 1013904223u;
    }
    u3.Store(520, idle);
    u3.Store(512, spins);
    u3.Store(516, size.y);
    u0[uint2(0, 0)] = float4(answer == size.x ? 1.0 : 0.0, 0.0, 0.0, 0.0);
}
)HLSL";

} // namespace dlss5nr
