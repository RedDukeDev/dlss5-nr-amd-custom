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

static const float3 LUMA = float3(0.2126, 0.7152, 0.0722);

Texture2D<float4> t0 : register(t0);
Texture2D<float4> t1 : register(t1);
Texture2D<float4> t2 : register(t2);
Texture2D<float4> t3 : register(t3);
Texture2D<float4> t4 : register(t4);
Texture2D<float4> t5 : register(t5);
Texture2D<float4> t6 : register(t6);
Texture2D<float4> t7 : register(t7);
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

// Moves a displacement map (t0) on by one frame, into u0: the game's motion
// vectors (t2) say where each pixel was in the previous frame, and the map
// there says where that was then. FLAG_IDENTITY: the earlier frame is the
// previous one, and the map starts from nothing.
//
// Occlusion: the depth of the previous frame (t3) where the pixel came from
// has to agree with this pixel's depth (t1). Where it does not, the pixel was
// covered then, and what the map holds there belongs to whatever covered it.
// FLAG_KEEP_DEPTH also writes this frame's depth (u1), for the next frame.
[numthreads(8, 8, 1)]
void track_main(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= size)) return;
    float2 uv = (float2(id.xy) + 0.5) * inv_size;
    float2 mv = t2.SampleLevel(point_clamp, uv * uv2_scale, 0).rg * motion_to_px;
    float depth = t1.SampleLevel(point_clamp, uv * uv1_scale, 0).r;
    if (flags & FLAG_KEEP_DEPTH) u1[id.xy] = float4(depth, 0, 0, 0);

    float2 from = float2(id.xy) + 0.5 + mv;
    float2 d = GONE;
    if (all(from >= 0.0) && all(from < float2(size))) {
        int2 at = int2(from);
        float before = t3.Load(int3(at, 0)).r;
        bool seen = !(flags & FLAG_DEPTH_CHECK) ||
                    abs(before - depth) <= 0.02 * max(max(abs(before), abs(depth)), 1e-6);
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
    u2[id.xy] = float4(gone(d) ? 0.0 : d, 0, 0);
}

// ----------------------------------------------------------------- residual
// A fresh result: network output (t0) minus network input (t1), both in the
// proxy's linear space, into u0. It stays where the network put it, in the
// pixels of the frame it was captured in; the composition finds it through
// that slot's displacement map. It is never blended with older answers, which
// the network re-decides with every framing. The capture's depth (t2) goes
// with it, in alpha: it says what surface each of its pixels showed.
[numthreads(8, 8, 1)]
void refresh_main(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= size)) return;
    float3 r = from_net(t0.Load(int3(id.xy, 0)).rgb) - from_net(t1.Load(int3(id.xy, 0)).rgb);
    u0[id.xy] = float4(r, t2.Load(int3(id.xy, 0)).r);
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
static const int REACHES = 7;
static const float REACH[REACHES] = {6, 12, 24, 48, 96, 192, 384};   // pixels of the network's size

bool same_surface(float a, float b) { return abs(a - b) <= 0.05 * max(max(abs(a), abs(b)), 1e-6); }

// The network's own answer for this point, from a result (its residual and
// its displacement map) that did see it: read where the borrowed displacement
// leads, and taken only if what that capture showed there was this pixel's
// surface -- its depth then (the residual's alpha) against the depth now.
// Nearest first.
bool borrow(Texture2D<float4> residual, Texture2D<float4> map, float2 uv, float depth, out float3 r)
{
    r = 0.0;
    bool found = false;
    [loop] for (int j = 0; j < REACHES && !found; ++j) {
        [loop] for (int i = 0; i < 8 && !found; ++i) {
            float2 spot = uv + DIRECTIONS[i] * REACH[j] * uv1_scale;
            if (any(spot < 0.0) || any(spot > 1.0)) continue;
            float2 d = map.SampleLevel(point_clamp, spot, 0).rg;
            if (gone(d) || !same_surface(t5.SampleLevel(point_clamp, spot, 0).r, depth)) continue;
            float2 at = uv + d * uv1_scale;
            if (any(at < 0.0) || any(at > 1.0)) continue;
            if (!same_surface(residual.SampleLevel(point_clamp, at, 0).a, depth)) continue;
            r = residual.SampleLevel(linear_clamp, at, 0).rgb;
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
bool fill_in(float2 uv, float depth, out float3 gain)
{
    float3 sum = 0.0;
    float count = 0.0;
    [loop] for (int i = 0; i < 8; ++i) {
        [loop] for (int j = 0; j < REACHES; ++j) {
            float2 spot = uv + DIRECTIONS[i] * REACH[j] * uv1_scale;
            if (any(spot < 0.0) || any(spot > 1.0)) break;
            float2 d = t4.SampleLevel(point_clamp, spot, 0).rg;
            if (gone(d) || !same_surface(t5.SampleLevel(point_clamp, spot, 0).r, depth)) continue;
            float2 at = spot + d * uv1_scale;
            if (any(at < 0.0) || any(at > 1.0)) continue;
            sum += t1.SampleLevel(linear_clamp, at, 0).rgb;
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
// was found.
[numthreads(8, 8, 1)]
void compose_main(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= size)) return;
    float4 c = t0.Load(int3(id.xy, 0));
    float4 result = c;
    if (!(flags & FLAG_COPY)) {
        float2 uv = (float2(id.xy) + 0.5) * inv_size;
        float2 d = (flags & FLAG_IDENTITY) ? 0.0 : t4.SampleLevel(point_clamp, uv, 0).rg;
        float2 at = uv + d * uv1_scale;
        float wp = white_point();
        float3 frame = max(c.rgb, 0.0) / wp;
        float3 r = 0.0;
        float3 tint = 0.0;
        if (!gone(d) && all(at >= 0.0) && all(at <= 1.0)) {
            r = t3.SampleLevel(linear_clamp, at, 0).rgb;
            // A result that has just arrived takes over from the one before over
            // a few frames, not at once: its detail differs from the old one's
            // wherever the network decided otherwise for the new framing.
            if ((flags & FLAG_PREVIOUS) && blend < 1.0) {
                float2 od = t6.SampleLevel(point_clamp, uv, 0).rg;
                float2 there = uv + od * uv1_scale;
                if (!gone(od) && all(there >= 0.0) && all(there <= 1.0))
                    r = lerp(t2.SampleLevel(linear_clamp, there, 0).rgb, r, blend);
            }
        } else if (flags & FLAG_FILL) {
            float depth = t5.SampleLevel(point_clamp, uv, 0).r;
            float2 older = (flags & FLAG_PREVIOUS) ? t6.SampleLevel(point_clamp, uv, 0).rg : GONE;
            float2 there = uv + older * uv1_scale;
            if (borrow(t3, t4, uv, depth, r)) {
                tint = float3(0.0, 0.25, 0.25);
            } else if (!gone(older) && all(there >= 0.0) && all(there <= 1.0)) {
                r = t2.SampleLevel(linear_clamp, there, 0).rgb;
                tint = float3(0.0, 0.0, 0.3);
            } else if ((flags & FLAG_PREVIOUS) && borrow(t2, t6, uv, depth, r)) {
                tint = float3(0.0, 0.0, 0.3);
            } else {
                float3 gain = 1.0;
                tint = fill_in(uv, depth, gain) ? float3(0.0, 0.25, 0.0) : float3(0.25, 0.0, 0.0);
                r = proxy_of(frame) * (gain - 1.0);
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
)HLSL";

} // namespace dlss5nr
