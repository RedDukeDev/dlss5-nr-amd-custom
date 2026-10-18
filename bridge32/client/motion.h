// Motion estimated from two pictures: for every block of 16x16 pixels of the picture now, where in
// the old picture it came from. Runs on the game's own Direct3D 9 device, in a few passes over
// pictures of the luminance at 1/4 and 1/16 size: a wide search on the small one, a fine one on
// the larger, around what the first found. See shaders/me_coarse.hlsl and me_refine.hlsl.
#pragma once
#include <windows.h>
#include <d3d9.h>

class Motion {
public:
    // False when the device cannot do it (Shader Model 3 and half-float render targets are needed).
    bool create(IDirect3DDevice9 *device, UINT width, UINT height);
    void release();
    bool ready() const { return ready_; }

    // Searches. `old_changed` says the old picture is not the one of the last call, so its pyramid
    // is built again. Changes the render target, the viewport and the shader state of the device:
    // the caller keeps them in a state block.
    void update(IDirect3DDevice9 *device, IDirect3DTexture9 *now, IDirect3DTexture9 *old, bool old_changed);

    // A16B16G16R16F, one texel a block: x and y in pixels (where in the old picture the block was),
    // z the cost of the match.
    IDirect3DTexture9 *field() const { return median ? smooth_.tex : fine_.tex; }
    bool filterable() const { return filterable_; }
    UINT grid_width() const { return fine_.w; }
    UINT grid_height() const { return fine_.h; }
    IDirect3DSurface9 *field_surface() const { return median ? smooth_.surf : fine_.surf; }

private:
    struct Target {
        IDirect3DTexture9 *tex = nullptr;
        IDirect3DSurface9 *surf = nullptr;
        UINT w = 0, h = 0;
        void release();
    };
    bool make(IDirect3DDevice9 *device, Target &t, UINT w, UINT h, D3DFORMAT format);
    void pass(IDirect3DDevice9 *device, Target &target, IDirect3DPixelShader9 *ps, bool clip_space,
              IDirect3DBaseTexture9 *a, IDirect3DBaseTexture9 *b, IDirect3DBaseTexture9 *c, const float *c0, const float *c1);

    bool ready_ = false;
    bool filterable_ = false;
    UINT w_ = 0, h_ = 0;
    Target s4_now_, s4_old_, s16_now_, s16_old_, coarse_, fine_, smooth_;
    IDirect3DPixelShader9 *down_ = nullptr, *coarse_ps_ = nullptr, *refine_ps_ = nullptr, *median_ps_ = nullptr;
    IDirect3DVertexShader9 *quad_vs_ = nullptr;

public:
    // Tunables, read by update().
    float coarse_penalty = 0.001f;   // the cost of a texel of motion in the coarse search
    float fine_penalty = 0.0005f;
    bool median = true;             // take the stray vectors out of the field
};

// One quad over the whole target: with a vertex shader its corners are in clip space (with the half
// pixel shift Direct3D 9 wants), without they are pre-transformed pixel positions.
void draw_fullscreen_quad(IDirect3DDevice9 *device, UINT width, UINT height, bool clip_space);
