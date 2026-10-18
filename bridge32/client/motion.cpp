#include "motion.h"
#include <initializer_list>
#include "shaders/down.h"
#include "shaders/me_coarse.h"
#include "shaders/me_median.h"
#include "shaders/me_refine.h"
#include "shaders/quad_vs.h"

void draw_fullscreen_quad(IDirect3DDevice9 *device, UINT w, UINT h, bool clip_space) {
    if (clip_space) {
        struct Vertex {
            float x, y, z, rw, u, v;
        };
        const float dx = 1.0f / (float)w, dy = 1.0f / (float)h;
        const Vertex quad[4] = {{-1.0f - dx, 1.0f + dy, 0.0f, 1.0f, 0.0f, 0.0f},
                                {1.0f - dx, 1.0f + dy, 0.0f, 1.0f, 1.0f, 0.0f},
                                {-1.0f - dx, -1.0f + dy, 0.0f, 1.0f, 0.0f, 1.0f},
                                {1.0f - dx, -1.0f + dy, 0.0f, 1.0f, 1.0f, 1.0f}};
        device->SetFVF(D3DFVF_XYZW | D3DFVF_TEX1);
        device->DrawPrimitiveUP(D3DPT_TRIANGLESTRIP, 2, quad, sizeof(Vertex));
    } else {
        struct Vertex {
            float x, y, z, rhw, u, v;
        };
        const float l = -0.5f, t = -0.5f, r = (float)w - 0.5f, b = (float)h - 0.5f;
        const Vertex quad[4] = {{l, t, 0.0f, 1.0f, 0.0f, 0.0f},
                                {r, t, 0.0f, 1.0f, 1.0f, 0.0f},
                                {l, b, 0.0f, 1.0f, 0.0f, 1.0f},
                                {r, b, 0.0f, 1.0f, 1.0f, 1.0f}};
        device->SetFVF(D3DFVF_XYZRHW | D3DFVF_TEX1);
        device->DrawPrimitiveUP(D3DPT_TRIANGLESTRIP, 2, quad, sizeof(Vertex));
    }
}

void Motion::Target::release() {
    if (surf) surf->Release();
    if (tex) tex->Release();
    surf = nullptr;
    tex = nullptr;
}

bool Motion::make(IDirect3DDevice9 *device, Target &t, UINT w, UINT h, D3DFORMAT format) {
    t.w = w;
    t.h = h;
    return SUCCEEDED(device->CreateTexture(w, h, 1, D3DUSAGE_RENDERTARGET, format, D3DPOOL_DEFAULT, &t.tex, nullptr)) &&
           SUCCEEDED(t.tex->GetSurfaceLevel(0, &t.surf));
}

bool Motion::create(IDirect3DDevice9 *device, UINT width, UINT height) {
    release();
    D3DCAPS9 caps = {};
    device->GetDeviceCaps(&caps);
    if (caps.PixelShaderVersion < D3DPS_VERSION(3, 0) || caps.VertexShaderVersion < D3DVS_VERSION(3, 0)) return false;

    IDirect3D9 *d3d = nullptr;
    device->GetDirect3D(&d3d);
    D3DDEVICE_CREATION_PARAMETERS cp;
    device->GetCreationParameters(&cp);
    D3DDISPLAYMODE mode = {};
    d3d->GetAdapterDisplayMode(cp.AdapterOrdinal, &mode);
    const bool half_rt = SUCCEEDED(d3d->CheckDeviceFormat(cp.AdapterOrdinal, cp.DeviceType, mode.Format, D3DUSAGE_RENDERTARGET,
                                                          D3DRTYPE_TEXTURE, D3DFMT_A16B16G16R16F));
    filterable_ = SUCCEEDED(d3d->CheckDeviceFormat(cp.AdapterOrdinal, cp.DeviceType, mode.Format, D3DUSAGE_RENDERTARGET | D3DUSAGE_QUERY_FILTER,
                                                   D3DRTYPE_TEXTURE, D3DFMT_A16B16G16R16F));
    d3d->Release();
    if (!half_rt) return false;

    w_ = width;
    h_ = height;
    const UINT w16 = (width + 15) / 16, h16 = (height + 15) / 16;
    ready_ = make(device, s4_now_, w16 * 4, h16 * 4, D3DFMT_A8R8G8B8) && make(device, s4_old_, w16 * 4, h16 * 4, D3DFMT_A8R8G8B8) &&
             make(device, s16_now_, w16, h16, D3DFMT_A8R8G8B8) && make(device, s16_old_, w16, h16, D3DFMT_A8R8G8B8) &&
             make(device, coarse_, w16, h16, D3DFMT_A16B16G16R16F) && make(device, fine_, w16, h16, D3DFMT_A16B16G16R16F) && make(device, smooth_, w16, h16, D3DFMT_A16B16G16R16F) &&
             SUCCEEDED(device->CreatePixelShader((const DWORD *)kDown, &down_)) &&
             SUCCEEDED(device->CreatePixelShader((const DWORD *)kMeCoarse, &coarse_ps_)) &&
             SUCCEEDED(device->CreatePixelShader((const DWORD *)kMeRefine, &refine_ps_)) &&
             SUCCEEDED(device->CreatePixelShader((const DWORD *)kMeMedian, &median_ps_)) &&
             SUCCEEDED(device->CreateVertexShader((const DWORD *)kQuadVs, &quad_vs_));
    if (!ready_) release();
    return ready_;
}

void Motion::release() {
    for (Target *t : {&s4_now_, &s4_old_, &s16_now_, &s16_old_, &coarse_, &fine_, &smooth_}) t->release();
    if (down_) down_->Release();
    if (coarse_ps_) coarse_ps_->Release();
    if (refine_ps_) refine_ps_->Release();
    if (median_ps_) median_ps_->Release();
    if (quad_vs_) quad_vs_->Release();
    down_ = coarse_ps_ = refine_ps_ = median_ps_ = nullptr;
    quad_vs_ = nullptr;
    ready_ = false;
}

void Motion::pass(IDirect3DDevice9 *device, Target &target, IDirect3DPixelShader9 *ps, bool clip_space, IDirect3DBaseTexture9 *a,
                  IDirect3DBaseTexture9 *b, IDirect3DBaseTexture9 *c, const float *c0, const float *c1) {
    device->SetRenderTarget(0, target.surf);
    const D3DVIEWPORT9 vp = {0, 0, target.w, target.h, 0.0f, 1.0f};
    device->SetViewport(&vp);
    device->SetVertexShader(clip_space ? quad_vs_ : nullptr);
    device->SetPixelShader(ps);
    IDirect3DBaseTexture9 *textures[4] = {a, b, c, nullptr};
    for (DWORD i = 0; i < 4; ++i) device->SetTexture(i, textures[i]);
    device->SetPixelShaderConstantF(0, c0, 1);
    if (c1) device->SetPixelShaderConstantF(1, c1, 1);
    device->BeginScene();
    draw_fullscreen_quad(device, target.w, target.h, clip_space);
    device->EndScene();
}

void Motion::update(IDirect3DDevice9 *device, IDirect3DTexture9 *now, IDirect3DTexture9 *old, bool old_changed) {
    if (!ready_) return;
    device->SetRenderState(D3DRS_ZENABLE, D3DZB_FALSE);
    device->SetRenderState(D3DRS_ZWRITEENABLE, FALSE);
    device->SetRenderState(D3DRS_STENCILENABLE, FALSE);
    device->SetRenderState(D3DRS_CULLMODE, D3DCULL_NONE);
    device->SetRenderState(D3DRS_ALPHABLENDENABLE, FALSE);
    device->SetRenderState(D3DRS_ALPHATESTENABLE, FALSE);
    device->SetRenderState(D3DRS_SCISSORTESTENABLE, FALSE);
    device->SetRenderState(D3DRS_SRGBWRITEENABLE, FALSE);
    device->SetRenderState(D3DRS_COLORWRITEENABLE, 0xF);
    for (DWORD i = 0; i < 4; ++i) {
        device->SetSamplerState(i, D3DSAMP_MINFILTER, D3DTEXF_POINT);
        device->SetSamplerState(i, D3DSAMP_MAGFILTER, D3DTEXF_POINT);
        device->SetSamplerState(i, D3DSAMP_MIPFILTER, D3DTEXF_NONE);
        device->SetSamplerState(i, D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP);
        device->SetSamplerState(i, D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP);
        device->SetSamplerState(i, D3DSAMP_SRGBTEXTURE, FALSE);
    }
    const UINT w16 = s16_now_.w, h16 = s16_now_.h, w4 = s4_now_.w, h4 = s4_now_.h;
    const float spacing4[4] = {0.25f / (float)w4, 0.25f / (float)h4, 0, 0};
    const float spacing16[4] = {0.25f / (float)w16, 0.25f / (float)h16, 0, 0};
    if (old_changed) {
        pass(device, s4_old_, down_, false, old, nullptr, nullptr, spacing4, nullptr);
        pass(device, s16_old_, down_, false, s4_old_.tex, nullptr, nullptr, spacing16, nullptr);
    }
    pass(device, s4_now_, down_, false, now, nullptr, nullptr, spacing4, nullptr);
    pass(device, s16_now_, down_, false, s4_now_.tex, nullptr, nullptr, spacing16, nullptr);

    const float coarse_k[4] = {1.0f / (float)w16, 1.0f / (float)h16, coarse_penalty, 0};
    pass(device, coarse_, coarse_ps_, true, s16_now_.tex, s16_old_.tex, nullptr, coarse_k, nullptr);
    const float refine_k[4] = {1.0f / (float)w4, 1.0f / (float)h4, fine_penalty, 0};
    const float refine_scale[4] = {(float)w_ / (float)w16, (float)h_ / (float)h16, (float)w_ / (float)w4, (float)h_ / (float)h4};
    pass(device, fine_, refine_ps_, true, s4_now_.tex, s4_old_.tex, coarse_.tex, refine_k, refine_scale);
    if (median) {
        const float median_k[4] = {1.0f / (float)w16, 1.0f / (float)h16, 0, 0};
        pass(device, smooth_, median_ps_, true, fine_.tex, nullptr, nullptr, median_k, nullptr);
    }
}
