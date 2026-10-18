#include "dx11_overlay.h"
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <vector>
#include "core.h"
#include "menu.h"
#include "shaders/dx11_apply_ps.h"
#include "shaders/dx11_down_ps.h"
#include "shaders/dx11_vs.h"

namespace dx11 {
namespace {

template <class T> void release(T *&p) {
    if (p) p->Release();
    p = nullptr;
}

// What the game's backbuffer is made of, as far as this goes: eight bits a channel, in either order,
// typed, sRGB or typeless.
struct Format {
    bool ok = false;
    DXGI_FORMAT typeless = DXGI_FORMAT_UNKNOWN;   // the copy is made in this
    DXGI_FORMAT unorm = DXGI_FORMAT_UNKNOWN;      // what the copy is read as, and a multisampled frame resolved in
    DXGI_FORMAT target = DXGI_FORMAT_UNKNOWN;     // what the backbuffer is drawn through
    bool srgb = false;
};

Format classify(DXGI_FORMAT f) {
    Format r;
    switch (f) {
    case DXGI_FORMAT_R8G8B8A8_UNORM: case DXGI_FORMAT_R8G8B8A8_TYPELESS:
        r = {true, DXGI_FORMAT_R8G8B8A8_TYPELESS, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM, false};
        break;
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
        r = {true, DXGI_FORMAT_R8G8B8A8_TYPELESS, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, true};
        break;
    case DXGI_FORMAT_B8G8R8A8_UNORM: case DXGI_FORMAT_B8G8R8A8_TYPELESS:
        r = {true, DXGI_FORMAT_B8G8R8A8_TYPELESS, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_B8G8R8A8_UNORM, false};
        break;
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
        r = {true, DXGI_FORMAT_B8G8R8A8_TYPELESS, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, true};
        break;
    default:
        break;
    }
    return r;
}

// The pipeline state this draws in is not the game's: what it changes is kept and given back.
struct StateBackup {
    UINT scissor_count = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE, viewport_count = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
    D3D11_RECT scissors[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE];
    D3D11_VIEWPORT viewports[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE];
    ID3D11RasterizerState *rasterizer = nullptr;
    ID3D11BlendState *blend = nullptr;
    FLOAT blend_factor[4];
    UINT sample_mask = 0;
    ID3D11DepthStencilState *depth_stencil = nullptr;
    UINT stencil_ref = 0;
    ID3D11RenderTargetView *targets[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT] = {};
    ID3D11DepthStencilView *depth_view = nullptr;
    ID3D11InputLayout *layout = nullptr;
    D3D11_PRIMITIVE_TOPOLOGY topology = D3D11_PRIMITIVE_TOPOLOGY_UNDEFINED;
    ID3D11VertexShader *vs = nullptr;
    ID3D11PixelShader *ps = nullptr;
    ID3D11GeometryShader *gs = nullptr;
    ID3D11HullShader *hs = nullptr;
    ID3D11DomainShader *ds = nullptr;
    ID3D11ShaderResourceView *ps_views[2] = {};
    ID3D11SamplerState *ps_samplers[2] = {};
    ID3D11Buffer *ps_buffer = nullptr;

    void save(ID3D11DeviceContext *c) {
        c->RSGetScissorRects(&scissor_count, scissors);
        c->RSGetViewports(&viewport_count, viewports);
        c->RSGetState(&rasterizer);
        c->OMGetBlendState(&blend, blend_factor, &sample_mask);
        c->OMGetDepthStencilState(&depth_stencil, &stencil_ref);
        c->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, targets, &depth_view);
        c->IAGetInputLayout(&layout);
        c->IAGetPrimitiveTopology(&topology);
        c->VSGetShader(&vs, nullptr, nullptr);
        c->PSGetShader(&ps, nullptr, nullptr);
        c->GSGetShader(&gs, nullptr, nullptr);
        c->HSGetShader(&hs, nullptr, nullptr);
        c->DSGetShader(&ds, nullptr, nullptr);
        c->PSGetShaderResources(0, 2, ps_views);
        c->PSGetSamplers(0, 2, ps_samplers);
        c->PSGetConstantBuffers(0, 1, &ps_buffer);
    }

    void restore(ID3D11DeviceContext *c) {
        c->RSSetScissorRects(scissor_count, scissors);
        c->RSSetViewports(viewport_count, viewports);
        c->RSSetState(rasterizer);
        c->OMSetBlendState(blend, blend_factor, sample_mask);
        c->OMSetDepthStencilState(depth_stencil, stencil_ref);
        c->OMSetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, targets, depth_view);
        c->IASetInputLayout(layout);
        c->IASetPrimitiveTopology(topology);
        c->VSSetShader(vs, nullptr, 0);
        c->PSSetShader(ps, nullptr, 0);
        c->GSSetShader(gs, nullptr, 0);
        c->HSSetShader(hs, nullptr, 0);
        c->DSSetShader(ds, nullptr, 0);
        c->PSSetShaderResources(0, 2, ps_views);
        c->PSSetSamplers(0, 2, ps_samplers);
        c->PSSetConstantBuffers(0, 1, &ps_buffer);
        release(rasterizer);
        release(blend);
        release(depth_stencil);
        for (ID3D11RenderTargetView *&t : targets) release(t);
        release(depth_view);
        release(layout);
        release(vs);
        release(ps);
        release(gs);
        release(hs);
        release(ds);
        for (ID3D11ShaderResourceView *&v : ps_views) release(v);
        for (ID3D11SamplerState *&s : ps_samplers) release(s);
        release(ps_buffer);
    }
};

// What is kept for the swap chain whose frames are enhanced.
struct Overlay {
    IDXGISwapChain *swap_chain = nullptr;
    ID3D11Device *device = nullptr;
    HWND window = nullptr;
    bool broken = false;
    bool menu_ready = false;
    bool unsupported_logged = false;

    UINT w = 0, h = 0;          // the backbuffer
    UINT cw = 0, ch = 0;        // the network's picture
    Format format;
    bool multisampled = false;

    ID3D11Texture2D *copy = nullptr;
    ID3D11ShaderResourceView *copy_view = nullptr;
    ID3D11Texture2D *reduced = nullptr;
    ID3D11RenderTargetView *small_target = nullptr;
    ID3D11Texture2D *staging = nullptr;
    ID3D11Texture2D *gain = nullptr;
    ID3D11ShaderResourceView *gain_view = nullptr;
    ID3D11VertexShader *vs = nullptr;
    ID3D11PixelShader *down_ps = nullptr, *apply_ps = nullptr;
    ID3D11SamplerState *point_sampler = nullptr, *linear_sampler = nullptr;
    ID3D11Buffer *constants = nullptr;
    ID3D11BlendState *blend = nullptr;
    ID3D11RasterizerState *rasterizer = nullptr;
    ID3D11DepthStencilState *depth_stencil = nullptr;
    bool have_gain = false;
    unsigned frames = 0, sent = 0, applied = 0;
    DWORD last_report = 0;

    void release_all() {
        release(copy_view);
        release(copy);
        release(small_target);
        release(reduced);
        release(staging);
        release(gain_view);
        release(gain);
        release(vs);
        release(down_ps);
        release(apply_ps);
        release(point_sampler);
        release(linear_sampler);
        release(constants);
        release(blend);
        release(rasterizer);
        release(depth_stencil);
        have_gain = false;
        w = h = cw = ch = 0;
    }
};
Overlay g_overlay;

bool setup(Overlay &o, ID3D11Texture2D *back) {
    D3D11_TEXTURE2D_DESC bd;
    back->GetDesc(&bd);
    o.format = classify(bd.Format);
    if (!o.format.ok) {
        log("dx11: the backbuffer is format %d, which is not handled", (int)bd.Format);
        o.broken = true;
        return false;
    }
    o.w = bd.Width;
    o.h = bd.Height;
    o.multisampled = bd.SampleDesc.Count > 1;
    network_size(o.w, o.h, o.cw, o.ch);

    D3D11_TEXTURE2D_DESC d = {};
    d.Width = o.w;
    d.Height = o.h;
    d.MipLevels = 1;
    d.ArraySize = 1;
    d.Format = o.format.typeless;
    d.SampleDesc.Count = 1;
    d.Usage = D3D11_USAGE_DEFAULT;
    d.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SHADER_RESOURCE_VIEW_DESC sv = {};
    sv.Format = o.format.unorm;
    sv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    sv.Texture2D.MipLevels = 1;
    bool ok = SUCCEEDED(o.device->CreateTexture2D(&d, nullptr, &o.copy)) && SUCCEEDED(o.device->CreateShaderResourceView(o.copy, &sv, &o.copy_view));

    // The network's picture: drawn reduced, read back.
    d.Width = o.cw;
    d.Height = o.ch;
    d.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    d.BindFlags = D3D11_BIND_RENDER_TARGET;
    ok = ok && SUCCEEDED(o.device->CreateTexture2D(&d, nullptr, &o.reduced)) && SUCCEEDED(o.device->CreateRenderTargetView(o.reduced, nullptr, &o.small_target));
    d.BindFlags = 0;
    d.Usage = D3D11_USAGE_STAGING;
    d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ok = ok && SUCCEEDED(o.device->CreateTexture2D(&d, nullptr, &o.staging));

    // The gain: written by the CPU, read by the shader.
    d.Format = DXGI_FORMAT_R16G16B16A16_UNORM;
    d.Usage = D3D11_USAGE_DYNAMIC;
    d.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    d.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    sv.Format = DXGI_FORMAT_R16G16B16A16_UNORM;
    ok = ok && SUCCEEDED(o.device->CreateTexture2D(&d, nullptr, &o.gain)) && SUCCEEDED(o.device->CreateShaderResourceView(o.gain, &sv, &o.gain_view));

    ok = ok && SUCCEEDED(o.device->CreateVertexShader(kDx11Vs, sizeof kDx11Vs, nullptr, &o.vs)) &&
         SUCCEEDED(o.device->CreatePixelShader(kDx11DownPs, sizeof kDx11DownPs, nullptr, &o.down_ps)) &&
         SUCCEEDED(o.device->CreatePixelShader(kDx11ApplyPs, sizeof kDx11ApplyPs, nullptr, &o.apply_ps));

    D3D11_SAMPLER_DESC sd = {};
    sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    sd.ComparisonFunc = D3D11_COMPARISON_NEVER;
    sd.MaxLOD = D3D11_FLOAT32_MAX;
    sd.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
    ok = ok && SUCCEEDED(o.device->CreateSamplerState(&sd, &o.point_sampler));
    sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    ok = ok && SUCCEEDED(o.device->CreateSamplerState(&sd, &o.linear_sampler));

    const float k[4] = {o.format.srgb ? 1.0f : 0.0f, 0, 0, 0};
    D3D11_BUFFER_DESC bdesc = {};
    bdesc.ByteWidth = sizeof k;
    bdesc.Usage = D3D11_USAGE_IMMUTABLE;
    bdesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    D3D11_SUBRESOURCE_DATA init = {k};
    ok = ok && SUCCEEDED(o.device->CreateBuffer(&bdesc, &init, &o.constants));

    D3D11_BLEND_DESC blend = {};
    blend.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    D3D11_RASTERIZER_DESC rs = {};
    rs.FillMode = D3D11_FILL_SOLID;
    rs.CullMode = D3D11_CULL_NONE;
    rs.DepthClipEnable = TRUE;
    D3D11_DEPTH_STENCIL_DESC ds = {};
    ok = ok && SUCCEEDED(o.device->CreateBlendState(&blend, &o.blend)) && SUCCEEDED(o.device->CreateRasterizerState(&rs, &o.rasterizer)) &&
         SUCCEEDED(o.device->CreateDepthStencilState(&ds, &o.depth_stencil));
    if (!ok) {
        log("dx11: could not create what the overlay needs");
        o.release_all();
        o.broken = true;
        return false;
    }
    log("dx11: %ux%u, the network at %ux%u, format %d%s%s", o.w, o.h, o.cw, o.ch, (int)bd.Format, o.format.srgb ? " (sRGB)" : "",
        o.multisampled ? ", multisampled" : "");
    return true;
}

void draw_triangle(ID3D11DeviceContext *c, UINT w, UINT h, ID3D11PixelShader *ps) {
    const D3D11_VIEWPORT vp = {0.0f, 0.0f, (float)w, (float)h, 0.0f, 1.0f};
    c->RSSetViewports(1, &vp);
    c->IASetInputLayout(nullptr);
    c->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    c->VSSetShader(g_overlay.vs, nullptr, 0);
    c->GSSetShader(nullptr, nullptr, 0);
    c->HSSetShader(nullptr, nullptr, 0);
    c->DSSetShader(nullptr, nullptr, 0);
    c->PSSetShader(ps, nullptr, 0);
    c->Draw(3, 0);
}

// The frame, made reduced and read back, into the shared memory.
bool capture(Overlay &o, ID3D11DeviceContext *c, uint8_t *to) {
    ID3D11ShaderResourceView *views[2] = {o.copy_view, nullptr};
    ID3D11SamplerState *samplers[2] = {o.linear_sampler, nullptr};
    c->PSSetShaderResources(0, 2, views);
    c->PSSetSamplers(0, 2, samplers);
    c->OMSetRenderTargets(1, &o.small_target, nullptr);
    draw_triangle(c, o.cw, o.ch, o.down_ps);
    c->CopyResource(o.staging, o.reduced);
    D3D11_MAPPED_SUBRESOURCE m;
    if (FAILED(c->Map(o.staging, 0, D3D11_MAP_READ, 0, &m))) return false;
    for (UINT y = 0; y < o.ch; ++y) memcpy(to + (size_t)y * o.cw * 4, (const uint8_t *)m.pData + (size_t)y * m.RowPitch, (size_t)o.cw * 4);
    c->Unmap(o.staging, 0);
    return true;
}

bool upload_gain(Overlay &o, ID3D11DeviceContext *c, const uint8_t *from) {
    D3D11_MAPPED_SUBRESOURCE m;
    const HRESULT hr = c->Map(o.gain, 0, D3D11_MAP_WRITE_DISCARD, 0, &m);
    if (FAILED(hr)) {
        static int reported = 0;
        if (reported++ < 3) log("dx11: could not map the gain texture (%08lx)", (unsigned long)hr);
        return false;
    }
    for (UINT y = 0; y < o.ch; ++y) memcpy((uint8_t *)m.pData + (size_t)y * m.RowPitch, from + (size_t)y * o.cw * 8, (size_t)o.cw * 8);
    c->Unmap(o.gain, 0);
    return true;
}

void apply_gain(Overlay &o, ID3D11DeviceContext *c, ID3D11RenderTargetView *target) {
    ID3D11ShaderResourceView *views[2] = {o.copy_view, o.gain_view};
    ID3D11SamplerState *samplers[2] = {o.point_sampler, o.linear_sampler};
    c->PSSetShaderResources(0, 2, views);
    c->PSSetSamplers(0, 2, samplers);
    c->PSSetConstantBuffers(0, 1, &o.constants);
    c->OMSetRenderTargets(1, &target, nullptr);
    draw_triangle(c, o.w, o.h, o.apply_ps);
}

// For testing: a texture of the frame's size and format as a 32-bit BMP.
void dump_texture(Overlay &o, ID3D11DeviceContext *c, ID3D11Texture2D *texture, const wchar_t *name) {
    D3D11_TEXTURE2D_DESC d;
    texture->GetDesc(&d);
    d.Usage = D3D11_USAGE_STAGING;
    d.BindFlags = 0;
    d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ID3D11Texture2D *staging = nullptr;
    if (FAILED(o.device->CreateTexture2D(&d, nullptr, &staging))) return;
    c->CopyResource(staging, texture);
    D3D11_MAPPED_SUBRESOURCE m;
    if (SUCCEEDED(c->Map(staging, 0, D3D11_MAP_READ, 0, &m))) {
        FILE *f = _wfsopen((g_dir + name).c_str(), L"wb", _SH_DENYNO);
        if (f) {
            BITMAPFILEHEADER fh = {};
            BITMAPINFOHEADER ih = {};
            fh.bfType = 0x4D42;
            fh.bfOffBits = sizeof fh + sizeof ih;
            fh.bfSize = fh.bfOffBits + d.Width * d.Height * 4;
            ih.biSize = sizeof ih;
            ih.biWidth = (LONG)d.Width;
            ih.biHeight = -(LONG)d.Height;
            ih.biPlanes = 1;
            ih.biBitCount = 32;
            fwrite(&fh, sizeof fh, 1, f);
            fwrite(&ih, sizeof ih, 1, f);
            const bool rgba = o.format.unorm == DXGI_FORMAT_R8G8B8A8_UNORM;
            std::vector<uint8_t> row(d.Width * 4);
            for (UINT y = 0; y < d.Height; ++y) {
                const uint8_t *src = (const uint8_t *)m.pData + (size_t)y * m.RowPitch;
                for (UINT x = 0; x < d.Width; ++x) {
                    row[4 * x + 0] = src[4 * x + (rgba ? 2 : 0)];
                    row[4 * x + 1] = src[4 * x + 1];
                    row[4 * x + 2] = src[4 * x + (rgba ? 0 : 2)];
                    row[4 * x + 3] = 255;
                }
                fwrite(row.data(), 1, row.size(), f);
            }
            fclose(f);
        }
        c->Unmap(staging, 0);
    }
    staging->Release();
}

void run_bridge(Overlay &o, ID3D11DeviceContext *c, ID3D11Texture2D *back, ID3D11RenderTargetView *target) {
    if (!g_bridge.start()) {
        o.broken = true;
        return;
    }
    bridge::Header *h = g_bridge.header;
    if (h->hostState == bridge::kFailed) {
        log("bridge: the host failed: %s", h->message);
        o.broken = true;
        return;
    }
    ++o.frames;
    if (h->hostState != bridge::kRunning) return;
    if (!g_bridge.map(o.cw, o.ch)) return;

    auto consume = [&]() {
        if (h->resultSeq == g_bridge.seen_result || h->resultWidth != o.cw || h->resultHeight != o.ch) return;
        if (upload_gain(o, c, g_bridge.view + bridge::result_offset(o.cw, o.ch))) {
            o.have_gain = true;
            ++o.applied;
        }
        g_bridge.seen_result = h->resultSeq;
    };
    consume();

    // A copy of the frame: what the network sees, and what the gain is applied to, since the
    // backbuffer cannot be read while it is written.
    if (o.multisampled) c->ResolveSubresource(o.copy, 0, back, 0, o.format.unorm);
    else c->CopyResource(o.copy, back);

    StateBackup saved;
    saved.save(c);
    c->OMSetBlendState(o.blend, nullptr, 0xFFFFFFFF);
    c->OMSetDepthStencilState(o.depth_stencil, 0);
    c->RSSetState(o.rasterizer);
    c->RSSetScissorRects(0, nullptr);

    bool sent = false;
    if (InterlockedCompareExchange(&h->hostReady, 0, 1) == 1) {
        h->width = o.cw;
        h->height = o.ch;
        ResetEvent(g_bridge.result_event);   // whatever it was signalled for is old
        if (capture(o, c, g_bridge.view + bridge::capture_offset())) {
            InterlockedIncrement(&h->captureSeq);
            SetEvent(g_bridge.capture_event);
            ++o.sent;
            sent = true;
        } else {
            InterlockedExchange(&h->hostReady, 1);
        }
    }
    // Waiting: the answer to this very frame is applied to it.
    if (sent && g_config.options.wait) {
        if (WaitForSingleObject(g_bridge.result_event, 500) == WAIT_OBJECT_0) consume();
        else log("bridge: the network did not answer within 500 ms");
    }
    const bool dump = g_config.dump_frame > 0 && o.frames == (unsigned)g_config.dump_frame;
    if (dump) dump_texture(o, c, o.copy, L"dump_plain.bmp");
    if (o.have_gain) apply_gain(o, c, target);
    saved.restore(c);
    if (dump) {
        c->CopyResource(o.copy, back);   // after the gain
        dump_texture(o, c, o.copy, L"dump_final.bmp");
        log("dumped frame %u (gain applied: %d)", o.frames, (int)o.have_gain);
    }

    const DWORD now = GetTickCount();
    if (now - o.last_report > 10000) {
        log("bridge: %u frames, %u captures sent, %u results applied, network %.0f ms", o.frames, o.sent, o.applied, h->evaluationMs);
        o.last_report = now;
    }
}

} // namespace

void on_present(IDXGISwapChain *swap_chain) {
    Overlay &o = g_overlay;
    if (o.broken) return;
    if (!menu::visible() && read_config() && g_bridge.header) {
        g_bridge.header->settings = g_config.settings;
        InterlockedIncrement(&g_bridge.header->settingsSeq);
    }
    if (o.swap_chain && o.swap_chain != swap_chain) return;   // one swap chain is enhanced: the first

    ID3D11Device *device = nullptr;
    if (FAILED(swap_chain->GetDevice(__uuidof(ID3D11Device), (void **)&device))) {
        if (!o.unsupported_logged) log("dx11: the swap chain's device is not a Direct3D 11 one (Direct3D 10 is not handled yet)");
        o.unsupported_logged = true;
        return;
    }
    DXGI_SWAP_CHAIN_DESC sd = {};
    swap_chain->GetDesc(&sd);
    if (sd.BufferDesc.Width < 320 || sd.BufferDesc.Height < 240) {   // a swap chain for something reduced: not the game's picture
        device->Release();
        return;
    }
    if (!o.swap_chain) {
        o.swap_chain = swap_chain;
        o.device = device;
        o.window = sd.OutputWindow;
        log("dx11: swap chain %ux%u, window %p", sd.BufferDesc.Width, sd.BufferDesc.Height, (void *)o.window);
    }
    ID3D11DeviceContext *c = nullptr;
    device->GetImmediateContext(&c);
    ID3D11Texture2D *back = nullptr;
    if (c && SUCCEEDED(swap_chain->GetBuffer(0, __uuidof(ID3D11Texture2D), (void **)&back))) {
        D3D11_TEXTURE2D_DESC bd;
        back->GetDesc(&bd);
        const Format f = classify(bd.Format);
        ID3D11RenderTargetView *target = nullptr;
        if (!f.ok && !o.unsupported_logged) {
            log("dx11: the backbuffer is format %d, which is not handled", (int)bd.Format);
            o.unsupported_logged = true;
        }
        if (f.ok) {
            D3D11_RENDER_TARGET_VIEW_DESC rv = {};
            rv.Format = f.target;
            rv.ViewDimension = bd.SampleDesc.Count > 1 ? D3D11_RTV_DIMENSION_TEXTURE2DMS : D3D11_RTV_DIMENSION_TEXTURE2D;
            device->CreateRenderTargetView(back, &rv, &target);
        }
        if (!o.menu_ready && o.window && target) {
            menu::init(device, c, o.window);
            o.menu_ready = true;
            if (g_config.open_menu) menu::open();
        }
        if (g_config.enabled && target) {
            if (o.copy && (o.w != bd.Width || o.h != bd.Height)) o.release_all();
            if (!o.copy) setup(o, back);
            UINT cw, ch;
            network_size(bd.Width, bd.Height, cw, ch);
            if (o.copy && (cw != o.cw || ch != o.ch)) {   // the network's resolution was changed
                o.release_all();
                setup(o, back);
            }
            if (o.copy && !o.broken) run_bridge(o, c, back, target);
        }
        if (menu::visible() && target) {
            menu::Status status = {};
            status.enabled = g_config.enabled;
            status.header = g_bridge.started && !g_bridge.failed ? g_bridge.header : nullptr;
            status.frames = o.frames;
            status.sent = o.sent;
            status.applied = o.applied;
            status.problem = o.broken && !status.header ? "The bridge cannot run here: see dlss5nr_bridge_dxgi.log" : nullptr;
            status.mask_available = false;
            menu::frame(c, target, bd.Width, bd.Height, status);
            if (g_config.dump_frame > 0 && o.frames == (unsigned)g_config.dump_frame && o.copy) dump_texture(o, c, back, L"dump_menu.bmp");
        }
        release(target);
        back->Release();
    }
    if (c) c->Release();
    if (o.device != device) device->Release();
}

} // namespace dx11
