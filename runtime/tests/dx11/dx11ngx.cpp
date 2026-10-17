// A Direct3D 11 test "game" that upscales through the NGX API, as a DLSS game does, with OptiScaler
// answering it (loaded as dxgi.dll beside this exe): the scene is drawn at render resolution with
// colour, depth and motion vectors, EvaluateFeature produces the output, and that is presented.
// usage: dx11ngx [seconds=14] [capture_at=10] [out.bmp] [out_w=1920] [out_h=1080] [render_percent=50] [freeze_time]
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <dxgi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>

#include "nvsdk_ngx.h"
#include "nvsdk_ngx_params.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")

static const char *hlsl = R"(
cbuffer C : register(b0) { float2 res; float t; float moving; float2 jitter; float2 pad; };
float4 vs(uint id : SV_VertexID) : SV_Position { float2 p = float2((id << 1) & 2, id & 2); return float4(p * 2 - 1, 0, 1); }
float hash(float2 p) { return frac(sin(dot(p, float2(127.1, 311.7))) * 43758.5453); }
float noise(float2 p) { float2 i = floor(p), f = frac(p); f = f * f * (3 - 2 * f);
  return lerp(lerp(hash(i), hash(i + float2(1, 0)), f.x), lerp(hash(i + float2(0, 1)), hash(i + 1), f.x), f.y); }
struct Out { float4 color : SV_Target0; float depth : SV_Target1; float2 mv : SV_Target2; };
Out ps(float4 pos : SV_Position) {
  Out o; float2 uv = (pos.xy + jitter) / res; float2 q = uv * float2(res.x / res.y, 1);
  float3 sky = lerp(float3(0.25, 0.45, 0.85), float3(0.9, 0.75, 0.6), pow(uv.y, 1.5));
  float h = 0.55 + 0.08 * noise(q * 6) + 0.05 * sin(q.x * 5);
  float3 c = sky; float depth = 1.0; float2 mv = 0;
  if (uv.y > h) { float3 g = lerp(float3(0.15, 0.3, 0.1), float3(0.35, 0.28, 0.15), noise(q * 40)); c = g * (0.6 + 0.4 * noise(q * 160)); depth = 0.2 + 0.7 * (1 - uv.y); }
  float2 sp = float2(0.3 + 0.4 * frac(t * 0.1), 0.62); float2 d = (uv - sp) * float2(res.x / res.y, 1);
  float r = length(d); if (r < 0.1) { float z = sqrt(1 - (r / 0.1) * (r / 0.1)); float3 n = float3(d / 0.1, z);
    float l = saturate(dot(n, normalize(float3(-0.4, -0.6, 0.7)))); c = float3(0.8, 0.25, 0.2) * (0.15 + 0.85 * l) + pow(l, 40) * 0.5; depth = 0.15;
    mv = float2(-0.04 * res.x / 60.0 * moving, 0); }
  float2 ck = floor(q * float2(24, 12)); if (uv.y > 0.85) c = lerp(c, ((int(ck.x + ck.y) & 1) ? 0.75 : 0.2).xxx, 0.7);
  if (uv.x < 0.18 && uv.y < 0.06) c = float3(0.05, 0.05, 0.08) + 0.9 * step(frac(uv.x * 40), 0.45) * step(0.01, frac(uv.y * 20));
  o.color = float4(c, 1); o.depth = depth; o.mv = mv; return o;
}
)";

static LRESULT CALLBACK proc(HWND h, UINT m, WPARAM w, LPARAM l) {
    if (m == WM_DESTROY) { PostQuitMessage(0); return 0; }
    return DefWindowProcW(h, m, w, l);
}

static void capture(HWND hwnd, int w, int h, const char *path) {
    POINT origin = {0, 0}; ClientToScreen(hwnd, &origin);
    HDC screen = GetDC(nullptr), mem = CreateCompatibleDC(screen);
    BITMAPINFO bi = {}; bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER); bi.bmiHeader.biWidth = w; bi.bmiHeader.biHeight = -h;
    bi.bmiHeader.biPlanes = 1; bi.bmiHeader.biBitCount = 32; bi.bmiHeader.biCompression = BI_RGB;
    void *bits = nullptr; HBITMAP dib = CreateDIBSection(screen, &bi, DIB_RGB_COLORS, &bits, nullptr, 0);
    SelectObject(mem, dib); BitBlt(mem, 0, 0, w, h, screen, origin.x, origin.y, SRCCOPY | CAPTUREBLT);
    BITMAPFILEHEADER fh = {}; fh.bfType = 0x4D42; fh.bfOffBits = sizeof(fh) + sizeof(bi.bmiHeader); fh.bfSize = fh.bfOffBits + w * h * 4;
    BITMAPINFOHEADER ih = bi.bmiHeader; ih.biHeight = -h;
    FILE *f = fopen(path, "wb"); fwrite(&fh, sizeof fh, 1, f); fwrite(&ih, sizeof ih, 1, f); fwrite(bits, 4, (size_t) w * h, f); fclose(f);
    DeleteObject(dib); DeleteDC(mem); ReleaseDC(nullptr, screen);
}

static float halton(int i, int b) { float f = 1, r = 0; while (i > 0) { f /= b; r += f * (i % b); i /= b; } return r; }

typedef NVSDK_NGX_Result (*PFN_Init)(unsigned long long, const wchar_t *, ID3D11Device *, const NVSDK_NGX_FeatureCommonInfo *, NVSDK_NGX_Version);
typedef NVSDK_NGX_Result (*PFN_Caps)(NVSDK_NGX_Parameter **);
typedef NVSDK_NGX_Result (*PFN_Create)(ID3D11DeviceContext *, NVSDK_NGX_Feature, NVSDK_NGX_Parameter *, NVSDK_NGX_Handle **);
typedef NVSDK_NGX_Result (*PFN_Eval)(ID3D11DeviceContext *, const NVSDK_NGX_Handle *, NVSDK_NGX_Parameter *, PFN_NVSDK_NGX_ProgressCallback);

int main(int argc, char **argv) {
    double seconds = argc > 1 ? atof(argv[1]) : 14, capture_at = argc > 2 ? atof(argv[2]) : 10;
    const char *out = argc > 3 ? argv[3] : "shot.bmp";
    int W = argc > 4 ? atoi(argv[4]) : 1920, H = argc > 5 ? atoi(argv[5]) : 1080;
    int percent = argc > 6 ? atoi(argv[6]) : 50;
    double freeze = argc > 7 ? atof(argv[7]) : -1;
    int RW = W * percent / 100, RH = H * percent / 100;

    WNDCLASSW wc = {}; wc.lpfnWndProc = proc; wc.hInstance = GetModuleHandleW(nullptr); wc.lpszClassName = L"dx11ngx"; wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    RegisterClassW(&wc);
    RECT rc = {0, 0, W, H}; AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);
    HWND hwnd = CreateWindowW(L"dx11ngx", L"dx11ngx", WS_OVERLAPPEDWINDOW | WS_VISIBLE, 20, 20, rc.right - rc.left, rc.bottom - rc.top, nullptr, nullptr, wc.hInstance, nullptr);

    IDXGIFactory1 *factory; if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void **) &factory))) return 1;
    IDXGIAdapter1 *adapter = nullptr, *pick = nullptr; SIZE_T best = 0;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) == S_OK; ++i) {
        DXGI_ADAPTER_DESC1 d; adapter->GetDesc1(&d);
        if (!(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) && d.DedicatedVideoMemory > best) { best = d.DedicatedVideoMemory; pick = adapter; }
    }
    ID3D11Device *dev; ID3D11DeviceContext *ctx;
    if (FAILED(D3D11CreateDevice(pick, D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &dev, nullptr, &ctx))) return 2;
    DXGI_SWAP_CHAIN_DESC sd = {}; sd.BufferDesc.Width = W; sd.BufferDesc.Height = H; sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.SampleDesc.Count = 1; sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT | DXGI_USAGE_SHADER_INPUT; sd.BufferCount = 2;
    sd.OutputWindow = hwnd; sd.Windowed = TRUE; sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    IDXGISwapChain *sc; if (FAILED(factory->CreateSwapChain(dev, &sd, &sc))) return 3;
    ID3D11Texture2D *bb; sc->GetBuffer(0, __uuidof(ID3D11Texture2D), (void **) &bb);

    auto tex = [&](int w, int h, DXGI_FORMAT f, UINT bind) { D3D11_TEXTURE2D_DESC d = {(UINT) w, (UINT) h, 1, 1, f, {1, 0}, D3D11_USAGE_DEFAULT, bind, 0, 0}; ID3D11Texture2D *t; dev->CreateTexture2D(&d, nullptr, &t); return t; };
    ID3D11Texture2D *color = tex(RW, RH, DXGI_FORMAT_R8G8B8A8_UNORM, D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE);
    ID3D11Texture2D *depth = tex(RW, RH, DXGI_FORMAT_R32_FLOAT, D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE);
    ID3D11Texture2D *mv = tex(RW, RH, DXGI_FORMAT_R16G16_FLOAT, D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE);
    ID3D11Texture2D *output = tex(W, H, DXGI_FORMAT_R8G8B8A8_UNORM, D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET);
    ID3D11RenderTargetView *rtv[3]; dev->CreateRenderTargetView(color, nullptr, &rtv[0]); dev->CreateRenderTargetView(depth, nullptr, &rtv[1]); dev->CreateRenderTargetView(mv, nullptr, &rtv[2]);

    ID3DBlob *vsb, *psb, *err;
    if (FAILED(D3DCompile(hlsl, strlen(hlsl), 0, 0, 0, "vs", "vs_5_0", 0, 0, &vsb, &err))) { puts((char *) err->GetBufferPointer()); return 4; }
    if (FAILED(D3DCompile(hlsl, strlen(hlsl), 0, 0, 0, "ps", "ps_5_0", 0, 0, &psb, &err))) { puts((char *) err->GetBufferPointer()); return 4; }
    ID3D11VertexShader *vs; ID3D11PixelShader *ps;
    dev->CreateVertexShader(vsb->GetBufferPointer(), vsb->GetBufferSize(), 0, &vs); dev->CreatePixelShader(psb->GetBufferPointer(), psb->GetBufferSize(), 0, &ps);
    D3D11_BUFFER_DESC bd = {32, D3D11_USAGE_DEFAULT, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0}; ID3D11Buffer *cb; dev->CreateBuffer(&bd, nullptr, &cb);
    D3D11_RASTERIZER_DESC rs = {}; rs.FillMode = D3D11_FILL_SOLID; rs.CullMode = D3D11_CULL_NONE; ID3D11RasterizerState *rst; dev->CreateRasterizerState(&rs, &rst);

    // NGX, through the module that answers it: OptiScaler, loaded as dxgi.dll
    HMODULE ngx = GetModuleHandleW(L"dxgi.dll");
    auto Init = (PFN_Init) GetProcAddress(ngx, "NVSDK_NGX_D3D11_Init");
    auto Caps = (PFN_Caps) GetProcAddress(ngx, "NVSDK_NGX_D3D11_GetCapabilityParameters");
    auto Create = (PFN_Create) GetProcAddress(ngx, "NVSDK_NGX_D3D11_CreateFeature");
    auto Eval = (PFN_Eval) GetProcAddress(ngx, "NVSDK_NGX_D3D11_EvaluateFeature");
    if (!Init || !Caps || !Create || !Eval) { puts("the NGX D3D11 entry points are not exported by dxgi.dll"); return 5; }
    puts("init..."); fflush(stdout);
    NVSDK_NGX_Result r = Init(0x1234, L".", dev, nullptr, NVSDK_NGX_Version_API);
    printf("init %x\n", (unsigned) r); fflush(stdout);
    NVSDK_NGX_Parameter *params = nullptr; r = Caps(&params);
    printf("caps %x %p\n", (unsigned) r, (void *) params); fflush(stdout);
    params->Set(NVSDK_NGX_Parameter_Width, (unsigned) RW); params->Set(NVSDK_NGX_Parameter_Height, (unsigned) RH);
    params->Set(NVSDK_NGX_Parameter_OutWidth, (unsigned) W); params->Set(NVSDK_NGX_Parameter_OutHeight, (unsigned) H);
    params->Set(NVSDK_NGX_Parameter_PerfQualityValue, (int) NVSDK_NGX_PerfQuality_Value_Balanced);
    params->Set(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags, (int) 0);
    NVSDK_NGX_Handle *handle = nullptr;
    r = Create(ctx, NVSDK_NGX_Feature_SuperSampling, params, &handle);
    printf("NGX create: %x\n", (unsigned) r); fflush(stdout);
    if (!handle) return 6;

    using clk = std::chrono::steady_clock; auto start = clk::now(), tick = start; long frames = 0, lastFrames = 0; bool shot = false;
    MSG msg; int index = 0;
    for (;;) {
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) { if (msg.message == WM_QUIT) return 0; TranslateMessage(&msg); DispatchMessageW(&msg); }
        double t = std::chrono::duration<double>(clk::now() - start).count();
        if (t > seconds) break;
        float jx = halton(index % 16 + 1, 2) - 0.5f, jy = halton(index % 16 + 1, 3) - 0.5f; ++index;
        float c[8] = {(float) RW, (float) RH, (float) (freeze >= 0 ? freeze : t), freeze >= 0 ? 0.f : 1.f, jx, jy, 0, 0};
        ctx->UpdateSubresource(cb, 0, nullptr, c, 0, 0);
        D3D11_VIEWPORT vp = {0, 0, (float) RW, (float) RH, 0, 1};
        ctx->OMSetRenderTargets(3, rtv, nullptr); ctx->RSSetViewports(1, &vp); ctx->RSSetState(rst);
        ctx->VSSetShader(vs, 0, 0); ctx->PSSetShader(ps, 0, 0); ctx->PSSetConstantBuffers(0, 1, &cb);
        ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST); ctx->Draw(3, 0);
        ID3D11RenderTargetView *none[3] = {}; ctx->OMSetRenderTargets(3, none, nullptr);

        params->Set(NVSDK_NGX_Parameter_Color, (ID3D11Resource *) color); params->Set(NVSDK_NGX_Parameter_Output, (ID3D11Resource *) output);
        params->Set(NVSDK_NGX_Parameter_Depth, (ID3D11Resource *) depth); params->Set(NVSDK_NGX_Parameter_MotionVectors, (ID3D11Resource *) mv);
        params->Set(NVSDK_NGX_Parameter_Jitter_Offset_X, jx); params->Set(NVSDK_NGX_Parameter_Jitter_Offset_Y, jy);
        params->Set(NVSDK_NGX_Parameter_Sharpness, 0.0f); params->Set(NVSDK_NGX_Parameter_Reset, frames == 0 ? 1 : 0);
        params->Set(NVSDK_NGX_Parameter_MV_Scale_X, 1.0f); params->Set(NVSDK_NGX_Parameter_MV_Scale_Y, 1.0f);
        params->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width, (unsigned) RW); params->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height, (unsigned) RH);
        Eval(ctx, handle, params, nullptr);

        ctx->CopyResource(bb, output);
        sc->Present(getenv("NOVSYNC") ? 0 : 1, 0); ++frames;
        if (!shot && t >= capture_at) { Sleep(150); capture(hwnd, W, H, out); shot = true; }
        double dt = std::chrono::duration<double>(clk::now() - tick).count();
        if (dt >= 2.0) { printf("t=%.1f  %.1f fps\n", t, (frames - lastFrames) / dt); fflush(stdout); lastFrames = frames; tick = clk::now(); }
    }
    return 0;
}
