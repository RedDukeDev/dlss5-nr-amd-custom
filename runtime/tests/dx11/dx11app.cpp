// A Direct3D 11 test "game": a window, a flip-model swap chain, an animated procedural scene, and a
// capture of what the desktop shows of it (so what OptiScaler did to the back buffer is included).
// usage: dx11app [seconds=12] [capture_at=8] [out.bmp] [width=1920] [height=1080] [format: rgba|bgra|srgb|fp16] [freeze_time]
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <dxgi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")

static const char *hlsl = R"(
cbuffer C : register(b0) { float2 res; float t; float pad; };
float4 vs(uint id : SV_VertexID) : SV_Position { float2 p = float2((id << 1) & 2, id & 2); return float4(p * 2 - 1, 0, 1); }
float hash(float2 p) { return frac(sin(dot(p, float2(127.1, 311.7))) * 43758.5453); }
float noise(float2 p) { float2 i = floor(p), f = frac(p); f = f * f * (3 - 2 * f);
  return lerp(lerp(hash(i), hash(i + float2(1, 0)), f.x), lerp(hash(i + float2(0, 1)), hash(i + 1), f.x), f.y); }
float4 ps(float4 pos : SV_Position) : SV_Target {
  float2 uv = pos.xy / res; float2 q = uv * float2(res.x / res.y, 1);
  // sky to ground gradient, hills, a lit sphere moving across, a checker floor, and a few UI-like bars
  float3 sky = lerp(float3(0.25, 0.45, 0.85), float3(0.9, 0.75, 0.6), pow(uv.y, 1.5));
  float h = 0.55 + 0.08 * noise(q * 6) + 0.05 * sin(q.x * 5);
  float3 c = sky;
  if (uv.y > h) { float3 g = lerp(float3(0.15, 0.3, 0.1), float3(0.35, 0.28, 0.15), noise(q * 40)); c = g * (0.6 + 0.4 * noise(q * 160)); }
  float2 sp = float2(0.3 + 0.4 * frac(t * 0.1), 0.62); float2 d = (uv - sp) * float2(res.x / res.y, 1);
  float r = length(d); if (r < 0.1) { float z = sqrt(1 - (r / 0.1) * (r / 0.1)); float3 n = float3(d / 0.1, z);
    float l = saturate(dot(n, normalize(float3(-0.4, -0.6, 0.7)))); c = float3(0.8, 0.25, 0.2) * (0.15 + 0.85 * l) + pow(l, 40) * 0.5; }
  float2 ck = floor(q * float2(24, 12)); if (uv.y > 0.85) c = lerp(c, ((int(ck.x + ck.y) & 1) ? 0.75 : 0.2).xxx, 0.7);
  if (uv.x < 0.18 && uv.y < 0.06) c = float3(0.05, 0.05, 0.08) + 0.9 * step(frac(uv.x * 40), 0.45) * step(0.01, frac(uv.y * 20));
  return float4(c, 1);
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

int main(int argc, char **argv) {
    double seconds = argc > 1 ? atof(argv[1]) : 12, capture_at = argc > 2 ? atof(argv[2]) : 8;
    const char *out = argc > 3 ? argv[3] : "shot.bmp";
    int W = argc > 4 ? atoi(argv[4]) : 1920, H = argc > 5 ? atoi(argv[5]) : 1080;
    const char *fmt = argc > 6 ? argv[6] : "rgba";
    double freeze = argc > 7 ? atof(argv[7]) : -1;
    DXGI_FORMAT format = DXGI_FORMAT_R8G8B8A8_UNORM;
    if (!strcmp(fmt, "bgra")) format = DXGI_FORMAT_B8G8R8A8_UNORM;
    if (!strcmp(fmt, "srgb")) format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    if (!strcmp(fmt, "fp16")) format = DXGI_FORMAT_R16G16B16A16_FLOAT;

    WNDCLASSW wc = {}; wc.lpfnWndProc = proc; wc.hInstance = GetModuleHandleW(nullptr); wc.lpszClassName = L"dx11app"; wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    RegisterClassW(&wc);
    RECT rc = {0, 0, W, H}; AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);
    HWND hwnd = CreateWindowW(L"dx11app", L"dx11app", WS_OVERLAPPEDWINDOW | WS_VISIBLE, 20, 20, rc.right - rc.left, rc.bottom - rc.top, nullptr, nullptr, wc.hInstance, nullptr);

    IDXGIFactory1 *factory; if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void **) &factory))) return 1;
    IDXGIAdapter1 *adapter = nullptr, *pick = nullptr; SIZE_T best = 0;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) == S_OK; ++i) {
        DXGI_ADAPTER_DESC1 d; adapter->GetDesc1(&d);
        if (!(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) && d.DedicatedVideoMemory > best) { best = d.DedicatedVideoMemory; pick = adapter; }
    }
    ID3D11Device *dev; ID3D11DeviceContext *ctx;
    if (FAILED(D3D11CreateDevice(pick, D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &dev, nullptr, &ctx))) return 2;
    DXGI_SWAP_CHAIN_DESC sd = {}; sd.BufferDesc.Width = W; sd.BufferDesc.Height = H; sd.BufferDesc.Format = format;
    sd.SampleDesc.Count = 1; sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT | DXGI_USAGE_SHADER_INPUT; sd.BufferCount = 2;
    sd.OutputWindow = hwnd; sd.Windowed = TRUE; sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    IDXGISwapChain *sc; if (FAILED(factory->CreateSwapChain(dev, &sd, &sc))) return 3;
    ID3D11Texture2D *bb; sc->GetBuffer(0, __uuidof(ID3D11Texture2D), (void **) &bb);
    D3D11_RENDER_TARGET_VIEW_DESC rd = {}; rd.Format = format; rd.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
    ID3D11RenderTargetView *rtv; dev->CreateRenderTargetView(bb, &rd, &rtv);
    ID3DBlob *vsb, *psb, *err;
    if (FAILED(D3DCompile(hlsl, strlen(hlsl), 0, 0, 0, "vs", "vs_5_0", 0, 0, &vsb, &err))) { puts((char *) err->GetBufferPointer()); return 4; }
    if (FAILED(D3DCompile(hlsl, strlen(hlsl), 0, 0, 0, "ps", "ps_5_0", 0, 0, &psb, &err))) { puts((char *) err->GetBufferPointer()); return 4; }
    ID3D11VertexShader *vs; ID3D11PixelShader *ps;
    dev->CreateVertexShader(vsb->GetBufferPointer(), vsb->GetBufferSize(), 0, &vs); dev->CreatePixelShader(psb->GetBufferPointer(), psb->GetBufferSize(), 0, &ps);
    D3D11_BUFFER_DESC bd = {16, D3D11_USAGE_DEFAULT, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0}; ID3D11Buffer *cb; dev->CreateBuffer(&bd, nullptr, &cb);
    D3D11_RASTERIZER_DESC rs = {}; rs.FillMode = D3D11_FILL_SOLID; rs.CullMode = D3D11_CULL_NONE; ID3D11RasterizerState *rst; dev->CreateRasterizerState(&rs, &rst);

    using clk = std::chrono::steady_clock; auto start = clk::now(); long frames = 0, lastFrames = 0; auto tick = start; bool shot = false;
    MSG msg;
    for (;;) {
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) { if (msg.message == WM_QUIT) return 0; TranslateMessage(&msg); DispatchMessageW(&msg); }
        double t = std::chrono::duration<double>(clk::now() - start).count();
        if (t > seconds) break;
        float c[4] = {(float) W, (float) H, (float) (freeze >= 0 ? freeze : t), 0}; ctx->UpdateSubresource(cb, 0, nullptr, c, 0, 0);
        D3D11_VIEWPORT vp = {0, 0, (float) W, (float) H, 0, 1};
        ctx->OMSetRenderTargets(1, &rtv, nullptr); ctx->RSSetViewports(1, &vp); ctx->RSSetState(rst);
        ctx->VSSetShader(vs, 0, 0); ctx->PSSetShader(ps, 0, 0); ctx->PSSetConstantBuffers(0, 1, &cb);
        ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST); ctx->Draw(3, 0);
        sc->Present(getenv("NOVSYNC") ? 0 : 1, 0); ++frames;
        if (!shot && t >= capture_at) { Sleep(150); capture(hwnd, W, H, out); shot = true; }
        double dt = std::chrono::duration<double>(clk::now() - tick).count();
        if (dt >= 2.0) { printf("t=%.1f  %.1f fps\n", t, (frames - lastFrames) / dt); fflush(stdout); lastFrames = frames; tick = clk::now(); }
        // sc->GetBuffer(0) is the same texture every frame in a flip-model chain for D3D11
    }
    return 0;
}
