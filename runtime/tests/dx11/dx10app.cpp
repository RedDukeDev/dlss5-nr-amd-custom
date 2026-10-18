// A Direct3D 10 test "game": a window, a DXGI swap chain on a Direct3D 10 device, an animated procedural scene
// drawn by a pixel shader, presented every frame; the window is captured to a BMP at a chosen time so what was
// done to the picture is in it. Meant to run with DXVK's d3d10core.dll, d3d11.dll and dxgi.dll beside it
// (DX10 -> DX11 -> Vulkan) and OptiScaler loaded as version.dll, which the program imports.
// usage: dx10app [seconds=12] [capture_at=8] [out.bmp] [width=1920] [height=1080] [freeze_time]
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d10.h>
#include <d3dcompiler.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#pragma comment(lib, "d3d10.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "version.lib")

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

// On a crash: the exception and a short stack, as module + offset, for finding where it was
static LONG WINAPI crashed(EXCEPTION_POINTERS *e) {
    printf("CRASH %08lx at %p\n", e->ExceptionRecord->ExceptionCode, e->ExceptionRecord->ExceptionAddress);
    void *frames[24]; USHORT n = CaptureStackBackTrace(0, 24, frames, nullptr);
    frames[0] = e->ExceptionRecord->ExceptionAddress;
    for (USHORT i = 0; i < n; ++i) {
        HMODULE m = nullptr; char name[MAX_PATH] = "?";
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCSTR) frames[i], &m)) GetModuleFileNameA(m, name, MAX_PATH);
        const char *base = strrchr(name, '\\'); printf("  %s+0x%llx\n", base ? base + 1 : name, (unsigned long long) ((char *) frames[i] - (char *) m));
    }
    fflush(stdout); return EXCEPTION_EXECUTE_HANDLER;
}

int main(int argc, char **argv) {
    SetUnhandledExceptionFilter(crashed);
    double seconds = argc > 1 ? atof(argv[1]) : 12, capture_at = argc > 2 ? atof(argv[2]) : 8;
    const char *out = argc > 3 ? argv[3] : "shot.bmp";
    int W = argc > 4 ? atoi(argv[4]) : 1920, H = argc > 5 ? atoi(argv[5]) : 1080;
    double freeze = argc > 6 ? atof(argv[6]) : -1;

    // a game imports version.dll; calling it makes this program load the proxy beside it
    DWORD handle = 0; GetFileVersionInfoSizeW(L"C:\\Windows\\System32\\kernel32.dll", &handle);

    WNDCLASSW wc = {}; wc.lpfnWndProc = proc; wc.hInstance = GetModuleHandleW(nullptr); wc.lpszClassName = L"dx10app"; wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    RegisterClassW(&wc);
    RECT rc = {0, 0, W, H}; AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);
    HWND hwnd = CreateWindowW(L"dx10app", L"dx10app", WS_OVERLAPPEDWINDOW | WS_VISIBLE, 20, 20, rc.right - rc.left, rc.bottom - rc.top, nullptr, nullptr, wc.hInstance, nullptr);

    DXGI_SWAP_CHAIN_DESC sd = {}; sd.BufferDesc.Width = W; sd.BufferDesc.Height = H; sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.SampleDesc.Count = 1; sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT; sd.BufferCount = 1;
    sd.OutputWindow = hwnd; sd.Windowed = TRUE; sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
    ID3D10Device *dev; IDXGISwapChain *sc;
    if (FAILED(D3D10CreateDeviceAndSwapChain(nullptr, D3D10_DRIVER_TYPE_HARDWARE, nullptr, 0, D3D10_SDK_VERSION, &sd, &sc, &dev))) { puts("D3D10CreateDeviceAndSwapChain failed"); return 2; }
    ID3D10Texture2D *bb; sc->GetBuffer(0, __uuidof(ID3D10Texture2D), (void **) &bb);
    ID3D10RenderTargetView *rtv; dev->CreateRenderTargetView(bb, nullptr, &rtv);
    ID3DBlob *vsb, *psb, *err;
    if (FAILED(D3DCompile(hlsl, strlen(hlsl), 0, 0, 0, "vs", "vs_4_0", 0, 0, &vsb, &err))) { puts((char *) err->GetBufferPointer()); return 4; }
    if (FAILED(D3DCompile(hlsl, strlen(hlsl), 0, 0, 0, "ps", "ps_4_0", 0, 0, &psb, &err))) { puts((char *) err->GetBufferPointer()); return 4; }
    ID3D10VertexShader *vs; ID3D10PixelShader *ps;
    dev->CreateVertexShader(vsb->GetBufferPointer(), vsb->GetBufferSize(), &vs); dev->CreatePixelShader(psb->GetBufferPointer(), psb->GetBufferSize(), &ps);
    D3D10_BUFFER_DESC bd = {16, D3D10_USAGE_DEFAULT, D3D10_BIND_CONSTANT_BUFFER, 0, 0}; ID3D10Buffer *cb; dev->CreateBuffer(&bd, nullptr, &cb);
    D3D10_RASTERIZER_DESC rs = {}; rs.FillMode = D3D10_FILL_SOLID; rs.CullMode = D3D10_CULL_NONE; ID3D10RasterizerState *rst; dev->CreateRasterizerState(&rs, &rst);

    using clk = std::chrono::steady_clock; auto start = clk::now(), tick = start; long frames = 0, lastFrames = 0; bool shot = false;
    MSG msg;
    for (;;) {
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) { if (msg.message == WM_QUIT) return 0; TranslateMessage(&msg); DispatchMessageW(&msg); }
        double t = std::chrono::duration<double>(clk::now() - start).count();
        if (t > seconds) break;
        float c[4] = {(float) W, (float) H, (float) (freeze >= 0 ? freeze : t), 0}; dev->UpdateSubresource(cb, 0, nullptr, c, 0, 0);
        D3D10_VIEWPORT vp = {0, 0, (UINT) W, (UINT) H, 0.0f, 1.0f};
        dev->OMSetRenderTargets(1, &rtv, nullptr); dev->RSSetViewports(1, &vp); dev->RSSetState(rst);
        dev->VSSetShader(vs); dev->PSSetShader(ps); dev->PSSetConstantBuffers(0, 1, &cb);
        dev->IASetPrimitiveTopology(D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST); dev->Draw(3, 0);
        sc->Present(getenv("NOVSYNC") ? 0 : 1, 0); ++frames;
        if (!shot && t >= capture_at) { Sleep(150); capture(hwnd, W, H, out); shot = true; }
        double dt = std::chrono::duration<double>(clk::now() - tick).count();
        if (dt >= 2.0) { printf("t=%.1f  %.1f fps\n", t, (frames - lastFrames) / dt); fflush(stdout); lastFrames = frames; tick = clk::now(); }
    }
    return 0;
}
