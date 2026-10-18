// A Direct3D 9 test "game": a window, a windowed device, an animated procedural scene drawn by a pixel
// shader over a full-screen quad, presented every frame; the window is captured to a BMP at a chosen time so
// what was done to the picture is in it. Meant to run with DXVK's d3d9.dll beside it (DX9 -> Vulkan) and
// OptiScaler loaded as version.dll, which the program imports.
// usage: dx9app [seconds=12] [capture_at=8] [out.bmp] [width=1920] [height=1080] [freeze_time]
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d9.h>
#include <d3dcompiler.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#pragma comment(lib, "d3d9.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "version.lib")

static const char *hlsl = R"(
float4 prm : register(c0);   // res.x, res.y, time, 0
float4 vs(float4 p : POSITION) : POSITION { return p; }
float hash(float2 p) { return frac(sin(dot(p, float2(127.1, 311.7))) * 43758.5453); }
float noise(float2 p) { float2 i = floor(p), f = frac(p); f = f * f * (3 - 2 * f);
  return lerp(lerp(hash(i), hash(i + float2(1, 0)), f.x), lerp(hash(i + float2(0, 1)), hash(i + float2(1, 1)), f.x), f.y); }
float4 ps(float2 vpos : VPOS) : COLOR0 {
  float2 res = prm.xy; float t = prm.z;
  float2 uv = (vpos + 0.5) / res; float2 q = uv * float2(res.x / res.y, 1);
  float3 c = lerp(float3(0.25, 0.45, 0.85), float3(0.9, 0.75, 0.6), pow(uv.y, 1.5));
  float h = 0.55 + 0.08 * noise(q * 6) + 0.05 * sin(q.x * 5);
  if (uv.y > h) { float3 g = lerp(float3(0.15, 0.3, 0.1), float3(0.35, 0.28, 0.15), noise(q * 40)); c = g * (0.6 + 0.4 * noise(q * 160)); }
  float2 sp = float2(0.3 + 0.4 * frac(t * 0.1), 0.62); float2 d = (uv - sp) * float2(res.x / res.y, 1);
  float r = length(d);
  if (r < 0.1) { float z = sqrt(1 - (r / 0.1) * (r / 0.1)); float3 n = float3(d / 0.1, z);
    float l = saturate(dot(n, normalize(float3(-0.4, -0.6, 0.7)))); c = float3(0.8, 0.25, 0.2) * (0.15 + 0.85 * l) + pow(l, 40) * 0.5; }
  float2 ck = floor(q * float2(24, 12));
  if (uv.y > 0.85) c = lerp(c, (fmod(ck.x + ck.y, 2.0) > 0.5 ? 0.75 : 0.2).xxx, 0.7);
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
    double freeze = argc > 6 ? atof(argv[6]) : -1;

    // a game imports version.dll; calling it makes this program load the proxy beside it
    DWORD handle = 0; GetFileVersionInfoSizeW(L"C:\\Windows\\System32\\kernel32.dll", &handle);

    WNDCLASSW wc = {}; wc.lpfnWndProc = proc; wc.hInstance = GetModuleHandleW(nullptr); wc.lpszClassName = L"dx9app"; wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    RegisterClassW(&wc);
    RECT rc = {0, 0, W, H}; AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);
    HWND hwnd = CreateWindowW(L"dx9app", L"dx9app", WS_OVERLAPPEDWINDOW | WS_VISIBLE, 20, 20, rc.right - rc.left, rc.bottom - rc.top, nullptr, nullptr, wc.hInstance, nullptr);

    IDirect3D9 *d3d = Direct3DCreate9(D3D_SDK_VERSION); if (!d3d) return 1;
    D3DPRESENT_PARAMETERS pp = {}; pp.BackBufferWidth = W; pp.BackBufferHeight = H; pp.BackBufferFormat = D3DFMT_X8R8G8B8; pp.BackBufferCount = 1;
    pp.SwapEffect = D3DSWAPEFFECT_DISCARD; pp.hDeviceWindow = hwnd; pp.Windowed = TRUE;
    pp.PresentationInterval = getenv("NOVSYNC") ? D3DPRESENT_INTERVAL_IMMEDIATE : D3DPRESENT_INTERVAL_ONE;
    IDirect3DDevice9 *dev;
    if (FAILED(d3d->CreateDevice(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, hwnd, D3DCREATE_HARDWARE_VERTEXPROCESSING, &pp, &dev))) { puts("CreateDevice failed"); return 2; }

    ID3DBlob *vsb, *psb, *err;
    if (FAILED(D3DCompile(hlsl, strlen(hlsl), 0, 0, 0, "vs", "vs_3_0", 0, 0, &vsb, &err))) { puts((char *) err->GetBufferPointer()); return 4; }
    if (FAILED(D3DCompile(hlsl, strlen(hlsl), 0, 0, 0, "ps", "ps_3_0", 0, 0, &psb, &err))) { puts((char *) err->GetBufferPointer()); return 4; }
    IDirect3DVertexShader9 *vs; IDirect3DPixelShader9 *ps;
    dev->CreateVertexShader((DWORD *) vsb->GetBufferPointer(), &vs); dev->CreatePixelShader((DWORD *) psb->GetBufferPointer(), &ps);
    D3DVERTEXELEMENT9 elements[] = {{0, 0, D3DDECLTYPE_FLOAT4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0}, D3DDECL_END()};
    IDirect3DVertexDeclaration9 *decl; dev->CreateVertexDeclaration(elements, &decl);
    float quad[4][4] = {{-1, -1, 0, 1}, {-1, 1, 0, 1}, {1, -1, 0, 1}, {1, 1, 0, 1}};

    using clk = std::chrono::steady_clock; auto start = clk::now(), tick = start; long frames = 0, lastFrames = 0; bool shot = false;
    MSG msg;
    for (;;) {
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) { if (msg.message == WM_QUIT) return 0; TranslateMessage(&msg); DispatchMessageW(&msg); }
        double t = std::chrono::duration<double>(clk::now() - start).count();
        if (t > seconds) break;
        float prm[4] = {(float) W, (float) H, (float) (freeze >= 0 ? freeze : t), 0};
        dev->BeginScene();
        dev->SetVertexDeclaration(decl); dev->SetVertexShader(vs); dev->SetPixelShader(ps); dev->SetPixelShaderConstantF(0, prm, 1);
        dev->DrawPrimitiveUP(D3DPT_TRIANGLESTRIP, 2, quad, sizeof(quad[0]));
        dev->EndScene();
        dev->Present(nullptr, nullptr, nullptr, nullptr); ++frames;
        if (!shot && t >= capture_at) { Sleep(150); capture(hwnd, W, H, out); shot = true; }
        double dt = std::chrono::duration<double>(clk::now() - tick).count();
        if (dt >= 2.0) { printf("t=%.1f  %.1f fps\n", t, (frames - lastFrames) / dt); fflush(stdout); lastFrames = frames; tick = clk::now(); }
    }
    return 0;
}
