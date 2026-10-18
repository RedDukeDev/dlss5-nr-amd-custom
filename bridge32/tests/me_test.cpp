// Offline test of the motion search: two pictures (32-bit BMPs, as the client's debug dump writes them)
// go through client/motion.cpp on a real Direct3D 9 device, and what it finds is compared with the
// motion the pictures are known to have.
//
//   me_test now.bmp old.bmp expected_dx expected_dy [field.bin]
//
// "now" is the picture the game shows, "old" the one the network saw; the motion of a block is where in
// the old picture it was, so a scene that moved left by 60 pixels has an expected_dx of +60.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d9.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <vector>
#include "../client/motion.h"

namespace {

bool load_bmp(const char *path, UINT &w, UINT &h, std::vector<uint8_t> &pixels) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    BITMAPFILEHEADER fh;
    BITMAPINFOHEADER ih;
    if (fread(&fh, sizeof fh, 1, f) != 1 || fread(&ih, sizeof ih, 1, f) != 1 || ih.biBitCount != 32) {
        fclose(f);
        return false;
    }
    w = (UINT)ih.biWidth;
    h = (UINT)abs(ih.biHeight);
    pixels.resize((size_t)w * h * 4);
    fseek(f, (long)fh.bfOffBits, SEEK_SET);
    const bool ok = fread(pixels.data(), 1, pixels.size(), f) == pixels.size();
    fclose(f);
    if (ih.biHeight > 0) {   // bottom-up: flip
        std::vector<uint8_t> flipped(pixels.size());
        for (UINT y = 0; y < h; ++y) memcpy(&flipped[(size_t)y * w * 4], &pixels[(size_t)(h - 1 - y) * w * 4], (size_t)w * 4);
        pixels.swap(flipped);
    }
    return ok;
}

float half_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    const int e = (h >> 10) & 31;
    const uint32_t m = h & 0x3FF;
    if (e == 0) {
        const float v = m / 16777216.0f;
        return (h & 0x8000) ? -v : v;
    }
    const uint32_t x = e == 31 ? (sign | 0x7F800000 | (m << 13)) : (sign | ((uint32_t)(e - 15 + 127) << 23) | (m << 13));
    float f;
    memcpy(&f, &x, 4);
    return f;
}

IDirect3DTexture9 *make_texture(IDirect3DDevice9 *device, UINT w, UINT h, const std::vector<uint8_t> &pixels) {
    IDirect3DTexture9 *tex = nullptr;
    if (FAILED(device->CreateTexture(w, h, 1, D3DUSAGE_DYNAMIC, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT, &tex, nullptr))) return nullptr;
    D3DLOCKED_RECT rect;
    if (FAILED(tex->LockRect(0, &rect, nullptr, D3DLOCK_DISCARD))) return nullptr;
    for (UINT y = 0; y < h; ++y) memcpy((uint8_t *)rect.pBits + (size_t)y * rect.Pitch, &pixels[(size_t)y * w * 4], (size_t)w * 4);
    tex->UnlockRect(0);
    return tex;
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: me_test now.bmp old.bmp expected_dx expected_dy [field.bin]\n");
        return 2;
    }
    UINT w, h, w2, h2;
    std::vector<uint8_t> now_pixels, old_pixels;
    if (!load_bmp(argv[1], w, h, now_pixels) || !load_bmp(argv[2], w2, h2, old_pixels) || w != w2 || h != h2) {
        fprintf(stderr, "the pictures could not be read, or differ in size\n");
        return 1;
    }
    const float expected_dx = (float)atof(argv[3]), expected_dy = (float)atof(argv[4]);

    WNDCLASSA wc = {};
    wc.lpfnWndProc = DefWindowProcA;
    wc.hInstance = GetModuleHandleA(nullptr);
    wc.lpszClassName = "me_test";
    RegisterClassA(&wc);
    HWND window = CreateWindowA("me_test", "me_test", WS_OVERLAPPEDWINDOW, 0, 0, 64, 64, nullptr, nullptr, wc.hInstance, nullptr);
    IDirect3D9 *d3d = Direct3DCreate9(D3D_SDK_VERSION);
    D3DPRESENT_PARAMETERS pp = {};
    pp.Windowed = TRUE;
    pp.SwapEffect = D3DSWAPEFFECT_DISCARD;
    pp.BackBufferFormat = D3DFMT_X8R8G8B8;
    pp.BackBufferWidth = 64;
    pp.BackBufferHeight = 64;
    pp.hDeviceWindow = window;
    IDirect3DDevice9 *device = nullptr;
    if (!d3d || FAILED(d3d->CreateDevice(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, window, D3DCREATE_HARDWARE_VERTEXPROCESSING, &pp, &device))) {
        fprintf(stderr, "no Direct3D 9 device\n");
        return 1;
    }
    IDirect3DTexture9 *now = make_texture(device, w, h, now_pixels), *old = make_texture(device, w, h, old_pixels);
    Motion motion;
    if (!now || !old || !motion.create(device, w, h)) {
        fprintf(stderr, "the motion search is not available on this device\n");
        return 1;
    }
    if (const char *v = getenv("ME_COARSE_PENALTY")) motion.coarse_penalty = (float)atof(v);
    if (const char *v = getenv("ME_FINE_PENALTY")) motion.fine_penalty = (float)atof(v);
    if (const char *v = getenv("ME_MEDIAN")) motion.median = atoi(v) != 0;
    motion.update(device, now, old, true);

    IDirect3DSurface9 *system = nullptr;
    device->CreateOffscreenPlainSurface(motion.grid_width(), motion.grid_height(), D3DFMT_A16B16G16R16F, D3DPOOL_SYSTEMMEM, &system, nullptr);
    if (!system || FAILED(device->GetRenderTargetData(motion.field_surface(), system))) {
        fprintf(stderr, "could not read the field back\n");
        return 1;
    }
    D3DLOCKED_RECT rect;
    system->LockRect(&rect, nullptr, D3DLOCK_READONLY);
    const UINT gw = motion.grid_width(), gh = motion.grid_height();
    std::vector<float> field((size_t)gw * gh * 3);
    for (UINT y = 0; y < gh; ++y) {
        const uint16_t *row = (const uint16_t *)((const uint8_t *)rect.pBits + (size_t)y * rect.Pitch);
        for (UINT x = 0; x < gw; ++x)
            for (int c = 0; c < 3; ++c) field[((size_t)y * gw + x) * 3 + c] = half_to_float(row[4 * x + c]);
    }
    system->UnlockRect();

    // Blocks that have texture to be matched on: the picture's own variation inside the block, in the
    // new picture. A flat block has no motion to be found, whatever it is given.
    std::vector<char> textured((size_t)gw * gh, 0);
    for (UINT by = 0; by < gh; ++by)
        for (UINT bx = 0; bx < gw; ++bx) {
            float lo = 255.0f, hi = 0.0f;
            for (UINT y = by * 16; y < std::min(h, by * 16 + 16); ++y)
                for (UINT x = bx * 16; x < std::min(w, bx * 16 + 16); ++x) {
                    const uint8_t *p = &now_pixels[((size_t)y * w + x) * 4];
                    const float l = 0.299f * p[2] + 0.587f * p[1] + 0.114f * p[0];
                    lo = std::min(lo, l);
                    hi = std::max(hi, l);
                }
            textured[(size_t)by * gw + bx] = hi - lo > 12.0f;
        }

    // The rows of blocks the figures are about (ME_ROWS=first-last): a part of the picture whose motion is
    // known and which is not periodic or flat.
    UINT row_first = 0, row_last = gh - 1;
    if (const char *v = getenv("ME_ROWS")) sscanf(v, "%u-%u", &row_first, &row_last);
    unsigned counted = 0, right = 0, near_right = 0, zero = 0;
    std::vector<float> xs;
    for (UINT i = 0; i < gw * gh; ++i) {
        if (!textured[i] || i / gw < row_first || i / gw > row_last) continue;
        ++counted;
        const float dx = field[i * 3], dy = field[i * 3 + 1];
        xs.push_back(dx);
        if (fabsf(dx - expected_dx) <= 2.0f && fabsf(dy - expected_dy) <= 2.0f) ++right;
        else if (fabsf(dx - expected_dx) <= 8.0f && fabsf(dy - expected_dy) <= 8.0f) ++near_right;
        if (fabsf(dx) <= 1.0f && fabsf(dy) <= 1.0f) ++zero;
    }
    std::sort(xs.begin(), xs.end());
    printf("%ux%u, grid %ux%u; %u textured blocks of %u\n", w, h, gw, gh, counted, gw * gh);
    printf("expected %.0f,%.0f px: within 2 px %.1f%%, within 8 px %.1f%%, at zero %.1f%%; median x %.1f\n", expected_dx, expected_dy,
           counted ? 100.0 * right / counted : 0.0, counted ? 100.0 * (right + near_right) / counted : 0.0,
           counted ? 100.0 * zero / counted : 0.0, xs.empty() ? 0.0f : xs[xs.size() / 2]);
    if (argc > 5) {
        FILE *f = fopen(argv[5], "wb");
        const uint32_t dims[2] = {gw, gh};
        fwrite(dims, 4, 2, f);
        fwrite(field.data(), 4, field.size(), f);
        fclose(f);
    }
    return 0;
}
