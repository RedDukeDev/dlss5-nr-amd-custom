// A Vulkan test "game" that upscales through the NGX Vulkan API, as a DLSS game does, with OptiScaler
// answering it (loaded as dxgi.dll beside this exe). The scene is made on the CPU at render resolution
// with its depth and motion vectors, copied into images, upscaled by EvaluateFeature into an output
// image, and that is copied to the swapchain and presented.
// usage: vkngx [seconds=14] [capture_at=10] [out.bmp] [out_w=1920] [out_h=1080] [render_percent=58] [freeze_time]
#define VK_USE_PLATFORM_WIN32_KHR
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <dxgi.h>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_win32.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <vector>
#include <algorithm>

#include "nvsdk_ngx.h"
#include "nvsdk_ngx_vk.h"
#include "nvsdk_ngx_params.h"
#include "nvsdk_ngx_helpers_vk.h"

#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")

#define CHECK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { printf("%s failed: %d (line %d)\n", #x, (int) r_, __LINE__); exit(3); } } while (0)

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

static float hash2(float x, float y) { float s = sinf(x * 127.1f + y * 311.7f) * 43758.5453f; return s - floorf(s); }
static float noise2(float x, float y) {
    float ix = floorf(x), iy = floorf(y), fx = x - ix, fy = y - iy; fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy);
    float a = hash2(ix, iy), b = hash2(ix + 1, iy), c = hash2(ix, iy + 1), d = hash2(ix + 1, iy + 1);
    return (a + (b - a) * fx) + ((c + (d - c) * fx) - (a + (b - a) * fx)) * fy;
}
static float mixf(float a, float b, float t) { return a + (b - a) * t; }
static float clamp01(float x) { return x < 0 ? 0 : x > 1 ? 1 : x; }

// the scene, in RGB 0..1, with its depth; the ball is left out when asked (the background is made once)
static void shade(float u, float v, float aspect, float t, bool ball, float rgb[3], float *depth) {
    float qx = u * aspect, qy = v;
    float skyT = powf(v, 1.5f);
    float c[3] = {mixf(0.25f, 0.9f, skyT), mixf(0.45f, 0.75f, skyT), mixf(0.85f, 0.6f, skyT)};
    float h = 0.55f + 0.08f * noise2(qx * 6, qy * 6) + 0.05f * sinf(qx * 5);
    float d = 1.0f;
    if (v > h) {
        float n = noise2(qx * 40, qy * 40), k = 0.6f + 0.4f * noise2(qx * 160, qy * 160);
        c[0] = mixf(0.15f, 0.35f, n) * k; c[1] = mixf(0.3f, 0.28f, n) * k; c[2] = mixf(0.1f, 0.15f, n) * k; d = 0.2f + 0.7f * (1 - v);
    }
    float spx = 0.3f + 0.4f * ((t * 0.1f) - floorf(t * 0.1f)), spy = 0.62f;
    float dx = (u - spx) * aspect, dy = v - spy, r = sqrtf(dx * dx + dy * dy);
    if (ball && r < 0.1f) {
        float z = sqrtf(1 - (r / 0.1f) * (r / 0.1f)), nx = dx / 0.1f, ny = dy / 0.1f;
        float lx = -0.4f, ly = -0.6f, lz = 0.7f, ll = sqrtf(lx * lx + ly * ly + lz * lz);
        float l = clamp01((nx * lx + ny * ly + z * lz) / ll), sp = powf(l, 40) * 0.5f;
        c[0] = 0.8f * (0.15f + 0.85f * l) + sp; c[1] = 0.25f * (0.15f + 0.85f * l) + sp; c[2] = 0.2f * (0.15f + 0.85f * l) + sp; d = 0.15f;
    }
    if (v > 0.85f) {
        int ck = (int) floorf(qx * 24) + (int) floorf(qy * 12); float g = (ck & 1) ? 0.75f : 0.2f;
        for (int i = 0; i < 3; ++i) c[i] = mixf(c[i], g, 0.7f);
    }
    if (u < 0.18f && v < 0.06f) {
        float bar = (u * 40 - floorf(u * 40)) < 0.45f ? 1.f : 0.f, row = (v * 20 - floorf(v * 20)) > 0.01f ? 1.f : 0.f;
        for (int i = 0; i < 3; ++i) c[i] = (i == 2 ? 0.08f : 0.05f) + 0.9f * bar * row;
    }
    rgb[0] = c[0]; rgb[1] = c[1]; rgb[2] = c[2]; *depth = d;
}

static float halton(int i, int b) { float f = 1, r = 0; while (i > 0) { f /= b; r += f * (i % b); i /= b; } return r; }

// half precision of a float, enough for motion vectors
static uint16_t toHalf(float f) {
    uint32_t x; memcpy(&x, &f, 4); uint32_t sign = (x >> 16) & 0x8000; int exp = (int) ((x >> 23) & 0xFF) - 127 + 15; uint32_t man = x & 0x7FFFFF;
    if (exp <= 0) return (uint16_t) sign; if (exp >= 31) return (uint16_t) (sign | 0x7C00);
    return (uint16_t) (sign | (exp << 10) | (man >> 13));
}

struct Image { VkImage image; VkDeviceMemory memory; VkImageView view; VkFormat format; uint32_t w, h; };

int main(int argc, char **argv) {
    double seconds = argc > 1 ? atof(argv[1]) : 14, capture_at = argc > 2 ? atof(argv[2]) : 10;
    const char *out = argc > 3 ? argv[3] : "shot.bmp";
    int W = argc > 4 ? atoi(argv[4]) : 1920, H = argc > 5 ? atoi(argv[5]) : 1080;
    int percent = argc > 6 ? atoi(argv[6]) : 58;
    double freeze = argc > 7 ? atof(argv[7]) : -1;
    int RW = W * percent / 100, RH = H * percent / 100;

    IDXGIFactory1 *factory = nullptr; CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void **) &factory);

    WNDCLASSW wc = {}; wc.lpfnWndProc = proc; wc.hInstance = GetModuleHandleW(nullptr); wc.lpszClassName = L"vkngx"; wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    RegisterClassW(&wc);
    RECT rc = {0, 0, W, H}; AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);
    HWND hwnd = CreateWindowW(L"vkngx", L"vkngx", WS_OVERLAPPEDWINDOW | WS_VISIBLE, 20, 20, rc.right - rc.left, rc.bottom - rc.top, nullptr, nullptr, wc.hInstance, nullptr);

    HMODULE loader = LoadLibraryW(L"vulkan-1.dll");
    auto gipa = (PFN_vkGetInstanceProcAddr) GetProcAddress(loader, "vkGetInstanceProcAddr");
    auto vkCreateInstance_ = (PFN_vkCreateInstance) gipa(nullptr, "vkCreateInstance");
    const char *instExt[] = {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME};
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "vkngx", 1, "none", 1, VK_API_VERSION_1_2};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0, &app, 0, nullptr, 2, instExt};
    VkInstance inst; CHECK(vkCreateInstance_(&ici, nullptr, &inst));
#define IFN(name) auto name = (PFN_##name) gipa(inst, #name)
    IFN(vkEnumeratePhysicalDevices); IFN(vkGetPhysicalDeviceProperties); IFN(vkGetPhysicalDeviceQueueFamilyProperties);
    IFN(vkCreateWin32SurfaceKHR); IFN(vkGetPhysicalDeviceSurfaceSupportKHR); IFN(vkGetPhysicalDeviceSurfaceCapabilitiesKHR);
    IFN(vkCreateDevice); IFN(vkGetDeviceProcAddr); IFN(vkGetPhysicalDeviceMemoryProperties);
    VkSurfaceKHR surface; VkWin32SurfaceCreateInfoKHR sci = {VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR, nullptr, 0, wc.hInstance, hwnd};
    CHECK(vkCreateWin32SurfaceKHR(inst, &sci, nullptr, &surface));
    uint32_t n = 0; vkEnumeratePhysicalDevices(inst, &n, nullptr); std::vector<VkPhysicalDevice> pds(n); vkEnumeratePhysicalDevices(inst, &n, pds.data());
    VkPhysicalDevice pd = nullptr; uint32_t family = 0;
    for (auto p : pds) {
        VkPhysicalDeviceProperties pr; vkGetPhysicalDeviceProperties(p, &pr);
        if (pr.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) continue;
        uint32_t qn = 0; vkGetPhysicalDeviceQueueFamilyProperties(p, &qn, nullptr); std::vector<VkQueueFamilyProperties> qf(qn); vkGetPhysicalDeviceQueueFamilyProperties(p, &qn, qf.data());
        for (uint32_t i = 0; i < qn; ++i) { VkBool32 ok = 0; vkGetPhysicalDeviceSurfaceSupportKHR(p, i, surface, &ok); if (ok && (qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) { pd = p; family = i; break; } }
        if (pd) { printf("GPU: %s\n", pr.deviceName); break; }
    }
    if (!pd) { puts("no suitable GPU"); return 2; }
    float prio = 1.0f; VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, family, 1, &prio};
    const char *devExt[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, nullptr, 0, 1, &qci, 0, nullptr, 1, devExt, nullptr};
    VkDevice dev; CHECK(vkCreateDevice(pd, &dci, nullptr, &dev));
#define DFN(name) auto name = (PFN_##name) vkGetDeviceProcAddr(dev, #name)
    DFN(vkGetDeviceQueue); DFN(vkCreateSwapchainKHR); DFN(vkGetSwapchainImagesKHR); DFN(vkAcquireNextImageKHR); DFN(vkQueuePresentKHR);
    DFN(vkCreateCommandPool); DFN(vkAllocateCommandBuffers); DFN(vkBeginCommandBuffer); DFN(vkEndCommandBuffer); DFN(vkResetCommandBuffer);
    DFN(vkCmdPipelineBarrier); DFN(vkCmdCopyBufferToImage); DFN(vkCmdCopyImage); DFN(vkQueueSubmit); DFN(vkCreateSemaphore); DFN(vkCreateFence);
    DFN(vkWaitForFences); DFN(vkResetFences); DFN(vkCreateBuffer); DFN(vkGetBufferMemoryRequirements); DFN(vkAllocateMemory);
    DFN(vkBindBufferMemory); DFN(vkMapMemory); DFN(vkDeviceWaitIdle); DFN(vkCreateImage); DFN(vkGetImageMemoryRequirements);
    DFN(vkBindImageMemory); DFN(vkCreateImageView);
    VkQueue queue; vkGetDeviceQueue(dev, family, 0, &queue);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    auto findType = [&](uint32_t bits, VkMemoryPropertyFlags flags) { for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & flags) == flags) return i; return 0xFFFFFFFFu; };

    // swapchain: RGBA8, the same format as the upscaler's output
    VkSurfaceCapabilitiesKHR caps; vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pd, surface, &caps);
    VkSwapchainCreateInfoKHR sc = {VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    sc.surface = surface; sc.minImageCount = std::max(3u, caps.minImageCount); sc.imageFormat = VK_FORMAT_R8G8B8A8_UNORM; sc.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    sc.imageExtent = {(uint32_t) W, (uint32_t) H}; sc.imageArrayLayers = 1; sc.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    sc.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE; sc.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR; sc.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sc.presentMode = getenv("NOVSYNC") ? VK_PRESENT_MODE_IMMEDIATE_KHR : VK_PRESENT_MODE_FIFO_KHR; sc.clipped = VK_TRUE;
    VkSwapchainKHR swap; CHECK(vkCreateSwapchainKHR(dev, &sc, nullptr, &swap));
    uint32_t count = 0; vkGetSwapchainImagesKHR(dev, swap, &count, nullptr); std::vector<VkImage> images(count); vkGetSwapchainImagesKHR(dev, swap, &count, images.data());

    auto makeImage = [&](VkFormat format, uint32_t w, uint32_t h, VkImageUsageFlags usage) {
        Image im = {}; im.format = format; im.w = w; im.h = h;
        VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, nullptr, 0, VK_IMAGE_TYPE_2D, format, {w, h, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL, usage, VK_SHARING_MODE_EXCLUSIVE, 0, nullptr, VK_IMAGE_LAYOUT_UNDEFINED};
        CHECK(vkCreateImage(dev, &ci, nullptr, &im.image));
        VkMemoryRequirements mr; vkGetImageMemoryRequirements(dev, im.image, &mr);
        VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, mr.size, findType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
        CHECK(vkAllocateMemory(dev, &mai, nullptr, &im.memory)); CHECK(vkBindImageMemory(dev, im.image, im.memory, 0));
        VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, nullptr, 0, im.image, VK_IMAGE_VIEW_TYPE_2D, format, {}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        CHECK(vkCreateImageView(dev, &vi, nullptr, &im.view));
        return im;
    };
    const VkImageUsageFlags inUsage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    Image color = makeImage(VK_FORMAT_R8G8B8A8_UNORM, RW, RH, inUsage);
    Image depth = makeImage(VK_FORMAT_R32_SFLOAT, RW, RH, inUsage);
    Image motion = makeImage(VK_FORMAT_R16G16_SFLOAT, RW, RH, inUsage);
    Image output = makeImage(VK_FORMAT_R8G8B8A8_UNORM, W, H, inUsage);

    auto makeStaging = [&](VkDeviceSize bytes, uint8_t **mapped) {
        VkBuffer b; VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE};
        CHECK(vkCreateBuffer(dev, &bci, nullptr, &b)); VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, b, &mr);
        VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, mr.size, findType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
        VkDeviceMemory m; CHECK(vkAllocateMemory(dev, &mai, nullptr, &m)); CHECK(vkBindBufferMemory(dev, b, m, 0)); CHECK(vkMapMemory(dev, m, 0, bytes, 0, (void **) mapped));
        return b;
    };
    uint8_t *colorPix, *depthPix, *motionPix;
    VkBuffer colorBuf = makeStaging((VkDeviceSize) RW * RH * 4, &colorPix), depthBuf = makeStaging((VkDeviceSize) RW * RH * 4, &depthPix), motionBuf = makeStaging((VkDeviceSize) RW * RH * 4, &motionPix);

    // the static background at render resolution (and its depth), drawn once; the ball is drawn over it each frame
    float aspect = (float) RW / RH;
    std::vector<uint8_t> bgColor((size_t) RW * RH * 4); std::vector<float> bgDepth((size_t) RW * RH);
    auto toByte = [](float x) { return (uint8_t) (clamp01(x) * 255.0f + 0.5f); };
    for (int y = 0; y < RH; ++y)
        for (int x = 0; x < RW; ++x) {
            float rgb[3], d; shade((x + 0.5f) / RW, (y + 0.5f) / RH, aspect, 0.0f, false, rgb, &d);
            uint8_t *p = &bgColor[((size_t) y * RW + x) * 4]; p[0] = toByte(rgb[0]); p[1] = toByte(rgb[1]); p[2] = toByte(rgb[2]); p[3] = 255; bgDepth[(size_t) y * RW + x] = d;
        }

    // NGX, through the module that answers it: OptiScaler, loaded as dxgi.dll
    HMODULE ngx = GetModuleHandleW(L"dxgi.dll");
    typedef NVSDK_NGX_Result (*PFN_Init)(unsigned long long, const wchar_t *, VkInstance, VkPhysicalDevice, VkDevice, PFN_vkGetInstanceProcAddr, PFN_vkGetDeviceProcAddr, const NVSDK_NGX_FeatureCommonInfo *, NVSDK_NGX_Version);
    typedef NVSDK_NGX_Result (*PFN_Caps)(NVSDK_NGX_Parameter **);
    typedef NVSDK_NGX_Result (*PFN_Create)(VkDevice, VkCommandBuffer, NVSDK_NGX_Feature, NVSDK_NGX_Parameter *, NVSDK_NGX_Handle **);
    typedef NVSDK_NGX_Result (*PFN_Eval)(VkCommandBuffer, const NVSDK_NGX_Handle *, NVSDK_NGX_Parameter *, PFN_NVSDK_NGX_ProgressCallback);
    auto Init = (PFN_Init) GetProcAddress(ngx, "NVSDK_NGX_VULKAN_Init");
    auto Caps = (PFN_Caps) GetProcAddress(ngx, "NVSDK_NGX_VULKAN_GetCapabilityParameters");
    auto Create = (PFN_Create) GetProcAddress(ngx, "NVSDK_NGX_VULKAN_CreateFeature1");
    auto Eval = (PFN_Eval) GetProcAddress(ngx, "NVSDK_NGX_VULKAN_EvaluateFeature");
    if (!Init || !Caps || !Create || !Eval) { puts("the NGX Vulkan entry points are not exported by dxgi.dll"); return 5; }
    NVSDK_NGX_Result r = Init(0x1234, L".", inst, pd, dev, gipa, vkGetDeviceProcAddr, nullptr, NVSDK_NGX_Version_API);
    printf("NGX init: %x\n", (unsigned) r); fflush(stdout);
    NVSDK_NGX_Parameter *params = nullptr; r = Caps(&params);
    printf("NGX caps: %x\n", (unsigned) r); fflush(stdout);
    params->Set(NVSDK_NGX_Parameter_Width, (unsigned) RW); params->Set(NVSDK_NGX_Parameter_Height, (unsigned) RH);
    params->Set(NVSDK_NGX_Parameter_OutWidth, (unsigned) W); params->Set(NVSDK_NGX_Parameter_OutHeight, (unsigned) H);
    params->Set(NVSDK_NGX_Parameter_PerfQualityValue, (int) NVSDK_NGX_PerfQuality_Value_Balanced);
    params->Set(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags, (int) 0);

    VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, family};
    VkCommandPool pool; CHECK(vkCreateCommandPool(dev, &cpi, nullptr, &pool));
    VkCommandBufferAllocateInfo cbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 2};
    VkCommandBuffer cmds[2]; CHECK(vkAllocateCommandBuffers(dev, &cbi, cmds));
    VkSemaphoreCreateInfo semi = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO}; VkFenceCreateInfo fi = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, VK_FENCE_CREATE_SIGNALED_BIT};
    VkSemaphore acquired, rendered; VkFence frameFence; CHECK(vkCreateSemaphore(dev, &semi, nullptr, &acquired)); CHECK(vkCreateSemaphore(dev, &semi, nullptr, &rendered)); CHECK(vkCreateFence(dev, &fi, nullptr, &frameFence));

    auto barrier = [&](VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to, VkAccessFlags sa, VkAccessFlags da) {
        VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, sa, da, from, to, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    };
    const VkAccessFlags any = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    NVSDK_NGX_Handle *handle = nullptr; bool created = false;

    using clk = std::chrono::steady_clock; auto start = clk::now(), tick = start; long frames = 0, lastFrames = 0; bool shot = false;
    MSG msg; int index = 0;
    for (;;) {
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) { if (msg.message == WM_QUIT) return 0; TranslateMessage(&msg); DispatchMessageW(&msg); }
        double t = std::chrono::duration<double>(clk::now() - start).count();
        if (t > seconds) break;
        CHECK(vkWaitForFences(dev, 1, &frameFence, VK_TRUE, UINT64_MAX)); CHECK(vkResetFences(dev, 1, &frameFence));

        // this frame: background, ball, depth and motion at render resolution
        double st = freeze >= 0 ? freeze : t;
        float jx = halton(index % 16 + 1, 2) - 0.5f, jy = halton(index % 16 + 1, 3) - 0.5f; ++index;
        memcpy(colorPix, bgColor.data(), (size_t) RW * RH * 4); memcpy(depthPix, bgDepth.data(), (size_t) RW * RH * 4); memset(motionPix, 0, (size_t) RW * RH * 4);
        {
            float spx = 0.3f + 0.4f * (float) (st * 0.1 - floor(st * 0.1)), spy = 0.62f;
            int x0 = std::max(0, (int) ((spx - 0.1f / aspect) * RW) - 2), x1 = std::min(RW - 1, (int) ((spx + 0.1f / aspect) * RW) + 2);
            int y0 = std::max(0, (int) ((spy - 0.1f) * RH) - 2), y1 = std::min(RH - 1, (int) ((spy + 0.1f) * RH) + 2);
            // the ball's motion in pixels a frame, towards where it was
            uint16_t mvx = toHalf(freeze >= 0 ? 0.f : -0.04f * RW / 60.0f), mvy = 0;
            for (int y = y0; y <= y1; ++y)
                for (int x = x0; x <= x1; ++x) {
                    float rgb[3], d; shade((x + 0.5f) / RW, (y + 0.5f) / RH, aspect, (float) st, true, rgb, &d);
                    if (d != 0.15f) continue;
                    uint8_t *p = colorPix + ((size_t) y * RW + x) * 4; p[0] = toByte(rgb[0]); p[1] = toByte(rgb[1]); p[2] = toByte(rgb[2]); p[3] = 255;
                    ((float *) depthPix)[(size_t) y * RW + x] = d;
                    uint16_t *mv = (uint16_t *) (motionPix + ((size_t) y * RW + x) * 4); mv[0] = mvx; mv[1] = mvy;
                }
        }

        uint32_t imageIndex = 0; { VkResult ar = vkAcquireNextImageKHR(dev, swap, UINT64_MAX, acquired, VK_NULL_HANDLE, &imageIndex); if (ar != VK_SUCCESS && ar != VK_SUBOPTIMAL_KHR) { printf("acquire failed: %d\n", (int) ar); return 3; } }
        VkCommandBuffer cmd = cmds[0];
        vkResetCommandBuffer(cmd, 0);
        VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
        vkBeginCommandBuffer(cmd, &bi);
        Image *ins[3] = {&color, &depth, &motion}; VkBuffer bufs[3] = {colorBuf, depthBuf, motionBuf};
        for (int i = 0; i < 3; ++i) {
            barrier(cmd, ins[i]->image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
            VkBufferImageCopy copy = {0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {(uint32_t) RW, (uint32_t) RH, 1}};
            vkCmdCopyBufferToImage(cmd, bufs[i], ins[i]->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
            barrier(cmd, ins[i]->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT, any);
        }
        barrier(cmd, output.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0, any);

        VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        NVSDK_NGX_Resource_VK rColor = NVSDK_NGX_Create_ImageView_Resource_VK(color.view, color.image, range, color.format, RW, RH, false);
        NVSDK_NGX_Resource_VK rDepth = NVSDK_NGX_Create_ImageView_Resource_VK(depth.view, depth.image, range, depth.format, RW, RH, false);
        NVSDK_NGX_Resource_VK rMotion = NVSDK_NGX_Create_ImageView_Resource_VK(motion.view, motion.image, range, motion.format, RW, RH, false);
        NVSDK_NGX_Resource_VK rOutput = NVSDK_NGX_Create_ImageView_Resource_VK(output.view, output.image, range, output.format, W, H, true);
        if (!created) {
            r = Create(dev, cmd, NVSDK_NGX_Feature_SuperSampling, params, &handle); created = true;
            printf("NGX create: %x\n", (unsigned) r); fflush(stdout);
            if (!handle) return 6;
        }
        params->Set(NVSDK_NGX_Parameter_Color, &rColor); params->Set(NVSDK_NGX_Parameter_Output, &rOutput);
        params->Set(NVSDK_NGX_Parameter_Depth, &rDepth); params->Set(NVSDK_NGX_Parameter_MotionVectors, &rMotion);
        params->Set(NVSDK_NGX_Parameter_Jitter_Offset_X, jx); params->Set(NVSDK_NGX_Parameter_Jitter_Offset_Y, jy);
        params->Set(NVSDK_NGX_Parameter_Sharpness, 0.0f); params->Set(NVSDK_NGX_Parameter_Reset, frames == 0 ? 1 : 0);
        params->Set(NVSDK_NGX_Parameter_MV_Scale_X, 1.0f); params->Set(NVSDK_NGX_Parameter_MV_Scale_Y, 1.0f);
        params->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width, (unsigned) RW); params->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height, (unsigned) RH);
        Eval(cmd, handle, params, nullptr);

        // the output to the swapchain image
        barrier(cmd, output.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, any, VK_ACCESS_TRANSFER_READ_BIT);
        barrier(cmd, images[imageIndex], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
        VkImageCopy ic = {{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {(uint32_t) W, (uint32_t) H, 1}};
        vkCmdCopyImage(cmd, output.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, images[imageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &ic);
        barrier(cmd, images[imageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_TRANSFER_WRITE_BIT, 0);
        vkEndCommandBuffer(cmd);
        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 1, &acquired, &waitStage, 1, &cmd, 1, &rendered};
        CHECK(vkQueueSubmit(queue, 1, &si, frameFence));
        VkPresentInfoKHR pi = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, nullptr, 1, &rendered, 1, &swap, &imageIndex};
        VkResult pr = vkQueuePresentKHR(queue, &pi);
        if (pr != VK_SUCCESS && pr != VK_SUBOPTIMAL_KHR) { printf("present failed: %d\n", (int) pr); return 4; }
        ++frames;
        if (!shot && t >= capture_at) { Sleep(150); capture(hwnd, W, H, out); shot = true; }
        double dt = std::chrono::duration<double>(clk::now() - tick).count();
        if (dt >= 2.0) { printf("t=%.1f  %.1f fps\n", t, (frames - lastFrames) / dt); fflush(stdout); lastFrames = frames; tick = clk::now(); }
    }
    vkDeviceWaitIdle(dev);
    return 0;
}
