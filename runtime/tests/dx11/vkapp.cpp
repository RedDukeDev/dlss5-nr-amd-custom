// A Vulkan test "game": a window, a swapchain, and a scene drawn on the CPU and uploaded by copy
// (so it needs no shaders), presented every frame. OptiScaler, loaded as dxgi.dll beside the exe,
// hooks the Vulkan loader this program loads, as it does a game's. The window is captured to a BMP at a
// chosen time, so what OptiScaler did to the swapchain image is in it.
// usage: vkapp [seconds=12] [capture_at=8] [out.bmp] [width=1920] [height=1080] [freeze_time] [format: bgra|rgba]
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

// the scene, in RGB 0..1: the same one the Direct3D 11 test programs draw
static bool g_noBall = false;
static void shade(float u, float v, float aspect, float t, float rgb[3]) {
    float qx = u * aspect, qy = v;
    float skyT = powf(v, 1.5f);
    float c[3] = {mixf(0.25f, 0.9f, skyT), mixf(0.45f, 0.75f, skyT), mixf(0.85f, 0.6f, skyT)};
    float h = 0.55f + 0.08f * noise2(qx * 6, qy * 6) + 0.05f * sinf(qx * 5);
    if (v > h) {
        float n = noise2(qx * 40, qy * 40), k = 0.6f + 0.4f * noise2(qx * 160, qy * 160);
        c[0] = mixf(0.15f, 0.35f, n) * k; c[1] = mixf(0.3f, 0.28f, n) * k; c[2] = mixf(0.1f, 0.15f, n) * k;
    }
    float spx = 0.3f + 0.4f * ((t * 0.1f) - floorf(t * 0.1f)), spy = 0.62f;
    float dx = (u - spx) * aspect, dy = v - spy, r = sqrtf(dx * dx + dy * dy);
    if (!g_noBall && r < 0.1f) {
        float z = sqrtf(1 - (r / 0.1f) * (r / 0.1f)), nx = dx / 0.1f, ny = dy / 0.1f;
        float lx = -0.4f, ly = -0.6f, lz = 0.7f, ll = sqrtf(lx * lx + ly * ly + lz * lz);
        float l = clamp01((nx * lx + ny * ly + z * lz) / ll), sp = powf(l, 40) * 0.5f;
        c[0] = 0.8f * (0.15f + 0.85f * l) + sp; c[1] = 0.25f * (0.15f + 0.85f * l) + sp; c[2] = 0.2f * (0.15f + 0.85f * l) + sp;
    }
    if (v > 0.85f) {
        int ck = (int) floorf(qx * 24) + (int) floorf(qy * 12); float g = (ck & 1) ? 0.75f : 0.2f;
        for (int i = 0; i < 3; ++i) c[i] = mixf(c[i], g, 0.7f);
    }
    if (u < 0.18f && v < 0.06f) {
        float bar = (u * 40 - floorf(u * 40)) < 0.45f ? 1.f : 0.f, row = (v * 20 - floorf(v * 20)) > 0.01f ? 1.f : 0.f;
        for (int i = 0; i < 3; ++i) c[i] = (i == 2 ? 0.08f : 0.05f) + 0.9f * bar * row;
    }
    rgb[0] = c[0]; rgb[1] = c[1]; rgb[2] = c[2];
}

int main(int argc, char **argv) {
    double seconds = argc > 1 ? atof(argv[1]) : 12, capture_at = argc > 2 ? atof(argv[2]) : 8;
    const char *out = argc > 3 ? argv[3] : "shot.bmp";
    int W = argc > 4 ? atoi(argv[4]) : 1920, H = argc > 5 ? atoi(argv[5]) : 1080;
    double freeze = argc > 6 ? atof(argv[6]) : -1;
    bool rgbaFormat = argc > 7 && !strcmp(argv[7], "rgba");

    // load the proxy: a game imports dxgi.dll, this makes the program do as well
    IDXGIFactory1 *factory = nullptr; CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void **) &factory);

    WNDCLASSW wc = {}; wc.lpfnWndProc = proc; wc.hInstance = GetModuleHandleW(nullptr); wc.lpszClassName = L"vkapp"; wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    RegisterClassW(&wc);
    RECT rc = {0, 0, W, H}; AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);
    HWND hwnd = CreateWindowW(L"vkapp", L"vkapp", WS_OVERLAPPEDWINDOW | WS_VISIBLE, 20, 20, rc.right - rc.left, rc.bottom - rc.top, nullptr, nullptr, wc.hInstance, nullptr);

    HMODULE loader = LoadLibraryW(L"vulkan-1.dll");
    auto gipa = (PFN_vkGetInstanceProcAddr) GetProcAddress(loader, "vkGetInstanceProcAddr");
    auto vkCreateInstance_ = (PFN_vkCreateInstance) gipa(nullptr, "vkCreateInstance");

    const char *instExt[] = {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME};
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "vkapp", 1, "none", 1, VK_API_VERSION_1_2};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0, &app, 0, nullptr, 2, instExt};
    VkInstance inst; CHECK(vkCreateInstance_(&ici, nullptr, &inst));
#define IFN(name) auto name = (PFN_##name) gipa(inst, #name)
    IFN(vkEnumeratePhysicalDevices); IFN(vkGetPhysicalDeviceProperties); IFN(vkGetPhysicalDeviceQueueFamilyProperties);
    IFN(vkCreateWin32SurfaceKHR); IFN(vkGetPhysicalDeviceSurfaceSupportKHR); IFN(vkGetPhysicalDeviceSurfaceCapabilitiesKHR);
    IFN(vkGetPhysicalDeviceSurfaceFormatsKHR); IFN(vkCreateDevice); IFN(vkGetDeviceProcAddr); IFN(vkGetPhysicalDeviceMemoryProperties);
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
    DFN(vkCmdPipelineBarrier); DFN(vkCmdCopyBufferToImage); DFN(vkQueueSubmit); DFN(vkCreateSemaphore); DFN(vkCreateFence);
    DFN(vkWaitForFences); DFN(vkResetFences); DFN(vkCreateBuffer); DFN(vkGetBufferMemoryRequirements); DFN(vkAllocateMemory);
    DFN(vkBindBufferMemory); DFN(vkMapMemory); DFN(vkDeviceWaitIdle);
    VkQueue queue; vkGetDeviceQueue(dev, family, 0, &queue);

    VkSurfaceCapabilitiesKHR caps; vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pd, surface, &caps);
    VkFormat want = rgbaFormat ? VK_FORMAT_R8G8B8A8_UNORM : VK_FORMAT_B8G8R8A8_UNORM;
    VkSwapchainCreateInfoKHR sc = {VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    sc.surface = surface; sc.minImageCount = std::max(3u, caps.minImageCount); sc.imageFormat = want; sc.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    sc.imageExtent = {(uint32_t) W, (uint32_t) H}; sc.imageArrayLayers = 1; sc.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    sc.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE; sc.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR; sc.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sc.presentMode = getenv("NOVSYNC") ? VK_PRESENT_MODE_IMMEDIATE_KHR : VK_PRESENT_MODE_FIFO_KHR; sc.clipped = VK_TRUE;
    VkSwapchainKHR swap; CHECK(vkCreateSwapchainKHR(dev, &sc, nullptr, &swap));
    uint32_t count = 0; vkGetSwapchainImagesKHR(dev, swap, &count, nullptr); std::vector<VkImage> images(count); vkGetSwapchainImagesKHR(dev, swap, &count, images.data());

    // staging buffer, host visible and coherent, holding one frame
    VkDeviceSize bytes = (VkDeviceSize) W * H * 4;
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE};
    VkBuffer staging; CHECK(vkCreateBuffer(dev, &bci, nullptr, &staging));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, staging, &mr);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    uint32_t type = 0xFFFFFFFF;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((mr.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) == (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) { type = i; break; }
    VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, mr.size, type};
    VkDeviceMemory mem; CHECK(vkAllocateMemory(dev, &mai, nullptr, &mem)); CHECK(vkBindBufferMemory(dev, staging, mem, 0));
    uint8_t *pixels; CHECK(vkMapMemory(dev, mem, 0, bytes, 0, (void **) &pixels));

    // the static background, drawn once; each frame copies it and draws the moving ball over it
    std::vector<uint8_t> background((size_t) W * H * 4);
    float aspect = (float) W / H;
    g_noBall = true;
    auto toByte = [](float x) { return (uint8_t) (clamp01(x) * 255.0f + 0.5f); };
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float rgb[3]; shade((x + 0.5f) / W, (y + 0.5f) / H, aspect, 0.0f, rgb);
            uint8_t *p = &background[((size_t) y * W + x) * 4];
            if (rgbaFormat) { p[0] = toByte(rgb[0]); p[1] = toByte(rgb[1]); p[2] = toByte(rgb[2]); } else { p[0] = toByte(rgb[2]); p[1] = toByte(rgb[1]); p[2] = toByte(rgb[0]); }
            p[3] = 255;
        }

    g_noBall = false;
    VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, family};
    VkCommandPool pool; CHECK(vkCreateCommandPool(dev, &cpi, nullptr, &pool));
    VkCommandBufferAllocateInfo cbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VkCommandBuffer cmd; CHECK(vkAllocateCommandBuffers(dev, &cbi, &cmd));
    VkSemaphoreCreateInfo semi = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO}; VkFenceCreateInfo fi = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, VK_FENCE_CREATE_SIGNALED_BIT};
    VkSemaphore acquired, rendered; VkFence frameFence; CHECK(vkCreateSemaphore(dev, &semi, nullptr, &acquired)); CHECK(vkCreateSemaphore(dev, &semi, nullptr, &rendered)); CHECK(vkCreateFence(dev, &fi, nullptr, &frameFence));
    std::vector<bool> initialised(count, false);

    using clk = std::chrono::steady_clock; auto start = clk::now(), tick = start; long frames = 0, lastFrames = 0; bool shot = false;
    MSG msg;
    for (;;) {
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) { if (msg.message == WM_QUIT) return 0; TranslateMessage(&msg); DispatchMessageW(&msg); }
        double t = std::chrono::duration<double>(clk::now() - start).count();
        if (t > seconds) break;

        CHECK(vkWaitForFences(dev, 1, &frameFence, VK_TRUE, UINT64_MAX)); CHECK(vkResetFences(dev, 1, &frameFence));
        // this frame's picture: the background and the ball
        memcpy(pixels, background.data(), (size_t) W * H * 4);
        double st = freeze >= 0 ? freeze : t;
        {
            float spx = 0.3f + 0.4f * (float) (st * 0.1 - floor(st * 0.1)), spy = 0.62f;
            int x0 = std::max(0, (int) ((spx - 0.1f / aspect) * W) - 2), x1 = std::min(W - 1, (int) ((spx + 0.1f / aspect) * W) + 2);
            int y0 = std::max(0, (int) ((spy - 0.1f) * H) - 2), y1 = std::min(H - 1, (int) ((spy + 0.1f) * H) + 2);
            for (int y = y0; y <= y1; ++y)
                for (int x = x0; x <= x1; ++x) {
                    float rgb[3]; shade((x + 0.5f) / W, (y + 0.5f) / H, aspect, (float) st, rgb);
                    uint8_t *p = pixels + ((size_t) y * W + x) * 4;
                    if (rgbaFormat) { p[0] = toByte(rgb[0]); p[1] = toByte(rgb[1]); p[2] = toByte(rgb[2]); } else { p[0] = toByte(rgb[2]); p[1] = toByte(rgb[1]); p[2] = toByte(rgb[0]); }
                    p[3] = 255;
                }
        }

        uint32_t index = 0; { VkResult ar = vkAcquireNextImageKHR(dev, swap, UINT64_MAX, acquired, VK_NULL_HANDLE, &index); if (ar != VK_SUCCESS && ar != VK_SUBOPTIMAL_KHR) { printf("acquire failed: %d\n", (int) ar); return 3; } }
        vkResetCommandBuffer(cmd, 0);
        VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
        vkBeginCommandBuffer(cmd, &bi);
        VkImageMemoryBarrier toDst = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, images[index], {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toDst);
        VkBufferImageCopy copy = {0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {(uint32_t) W, (uint32_t) H, 1}};
        vkCmdCopyBufferToImage(cmd, staging, images[index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        VkImageMemoryBarrier toPresent = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, 0, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, images[index], {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &toPresent);
        vkEndCommandBuffer(cmd);
        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 1, &acquired, &waitStage, 1, &cmd, 1, &rendered};
        CHECK(vkQueueSubmit(queue, 1, &si, frameFence));
        VkPresentInfoKHR pi = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, nullptr, 1, &rendered, 1, &swap, &index};
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
