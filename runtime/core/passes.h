// The Direct3D 12 side of the runtime: one root signature, the compute
// pipelines of shaders.h, and a ring of shader-visible descriptors, all
// recorded into the game's own command list.

#pragma once

#include <d3d12.h>

#include <cstdint>
#include <string>

namespace dlss5nr {

enum class Pass { Exposure, Capture, Refresh, Soften, Track, Compose, Count };

// Mirrors the cbuffer in shaders.h, 24 dwords.
struct Constants {
    uint32_t size[2];
    float inv_size[2];
    float uv0_scale[2];
    float uv1_scale[2];
    float uv2_scale[2];
    float motion_to_px[2];
    uint32_t encoding;
    float colour_strength;
    float exposure_mul;
    float pre_exposure;
    float strength;
    float max_ratio;
    float fade;
    uint32_t flags;
    uint32_t exposure_mode;
    float adaptation;
    float blend;
    uint32_t pad;
};
static_assert(sizeof(Constants) == 24 * 4, "24 root constants, as the cbuffer in shaders.h");

enum : uint32_t {
    FLAG_RESET = 1,
    FLAG_COPY = 4,
    FLAG_DEPTH_INVERTED = 8,
    FLAG_REPLACE = 16,
    FLAG_IDENTITY = 32,
    FLAG_KEEP_DEPTH = 64,
    FLAG_DEPTH_CHECK = 128,
    FLAG_FILL = 256,
    FLAG_SHOW_TRACKING = 512,
    FLAG_PREVIOUS = 1024,
};

struct Bindings {
    ID3D12Resource *srv[8] = {};
    ID3D12Resource *uav[3] = {};
    ID3D12Resource *exposure_buffer = nullptr;   // u3, raw
};

class Passes {
public:
    bool init(ID3D12Device *device, std::string &error);
    void shutdown();

    // Binds the root signature, the descriptor heap, the pipeline and the
    // views, and dispatches enough 8 x 8 groups to cover width x height (the
    // exposure pass is one group whatever the size).
    void dispatch(ID3D12GraphicsCommandList *cmd, Pass pass, const Constants &constants,
                  const Bindings &bindings, uint32_t width, uint32_t height);

private:
    void write_srv(D3D12_CPU_DESCRIPTOR_HANDLE at, ID3D12Resource *resource);
    void write_uav(D3D12_CPU_DESCRIPTOR_HANDLE at, ID3D12Resource *resource);

    ID3D12Device *device_ = nullptr;
    ID3D12RootSignature *root_ = nullptr;
    ID3D12PipelineState *pipelines_[(int)Pass::Count] = {};
    ID3D12DescriptorHeap *heap_ = nullptr;
    UINT increment_ = 0;
    UINT next_ = 0;
};

// The view format for reading or writing a texture of this storage format:
// typeless and depth formats become their readable form, sRGB becomes UNORM
// so values are read and written as stored. Unknown on formats the passes
// cannot use.
DXGI_FORMAT view_format(DXGI_FORMAT storage);

} // namespace dlss5nr
