#pragma once
// Converting one texture into another of a different format, on the GPU, with one small compute shader: what the
// upscalers do to the game's inputs and output, and what the frame generation does the same way.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>

#include <cstdint>
#include <string>

#include "../include/dlss5nr.h"

namespace dlss5nr {

// The view format for reading or writing a texture of this storage format:
// typeless and depth formats become their readable form, sRGB becomes UNORM
// so values are read and written as stored. Unknown on formats the passes
// cannot use.
DXGI_FORMAT view_format(DXGI_FORMAT storage);

enum Variant { ToFloat4 = 0, ToFloat2 = 1, ToFloat = 2, VariantCount };

DXGI_FORMAT readable_format(DXGI_FORMAT f);

struct Converter {
    ID3D12Device *device = nullptr;
    ID3D12RootSignature *root = nullptr;
    ID3D12PipelineState *pipelines[VariantCount] = {};
    ID3D12DescriptorHeap *heap = nullptr;
    UINT increment = 0;
    UINT next = 0;

    bool init(ID3D12Device *d, std::string &error);
    void shutdown();
    // src (in a state a shader can read) to dst (unordered access), over dst's size.
    bool dispatch(ID3D12GraphicsCommandList *cmd, Variant variant, ID3D12Resource *src, ID3D12Resource *dst,
                  dlss5nr_point src_origin, dlss5nr_point dst_origin, uint32_t region_w, uint32_t region_h,
                  std::string &error);
};

void transition(ID3D12GraphicsCommandList *cmd, ID3D12Resource *r, uint32_t from, uint32_t to);

} // namespace dlss5nr
