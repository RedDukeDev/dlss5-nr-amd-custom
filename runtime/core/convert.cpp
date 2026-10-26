// The format converter: see convert.h.

#include "convert.h"

#include <d3dcompiler.h>

#include <cstdio>

#pragma comment(lib, "d3dcompiler.lib")

namespace dlss5nr {

namespace {

template <class T> void release(T *&p) {
    if (p) p->Release();
    p = nullptr;
}

// One thread group covers 8 x 8 pixels. The destination decides how far it goes; a source smaller than the
// destination is clamped at its edge.
const char kConvert[] = R"HLSL(
cbuffer Constants : register(b0) {
    uint2 src_origin;   // where the region starts in the source
    uint2 dst_origin;   // and in the destination
    uint2 region;       // its size
    uint2 src_size;     // the textures'
    uint2 dst_size;
};
Texture2D<float4> src : register(t0);
#if DST_FLOAT4
RWTexture2D<float4> dst : register(u0);
#elif DST_FLOAT2
RWTexture2D<float2> dst : register(u0);
#else
RWTexture2D<float> dst : register(u0);
#endif
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= region)) return;
    const uint2 to = dst_origin + id.xy;
    if (any(to >= dst_size)) return;
    float4 v = src.Load(int3(min(src_origin + id.xy, src_size - 1), 0));
#if DST_FLOAT4
    dst[to] = v;
#elif DST_FLOAT2
    dst[to] = v.xy;
#else
    dst[to] = v.x;
#endif
}
)HLSL";

constexpr UINT kTables = 256;   // a table is the source's view and the destination's

} // namespace

DXGI_FORMAT view_format(DXGI_FORMAT f) {
    switch (f) {
    case DXGI_FORMAT_R16G16B16A16_TYPELESS: return DXGI_FORMAT_R16G16B16A16_FLOAT;
    case DXGI_FORMAT_R32G32B32A32_TYPELESS: return DXGI_FORMAT_R32G32B32A32_FLOAT;
    case DXGI_FORMAT_R32G32B32_TYPELESS: return DXGI_FORMAT_R32G32B32_FLOAT;
    case DXGI_FORMAT_R8G8B8A8_TYPELESS:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return DXGI_FORMAT_R8G8B8A8_UNORM;
    case DXGI_FORMAT_B8G8R8A8_TYPELESS:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: return DXGI_FORMAT_B8G8R8A8_UNORM;
    case DXGI_FORMAT_B8G8R8X8_TYPELESS:
    case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB: return DXGI_FORMAT_B8G8R8X8_UNORM;
    case DXGI_FORMAT_R10G10B10A2_TYPELESS: return DXGI_FORMAT_R10G10B10A2_UNORM;
    case DXGI_FORMAT_R16G16_TYPELESS: return DXGI_FORMAT_R16G16_FLOAT;
    case DXGI_FORMAT_R32G32_TYPELESS: return DXGI_FORMAT_R32G32_FLOAT;
    case DXGI_FORMAT_R32_TYPELESS:
    case DXGI_FORMAT_D32_FLOAT: return DXGI_FORMAT_R32_FLOAT;
    case DXGI_FORMAT_R24G8_TYPELESS:
    case DXGI_FORMAT_D24_UNORM_S8_UINT: return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
    case DXGI_FORMAT_R32G8X24_TYPELESS:
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
    case DXGI_FORMAT_R16_TYPELESS:
    case DXGI_FORMAT_D16_UNORM: return DXGI_FORMAT_R16_UNORM;
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
    case DXGI_FORMAT_R32G32B32_FLOAT:
    case DXGI_FORMAT_R11G11B10_FLOAT:
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8X8_UNORM:
    case DXGI_FORMAT_R10G10B10A2_UNORM:
    case DXGI_FORMAT_R16G16B16A16_UNORM:
    case DXGI_FORMAT_R16G16_FLOAT:
    case DXGI_FORMAT_R32G32_FLOAT:
    case DXGI_FORMAT_R16G16_SNORM:
    case DXGI_FORMAT_R32_FLOAT:
    case DXGI_FORMAT_R16_FLOAT:
    case DXGI_FORMAT_R16_UNORM:
    case DXGI_FORMAT_R9G9B9E5_SHAREDEXP: return f;
    default: return DXGI_FORMAT_UNKNOWN;
    }
}

// The view format for a texture of this storage format: the runtime's own list, plus the single and double
// channel 8 and 16 bit formats the masks and guides of games are kept in.
DXGI_FORMAT readable_format(DXGI_FORMAT f) {
    switch (f) {
    case DXGI_FORMAT_R8_TYPELESS: return DXGI_FORMAT_R8_UNORM;
    case DXGI_FORMAT_R8_UNORM:
    case DXGI_FORMAT_R8G8_UNORM:
    case DXGI_FORMAT_R16G16_UNORM:
    case DXGI_FORMAT_R16_SNORM: return f;
    case DXGI_FORMAT_R8G8_TYPELESS: return DXGI_FORMAT_R8G8_UNORM;
    default: return view_format(f);
    }
}

bool Converter::init(ID3D12Device *d, std::string &error) {
    device = d;
    D3D12_DESCRIPTOR_RANGE ranges[2] = {};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 1;
    ranges[0].OffsetInDescriptorsFromTableStart = 0;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 1;
    ranges[1].OffsetInDescriptorsFromTableStart = 1;
    D3D12_ROOT_PARAMETER params[2] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.Num32BitValues = 10;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 2;
    params[1].DescriptorTable.pDescriptorRanges = ranges;
    D3D12_ROOT_SIGNATURE_DESC desc{};
    desc.NumParameters = 2;
    desc.pParameters = params;
    ID3DBlob *blob = nullptr, *errors = nullptr;
    if (FAILED(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1_0, &blob, &errors)) ||
        FAILED(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root)))) {
        error = "could not create the conversion root signature";
        release(blob);
        release(errors);
        return false;
    }
    release(blob);
    release(errors);

    static const char *const kDefine[VariantCount] = {"DST_FLOAT4", "DST_FLOAT2", "DST_FLOAT1"};
    for (int v = 0; v < VariantCount; ++v) {
        D3D_SHADER_MACRO macros[] = {{kDefine[v], "1"}, {nullptr, nullptr}};
        ID3DBlob *code = nullptr;
        errors = nullptr;
        if (FAILED(D3DCompile(kConvert, sizeof kConvert - 1, nullptr, macros, nullptr, "main", "cs_5_0",
                              D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &errors))) {
            error = std::string("the conversion shader does not compile: ") +
                    (errors ? (const char *)errors->GetBufferPointer() : "?");
            release(errors);
            return false;
        }
        release(errors);
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
        pso.pRootSignature = root;
        pso.CS = {code->GetBufferPointer(), code->GetBufferSize()};
        const HRESULT made = device->CreateComputePipelineState(&pso, IID_PPV_ARGS(&pipelines[v]));
        code->Release();
        if (FAILED(made)) {
            error = "could not create the conversion pipeline";
            return false;
        }
    }
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
    heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap_desc.NumDescriptors = kTables * 2;
    heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap)))) {
        error = "could not create the conversion descriptor heap";
        return false;
    }
    increment = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    return true;
}

void Converter::shutdown() {
    for (auto &p : pipelines) release(p);
    release(root);
    release(heap);
}

bool Converter::dispatch(ID3D12GraphicsCommandList *cmd, Variant variant, ID3D12Resource *src, ID3D12Resource *dst,
              dlss5nr_point src_origin, dlss5nr_point dst_origin, uint32_t region_w, uint32_t region_h,
              std::string &error) {
    const D3D12_RESOURCE_DESC sd = src->GetDesc(), dd = dst->GetDesc();
    const DXGI_FORMAT src_format = readable_format(sd.Format), dst_format = readable_format(dd.Format);
    if (src_format == DXGI_FORMAT_UNKNOWN || dst_format == DXGI_FORMAT_UNKNOWN) {
        char text[128];
        snprintf(text, sizeof text, "cannot convert between DXGI formats %d and %d", (int)sd.Format, (int)dd.Format);
        error = text;
        return false;
    }
    const UINT table = next++ % kTables;
    D3D12_CPU_DESCRIPTOR_HANDLE cpu = heap->GetCPUDescriptorHandleForHeapStart();
    cpu.ptr += (SIZE_T)table * 2 * increment;
    D3D12_GPU_DESCRIPTOR_HANDLE gpu = heap->GetGPUDescriptorHandleForHeapStart();
    gpu.ptr += (UINT64)table * 2 * increment;

    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = src_format;
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Texture2D.MipLevels = 1;
    device->CreateShaderResourceView(src, &srv, cpu);
    cpu.ptr += increment;
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    uav.Format = dst_format;
    uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    device->CreateUnorderedAccessView(dst, nullptr, &uav, cpu);

    const uint32_t constants[10] = {src_origin.x,  src_origin.y, dst_origin.x, dst_origin.y,
                                    region_w,      region_h,     (uint32_t)sd.Width, sd.Height,
                                    (uint32_t)dd.Width, dd.Height};
    ID3D12DescriptorHeap *heaps[] = {heap};
    cmd->SetDescriptorHeaps(1, heaps);
    cmd->SetComputeRootSignature(root);
    cmd->SetPipelineState(pipelines[variant]);
    cmd->SetComputeRoot32BitConstants(0, 10, constants, 0);
    cmd->SetComputeRootDescriptorTable(1, gpu);
    cmd->Dispatch((region_w + 7) / 8, (region_h + 7) / 8, 1);
    return true;
}

void transition(ID3D12GraphicsCommandList *cmd, ID3D12Resource *r, uint32_t from, uint32_t to) {
    if (from == to) return;
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = r;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = (D3D12_RESOURCE_STATES)from;
    b.Transition.StateAfter = (D3D12_RESOURCE_STATES)to;
    cmd->ResourceBarrier(1, &b);
}

} // namespace dlss5nr
