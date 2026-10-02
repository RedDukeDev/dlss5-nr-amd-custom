#include "passes.h"

#include <d3dcompiler.h>

#include <cstdio>
#include <cstring>

#include "shaders.h"
#include "shaders_dxbc.h"

#pragma comment(lib, "d3dcompiler.lib")

namespace dlss5nr {
namespace {

constexpr UINT kSrvCount = 8;
constexpr UINT kUavCount = 4;
constexpr UINT kTableSize = kSrvCount + kUavCount;
// A dispatch takes one table of twelve. A frame records at most twelve dispatches,
// so the ring covers a hundred frames in flight before it reuses a
// table the GPU may still read.
constexpr UINT kRingTables = 1024;

const char *const kEntry[] = {"exposure_main", "capture_main", "refresh_main", "soften_main", "track_main",
                              "compose_main"};

template <typename T>
void release(T *&object) {
    if (object) object->Release();
    object = nullptr;
}

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

bool Passes::init(ID3D12Device *device, std::string &error) {
    device_ = device;

    D3D12_DESCRIPTOR_RANGE ranges[2] = {};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = kSrvCount;
    ranges[0].OffsetInDescriptorsFromTableStart = 0;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = kUavCount;
    ranges[1].OffsetInDescriptorsFromTableStart = kSrvCount;

    D3D12_ROOT_PARAMETER params[2] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.Num32BitValues = sizeof(Constants) / 4;
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 2;
    params[1].DescriptorTable.pDescriptorRanges = ranges;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_STATIC_SAMPLER_DESC samplers[2] = {};
    for (int i = 0; i < 2; ++i) {
        samplers[i].Filter = i == 0 ? D3D12_FILTER_MIN_MAG_MIP_LINEAR : D3D12_FILTER_MIN_MAG_MIP_POINT;
        samplers[i].AddressU = samplers[i].AddressV = samplers[i].AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        samplers[i].MaxLOD = D3D12_FLOAT32_MAX;
        samplers[i].ShaderRegister = i;
        samplers[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }

    D3D12_ROOT_SIGNATURE_DESC root{};
    root.NumParameters = 2;
    root.pParameters = params;
    root.NumStaticSamplers = 2;
    root.pStaticSamplers = samplers;
    ID3DBlob *blob = nullptr, *messages = nullptr;
    if (FAILED(D3D12SerializeRootSignature(&root, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &messages))) {
        error = messages ? (const char *)messages->GetBufferPointer() : "root signature serialisation failed";
        release(messages);
        return false;
    }
    const HRESULT hr =
        device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root_));
    release(blob);
    if (FAILED(hr)) {
        error = "CreateRootSignature failed";
        return false;
    }

    for (int i = 0; i < (int)Pass::Count; ++i) {
        ID3DBlob *code = nullptr, *log = nullptr;
        const void *bytes = nullptr;
        size_t size = 0;
        if (FAILED(D3DCompile(kShaderSource, sizeof kShaderSource - 1, "dlss5nr", nullptr, nullptr, kEntry[i],
                              "cs_5_1", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &log))) {
            // The system's compiler cannot build these everywhere (Wine's, which is
            // vkd3d's, has no RWByteAddressBuffer.Load): use the same shaders
            // compiled beforehand, as long as they are of this source.
            const std::string failure = std::string("compiling ") + kEntry[i] + ": " +
                                        (log ? (const char *)log->GetBufferPointer() : "?");
            release(log);
            if (kDxbcSourceHash != shader_hash(kShaderSource) || !kDxbc[i].size) {
                error = failure + " (and the shipped bytecode is not of this source)";
                return false;
            }
            bytes = kDxbc[i].data;
            size = kDxbc[i].size;
        } else {
            bytes = code->GetBufferPointer();
            size = code->GetBufferSize();
        }
        release(log);
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
        pso.pRootSignature = root_;
        pso.CS = {bytes, size};
        const HRESULT made = device->CreateComputePipelineState(&pso, IID_PPV_ARGS(&pipelines_[i]));
        release(code);
        if (FAILED(made)) {
            error = std::string("CreateComputePipelineState failed for ") + kEntry[i];
            return false;
        }
    }

    D3D12_DESCRIPTOR_HEAP_DESC heap{};
    heap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap.NumDescriptors = kTableSize * kRingTables;
    heap.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(device->CreateDescriptorHeap(&heap, IID_PPV_ARGS(&heap_)))) {
        error = "CreateDescriptorHeap failed";
        return false;
    }
    increment_ = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    return true;
}

void Passes::shutdown() {
    for (auto &p : pipelines_) release(p);
    release(root_);
    release(heap_);
}

void Passes::write_srv(D3D12_CPU_DESCRIPTOR_HANDLE at, ID3D12Resource *resource) {
    D3D12_SHADER_RESOURCE_VIEW_DESC view{};
    view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    view.Texture2D.MipLevels = 1;
    if (resource) {
        view.Format = view_format(resource->GetDesc().Format);
    } else {
        // A null view still needs a format and a dimension; it reads zero.
        view.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    }
    device_->CreateShaderResourceView(resource, &view, at);
}

void Passes::write_uav(D3D12_CPU_DESCRIPTOR_HANDLE at, ID3D12Resource *resource) {
    D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
    view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    view.Format = resource ? view_format(resource->GetDesc().Format) : DXGI_FORMAT_R8G8B8A8_UNORM;
    device_->CreateUnorderedAccessView(resource, nullptr, &view, at);
}

void Passes::dispatch(ID3D12GraphicsCommandList *cmd, Pass pass, const Constants &constants,
                      const Bindings &b, uint32_t width, uint32_t height) {
    const UINT table = (next_++ % kRingTables) * kTableSize;
    D3D12_CPU_DESCRIPTOR_HANDLE cpu = heap_->GetCPUDescriptorHandleForHeapStart();
    D3D12_GPU_DESCRIPTOR_HANDLE gpu = heap_->GetGPUDescriptorHandleForHeapStart();
    cpu.ptr += (SIZE_T)table * increment_;
    gpu.ptr += (UINT64)table * increment_;

    for (UINT i = 0; i < kSrvCount; ++i) {
        D3D12_CPU_DESCRIPTOR_HANDLE at = cpu;
        at.ptr += (SIZE_T)i * increment_;
        write_srv(at, b.srv[i]);
    }
    for (UINT i = 0; i < 3; ++i) {
        D3D12_CPU_DESCRIPTOR_HANDLE at = cpu;
        at.ptr += (SIZE_T)(kSrvCount + i) * increment_;
        write_uav(at, b.uav[i]);
    }
    {
        D3D12_CPU_DESCRIPTOR_HANDLE at = cpu;
        at.ptr += (SIZE_T)(kSrvCount + 3) * increment_;
        D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
        view.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        view.Format = DXGI_FORMAT_R32_TYPELESS;
        view.Buffer.NumElements = 4;
        view.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        device_->CreateUnorderedAccessView(b.exposure_buffer, nullptr, &view, at);
    }

    ID3D12DescriptorHeap *heaps[] = {heap_};
    cmd->SetDescriptorHeaps(1, heaps);
    cmd->SetComputeRootSignature(root_);
    cmd->SetPipelineState(pipelines_[(int)pass]);
    cmd->SetComputeRoot32BitConstants(0, sizeof(Constants) / 4, &constants, 0);
    cmd->SetComputeRootDescriptorTable(1, gpu);
    if (pass == Pass::Exposure)
        cmd->Dispatch(1, 1, 1);
    else
        cmd->Dispatch((width + 7) / 8, (height + 7) / 8, 1);
}

} // namespace dlss5nr
