// NGX CUDA surface, as exported by nvngx_dlss.dll.
//
// Everything here was derived from the shipped DLL rather than from an SDK, so
// the provenance of each declaration is marked:
//
//   [RE]      verified by disassembling the export
//   [LAYOUT]  ABI-critical layout that must match, inferred from the NGX pattern
//
// The plain NVSDK_NGX_CUDA_Init export is a dead stub -- it is literally
// `mov eax, 0xBAD00001; ret`, and D3D11_Init and D3D12_Init share that same
// address. Only the Init1 / Init_Ext / Init_Ext1 variants carry real code, so
// this header only declares the _Ext form.

#pragma once

#include <cstdint>

typedef uint32_t NVSDK_NGX_Result;

#define NVSDK_NGX_SUCCEED(r) (((r) & 0xFFF00000) != 0xBAD00000)

// [RE] the stub Init variants return this, so it is a real code, not a guess.
constexpr NVSDK_NGX_Result NVSDK_NGX_Result_Success = 0x1;
constexpr NVSDK_NGX_Result NVSDK_NGX_Result_Fail = 0xBAD00000;
constexpr NVSDK_NGX_Result NVSDK_NGX_Result_FeatureNotSupported = 0xBAD00001;

// [LAYOUT] Feature ids. SuperSampling is the one DLSS upscaling uses.
enum NVSDK_NGX_Feature : uint32_t {
    NVSDK_NGX_Feature_SuperSampling = 1,
    NVSDK_NGX_Feature_NeuralRendering = 18,
};

enum NVSDK_NGX_Version : uint32_t {
    NVSDK_NGX_Version_API = 0x15,
};

// [LAYOUT] Perf/quality selector, passed as "PerfQualityValue".
enum NVSDK_NGX_PerfQuality_Value : int32_t {
    NVSDK_NGX_PerfQuality_Value_MaxPerf = 0,
    NVSDK_NGX_PerfQuality_Value_Balanced = 1,
    NVSDK_NGX_PerfQuality_Value_MaxQuality = 2,
    NVSDK_NGX_PerfQuality_Value_UltraPerformance = 3,
    NVSDK_NGX_PerfQuality_Value_UltraQuality = 4,
    NVSDK_NGX_PerfQuality_Value_DLAA = 5,
};

// [LAYOUT] Creation flags, set under "DLSS.Feature.Create.Flags".
enum NVSDK_NGX_DLSS_Feature_Flags : uint32_t {
    NVSDK_NGX_DLSS_Feature_Flags_IsHDR = 1 << 0,
    NVSDK_NGX_DLSS_Feature_Flags_MVLowRes = 1 << 1,
    NVSDK_NGX_DLSS_Feature_Flags_MVJittered = 1 << 2,
    NVSDK_NGX_DLSS_Feature_Flags_DepthInverted = 1 << 3,
    NVSDK_NGX_DLSS_Feature_Flags_DoSharpening = 1 << 4,
    NVSDK_NGX_DLSS_Feature_Flags_AutoExposure = 1 << 5,
};

struct NVSDK_NGX_Handle {
    uint32_t Id;
};

// [LAYOUT] NVSDK_NGX_Parameter is a C++ interface: the DLL hands back a pointer
// to an object whose vtable has this exact order. Getting the order wrong calls
// the wrong slot, so nothing here may be reordered, and the whole block must be
// declared even though this addon only uses a few entries.
//
// The D3D11/D3D12 resource setters keep their slots so the numbering stays
// right. They are declared against forward-declared interface types rather than
// void*, both to avoid pulling in the D3D headers and because collapsing them
// to void* would merge three distinct overloads into one and shift every slot
// after them.
struct ID3D11Resource;
struct ID3D12Resource;

class NVSDK_NGX_Parameter {
public:
#ifdef __MINGW32__
    // MSVC lays the overloads of one virtual function out in the reverse of the
    // order they are declared in; GCC keeps the declared order. The object on the
    // other side is built by MSVC, so for GCC the overloads are declared reversed.
    virtual void Set(const char *InName, void *InValue) = 0;
    virtual void Set(const char *InName, ID3D12Resource *InValue) = 0;
    virtual void Set(const char *InName, ID3D11Resource *InValue) = 0;
    virtual void Set(const char *InName, int InValue) = 0;
    virtual void Set(const char *InName, unsigned int InValue) = 0;
    virtual void Set(const char *InName, double InValue) = 0;
    virtual void Set(const char *InName, float InValue) = 0;
    virtual void Set(const char *InName, unsigned long long InValue) = 0;
    virtual NVSDK_NGX_Result Get(const char *InName, void **OutValue) const = 0;
    virtual NVSDK_NGX_Result Get(const char *InName, ID3D12Resource **OutValue) const = 0;
    virtual NVSDK_NGX_Result Get(const char *InName, ID3D11Resource **OutValue) const = 0;
    virtual NVSDK_NGX_Result Get(const char *InName, int *OutValue) const = 0;
    virtual NVSDK_NGX_Result Get(const char *InName, unsigned int *OutValue) const = 0;
    virtual NVSDK_NGX_Result Get(const char *InName, double *OutValue) const = 0;
    virtual NVSDK_NGX_Result Get(const char *InName, float *OutValue) const = 0;
    virtual NVSDK_NGX_Result Get(const char *InName, unsigned long long *OutValue) const = 0;
#else
    virtual void Set(const char *InName, unsigned long long InValue) = 0;
    virtual void Set(const char *InName, float InValue) = 0;
    virtual void Set(const char *InName, double InValue) = 0;
    virtual void Set(const char *InName, unsigned int InValue) = 0;
    virtual void Set(const char *InName, int InValue) = 0;
    virtual void Set(const char *InName, ID3D11Resource *InValue) = 0;
    virtual void Set(const char *InName, ID3D12Resource *InValue) = 0;
    virtual void Set(const char *InName, void *InValue) = 0;
    virtual NVSDK_NGX_Result Get(const char *InName, unsigned long long *OutValue) const = 0;
    virtual NVSDK_NGX_Result Get(const char *InName, float *OutValue) const = 0;
    virtual NVSDK_NGX_Result Get(const char *InName, double *OutValue) const = 0;
    virtual NVSDK_NGX_Result Get(const char *InName, unsigned int *OutValue) const = 0;
    virtual NVSDK_NGX_Result Get(const char *InName, int *OutValue) const = 0;
    virtual NVSDK_NGX_Result Get(const char *InName, ID3D11Resource **OutValue) const = 0;
    virtual NVSDK_NGX_Result Get(const char *InName, ID3D12Resource **OutValue) const = 0;
    virtual NVSDK_NGX_Result Get(const char *InName, void **OutValue) const = 0;
#endif
    virtual void Reset() = 0;
};

typedef void (*PFN_NVSDK_NGX_ProgressCallback)(float InCurrentProgress, bool &OutShouldCancel);

// [RE] Four arguments. The disassembly also reads [rsp+0x2A8], which after the
// prologue (five pushes plus sub rsp,0x280) is the return address rather than a
// fifth argument: it goes to GetModuleHandleExW(FROM_ADDRESS) to identify the
// caller, which must be nvngx.dll. See ngx_runtime.cpp.
typedef NVSDK_NGX_Result (*PFN_NVSDK_NGX_CUDA_Init_Ext)(
    unsigned long long InApplicationId,
    const wchar_t *InApplicationDataPath,
    NVSDK_NGX_Version InSDKVersion,
    const void *InFeatureInfo);

// [RE] Init_Ext1 takes one extra pointer, in third position, which pushes the
// version to the fourth. Both forms call the same internal at 0x1800AFC30, and
// the only difference in the call is the fifth argument:
//
//     Init_Ext   ->  mov [rsp+20h],r14   ; zero
//     Init_Ext1  ->  mov [rsp+20h],rbp   ; the extra pointer
//
// That fifth argument is the key the internal registers itself under, in the map
// EvaluateFeature later looks the feature up in (0x1800B0660). Initialising
// through Init_Ext therefore registers under a null key, and evaluation cannot
// find the entry -- it returns 0xBAD00002 before touching any resource.
//
// On the CUDA backend the pointer is the CUcontext.
typedef NVSDK_NGX_Result (*PFN_NVSDK_NGX_CUDA_Init_Ext1)(
    unsigned long long InApplicationId,
    const wchar_t *InApplicationDataPath,
    void *InCUDAContext,
    NVSDK_NGX_Version InSDKVersion,
    const void *InFeatureInfo);

// [RE] CreateFeature moves ecx->r8d, rdx->r9 and r8->[rsp+0x20], then calls an
// internal five-argument function with two leading nulls -- the device and
// command-list slots the graphics backends fill in and CUDA leaves empty.
typedef NVSDK_NGX_Result (*PFN_NVSDK_NGX_CUDA_CreateFeature)(
    NVSDK_NGX_Feature InFeatureID,
    NVSDK_NGX_Parameter *InParameters,
    NVSDK_NGX_Handle **OutHandle);

// [RE] CreateFeature1 takes the same extra pointer in first position, and uses
// it as the key into the very registry Init_Ext1 populates (both reach
// 0x1800B0660). The two families therefore have to be paired:
//
//     Init_Ext  registers under null  ->  CreateFeature  looks up null
//     Init_Ext1 registers under ctx   ->  CreateFeature1 looks up ctx
//
// Mixing them leaves the feature associated with a device the evaluation cannot
// find, and EvaluateFeature returns 0xBAD00002 before touching any resource.
//
// Passing null here instead makes CreateFeature1 read the value from the
// parameter block under the name "Input1".
typedef NVSDK_NGX_Result (*PFN_NVSDK_NGX_CUDA_CreateFeature1)(
    void *InCUDAContext,
    NVSDK_NGX_Feature InFeatureID,
    NVSDK_NGX_Parameter *InParameters,
    NVSDK_NGX_Handle **OutHandle);

// [RE] EvaluateFeature keeps rcx, rdx and r8 and touches no stack argument.
typedef NVSDK_NGX_Result (*PFN_NVSDK_NGX_CUDA_EvaluateFeature)(
    NVSDK_NGX_Handle *InFeatureHandle,
    const NVSDK_NGX_Parameter *InParameters,
    PFN_NVSDK_NGX_ProgressCallback InCallback);

typedef NVSDK_NGX_Result (*PFN_NVSDK_NGX_CUDA_ReleaseFeature)(NVSDK_NGX_Handle *InHandle);
typedef NVSDK_NGX_Result (*PFN_NVSDK_NGX_CUDA_Shutdown)(void);
typedef NVSDK_NGX_Result (*PFN_NVSDK_NGX_CUDA_PopulateParameters_Impl)(NVSDK_NGX_Parameter *InParameters);

// [RE] Validates a feature id without building anything: the snippet checks the
// id against the one it implements and returns FeatureNotSupported for any
// other, long before it starts compiling a network. That makes it the cheap way
// to discover which id a snippet answers to.
typedef NVSDK_NGX_Result (*PFN_NVSDK_NGX_CUDA_GetScratchBufferSize)(
    NVSDK_NGX_Feature InFeatureId,
    const NVSDK_NGX_Parameter *InParameters,
    size_t *OutSizeInBytes);

// Parameter names, taken from the strings of the shipped DLL rather than from
// documentation, so they match this exact build. The full list is in
// docs/ngx_params.txt.
namespace ngx_param {
inline constexpr const char *CreationNodeMask = "CreationNodeMask";
inline constexpr const char *VisibilityNodeMask = "VisibilityNodeMask";
inline constexpr const char *Width = "Width";
inline constexpr const char *Height = "Height";
inline constexpr const char *OutWidth = "OutWidth";
inline constexpr const char *OutHeight = "OutHeight";
inline constexpr const char *PerfQualityValue = "PerfQualityValue";
inline constexpr const char *FeatureCreateFlags = "DLSS.Feature.Create.Flags";
inline constexpr const char *EnableOutputSubrects = "DLSS.Enable.Output.Subrects";

inline constexpr const char *Color = "Color";
inline constexpr const char *Depth = "Depth";
inline constexpr const char *MotionVectors = "MotionVectors";
inline constexpr const char *Output = "Output";
inline constexpr const char *Reset = "Reset";
inline constexpr const char *JitterOffsetX = "Jitter.Offset.X";
inline constexpr const char *JitterOffsetY = "Jitter.Offset.Y";
inline constexpr const char *MVScaleX = "MV.Scale.X";
inline constexpr const char *MVScaleY = "MV.Scale.Y";
inline constexpr const char *Sharpness = "Sharpness";
inline constexpr const char *PreExposure = "DLSS.Pre.Exposure";
inline constexpr const char *ExposureScale = "DLSS.Exposure.Scale";
inline constexpr const char *RenderSubrectWidth = "DLSS.Render.Subrect.Dimensions.Width";
inline constexpr const char *RenderSubrectHeight = "DLSS.Render.Subrect.Dimensions.Height";
} // namespace ngx_param

// Parameter names of the neural rendering feature, from the strings of
// nvngx_dlssnr.dll. This is a different network from super resolution and it
// does not share a single parameter name with it -- everything is prefixed
// DLSSNR and spelled without separating dots.
//
// Its input set is the interesting part: colour, depth, motion vectors and an
// output, exactly what super resolution takes. It needs no G-buffer, no albedo
// and no hit distances, so a ReShade addon can feed it without per-game
// knowledge of the renderer's internals.
namespace nr_param {
inline constexpr const char *Enabled = "DLSSNR.Enabled";
inline constexpr const char *Width = "DLSSNR.Width";
inline constexpr const char *Height = "DLSSNR.Height";
inline constexpr const char *InputWidth = "DLSSNR.InputWidth";
inline constexpr const char *InputHeight = "DLSSNR.InputHeight";
inline constexpr const char *OutputWidth = "DLSSNR.OutputWidth";
inline constexpr const char *OutputHeight = "DLSSNR.OutputHeight";
inline constexpr const char *Upscaling = "DLSSNR.Upscaling";
// The network reads this name, not "DLSSNR.Scale": the trace shows
// ScalingRatio queried and the other spelling never asked for.
inline constexpr const char *ScalingRatio = "DLSSNR.ScalingRatio";
inline constexpr const char *RenderPreset = "DLSSNR.Hint.Render.Preset";

inline constexpr const char *Color = "DLSSNR.Color";
inline constexpr const char *Depth = "DLSSNR.Depth";
inline constexpr const char *DepthInverted = "DLSSNR.DepthInverted";
inline constexpr const char *MVec = "DLSSNR.MVec";
inline constexpr const char *MVecScaleX = "DLSSNR.MVecScaleX";
inline constexpr const char *MVecScaleY = "DLSSNR.MVecScaleY";
inline constexpr const char *Output = "DLSSNR.Output";
inline constexpr const char *Reset = "DLSSNR.Reset";
inline constexpr const char *ControlMask = "DLSSNR.ControlMask";
inline constexpr const char *UseAutoMask = "DLSSNR.UseAutoMask";
inline constexpr const char *Backbuffer = "DLSSNR.Backbuffer";

// The controls over the effect itself. These have no counterpart in super
// resolution and are what makes this feature an image alteration rather than an
// upscaler.
inline constexpr const char *Style = "DLSSNR.Style";
inline constexpr const char *Intensity = "DLSSNR.Intensity";
inline constexpr const char *LocalToneStrength = "DLSSNR.LocalToneStrength";
inline constexpr const char *LocalStructureStrength = "DLSSNR.LocalStructureStrength";
inline constexpr const char *SkinStructureStrength = "DLSSNR.SkinStructureStrength";
inline constexpr const char *UICorrection = "DLSSNR.UICorrection";
} // namespace nr_param
