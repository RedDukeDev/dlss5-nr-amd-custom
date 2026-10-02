// The slice of the CUDA driver API this addon needs, loaded by name from
// nvcuda.dll so the same binary works over NVIDIA's driver or over ZLUDA.
//
// Resolution tries the _v2 name first and falls back to the bare one. That
// order matters over ZLUDA: several bare names are v1-ABI stubs that return
// CUDA_ERROR_NOT_SUPPORTED, while the _v2 entry points carry the real code.

#pragma once

#include <cstddef>
#include <cstdint>

typedef int CUresult;
constexpr CUresult CUDA_SUCCESS = 0;

typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUarray;
typedef void *CUmipmappedArray;
typedef void *CUexternalMemory;
typedef unsigned long long CUdeviceptr;
typedef unsigned long long CUsurfObject;
typedef unsigned long long CUtexObject;

enum CUexternalMemoryHandleType : uint32_t {
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD = 1,
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32 = 2,
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT = 3,
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP = 4,
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE = 5,
};

// Laid out exactly as the CUDA header. The union is 16 bytes: getting it wrong
// silently misplaces size and flags, which shows up as a bare INVALID_VALUE
// out of cuImportExternalMemory.
struct CUDA_EXTERNAL_MEMORY_HANDLE_DESC {
    CUexternalMemoryHandleType type;
    union {
        int fd;
        struct {
            void *handle;
            const void *name;
        } win32;
        const void *nvSciBufObject;
    } handle;
    unsigned long long size;
    unsigned int flags;
    unsigned int reserved[16];
};

enum CUarray_format : uint32_t {
    CU_AD_FORMAT_UNSIGNED_INT8 = 0x01,
    CU_AD_FORMAT_HALF = 0x10,
    CU_AD_FORMAT_FLOAT = 0x20,
};

// The 2D descriptor, which is what cuArrayGetDescriptor returns. DLSS resolves
// that entry point, so it very likely interrogates the arrays it is handed.
struct CUDA_ARRAY_DESCRIPTOR {
    size_t Width;
    size_t Height;
    CUarray_format Format;
    unsigned int NumChannels;
};

// A 2D copy. Only the array-to-host direction is used here, to read back what
// the network wrote; the layout must still match the driver's in full.
enum CUmemorytype : uint32_t {
    CU_MEMORYTYPE_HOST = 1,
    CU_MEMORYTYPE_DEVICE = 2,
    CU_MEMORYTYPE_ARRAY = 3,
};

struct CUDA_MEMCPY2D {
    size_t srcXInBytes, srcY;
    CUmemorytype srcMemoryType;
    const void *srcHost;
    CUdeviceptr srcDevice;
    CUarray srcArray;
    size_t srcPitch;
    size_t dstXInBytes, dstY;
    CUmemorytype dstMemoryType;
    void *dstHost;
    CUdeviceptr dstDevice;
    CUarray dstArray;
    size_t dstPitch;
    size_t WidthInBytes, Height;
};

struct CUDA_ARRAY3D_DESCRIPTOR {
    size_t Width;
    size_t Height;
    size_t Depth;
    CUarray_format Format;
    unsigned int NumChannels;
    unsigned int Flags;
};

struct CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC {
    unsigned long long offset;
    CUDA_ARRAY3D_DESCRIPTOR arrayDesc;
    unsigned int numLevels;
    unsigned int reserved[16];
};

enum CUresourcetype : uint32_t {
    CU_RESOURCE_TYPE_ARRAY = 0,
    CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = 1,
    CU_RESOURCE_TYPE_LINEAR = 2,
    CU_RESOURCE_TYPE_PITCH2D = 3,
};

struct CUDA_RESOURCE_DESC {
    CUresourcetype resType;
    union {
        struct { CUarray hArray; } array;
        struct { CUmipmappedArray hMipmappedArray; } mipmap;
        struct { void *devPtr; CUarray_format format; unsigned int numChannels; size_t sizeInBytes; } linear;
        struct { void *devPtr; CUarray_format format; unsigned int numChannels; size_t width, height, pitchInBytes; } pitch2D;
        struct { int reserved[32]; } reserved;
    } res;
    unsigned int flags;
};

// CU_CTX_SCHED_AUTO
// Laid out as the CUDA header: three address modes, a filter mode, flags, then
// fields this code leaves at zero. The trailing reserved array matters -- the
// driver reads the whole structure, and a short one leaves it reading stack.
struct CUDA_TEXTURE_DESC {
    unsigned int addressMode[3];
    unsigned int filterMode;
    unsigned int flags;
    unsigned int maxAnisotropy;
    unsigned int mipmapFilterMode;
    float mipmapLevelBias;
    float minMipmapLevelClamp;
    float maxMipmapLevelClamp;
    float borderColor[4];
    int reserved[12];
};

constexpr unsigned int CU_CTX_SCHED_AUTO = 0;

// The two attributes that name the architecture. Only these two are needed, so
// the whole CUdevice_attribute enum is not worth copying in.
constexpr int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75;
constexpr int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76;

struct CudaApi {
    CUresult (*cuInit)(unsigned int);
    CUresult (*cuDeviceGet)(CUdevice *, int);
    CUresult (*cuDeviceGetCount)(int *);
    CUresult (*cuDeviceGetName)(char *, int, CUdevice);
    CUresult (*cuDeviceGetAttribute)(int *, int, CUdevice);
    CUresult (*cuCtxCreate)(CUcontext *, unsigned int, CUdevice);
    CUresult (*cuCtxDestroy)(CUcontext);
    CUresult (*cuCtxSetCurrent)(CUcontext);
    CUresult (*cuCtxSynchronize)(void);
    CUresult (*cuImportExternalMemory)(CUexternalMemory *, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC *);
    CUresult (*cuDestroyExternalMemory)(CUexternalMemory);
    CUresult (*cuExternalMemoryGetMappedMipmappedArray)(CUmipmappedArray *, CUexternalMemory,
                                                        const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC *);
    CUresult (*cuMipmappedArrayGetLevel)(CUarray *, CUmipmappedArray, unsigned int);
    CUresult (*cuMipmappedArrayDestroy)(CUmipmappedArray);
    CUresult (*cuSurfObjectCreate)(CUsurfObject *, const CUDA_RESOURCE_DESC *);
    CUresult (*cuSurfObjectDestroy)(CUsurfObject);
    CUresult (*cuTexObjectCreate)(CUtexObject *, const CUDA_RESOURCE_DESC *,
                                  const CUDA_TEXTURE_DESC *, const void *);
    CUresult (*cuTexObjectDestroy)(CUtexObject);
    CUresult (*cuArrayGetDescriptor)(CUDA_ARRAY_DESCRIPTOR *, CUarray);
    // Optional, and used only to read a result back for inspection.
    CUresult (*cuMemcpy2D)(const CUDA_MEMCPY2D *);
    // Optional, and used only where textures are copied through host memory
    // instead of shared (Network::set_copy_mode).
    CUresult (*cuArray3DCreate)(CUarray *, const CUDA_ARRAY3D_DESCRIPTOR *);
    CUresult (*cuArrayDestroy)(CUarray);
    // Optional, and only reported: how much device memory is left is the first
    // thing worth knowing when a feature refuses to be created.
    CUresult (*cuMemGetInfo)(size_t *, size_t *);
    CUresult (*cuMemAlloc)(CUdeviceptr *, size_t);
    CUresult (*cuGetErrorString)(CUresult, const char **);
};

// Fills `api` from an already-loaded nvcuda module. Returns the name of the
// first entry point that could not be resolved, or nullptr on success.
const char *cuda_api_load(void *nvcuda_module, CudaApi &api);
