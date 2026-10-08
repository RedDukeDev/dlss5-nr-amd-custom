// The network itself: ZLUDA, the gate, our NGX runtime and NVIDIA's snippet,
// driven through the snippet's CUDA interface.
//
// Everything here runs on the runtime's worker thread and nowhere else: the
// CUDA context is current on that thread only, and the snippet's calls can take
// tens of milliseconds. The game thread never calls into this file.

#pragma once

#include <cstdint>
#include <string>

struct ID3D12Device;
struct ID3D12Resource;

namespace dlss5nr {

struct NetworkPaths {
    std::wstring runtime_dir;   // holds zluda\ and kernels\ (see dlss5nr.h)
    std::wstring snippet;       // nvngx_dlssnr.dll
    std::wstring data_dir;      // logs; the translation cache lives in data_dir\ComputeCache
};

// One D3D12 texture made shareable and imported into CUDA.
struct SharedImage {
    ID3D12Resource *resource = nullptr;   // created by the runtime, D3D12_HEAP_FLAG_SHARED
    void *nt_handle = nullptr;
    uint32_t width = 0, height = 0;
    int dxgi_format = 0;
    bool writable = false;                 // the network writes it

    // Copy mode only (Network::set_copy_mode): the image lives in a CUDA array
    // of its own, and moves through a host-visible buffer, mapped for good.
    void *staging = nullptr;               // ID3D12Resource: readback (inputs) or upload (output)
    void *staging_ptr = nullptr;
    // Ordinary memory of the same size. ROCm copies wrongly, and without saying
    // so, to and from memory that a Direct3D driver has mapped (measured under
    // Wine: an array filled from the readback buffer held almost none of it), so
    // the copies to and from the arrays go through this one.
    void *host = nullptr;
    uint32_t row_pitch = 0;                // of the staging buffer
    uint32_t bytes_per_pixel = 0;
    bool owns_array = false;

    // Filled by Network::import, released by Network::release.
    void *external_memory = nullptr;
    void *mipmapped_array = nullptr;
    void *array = nullptr;
    unsigned long long surface = 0;
    unsigned long long texture = 0;
    // What the snippet's parameter points at: the object first, then the
    // geometry. [RE] It reads the first eight bytes and asks CUDA about them.
    struct Descriptor {
        unsigned long long object;
        uint32_t width, height, pitch, format;
    } descriptor{};
};

struct NetworkControls {
    int32_t style = 0;
    float intensity = 1.0f;
    float local_tone = 1.0f;
    float local_structure = 1.0f;
    float skin_structure = -1.0f;   // negative: automatic, follows local_structure

    bool operator==(const NetworkControls &o) const {
        return style == o.style && intensity == o.intensity && local_tone == o.local_tone &&
               local_structure == o.local_structure && skin_structure == o.skin_structure;
    }
    bool operator!=(const NetworkControls &o) const { return !(*this == o); }
};

struct EvaluateDesc {
    SharedImage *color = nullptr;
    SharedImage *depth = nullptr;
    SharedImage *motion = nullptr;
    SharedImage *output = nullptr;
    bool depth_inverted = false;
    bool reset = false;
    NetworkControls controls;
    // The network run again on its own result, passes times: each pass takes
    // the one before as its picture, with a blank history, and the last one
    // writes `output`. keep_local_tone: whether the later ones keep applying the
    // local tone (else it is zero for them).
    int passes = 1;
    bool keep_local_tone = true;
};

class Network {
public:
    // Loads the whole chain and initialises NGX. On failure, error says why
    // and nothing else may be called.
    bool load(const NetworkPaths &paths, std::string &error);

    // Creates, or recreates for a new size or new controls, the neural
    // rendering feature. The network reads its controls when the feature is
    // created and keeps them: a change only takes effect through a new
    // feature, which also starts its history over.
    bool create_feature(uint32_t width, uint32_t height, const NetworkControls &controls, std::string &error);

    // Where the textures cannot be shared with CUDA (Wine: a shared handle is
    // not a file descriptor ROCm can import), each evaluation instead copies
    // the inputs out to host memory and the output back, through buffers and a
    // queue of this class's own. Set before the first import.
    void set_copy_mode(bool on) { copy_mode_ = on; }

    // Imports a texture the game thread created. Idempotent.
    bool import(SharedImage &image, std::string &error);
    void release(SharedImage &image);

    // Runs the network once and waits for it: when this returns true the
    // output holds the result.
    bool evaluate(const EvaluateDesc &desc, std::string &error);

    void shutdown();

    uint32_t native_kernels() const { return native_kernels_; }
    const std::string &device_name() const { return device_name_; }

private:
    bool import_copy(SharedImage &image, std::string &error);
    bool copy_in(const EvaluateDesc &desc, std::string &error);
    bool copy_out(const EvaluateDesc &desc, std::string &error);
    // The image the passes hand each other when there are more than two of
    // them: a CUDA array of the output's size and format, made when first wanted.
    bool ensure_scratch(const SharedImage &like, std::string &error);
    SharedImage scratch_;

    bool loaded_ = false;
    bool copy_mode_ = false;
    uint32_t feature_width_ = 0, feature_height_ = 0;
    NetworkControls feature_controls_;
    uint32_t native_kernels_ = 0;
    std::string device_name_;
};

// Tells the network's log sink where to go. Set before load.
void set_network_log(void (*sink)(int level, const char *message));

} // namespace dlss5nr
