// The network itself: ZLUDA, the gate, our NGX runtime and NVIDIA's snippet,
// driven through the snippet's CUDA interface.
//
// Everything here runs on the runtime's worker thread and nowhere else: the
// CUDA context is current on that thread only, and the snippet's calls can take
// tens of milliseconds. The game thread never calls into this file, but for
// the functions that record copies into a command list (record_stage_in and
// record_stage_out), which touch no CUDA.

#pragma once

#include <cstdint>
#include <string>

struct ID3D12Device;
struct ID3D12Resource;
struct ID3D12GraphicsCommandList;

namespace dlss5nr {

// Which of NVIDIA's networks a Network instance hosts. A process runs one: the NGX runtime, the CUDA
// context and the stand-in NVAPI are single, and each network wants the GPU described its own way.
enum class Kind {
    NeuralRendering,     // nvngx_dlssnr.dll: alters the finished image
    SuperResolution,     // nvngx_dlss.dll: upscales from the game's low resolution inputs
    RayReconstruction,   // nvngx_dlssd.dll: upscales and denoises from the G-buffer
};

struct NetworkPaths {
    std::wstring runtime_dir;   // holds zluda\ and kernels\ (see dlss5nr.h)
    std::wstring snippet;       // nvngx_dlssnr.dll, nvngx_dlss.dll or nvngx_dlssd.dll, to match kind
    std::wstring data_dir;      // logs; the translation cache lives in data_dir\ComputeCache
    Kind kind = Kind::NeuralRendering;
    // A folder of zluda\ holding a copy of nvngx.dll for this network alone (empty: zluda\nvngx.dll itself).
    // Two networks in one process each need their own.
    std::wstring host_dir;
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
    int32_t preset = 0;             // 0: the library's default; read when the feature is created

    bool operator==(const NetworkControls &o) const {
        return style == o.style && intensity == o.intensity && local_tone == o.local_tone &&
               local_structure == o.local_structure && skin_structure == o.skin_structure &&
               preset == o.preset;
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
    // Copy mode: the game's own command list has already copied the inputs into
    // their staging buffers, and copies the output out of its buffer after: the
    // network only moves them between those and its arrays.
    bool staged_by_list = false;
};

// What an upscaling network is created for. Sizes in pixels; flags are NVIDIA's DLSS creation flags
// (HDR 1, low resolution motion vectors 2, jittered motion vectors 4, inverted depth 8, auto exposure 32).
struct UpscalerCreate {
    uint32_t render_width = 0, render_height = 0;
    uint32_t output_width = 0, output_height = 0;
    uint32_t flags = 1 | 8 | 32;
    int32_t quality = 2;    // NVSDK_NGX_PerfQuality_Value
    int32_t preset = 0;     // 0: the library's own choice
    bool operator==(const UpscalerCreate &o) const {
        return render_width == o.render_width && render_height == o.render_height &&
               output_width == o.output_width && output_height == o.output_height && flags == o.flags &&
               quality == o.quality && preset == o.preset;
    }
};

// One frame for an upscaling network. Images are the runtime's own shared textures. The optional ones
// (null: absent) are what ray reconstruction takes beyond super resolution.
struct UpscalerDesc {
    SharedImage *color = nullptr, *depth = nullptr, *motion = nullptr, *output = nullptr;
    SharedImage *diffuse_albedo = nullptr, *specular_albedo = nullptr, *normals = nullptr;
    SharedImage *roughness = nullptr, *hit_distance = nullptr;
    // Optional for both: the game's exposure (a texture of one texel), and the two masks that tell the
    // network where its history is not to be trusted (the "bias current colour" mask, what games and FSR call
    // the reactive mask, and the transparency mask). Null: the game has none.
    SharedImage *exposure = nullptr, *bias_mask = nullptr, *transparency_mask = nullptr;
    // The region of the images that was rendered this frame, from their top left corner: smaller than
    // the creation size when the game scales its resolution dynamically. 0: the creation size.
    uint32_t render_width = 0, render_height = 0;
    float jitter_x = 0.0f, jitter_y = 0.0f;
    float motion_scale_x = 1.0f, motion_scale_y = 1.0f;
    float pre_exposure = 1.0f, exposure_scale = 1.0f, sharpness = 0.0f;
    bool reset = false;
    // Ray reconstruction: column-major or row-major as the game keeps them, passed through untouched.
    const float *world_to_view = nullptr;   // 16 floats
    const float *view_to_clip = nullptr;    // 16 floats
    bool roughness_packed_in_normals = true;
};

struct Own;

class Network {
public:
    Network();
    ~Network();
    Network(const Network &) = delete;
    Network &operator=(const Network &) = delete;

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

    // Whether a texture of this device can be shared with CUDA here: made shared, imported and mapped as an array, then
    // let go. Under Wine that depends on the build (the bridge asks the server for the descriptor behind the handle).
    bool can_share(ID3D12Device *device);

    // Imports a texture the game thread created. Idempotent.
    bool import(SharedImage &image, std::string &error);
    // Copy mode, for a frame that waits inside its command list: the copies of
    // an image to its staging buffer and from it, recorded into that list. False
    // if the image has no staging buffer yet (the worker makes it when it first
    // imports the image).
    static bool staged(const SharedImage &image) { return image.staging != nullptr; }
    static void record_stage_in(ID3D12GraphicsCommandList *cmd, const SharedImage &image);
    static void record_stage_out(ID3D12GraphicsCommandList *cmd, const SharedImage &image);
    void release(SharedImage &image);
    // Copy mode, for the upscalers (their frames wait on the CPU): the textures into the network's
    // arrays, and the output back into its texture. record false: the staging buffers are already
    // filled by a list of the caller's.
    bool copy_mode() const { return copy_mode_; }
    bool copy_images_in(SharedImage *const *images, size_t count, bool record, std::string &error);
    bool copy_image_out(SharedImage &image, bool staged_by_list, std::string &error);

    // Runs the network once and waits for it: when this returns true the
    // output holds the result.
    bool evaluate(const EvaluateDesc &desc, std::string &error);

    // Super resolution and ray reconstruction (Kind::SuperResolution, Kind::RayReconstruction).
    // The feature is made again only when what it is made for changes.
    bool create_upscaler(const UpscalerCreate &create, std::string &error);
    // synchronise false: the evaluation is only queued on the CUDA side, to be waited for with a fence
    // (signal_fence after it); true waits for the GPU to be done, as evaluate() does.
    bool evaluate_upscaler(const UpscalerDesc &desc, std::string &error, bool synchronise = true);
    // A Direct3D 12 fence (a shared NT handle) as a CUDA semaphore; waits and signals are queued on the
    // CUDA side's own stream, in order with the network's kernels. False if the driver cannot.
    bool import_fence(void *nt_handle, void **semaphore, std::string &error);
    // Makes the network's CUDA context current on the calling thread: needed before importing or
    // releasing images from a thread other than the one that loaded.
    bool make_current(std::string &error);
    // The calling thread's default CUDA stream gets high priority (or goes back to the plain one): the
    // upscalers ask for it, so that they are not held up by the neural rendering on the same GPU.
    void set_thread_high_priority(bool high);
    void destroy_fence(void *semaphore);
    bool wait_fence(void *semaphore, uint64_t value, std::string &error);
    bool signal_fence(void *semaphore, uint64_t value, std::string &error);

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
    UpscalerCreate upscaler_;
    Kind kind_ = Kind::NeuralRendering;
    Own *own_ = nullptr;
    bool counted_ = false;   // this network is one of the process's users of the CUDA context
    uint32_t native_kernels_ = 0;
    std::string device_name_;
};

// Tells the network's log sink where to go. Set before load.
void set_network_log(void (*sink)(int level, const char *message));

} // namespace dlss5nr
