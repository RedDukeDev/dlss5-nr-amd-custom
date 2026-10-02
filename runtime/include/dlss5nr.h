/*
 * dlss5nr.h -- the runtime's C interface.
 *
 * DLSS 5 Neural Rendering on AMD GPUs, for a Direct3D 12 host (OptiScaler, a
 * ReShade add-on, a test harness). The host hands over the game's image, depth
 * and motion vectors inside the game's own command list; the runtime records
 * a few compute passes there and runs the network on a thread of its own. See
 * docs/DESIGN.md for why it is shaped like this.
 *
 * Threading: every function except dlss5nr_get_status must be called from the
 * thread that records the game's frame (the host's render thread). None of
 * them waits for the network.
 *
 * Every structure starts with its own size, so a host built against an older
 * header keeps working with a newer runtime.
 */
#ifndef DLSS5NR_H
#define DLSS5NR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(DLSS5NR_BUILD)
#define DLSS5NR_API __declspec(dllexport)
#else
#define DLSS5NR_API __declspec(dllimport)
#endif

struct ID3D12Device;
struct ID3D12CommandQueue;
struct ID3D12GraphicsCommandList;
struct ID3D12Resource;

#define DLSS5NR_VERSION 1

typedef struct dlss5nr_context dlss5nr_context;

typedef void (*dlss5nr_log_fn)(void *user, int level, const char *message);

enum dlss5nr_log_level {
    DLSS5NR_LOG_ERROR = 0,
    DLSS5NR_LOG_WARNING = 1,
    DLSS5NR_LOG_INFO = 2,
    DLSS5NR_LOG_DEBUG = 3,
};

typedef struct dlss5nr_create_info {
    uint32_t struct_size;
    ID3D12Device *device;
    /* The directory the runtime's own files are in: zluda\ (ZLUDA as
     * zluda_real.dll, the gate as nvcuda.dll, nvapi64.dll, nvngx.dll),
     * kernels\ (the native kernels and kernels.txt).
     * Null means the directory of dlss5nr_runtime.dll. */
    const wchar_t *runtime_dir;
    /* NVIDIA's nvngx_dlssnr.dll. Null means runtime_dir\nvngx_dlssnr.dll. */
    const wchar_t *snippet_path;
    /* Where the translation cache and the logs go. Null means
     * runtime_dir\cache. Created if missing. */
    const wchar_t *data_dir;
    dlss5nr_log_fn log;
    void *log_user;
} dlss5nr_create_info;

enum dlss5nr_exposure_mode {
    /* Measured from the image, smoothed over time. */
    DLSS5NR_EXPOSURE_AUTO = 0,
    /* The game's exposure texture, when the frame carries one; else auto. */
    DLSS5NR_EXPOSURE_GAME = 1,
    /* dlss5nr_settings.exposure itself. */
    DLSS5NR_EXPOSURE_FIXED = 2,
};

enum dlss5nr_encoding {
    /* Linear when the frame says HDR, display encoded otherwise. */
    DLSS5NR_ENCODING_AUTO = 0,
    /* Scene-referred linear light: divided by its white point, highlights
     * rolled off, sRGB encoded on the way in. */
    DLSS5NR_ENCODING_LINEAR = 1,
    /* Already display encoded in [0, 1], as the network wants it. */
    DLSS5NR_ENCODING_DISPLAY = 2,
};

enum dlss5nr_composition {
    /* The network's answer as a luminance ratio against the game's image. */
    DLSS5NR_COMPOSITION_RATIO = 0,
    /* The network's answer as it is, laid on the frame's own proxy and
     * scaled back by the white point: for testing and comparison. */
    DLSS5NR_COMPOSITION_REPLACE = 1,
};

typedef struct dlss5nr_settings {
    uint32_t struct_size;
    /* 0 turns the effect off: dlss5nr_process then only copies. */
    int32_t enabled;
    /* The network's resolution as a fraction of the image it runs on,
     * 0.25 .. 1. Its cost is proportional to its pixels. */
    float resolution_scale;

    /* The network's own controls. */
    int32_t style;               /* 0 default, 1 natural, 2 cinematic */
    float intensity;             /* 0 .. 1 */
    float local_tone;            /* 0 .. 1 */
    float local_structure;       /* 0 .. 1 */
    float skin_structure;        /* 0 .. 1: the detail added to faces and skin; negative is
                                    automatic and follows local_structure (the network's own
                                    default). Works through the network's character mask. */

    /* How the network's answer is composed with the game's image: as a
     * luminance ratio against it, bounded both ways by max_ratio, and faded
     * out when no fresh result has arrived for max_age frames. */
    float detail_strength;       /* 0 .. 2: 0 the game's image, 1 the network's */
    float max_ratio;             /* 1 .. 8: the most a pixel is brightened or darkened */
    int32_t max_age;             /* frames */

    /* How the game's colour reaches the network's display space. */
    int32_t encoding;            /* enum dlss5nr_encoding */
    float colour_strength;       /* 0 .. 2: 0 the game's hue, 1 the network's */
    int32_t exposure_mode;       /* enum dlss5nr_exposure_mode */
    float exposure;              /* the fixed exposure, or a multiplier on the others */

    /* enum dlss5nr_composition. */
    int32_t composition;

    /* 1: a result follows the scene along the motion vectors, from the frame
     * it was captured in to the one it is composed into. 0: it is composed
     * where the network put it, lagging behind. */
    int32_t follow_motion;
    /* 1: the network keeps its own temporal history between evaluations,
     * given the motion since the previous one. 0, the default: it starts over
     * every time, and its answer for a frame is the one it gives that frame
     * alone. */
    int32_t network_history;
    /* 1: tints the pixels the network never saw, because they were hidden or
     * outside the image in the frame of the capture: green where they were
     * filled in from around them, red where not. For diagnosis. */
    int32_t show_tracking;
    /* 1: dlss5nr_present does not return until the network has answered for
     * the frame just captured. Every frame then gets a result one frame old,
     * and the frame rate is the network's. 0: the game never waits, and a
     * result is as many frames old as its evaluation took. */
    int32_t wait_for_network;
} dlss5nr_settings;

typedef struct dlss5nr_frame {
    uint32_t struct_size;
    ID3D12GraphicsCommandList *command_list;

    /* The image the network enhances, and where the result goes: same size,
     * different resources. input is read as a shader resource, output
     * written as an unordered access view. States are D3D12_RESOURCE_STATES,
     * and each resource is returned in the state it came in. */
    ID3D12Resource *input;
    uint32_t input_state;
    ID3D12Resource *output;
    uint32_t output_state;
    uint32_t width, height;            /* the region of input and output used */

    /* Depth and motion vectors, at their own resolution (usually the render
     * resolution, which is also the image's when the network runs before the
     * upscaler). Both may be null, for a host that only has the finished
     * image: the network then runs without them and without its history, and
     * its result cannot follow the scene, so it trails behind whatever moves
     * unless wait_for_network is set. */
    ID3D12Resource *depth;
    uint32_t depth_state;
    ID3D12Resource *motion;
    uint32_t motion_state;
    uint32_t depth_width, depth_height;
    uint32_t motion_width, motion_height;
    /* Motion vectors times these are pixels at motion_width x motion_height,
     * pointing from the current frame to the previous one. */
    float motion_scale_x, motion_scale_y;
    int32_t depth_inverted;

    /* Optional: a 1x1 texture with the game's exposure, and the game's
     * pre-exposure. */
    ID3D12Resource *exposure;
    uint32_t exposure_state;
    float pre_exposure;

    /* The image is scene-referred HDR. */
    int32_t hdr;
    /* A cut: history, residual and exposure start over. */
    int32_t reset;
} dlss5nr_frame;

enum dlss5nr_state {
    DLSS5NR_STATE_STARTING = 0,   /* loading, not yet running */
    DLSS5NR_STATE_RUNNING = 1,
    DLSS5NR_STATE_FAILED = 2,     /* see message; the effect stays off */
};

typedef struct dlss5nr_status {
    uint32_t struct_size;
    int32_t state;                /* enum dlss5nr_state */
    char message[256];
    uint32_t network_width, network_height;
    float evaluation_ms;          /* the network's last run, on its own thread */
    float evaluations_per_second;
    uint32_t result_age;          /* frames since the composed result was fresh */
    uint32_t native_kernels;      /* kernels the gate runs natively */
    uint32_t results;             /* network results composed so far */
    uint32_t result_latency;      /* frames between the capture of the composed
                                   * result and the frame it was first composed in */
} dlss5nr_status;

/* Starts the runtime. The network loads on a thread of its own, from the
 * host's first call into the runtime after this one: this returns at once,
 * and until the network is running dlss5nr_process copies input to output. */
DLSS5NR_API int dlss5nr_create(const dlss5nr_create_info *info, dlss5nr_context **out);

/* Stops the network thread and releases everything. The host must have
 * waited for the GPU to finish the frames that used the runtime. */
DLSS5NR_API void dlss5nr_destroy(dlss5nr_context *context);

/* Fills settings with the defaults. */
DLSS5NR_API void dlss5nr_default_settings(dlss5nr_settings *settings);

/* Takes effect from the next dlss5nr_process. */
DLSS5NR_API void dlss5nr_set_settings(dlss5nr_context *context, const dlss5nr_settings *settings);

/* Records into frame->command_list: capturing the frame for the network, when
 * the network is free to take it, and composing its latest result from input
 * into output. Returns 0 on success; on failure nothing was recorded and the
 * host should write output itself. Changes the descriptor heaps and the
 * compute state of the command list, as any upscaler does. */
DLSS5NR_API int dlss5nr_process(dlss5nr_context *context, const dlss5nr_frame *frame);

/* Tells the runtime that everything recorded for this frame has been
 * submitted to queue, which must be the queue the command lists went to.
 * Call it at Present. */
DLSS5NR_API void dlss5nr_present(dlss5nr_context *context, ID3D12CommandQueue *queue);

/* Safe from any thread. */
DLSS5NR_API void dlss5nr_get_status(dlss5nr_context *context, dlss5nr_status *status);

/* The id of the thread that loads the network. A host that intercepts library
 * loads (OptiScaler does, to stand in for nvngx.dll and nvapi64.dll) must let
 * this thread's loads through untouched: the network needs our nvngx.dll and
 * ZLUDA's nvapi64.dll, not the host's stand-ins. Safe from any thread. */
DLSS5NR_API uint32_t dlss5nr_loader_thread(dlss5nr_context *context);

#ifdef __cplusplus
}
#endif

#endif /* DLSS5NR_H */
