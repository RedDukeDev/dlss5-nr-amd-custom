#pragma once

#include <stdint.h>

#define DLSSNR_HIP_NOT_INITIALIZED 3
#define DLSSNR_HIP_NOT_SUPPORTED 801

/* One native function the Windows side can call: it gets the pointer to the
 * argument structure of its HIP function. */
typedef int32_t (*dlssnr_entry)(void *args);

/* What the Windows side's handle points at: one entry per forwarded function, in
 * the order of the generated enumeration (generated/thunks.c). */
extern const dlssnr_entry dlssnr_funcs[];

/* The real function behind entry `index`, from the HIP library, loaded when the
 * first one is asked for; null when the library has no such function. */
void *dlssnr_real(int index);

/* Names every call, with the arguments of the few worth reading, when
 * DLSSNR_BRIDGE_TRACE is set. `args` is the argument structure of the call. */
void dlssnr_trace_args(int index, const void *args);

/* Diagnostics, when DLSSNR_BRIDGE_LOG is set. */
void dlssnr_log(const char *format, ...);
