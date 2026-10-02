#pragma once

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdint.h>

#define HIPAPI __declspec(dllexport)

/* What a HIP call returns when it could not be made at all: hipErrorNotInitialized. */
#define DLSSNR_HIP_NOT_INITIALIZED 3

/* Hands one call to the native side: the number of the function in the
 * generated enumeration, and its argument structure. Returns 0 when the native
 * side cannot be reached. */
int dlssnr_call(int code, void *args);
