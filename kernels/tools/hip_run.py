#!/usr/bin/env python3
"""Run a native code object on the GPU through HIP, from Python.

The counterpart of zluda_run.py for kernels we write ourselves: load the .hsaco
build_kernel.py produces, hand it buffers of our choosing, launch, read back,
and time it with HIP events. No translation layer in between, so what comes out
is the native kernel's own behaviour, comparable byte for byte with a capture
of the translated one.
"""
import ctypes
import os

import numpy as np

DEFAULT_RUNTIME = r"C:\Windows\System32\amdhip64_7.dll"


class Hip:
    def __init__(self, runtime=DEFAULT_RUNTIME):
        self.hip = ctypes.CDLL(runtime)
        h = self.hip
        h.hipModuleLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
        h.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        h.hipMemcpyHtoD.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
        h.hipMemcpyDtoH.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
        h.hipMemset.argtypes = [ctypes.c_uint64, ctypes.c_int, ctypes.c_size_t]
        h.hipFree.argtypes = [ctypes.c_uint64]
        h.hipEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]
        h.hipEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        h.hipEventSynchronize.argtypes = [ctypes.c_void_p]
        self.check(h.hipInit(0), "hipInit")
        self.check(h.hipSetDevice(0), "hipSetDevice")

    @staticmethod
    def check(result, what):
        if result != 0:
            raise RuntimeError("%s -> %d" % (what, result))

    def module(self, path, name):
        image = open(path, "rb").read()
        self._image = ctypes.create_string_buffer(image, len(image))
        mod = ctypes.c_void_p()
        self.check(self.hip.hipModuleLoadData(ctypes.byref(mod), self._image), "hipModuleLoadData")
        fn = ctypes.c_void_p()
        self.check(self.hip.hipModuleGetFunction(ctypes.byref(fn), mod, name.encode()),
                   "hipModuleGetFunction(%s)" % name)
        return fn

    def alloc(self, nbytes):
        ptr = ctypes.c_uint64()
        self.check(self.hip.hipMalloc(ctypes.byref(ptr), nbytes), "hipMalloc")
        return ptr.value

    def upload(self, array):
        data = np.ascontiguousarray(array)
        ptr = self.alloc(data.nbytes)
        self.check(self.hip.hipMemcpyHtoD(ptr, data.ctypes.data, data.nbytes), "HtoD")
        return ptr

    def write(self, ptr, array):
        data = np.ascontiguousarray(array)
        self.check(self.hip.hipMemcpyHtoD(ptr, data.ctypes.data, data.nbytes), "HtoD")

    def download(self, ptr, nbytes):
        out = np.zeros(nbytes, dtype=np.uint8)
        self.check(self.hip.hipMemcpyDtoH(out.ctypes.data, ptr, nbytes), "DtoH")
        return out

    def launch(self, fn, grid, block, param_bytes, shared=0, sync=True):
        """param_bytes is the kernel's whole argument buffer, laid out as the
        kernel expects it: one struct passed by value, or several arguments
        packed with their alignment. It goes through HIP_LAUNCH_PARAM_BUFFER_*,
        which takes either shape."""
        blob = ctypes.create_string_buffer(bytes(param_bytes), len(param_bytes))
        size = ctypes.c_size_t(len(param_bytes))
        extra = (ctypes.c_void_p * 5)(1, ctypes.cast(blob, ctypes.c_void_p).value,
                                      2, ctypes.cast(ctypes.byref(size), ctypes.c_void_p).value, 3)
        self.check(self.hip.hipModuleLaunchKernel(fn, *grid, *block, shared, None, None, extra),
                   "hipModuleLaunchKernel")
        if sync:
            self.check(self.hip.hipDeviceSynchronize(), "hipDeviceSynchronize")

    def time(self, fn, grid, block, param_bytes, repeat=50, before=None):
        """Average milliseconds per launch over `repeat` launches, by HIP events.
        `before` runs ahead of every launch outside the timed region is not
        possible with events on one stream, so it is timed along; keep it cheap."""
        start, stop = ctypes.c_void_p(), ctypes.c_void_p()
        self.check(self.hip.hipEventCreate(ctypes.byref(start)), "hipEventCreate")
        self.check(self.hip.hipEventCreate(ctypes.byref(stop)), "hipEventCreate")
        self.launch(fn, grid, block, param_bytes)
        self.check(self.hip.hipEventRecord(start, None), "hipEventRecord")
        for _ in range(repeat):
            if before:
                before()
            self.launch(fn, grid, block, param_bytes, sync=False)
        self.check(self.hip.hipEventRecord(stop, None), "hipEventRecord")
        self.check(self.hip.hipEventSynchronize(stop), "hipEventSynchronize")
        ms = ctypes.c_float()
        self.check(self.hip.hipEventElapsedTime(ctypes.byref(ms), start, stop), "hipEventElapsedTime")
        return ms.value / repeat


if __name__ == "__main__":
    h = Hip()
    print("HIP runtime up:", os.path.basename(DEFAULT_RUNTIME))
