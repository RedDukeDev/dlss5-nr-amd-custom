#!/usr/bin/env python3
"""Run one PTX kernel on the GPU through ZLUDA, from Python, on inputs of our own.

A capture only ever shows what a whole kernel produced from the network's data.
To pin down how one instruction behaves -- here, how ZLUDA's mma.sync rounds --
it has to be run on its own, on inputs chosen to make the question answerable,
and its result compared element by element with ptxsim's. This loads ZLUDA's
nvcuda.dll by path through ctypes and drives it with the plain driver API:
module from PTX text, buffers, one launch, results back.

The kernel takes one parameter, a struct of pointers, the way the network's own
kernels do, so ptxsim runs exactly the same text.
"""
import ctypes
import os

import numpy as np
import workspace  # noqa: E402

DEFAULT_DRIVER = workspace.zluda()


class _Array3DDesc(ctypes.Structure):
    _fields_ = [("Width", ctypes.c_size_t), ("Height", ctypes.c_size_t), ("Depth", ctypes.c_size_t),
                ("Format", ctypes.c_uint), ("NumChannels", ctypes.c_uint), ("Flags", ctypes.c_uint)]


class _Memcpy2D(ctypes.Structure):
    _fields_ = [("srcXInBytes", ctypes.c_size_t), ("srcY", ctypes.c_size_t), ("srcMemoryType", ctypes.c_uint),
                ("srcHost", ctypes.c_void_p), ("srcDevice", ctypes.c_uint64), ("srcArray", ctypes.c_void_p),
                ("srcPitch", ctypes.c_size_t), ("dstXInBytes", ctypes.c_size_t), ("dstY", ctypes.c_size_t),
                ("dstMemoryType", ctypes.c_uint), ("dstHost", ctypes.c_void_p), ("dstDevice", ctypes.c_uint64),
                ("dstArray", ctypes.c_void_p), ("dstPitch", ctypes.c_size_t),
                ("WidthInBytes", ctypes.c_size_t), ("Height", ctypes.c_size_t)]


class _ResourceDesc(ctypes.Structure):
    _fields_ = [("resType", ctypes.c_int), ("hArray", ctypes.c_void_p), ("reserved", ctypes.c_int * 30),
                ("flags", ctypes.c_uint)]


class _TextureDesc(ctypes.Structure):
    _fields_ = [("addressMode", ctypes.c_int * 3), ("filterMode", ctypes.c_int), ("flags", ctypes.c_uint),
                ("maxAnisotropy", ctypes.c_uint), ("mipmapFilterMode", ctypes.c_int),
                ("mipmapLevelBias", ctypes.c_float), ("minMipmapLevelClamp", ctypes.c_float),
                ("maxMipmapLevelClamp", ctypes.c_float), ("borderColor", ctypes.c_float * 4),
                ("reserved", ctypes.c_int * 12)]


class Zluda:
    def __init__(self, driver=DEFAULT_DRIVER):
        directory = os.path.dirname(os.path.abspath(driver))
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(directory)
        self.cu = ctypes.CDLL(os.path.abspath(driver), winmode=0)
        self.check(self.cu.cuInit(0), "cuInit")
        dev = ctypes.c_int()
        self.check(self.cu.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
        self.ctx = ctypes.c_void_p()
        create = getattr(self.cu, "cuCtxCreate_v2", None) or self.cu.cuCtxCreate
        self.check(create(ctypes.byref(self.ctx), 0, dev), "cuCtxCreate")
        self.cu.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        self.cu.cuMemcpyHtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
        self.cu.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
        self.cu.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]

    @staticmethod
    def check(result, what):
        if result != 0:
            raise RuntimeError("%s -> %d" % (what, result))

    def module(self, ptx_text, name):
        mod = ctypes.c_void_p()
        image = ctypes.create_string_buffer(ptx_text.encode("utf-8") + b"\0")
        self.check(self.cu.cuModuleLoadData(ctypes.byref(mod), image), "cuModuleLoadData")
        fn = ctypes.c_void_p()
        self.check(self.cu.cuModuleGetFunction(ctypes.byref(fn), mod, name.encode()),
                   "cuModuleGetFunction")
        return fn

    def upload(self, array):
        data = np.ascontiguousarray(array)
        ptr = ctypes.c_uint64()
        self.check(self.cu.cuMemAlloc_v2(ctypes.byref(ptr), data.nbytes), "cuMemAlloc")
        self.check(self.cu.cuMemcpyHtoD_v2(ptr.value, data.ctypes.data, data.nbytes), "HtoD")
        return ptr.value

    def download(self, ptr, nbytes):
        out = np.zeros(nbytes, dtype=np.uint8)
        self.check(self.cu.cuMemcpyDtoH_v2(out.ctypes.data, ptr, nbytes), "DtoH")
        return out

    # ---------------------------------------------------------- images
    # A 2D array of float16 or float32 texels, 1, 2 or 4 channels, as the
    # driver API makes one, optionally usable as a surface.
    def array2d(self, texels, surface=False):
        data = np.ascontiguousarray(texels)
        height, width = data.shape[:2]
        channels = 1 if data.ndim == 2 else data.shape[2]
        fmt = {np.dtype(np.float16): 0x10, np.dtype(np.float32): 0x20}[data.dtype]
        desc = _Array3DDesc(width, height, 0, fmt, channels, 0x02 if surface else 0)
        arr = ctypes.c_void_p()
        self.check(self.cu.cuArray3DCreate_v2(ctypes.byref(arr), ctypes.byref(desc)), "cuArray3DCreate")
        row = width * channels * data.dtype.itemsize
        copy = _Memcpy2D()
        copy.srcMemoryType, copy.srcHost, copy.srcPitch = 1, data.ctypes.data, row   # host
        copy.dstMemoryType, copy.dstArray = 3, arr                                  # array
        copy.WidthInBytes, copy.Height = row, height
        self.check(self.cu.cuMemcpy2D_v2(ctypes.byref(copy)), "cuMemcpy2D")
        return arr

    def read_array2d(self, arr, shape, dtype):
        out = np.zeros(shape, dtype)
        copy = _Memcpy2D()
        row = shape[1] * (shape[2] if len(shape) > 2 else 1) * np.dtype(dtype).itemsize
        copy.srcMemoryType, copy.srcArray = 3, arr
        copy.dstMemoryType, copy.dstHost, copy.dstPitch = 1, out.ctypes.data, row
        copy.WidthInBytes, copy.Height = row, shape[0]
        self.check(self.cu.cuMemcpy2D_v2(ctypes.byref(copy)), "cuMemcpy2D")
        return out

    def texture(self, arr, linear=True, normalized=True, clamp=True):
        res = _ResourceDesc()
        res.resType, res.hArray = 0, arr
        tex = _TextureDesc()
        tex.addressMode[0] = tex.addressMode[1] = tex.addressMode[2] = 1 if clamp else 0
        tex.filterMode = 1 if linear else 0
        tex.flags = 2 if normalized else 0
        obj = ctypes.c_uint64()
        self.check(self.cu.cuTexObjectCreate(ctypes.byref(obj), ctypes.byref(res), ctypes.byref(tex), None),
                   "cuTexObjectCreate")
        return obj.value

    def surface(self, arr):
        res = _ResourceDesc()
        res.resType, res.hArray = 0, arr
        obj = ctypes.c_uint64()
        self.check(self.cu.cuSurfObjectCreate(ctypes.byref(obj), ctypes.byref(res)), "cuSurfObjectCreate")
        return obj.value

    def launch(self, fn, grid, block, param_bytes):
        blob = ctypes.create_string_buffer(bytes(param_bytes), len(param_bytes))
        params = (ctypes.c_void_p * 1)(ctypes.cast(blob, ctypes.c_void_p))
        self.check(self.cu.cuLaunchKernel(fn, *grid, *block, 0, None, params, None), "cuLaunchKernel")
        self.check(self.cu.cuCtxSynchronize(), "cuCtxSynchronize")
