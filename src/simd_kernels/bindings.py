"""ctypes bindings for the portable SIMD kernel library."""

import ctypes
import logging
import os
import platform
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)
_lib: Optional[ctypes.CDLL] = None


def _find_library() -> Optional[str]:
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(pkg_dir))
    system = platform.system()

    if system == "Darwin":
        names = ["libsimdkernels.dylib", "libsimdkernels.so"]
    elif system == "Windows":
        names = ["simdkernels.dll"]
    else:
        names = ["libsimdkernels.so"]

    for name in names:
        for directory in [pkg_dir, os.path.join(repo_root, "build"), repo_root]:
            path = os.path.join(directory, name)
            if os.path.exists(path):
                return path
    return None


def load_library() -> Optional[ctypes.CDLL]:
    global _lib
    if _lib is not None:
        return _lib

    path = _find_library()
    if path is None:
        logger.warning("SIMD library not found. Run 'make' to compile. Using PyTorch fallback.")
        return None

    try:
        _lib = ctypes.CDLL(path)
        _lib.simd_get_backend_name.restype = ctypes.c_char_p
        _lib.simd_get_backend.restype = ctypes.c_int
        _lib.simd_is_supported.restype = ctypes.c_int

        _lib.simd_binary_gemm.restype = None
        _lib.simd_binary_gemm.argtypes = [
            ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_float, ctypes.c_float,
        ]

        _lib.simd_ssm_scan.restype = None
        _lib.simd_ssm_scan.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]

        _lib.simd_popcount.restype = ctypes.c_int
        _lib.simd_popcount.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]

        backend = _lib.simd_get_backend_name().decode()
        logger.info("Loaded SIMD library: %s (backend: %s)", path, backend)
        return _lib
    except OSError as e:
        logger.error("Failed to load SIMD library: %s", e)
        return None


def is_available() -> bool:
    lib = load_library()
    return lib is not None and bool(lib.simd_is_supported())


def get_backend_name() -> str:
    lib = load_library()
    if lib is None:
        return "none (not compiled)"
    return lib.simd_get_backend_name().decode()


def _ptr(arr: np.ndarray, dtype):
    return arr.ctypes.data_as(ctypes.POINTER(dtype))
