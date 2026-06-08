"""Binary GEMM wrapper — calls native SIMD or falls back to PyTorch."""

import ctypes
import logging
from typing import Tuple

import numpy as np
import torch

from simd_kernels.bindings import load_library, _ptr

logger = logging.getLogger(__name__)


def pack_weights(weight: torch.Tensor) -> Tuple[np.ndarray, float]:
    """Pack float weights into binary format. Positive→1, negative→0."""
    w = weight.float().numpy()
    out_f, in_f = w.shape
    packed = np.zeros((out_f, (in_f + 7) // 8), dtype=np.uint8)
    alpha = float(np.mean(np.abs(w)))
    for i in range(out_f):
        for j in range(in_f):
            if w[i, j] >= 0:
                packed[i, j // 8] |= (1 << (7 - (j % 8)))
    return packed, alpha


def binary_gemm(x: torch.Tensor, w_packed: np.ndarray, alpha: float, beta: float = 1.0) -> torch.Tensor:
    """Binary GEMM: uses native SIMD kernel or PyTorch fallback."""
    lib = load_library()
    orig_shape = x.shape
    x_flat = x.float().reshape(-1, x.shape[-1])
    M, K = x_flat.shape
    N = w_packed.shape[0]

    if lib is not None and K % 8 == 0:
        x_np = np.ascontiguousarray(np.packbits((x_flat.numpy() >= 0).astype(np.uint8), axis=1))
        w_np = np.ascontiguousarray(w_packed)
        y_np = np.zeros((M, N), dtype=np.float32)

        lib.simd_binary_gemm(
            _ptr(x_np, ctypes.c_uint8), _ptr(w_np, ctypes.c_uint8),
            _ptr(y_np, ctypes.c_float),
            ctypes.c_int(M), ctypes.c_int(N), ctypes.c_int(K),
            ctypes.c_float(alpha), ctypes.c_float(beta),
        )
        out_shape = list(orig_shape[:-1]) + [N]
        return torch.from_numpy(y_np).reshape(out_shape)
    else:
        # PyTorch fallback
        w_float = np.zeros((N, K), dtype=np.float32)
        for i in range(N):
            for j in range(K):
                bit = (w_packed[i, j // 8] >> (7 - (j % 8))) & 1
                w_float[i, j] = 1.0 if bit else -1.0
        y = torch.nn.functional.linear(x_flat, torch.from_numpy(w_float)) * alpha * beta
        out_shape = list(orig_shape[:-1]) + [N]
        return y.reshape(out_shape)
