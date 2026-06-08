"""SSM scan wrapper — calls native SIMD or falls back to PyTorch."""

import ctypes
import logging
import numpy as np
import torch
from simd_kernels.bindings import load_library, _ptr

logger = logging.getLogger(__name__)


def simd_ssm_scan(x, A_bar, B_bar, C):
    """SSM scan using native SIMD kernel with PyTorch fallback."""
    lib = load_library()
    seq_len, d_inner = x.shape
    d_state = C.shape[-1]

    if lib is not None:
        x_np = np.ascontiguousarray(x.float().numpy())
        A_np = np.ascontiguousarray(A_bar.float().numpy())
        B_np = np.ascontiguousarray(B_bar.float().numpy())
        C_np = np.ascontiguousarray(C.float().numpy())
        y_np = np.zeros((seq_len, d_inner), dtype=np.float32)

        lib.simd_ssm_scan(
            _ptr(x_np, ctypes.c_float), _ptr(A_np, ctypes.c_float),
            _ptr(B_np, ctypes.c_float), _ptr(C_np, ctypes.c_float),
            _ptr(y_np, ctypes.c_float),
            ctypes.c_int(seq_len), ctypes.c_int(d_inner), ctypes.c_int(d_state),
        )
        return torch.from_numpy(y_np)
    else:
        h = torch.zeros(d_inner, d_state)
        outputs = []
        for t in range(seq_len):
            h = A_bar[t] * h + B_bar[t] * x[t].unsqueeze(-1)
            outputs.append(torch.einsum("n,dn->d", C[t], h))
        return torch.stack(outputs)
