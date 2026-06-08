"""1-bit convolution — PyTorch fallback (SIMD conv via im2col+GEMM planned)."""
import torch

def conv1bit_fallback(x, weight_binary, alpha, stride=1, padding=0):
    """1-bit conv using PyTorch. SIMD version uses im2col + binary_gemm."""
    return torch.nn.functional.conv2d(x, weight_binary, None, stride, padding) * alpha
