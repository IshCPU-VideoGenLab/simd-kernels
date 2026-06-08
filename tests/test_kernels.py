"""Tests for simd_kernels — run on any platform."""
import pytest
import torch
import numpy as np
from simd_kernels.binary_gemm import pack_weights, binary_gemm
from simd_kernels.ssm_scan import simd_ssm_scan
from simd_kernels.bindings import get_backend_name


class TestPackWeights:
    def test_shape(self):
        w = torch.randn(64, 256)
        packed, alpha = pack_weights(w)
        assert packed.shape == (64, 32)
        assert alpha > 0

    def test_sign_preserved(self):
        w = torch.tensor([[1.0, -1.0, 0.5, -0.3, 0.1, -0.2, 0.7, -0.8]])
        packed, _ = pack_weights(w)
        assert packed[0, 0] == 0xAA


class TestBinaryGemm:
    def test_output_shape(self):
        x = torch.randn(4, 64)
        w = torch.randn(16, 64)
        packed, alpha = pack_weights(w)
        y = binary_gemm(x, packed, alpha)
        assert y.shape == (4, 16)

    def test_no_nan(self):
        x = torch.randn(4, 128)
        w = torch.randn(32, 128)
        packed, alpha = pack_weights(w)
        y = binary_gemm(x, packed, alpha)
        assert not torch.isnan(y).any()

    def test_deterministic(self):
        x = torch.randn(2, 64)
        w = torch.randn(8, 64)
        packed, alpha = pack_weights(w)
        y1 = binary_gemm(x, packed, alpha)
        y2 = binary_gemm(x, packed, alpha)
        assert torch.allclose(y1, y2)


class TestSSMScan:
    def test_output_shape(self):
        S, D, N = 8, 16, 8
        y = simd_ssm_scan(torch.randn(S, D),
                          torch.ones(S, D, N) * 0.9,
                          torch.randn(S, D, N) * 0.1,
                          torch.randn(S, N))
        assert y.shape == (S, D)

    def test_matches_pytorch(self):
        S, D, N = 8, 4, 4
        x = torch.randn(S, D)
        A = torch.ones(S, D, N) * 0.5
        B = torch.randn(S, D, N) * 0.1
        C = torch.randn(S, N) * 0.1

        # PyTorch reference
        h = torch.zeros(D, N)
        ref = []
        for t in range(S):
            h = A[t] * h + B[t] * x[t].unsqueeze(-1)
            ref.append(torch.einsum("n,dn->d", C[t], h))
        ref = torch.stack(ref)

        result = simd_ssm_scan(x, A, B, C)
        assert torch.allclose(ref, result, atol=1e-4)


class TestBackendInfo:
    def test_returns_string(self):
        name = get_backend_name()
        assert isinstance(name, str)
        assert len(name) > 0
