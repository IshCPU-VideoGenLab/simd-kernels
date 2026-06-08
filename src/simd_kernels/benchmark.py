"""Benchmark PyTorch vs native SIMD kernels."""

import gc, logging, time
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import torch
from simd_kernels.binary_gemm import pack_weights, binary_gemm
from simd_kernels.bindings import get_backend_name

logger = logging.getLogger(__name__)


@dataclass
class BenchResult:
    name: str
    size: str
    pytorch_ms: float
    simd_ms: float
    backend: str

    @property
    def speedup(self) -> float:
        return self.pytorch_ms / self.simd_ms if self.simd_ms > 0 else 0.0


def benchmark_binary_gemm(sizes=None, warmup=3, steps=10) -> List[BenchResult]:
    if sizes is None:
        sizes = [(64, 128, 256), (128, 256, 512), (256, 512, 1024)]
    results = []
    backend = get_backend_name()

    for M, N, K in sizes:
        x = torch.randn(M, K)
        w = torch.randn(N, K)
        w_packed, alpha = pack_weights(w)
        w_bin = w.sign(); w_bin[w_bin == 0] = 1.0

        # PyTorch baseline
        for _ in range(warmup):
            torch.nn.functional.linear(x, w_bin) * alpha
        pt_times = []
        for _ in range(steps):
            gc.collect(); s = time.perf_counter()
            torch.nn.functional.linear(x, w_bin) * alpha
            pt_times.append((time.perf_counter() - s) * 1000)

        # SIMD kernel
        for _ in range(warmup):
            binary_gemm(x, w_packed, alpha)
        simd_times = []
        for _ in range(steps):
            gc.collect(); s = time.perf_counter()
            binary_gemm(x, w_packed, alpha)
            simd_times.append((time.perf_counter() - s) * 1000)

        results.append(BenchResult(
            name="binary_gemm", size=f"{M}x{N}x{K}",
            pytorch_ms=sum(pt_times)/len(pt_times),
            simd_ms=sum(simd_times)/len(simd_times),
            backend=backend,
        ))
    return results


def format_results(results: List[BenchResult]) -> str:
    lines = ["", "=" * 70, f"  SIMD Kernel Benchmarks (backend: {results[0].backend if results else '?'})",
             "=" * 70, "",
             f"  {'Kernel':<15} {'Size':<15} {'PyTorch':>10} {'SIMD':>10} {'Speedup':>8}",
             "  " + "-" * 60]
    for r in results:
        lines.append(f"  {r.name:<15} {r.size:<15} {r.pytorch_ms:>9.2f}ms {r.simd_ms:>9.2f}ms {r.speedup:>7.2f}x")
    lines.extend(["  " + "-" * 60, "", "=" * 70, ""])
    return "\n".join(lines)
