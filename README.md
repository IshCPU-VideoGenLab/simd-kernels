<p align="center">
  <img src="https://raw.githubusercontent.com/IshCPU-VideoGenLab/.github/main/logo.svg" alt="IshCPU-VideoGenLab" width="80">
</p>

# simd-kernels

[![CI](https://github.com/IshCPU-VideoGenLab/simd-kernels/actions/workflows/ci.yml/badge.svg)](https://github.com/IshCPU-VideoGenLab/simd-kernels/actions/workflows/ci.yml)

**Portable SIMD kernels for binary neural network inference — one API, runs on x86 (AVX2) and ARM (NEON).**

Part of [IshCPU-VideoGenLab](https://github.com/IshCPU-VideoGenLab).

---

## Architecture

```
┌──────────────────────────────────┐
│     simd_backend.h (public API)  │
│  binary_gemm / ssm_scan / pack  │
└──────┬──────────┬──────────┬─────┘
       │          │          │
  ┌────┴───┐ ┌───┴────┐ ┌───┴─────┐
  │  AVX2  │ │  NEON  │ │ Scalar  │
  │ x86-64 │ │ ARM64  │ │  (any)  │
  │ 256-bit│ │ 128-bit│ │ no SIMD │
  └────────┘ └────────┘ └─────────┘
```

Compile-time detection. `make` auto-selects the right backend. NEON has **native popcount** (`vcntq_u8`) — faster than AVX2's lookup table trick.

## Quick Start

```bash
python scripts/check_simd.py   # What does your CPU support?
make                           # Auto-detect and build
make test                      # Verify correctness
pytest tests/ -v               # Python-level tests
python scripts/run_benchmark.py
```

## Tested Platforms

| Platform | CPU | Backend | Status |
|----------|-----|---------|--------|
| MacBook Air | Apple M4 | NEON | Primary dev |
| HP Laptop 17 | Pentium Gold 7505 | AVX2 | Paper benchmark |
| Raspberry Pi 4 | Cortex-A72 | NEON | Tested |
| Any | Any | Scalar | Always works |

## Contributing

See the [Contributing Guide](https://github.com/IshCPU-VideoGenLab/.github/blob/main/CONTRIBUTING.md)
and [Version Control Guide](https://github.com/IshCPU-VideoGenLab/.github/blob/main/VERSION_CONTROL_GUIDE.md).

---

## License

MIT.
