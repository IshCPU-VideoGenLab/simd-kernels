# Phase 5 — simd-kernels Task Roadmap

---

## Milestone 1: Build System & Detection
- [ ] Makefile auto-detects x86 vs ARM and selects backend
- [ ] `check_simd.py` reports detected SIMD capabilities
- [ ] Library compiles on M4 MacBook (NEON)
- [ ] Library compiles on Pentium Gold (AVX2)
- [ ] Library compiles with `BACKEND=scalar` on any platform
- [ ] `make test` passes on all platforms

## Milestone 2: Binary GEMM
- [ ] NEON backend: `veorq_u8` + `vcntq_u8` binary GEMM
- [ ] AVX2 backend: `_mm256_xor_si256` + nibble LUT binary GEMM
- [ ] Scalar reference implementation
- [ ] C test: NEON/AVX2 output matches scalar exactly
- [ ] Python binding via ctypes
- [ ] Python test: output matches PyTorch `F.linear` on binary weights
- [ ] Benchmark: GOPS at various matrix sizes on M4 and Pentium Gold

## Milestone 3: SSM Scan
- [ ] NEON backend: `vfmaq_f32` vectorized state update
- [ ] AVX2 backend: `_mm256_fmadd_ps` vectorized state update
- [ ] Scalar reference
- [ ] C test: SIMD matches scalar
- [ ] Python binding and test
- [ ] Benchmark: speedup vs PyTorch loop

## Milestone 4: Integration
- [ ] Replace Phase 4 BitLinear forward with `simd_binary_gemm`
- [ ] Replace Phase 2 MambaBlock scan with `simd_ssm_scan`
- [ ] End-to-end benchmark on both platforms
- [ ] Produce cross-platform benchmark table for paper

## Milestone 5: Documentation
- [ ] docs/kernel_design.md with backend comparison
- [ ] README with benchmark results from both platforms
- [ ] Tag v0.1.0
