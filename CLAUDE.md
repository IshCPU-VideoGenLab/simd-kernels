# CLAUDE.md — simd-kernels

> Read by Claude Code at the start of every session.

---

## Project Identity

- **Org:** IshCPU-VideoGenLab
- **Repo:** simd-kernels (Phase 5 of 7)
- **Author:** Ishmael Affum Kwakye (Calyx)
- **GitHub:** calyxish
- **Institution:** University of Ghana, Legon

---

## What This Project Is

This is the **portable native kernel** phase. We write SIMD-accelerated C
kernels that compile and run on ANY modern CPU — x86 (Intel/AMD) or ARM
(Apple Silicon, Graviton, Raspberry Pi).

The key design decision: **one API, multiple backends, compile-time selection.**

```
┌─────────────────────────────────────────┐
│          simd_backend.h (API)           │
│  binary_gemm() / ssm_scan() / conv1b() │
└────────┬──────────┬──────────┬──────────┘
         │          │          │
    ┌────┴───┐ ┌────┴───┐ ┌───┴─────┐
    │  AVX2  │ │  NEON  │ │ Scalar  │
    │ x86-64 │ │ ARM64  │ │  (any)  │
    └────────┘ └────────┘ └─────────┘
```

At compile time, the preprocessor detects `__AVX2__` or `__ARM_NEON` and
selects the right backend. If neither is available, the scalar fallback
runs on literally any C compiler.

---

## Why This Design Is Better Than "avx2-kernels"

The original Phase 5 was AVX2-only. That locked it to x86. This design:

1. **Runs on the M4 MacBook** via ARM NEON backend
2. **Runs on the Pentium Gold** via AVX2 backend
3. **Runs on Raspberry Pi** via NEON backend
4. **Runs on AWS Graviton** via NEON backend
5. **Runs anywhere else** via scalar fallback
6. **Is more publishable** — "portable SIMD kernels" > "x86-only kernels"

### NEON Advantages on Apple Silicon

ARM NEON actually has some advantages over AVX2:

| Operation | AVX2 | NEON (M4) |
|-----------|------|-----------|
| Popcount | Lookup table trick (4 instructions) | `vcntq_u8` (1 native instruction) |
| Vector width | 256-bit | 128-bit (but 2 NEON units on M4) |
| FMA | `_mm256_fmadd_ps` (8 floats) | `vfmaq_f32` (4 floats, but dual-issue) |
| Memory | Separate CPU/GPU memory | Unified memory (zero-copy) |

The M4 has native popcount. We don't need the nibble lookup table hack.

---

## Hardware Targets

### Primary: Apple M4 (MacBook Air) — development + benchmarking
- ARM64 with NEON, AMX
- 10 CPU cores (4 performance + 6 efficiency)
- 16-24 GB unified memory
- Compile: `clang -O3 -march=native` (NEON auto-detected)

### Supported (CI-verified): commodity x86 with AVX2
- x86-64 with AVX2 (no AVX-512 assumed) — any modern Intel/AMD CPU
- Verified on every push by GitHub Actions (`make BACKEND=avx2 test`)
- Compile: `gcc -mavx2 -mfma -O3`

### Origin / proof-of-concept (retired): Intel Pentium Gold 7505
- x86-64 / AVX2, 2C/4T, 3.5 GHz, 16 GB DDR4 — the weakest-hardware case

### Fallback: Any CPU
- Pure C scalar implementation
- Compile: `gcc -O3` (no SIMD flags needed)

---

## Code Conventions

### C Code (csrc/)
- C11 standard
- All functions prefixed `simd_`
- Public API in `simd_backend.h` — NEVER include backend headers directly
- Backend selection via `#if defined(__AVX2__)` / `#elif defined(__ARM_NEON)`
- 32-byte aligned memory (AVX2) or 16-byte aligned (NEON) — use `simd_aligned_alloc()`
- Comments on every SIMD intrinsic explaining what it does
- Both backends tested against scalar reference

### Python Code (src/simd_kernels/)
- Python 3.9, ctypes bindings, numpy for data conversion
- Graceful fallback to PyTorch if C library not compiled

---

## File Structure

```
simd-kernels/
├── CLAUDE.md
├── README.md
├── LICENSE
├── Makefile
├── requirements.txt
├── setup.py
├── .gitignore
├── lessons.md
├── tasks/todo.md
├── .claude/{settings.json, commands/, rules/}
├── csrc/
│   ├── simd_backend.h         ← Unified API (the only header users include)
│   ├── simd_common.h          ← Shared types, alignment, detection
│   ├── backend_avx2.c         ← x86 AVX2 implementations
│   ├── backend_neon.c         ← ARM NEON implementations
│   ├── backend_scalar.c       ← Portable fallback
│   ├── binary_gemm.c          ← Dispatcher: calls the right backend
│   ├── ssm_scan.c             ← Dispatcher: calls the right backend
│   ├── conv1bit.c             ← Dispatcher: calls the right backend
│   └── test_kernels.c         ← C-level tests (run on any platform)
├── src/simd_kernels/
│   ├── __init__.py
│   ├── bindings.py            ← ctypes loader
│   ├── binary_gemm.py         ← Python wrapper
│   ├── ssm_scan.py            ← Python wrapper
│   ├── conv1bit.py            ← Python wrapper
│   ├── benchmark.py           ← PyTorch vs native benchmarks
│   └── cli.py
├── scripts/
│   ├── build.py
│   ├── check_simd.py          ← Detect available SIMD support
│   └── run_benchmark.py
├── tests/
│   ├── __init__.py
│   ├── test_binary_gemm.py
│   ├── test_ssm_scan.py
│   └── test_bindings.py
├── results/.gitkeep
└── docs/kernel_design.md
```

---

## Key Commands

```bash
# Detect SIMD capabilities
python scripts/check_simd.py

# Build (auto-detects AVX2 or NEON)
make

# Build with explicit backend
make BACKEND=neon    # Force ARM NEON
make BACKEND=avx2    # Force x86 AVX2
make BACKEND=scalar  # Force scalar fallback

# Test
make test
pytest tests/ -v

# Benchmark
python scripts/run_benchmark.py
```
