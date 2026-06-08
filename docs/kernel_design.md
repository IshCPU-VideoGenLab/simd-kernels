# Kernel Design — Portable SIMD for Binary Neural Networks

## Design Principle: One API, Multiple Backends

```c
// User code — same on any platform:
simd_binary_gemm(x, w, y, M, N, K, alpha, beta);

// Compiled on x86 → calls AVX2 backend (256-bit vectors)
// Compiled on ARM → calls NEON backend (128-bit vectors)
// Compiled anywhere → scalar fallback
```

The preprocessor detects `__AVX2__` or `__ARM_NEON` at compile time.
No runtime dispatch overhead. The right code is baked in.

## Binary GEMM: XNOR + Popcount

### The Operation

```
y[m][n] = (K - 2 * popcount(xor(x_bits[m], w_bits[n]))) * alpha * beta
```

### AVX2 Implementation (x86)

256 bits per instruction. Popcount via nibble lookup table (4 instructions):

```c
__m256i xor = _mm256_xor_si256(xv, wv);        // 256-bit XOR
// popcount via shuffle lookup:
__m256i lo = _mm256_shuffle_epi8(lut, xor & mask);
__m256i hi = _mm256_shuffle_epi8(lut, xor >> 4);
__m256i cnt = _mm256_add_epi8(lo, hi);
```

### NEON Implementation (ARM)

128 bits per instruction. **Native popcount** — simpler and arguably
more elegant than AVX2:

```c
uint8x16_t xor = veorq_u8(xv, wv);    // 128-bit XOR
uint8x16_t cnt = vcntq_u8(xor);        // Native popcount per byte!
int total = vaddlvq_u8(cnt);            // Horizontal sum
```

ARM's `vcntq_u8` does in 1 instruction what AVX2 needs 4 for.

### Performance Characteristics

| Aspect | AVX2 (Pentium Gold) | NEON (M4) |
|--------|-------------------|-----------|
| Vector width | 256 bits | 128 bits |
| Bits per XOR | 256 | 128 |
| Popcount method | Lookup table | Native `vcntq_u8` |
| Instructions per 256 bits | ~5 | ~6 (2 × 128-bit passes) |
| Execution units | 1 SIMD port | 2 NEON units (dual-issue) |
| Memory bandwidth | ~25 GB/s (single ch.) | ~100+ GB/s (unified) |

The M4 processes half the bits per instruction but has wider execution
resources and far higher memory bandwidth. Net throughput is comparable
or better.

## SSM Scan: Vectorized State Update

The scan is sequential across time (h[t] depends on h[t-1]).
Vectorization happens across the state dimension d_state.

### With d_state = 16:
- AVX2: 2 iterations of 8 floats = 16 elements ✓
- NEON: 4 iterations of 4 floats = 16 elements ✓

Both use fused multiply-add (FMA) for the state update:
```
h[d][s] = A[t][d][s] * h[d][s] + B[t][d][s] * x[t][d]
```

AVX2: `_mm256_fmadd_ps` (1 instruction, 8 floats)
NEON: `vfmaq_f32` (1 instruction, 4 floats)

## Why This Design Is Better Than AVX2-Only

1. **Portability** — runs on x86, ARM, and anything else
2. **Better paper** — "portable SIMD" > "x86-only"
3. **More hardware** — Apple Silicon, AWS Graviton, edge devices
4. **NEON advantages** — native popcount, unified memory
5. **Future-proof** — add SVE2, RISC-V V extensions as new backends
