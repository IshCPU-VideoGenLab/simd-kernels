/**
 * simd_backend.h — Portable SIMD kernel API.
 *
 * This is the ONLY header external code should include.
 * Backend selection (AVX2 / NEON / scalar) happens automatically
 * at compile time based on preprocessor detection.
 *
 * Part of IshCPU-VideoGenLab/simd-kernels
 * Author: Ishmael Affum Kwakye (Calyx)
 */

#ifndef SIMD_BACKEND_H
#define SIMD_BACKEND_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * Backend Detection
 * ================================================================ */

typedef enum {
    SIMD_BACKEND_SCALAR = 0,
    SIMD_BACKEND_AVX2   = 1,
    SIMD_BACKEND_NEON   = 2,
} SimdBackend;

/** Return which SIMD backend is active. */
SimdBackend simd_get_backend(void);

/** Return a human-readable backend name. */
const char* simd_get_backend_name(void);

/** Return 1 if the CPU supports the compiled backend at runtime. */
int simd_is_supported(void);

/* ================================================================
 * Memory Utilities
 * ================================================================ */

/** Allocate memory aligned for the active SIMD backend.
 *  AVX2: 32-byte aligned. NEON: 16-byte aligned. Scalar: 16-byte. */
void* simd_aligned_alloc(size_t size);

/** Free memory from simd_aligned_alloc. */
void  simd_aligned_free(void* ptr);

/* ================================================================
 * Binary GEMM: y = xnor_popcount(x_packed, w_packed) * alpha * beta
 *
 * The core operation for 1-bit neural network inference.
 * Both x and w are packed binary: 8 values per uint8 byte.
 * ================================================================ */

/**
 * Binary matrix multiplication via XNOR + popcount.
 *
 * @param x_packed  Packed activations, shape (M, K/8), row-major uint8
 * @param w_packed  Packed weights, shape (N, K/8), row-major uint8
 * @param y_out     Output float array, shape (M, N), pre-allocated
 * @param M         Rows of x (batch dimension)
 * @param N         Rows of w (output features)
 * @param K         Original feature dimension (must be multiple of 8)
 * @param alpha     Weight scale factor
 * @param beta      Activation scale factor
 */
void simd_binary_gemm(
    const uint8_t* x_packed,
    const uint8_t* w_packed,
    float* y_out,
    int M, int N, int K,
    float alpha, float beta
);

/* ================================================================
 * SSM Scan: h[t] = A[t]*h[t-1] + B[t]*x[t], y[t] = C[t]·h[t]
 *
 * Vectorized across d_state dimension.
 * Sequential across time (inherent dependency).
 * ================================================================ */

/**
 * Selective SSM sequential scan.
 *
 * @param x       Input, shape (seq_len, d_inner), row-major float32
 * @param A_bar   Discretized A, shape (seq_len, d_inner, d_state), float32
 * @param B_bar   Discretized B, shape (seq_len, d_inner, d_state), float32
 * @param C       Output matrix, shape (seq_len, d_state), float32
 * @param y_out   Output, shape (seq_len, d_inner), pre-allocated float32
 * @param seq_len Sequence length
 * @param d_inner Inner dimension
 * @param d_state SSM state dimension
 */
void simd_ssm_scan(
    const float* x,
    const float* A_bar,
    const float* B_bar,
    const float* C,
    float* y_out,
    int seq_len, int d_inner, int d_state
);

/* ================================================================
 * Bit Packing Utilities
 * ================================================================ */

/** Pack float weights to binary: positive → 1, negative → 0.
 *  Also computes alpha = mean(|weights|). */
void simd_pack_weights(
    const float* weights, uint8_t* packed_out,
    float* alpha_out, size_t n
);

/** Unpack binary weights to float {-1, +1}. */
void simd_unpack_weights(
    const uint8_t* packed, float* weights_out, size_t n
);

/** Popcount of a byte array. */
int simd_popcount(const uint8_t* data, size_t len);

#ifdef __cplusplus
}
#endif

#endif /* SIMD_BACKEND_H */
