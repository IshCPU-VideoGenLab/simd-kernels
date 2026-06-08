/**
 * binary_gemm.c + ssm_scan.c + conv1bit.c — Dispatchers.
 *
 * These are the public API implementations. Each function checks
 * which backend was compiled and calls the appropriate implementation.
 * This file compiles on ALL platforms.
 */

#include "simd_common.h"
#include "simd_backend.h"
#include <string.h>
#include <math.h>

/* ================================================================
 * Forward declarations for backend-specific functions
 * ================================================================ */

/* NEON backend */
#ifdef SIMD_USE_NEON
extern int  simd_popcount_neon(const uint8_t*, size_t);
extern void simd_binary_gemm_neon(const uint8_t*, const uint8_t*, float*, int, int, int, float, float);
extern void simd_ssm_scan_neon(const float*, const float*, const float*, const float*, float*, int, int, int);
#endif

/* AVX2 backend */
#ifdef SIMD_USE_AVX2
extern int  simd_popcount_avx2(const uint8_t*, size_t);
extern void simd_binary_gemm_avx2(const uint8_t*, const uint8_t*, float*, int, int, int, float, float);
extern void simd_ssm_scan_avx2(const float*, const float*, const float*, const float*, float*, int, int, int);
#endif

/* Scalar backend (always available) */
extern int  simd_popcount_scalar(const uint8_t*, size_t);
extern void simd_binary_gemm_scalar(const uint8_t*, const uint8_t*, float*, int, int, int, float, float);
extern void simd_ssm_scan_scalar(const float*, const float*, const float*, const float*, float*, int, int, int);

/* ================================================================
 * Backend Detection
 * ================================================================ */

SimdBackend simd_get_backend(void) {
#if defined(SIMD_USE_AVX2)
    return SIMD_BACKEND_AVX2;
#elif defined(SIMD_USE_NEON)
    return SIMD_BACKEND_NEON;
#else
    return SIMD_BACKEND_SCALAR;
#endif
}

const char* simd_get_backend_name(void) {
#if defined(SIMD_USE_AVX2)
    return "AVX2 (x86-64)";
#elif defined(SIMD_USE_NEON)
    return "NEON (ARM64)";
#else
    return "Scalar (portable)";
#endif
}

int simd_is_supported(void) {
    /* If it compiled with SIMD flags, it's supported */
#if defined(SIMD_USE_AVX2) || defined(SIMD_USE_NEON)
    return 1;
#else
    return 1;  /* Scalar always works */
#endif
}

/* ================================================================
 * Memory
 * ================================================================ */

void* simd_aligned_alloc(size_t size) {
    return simd_alloc_impl(size);
}

void simd_aligned_free(void* ptr) {
    simd_free_impl(ptr);
}

/* ================================================================
 * Dispatchers — call the right backend
 * ================================================================ */

int simd_popcount(const uint8_t* data, size_t len) {
#if defined(SIMD_USE_NEON)
    return simd_popcount_neon(data, len);
#elif defined(SIMD_USE_AVX2)
    return simd_popcount_avx2(data, len);
#else
    return simd_popcount_scalar(data, len);
#endif
}

void simd_binary_gemm(
    const uint8_t* x_packed, const uint8_t* w_packed,
    float* y_out, int M, int N, int K,
    float alpha, float beta
) {
#if defined(SIMD_USE_NEON)
    simd_binary_gemm_neon(x_packed, w_packed, y_out, M, N, K, alpha, beta);
#elif defined(SIMD_USE_AVX2)
    simd_binary_gemm_avx2(x_packed, w_packed, y_out, M, N, K, alpha, beta);
#else
    simd_binary_gemm_scalar(x_packed, w_packed, y_out, M, N, K, alpha, beta);
#endif
}

void simd_ssm_scan(
    const float* x, const float* A_bar, const float* B_bar,
    const float* C, float* y_out,
    int seq_len, int d_inner, int d_state
) {
#if defined(SIMD_USE_NEON)
    simd_ssm_scan_neon(x, A_bar, B_bar, C, y_out, seq_len, d_inner, d_state);
#elif defined(SIMD_USE_AVX2)
    simd_ssm_scan_avx2(x, A_bar, B_bar, C, y_out, seq_len, d_inner, d_state);
#else
    simd_ssm_scan_scalar(x, A_bar, B_bar, C, y_out, seq_len, d_inner, d_state);
#endif
}

/* ================================================================
 * Packing Utilities (platform-independent)
 * ================================================================ */

void simd_pack_weights(
    const float* weights, uint8_t* packed_out,
    float* alpha_out, size_t n
) {
    double sum_abs = 0.0;
    for (size_t i = 0; i < n; i++) sum_abs += fabs((double)weights[i]);
    *alpha_out = (float)(sum_abs / (double)n);

    size_t packed_len = (n + 7) / 8;
    memset(packed_out, 0, packed_len);
    for (size_t i = 0; i < n; i++) {
        if (weights[i] >= 0.0f)
            packed_out[i / 8] |= (1 << (7 - (i % 8)));
    }
}

void simd_unpack_weights(
    const uint8_t* packed, float* weights_out, size_t n
) {
    for (size_t i = 0; i < n; i++) {
        int bit = (packed[i / 8] >> (7 - (i % 8))) & 1;
        weights_out[i] = bit ? 1.0f : -1.0f;
    }
}
