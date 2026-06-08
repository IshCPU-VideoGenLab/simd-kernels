/**
 * simd_common.h — Backend detection and shared utilities.
 *
 * Detects at compile time which SIMD instruction set is available
 * and defines the alignment requirements accordingly.
 */

#ifndef SIMD_COMMON_H
#define SIMD_COMMON_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ================================================================
 * Compile-Time Backend Detection
 *
 * Priority: AVX2 > NEON > Scalar
 * User can override with -DFORCE_SCALAR, -DFORCE_AVX2, -DFORCE_NEON
 * ================================================================ */

#if defined(FORCE_SCALAR)
    #define SIMD_USE_SCALAR 1
    #define SIMD_ALIGNMENT 16

#elif defined(FORCE_AVX2) || (defined(__AVX2__) && !defined(FORCE_NEON))
    #define SIMD_USE_AVX2 1
    #define SIMD_ALIGNMENT 32
    #include <immintrin.h>

#elif defined(FORCE_NEON) || defined(__ARM_NEON) || defined(__ARM_NEON__)
    #define SIMD_USE_NEON 1
    #define SIMD_ALIGNMENT 16
    #include <arm_neon.h>

#else
    #define SIMD_USE_SCALAR 1
    #define SIMD_ALIGNMENT 16
#endif

/* ================================================================
 * Aligned Memory
 * ================================================================ */

static inline void* simd_alloc_impl(size_t size) {
    size_t aligned_size = (size + SIMD_ALIGNMENT - 1) & ~(size_t)(SIMD_ALIGNMENT - 1);
#if defined(_WIN32)
    return _aligned_malloc(aligned_size, SIMD_ALIGNMENT);
#elif defined(__APPLE__) || defined(__linux__)
    void* ptr = NULL;
    if (posix_memalign(&ptr, SIMD_ALIGNMENT, aligned_size) != 0) return NULL;
    return ptr;
#else
    return malloc(aligned_size);  /* Fallback, may not be aligned */
#endif
}

static inline void simd_free_impl(void* ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

/* ================================================================
 * Scalar Reference Helpers
 * ================================================================ */

/** Scalar popcount for a single byte. */
static inline int popcount_byte(uint8_t b) {
    int count = 0;
    while (b) { count += b & 1; b >>= 1; }
    return count;
}

/** Scalar popcount for a byte array. */
static inline int popcount_array_scalar(const uint8_t* data, size_t len) {
    int total = 0;
    for (size_t i = 0; i < len; i++) {
        total += popcount_byte(data[i]);
    }
    return total;
}

#endif /* SIMD_COMMON_H */
