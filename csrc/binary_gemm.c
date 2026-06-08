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
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

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

/* Run the compiled backend on a contiguous block of M rows. The output rows
 * of a GEMM are independent, so a worker can process any [m0, m1) slice by
 * offsetting the activation/output row pointers — no backend changes needed. */
static void binary_gemm_backend(
    const uint8_t* x_packed, const uint8_t* w_packed,
    float* y_out, int M, int N, int K, float alpha, float beta
) {
#if defined(SIMD_USE_NEON)
    simd_binary_gemm_neon(x_packed, w_packed, y_out, M, N, K, alpha, beta);
#elif defined(SIMD_USE_AVX2)
    simd_binary_gemm_avx2(x_packed, w_packed, y_out, M, N, K, alpha, beta);
#else
    simd_binary_gemm_scalar(x_packed, w_packed, y_out, M, N, K, alpha, beta);
#endif
}

#define SIMD_MAX_THREADS 16

typedef struct {
    const uint8_t* x_packed;
    const uint8_t* w_packed;
    float* y_out;
    int m0, m1, N, K;
    float alpha, beta;
} GemmTask;

static void* gemm_worker(void* arg) {
    GemmTask* t = (GemmTask*)arg;
    const int K_bytes = t->K / 8;
    binary_gemm_backend(
        t->x_packed + (size_t)t->m0 * (size_t)K_bytes,
        t->w_packed,
        t->y_out + (size_t)t->m0 * (size_t)t->N,
        t->m1 - t->m0, t->N, t->K, t->alpha, t->beta);
    return NULL;
}

/* Thread count: SIMD_NUM_THREADS env override, else online CPUs (capped). */
static int gemm_num_threads(void) {
    const char* env = getenv("SIMD_NUM_THREADS");
    if (env) {
        int n = atoi(env);
        if (n >= 1) return n > SIMD_MAX_THREADS ? SIMD_MAX_THREADS : n;
    }
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n < 1) n = 1;
    if (n > SIMD_MAX_THREADS) n = SIMD_MAX_THREADS;
    return (int)n;
}

void simd_binary_gemm(
    const uint8_t* x_packed, const uint8_t* w_packed,
    float* y_out, int M, int N, int K,
    float alpha, float beta
) {
    int T = gemm_num_threads();
    long long work = (long long)M * (long long)N * (long long)K;

    /* Serial for small problems: thread spawn cost (~0.1-0.5 ms) only pays off
     * once the work is large enough. Tuned threshold ~64M binary-MAC ops. */
    if (T <= 1 || M < 2 * T || work < (1LL << 26)) {
        binary_gemm_backend(x_packed, w_packed, y_out, M, N, K, alpha, beta);
        return;
    }
    if (T > M) T = M;

    pthread_t threads[SIMD_MAX_THREADS];
    GemmTask tasks[SIMD_MAX_THREADS];
    int created[SIMD_MAX_THREADS];
    int nt = 0, nc = 0;
    int rows_per = (M + T - 1) / T;

    for (int m0 = 0; m0 < M; m0 += rows_per) {
        int m1 = m0 + rows_per;
        if (m1 > M) m1 = M;
        tasks[nt].x_packed = x_packed; tasks[nt].w_packed = w_packed;
        tasks[nt].y_out = y_out;       tasks[nt].m0 = m0; tasks[nt].m1 = m1;
        tasks[nt].N = N; tasks[nt].K = K; tasks[nt].alpha = alpha; tasks[nt].beta = beta;
        nt++;
    }
    for (int i = 0; i < nt; i++) {
        if (pthread_create(&threads[i], NULL, gemm_worker, &tasks[i]) == 0)
            created[nc++] = i;
        else
            gemm_worker(&tasks[i]);   /* spawn failed: run this slice inline */
    }
    for (int i = 0; i < nc; i++)
        pthread_join(threads[created[i]], NULL);
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
