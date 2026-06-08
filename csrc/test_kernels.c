/**
 * test_kernels.c — Cross-platform kernel tests.
 * Compares the active SIMD backend against scalar reference.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "simd_backend.h"

/* Scalar references for comparison */
extern void simd_binary_gemm_scalar(const uint8_t*, const uint8_t*, float*, int, int, int, float, float);
extern void simd_ssm_scan_scalar(const float*, const float*, const float*, const float*, float*, int, int, int);
extern int  simd_popcount_scalar(const uint8_t*, size_t);

static int passed = 0, failed = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("  FAIL: %s\n", msg); failed++; } \
    else { passed++; } \
} while(0)

static void test_backend_info(void) {
    printf("Backend: %s\n", simd_get_backend_name());
    printf("Supported: %s\n\n", simd_is_supported() ? "YES" : "NO");
    passed++;
}

static void test_popcount(void) {
    printf("Test: popcount\n");

    uint8_t zeros[64]; memset(zeros, 0, 64);
    CHECK(simd_popcount(zeros, 64) == 0, "popcount(zeros)=0");

    uint8_t ones[64]; memset(ones, 0xFF, 64);
    CHECK(simd_popcount(ones, 64) == 512, "popcount(ones)=512");

    /* Random data: compare SIMD vs scalar */
    uint8_t data[128];
    for (int i = 0; i < 128; i++) data[i] = (uint8_t)(i * 7 + 13);
    int simd_count = simd_popcount(data, 128);
    int scalar_count = simd_popcount_scalar(data, 128);
    CHECK(simd_count == scalar_count, "popcount: SIMD matches scalar");
}

static void test_pack_unpack(void) {
    printf("Test: pack/unpack\n");
    float weights[64];
    for (int i = 0; i < 64; i++) weights[i] = (i % 3 == 0) ? -0.5f : 0.3f;

    uint8_t packed[8]; float alpha;
    simd_pack_weights(weights, packed, &alpha, 64);

    float unpacked[64];
    simd_unpack_weights(packed, unpacked, 64);

    int ok = 1;
    for (int i = 0; i < 64; i++) {
        float expected = (weights[i] >= 0.0f) ? 1.0f : -1.0f;
        if (unpacked[i] != expected) { ok = 0; break; }
    }
    CHECK(ok, "pack/unpack preserves signs");
    CHECK(alpha > 0.0f, "alpha > 0");
}

static void test_binary_gemm(void) {
    printf("Test: binary GEMM (SIMD vs scalar)\n");
    int M=4, N=8, K=256;
    int Kb = K/8;

    uint8_t *x = calloc(M*Kb, 1), *w = calloc(N*Kb, 1);
    float *y_simd = calloc(M*N, sizeof(float));
    float *y_ref  = calloc(M*N, sizeof(float));

    for (int i = 0; i < M*Kb; i++) x[i] = (uint8_t)(i*13+7);
    for (int i = 0; i < N*Kb; i++) w[i] = (uint8_t)(i*17+3);

    simd_binary_gemm(x, w, y_simd, M, N, K, 0.5f, 1.0f);
    simd_binary_gemm_scalar(x, w, y_ref, M, N, K, 0.5f, 1.0f);

    int ok = 1;
    for (int i = 0; i < M*N; i++) {
        if (fabs(y_simd[i] - y_ref[i]) > 0.01f) {
            printf("    [%d] simd=%.4f ref=%.4f\n", i, y_simd[i], y_ref[i]);
            ok = 0;
        }
    }
    CHECK(ok, "binary GEMM: SIMD matches scalar");

    free(x); free(w); free(y_simd); free(y_ref);
}

static void test_ssm_scan(void) {
    printf("Test: SSM scan (SIMD vs scalar)\n");
    int S=8, D=4, N=16;

    float *x = (float*)simd_aligned_alloc(S*D*sizeof(float));
    float *A = (float*)simd_aligned_alloc(S*D*N*sizeof(float));
    float *B = (float*)simd_aligned_alloc(S*D*N*sizeof(float));
    float *C = (float*)simd_aligned_alloc(S*N*sizeof(float));
    float *y_simd = (float*)simd_aligned_alloc(S*D*sizeof(float));
    float *y_ref  = (float*)calloc(S*D, sizeof(float));

    for (int i = 0; i < S*D; i++) x[i] = 0.1f * (float)(i%7-3);
    for (int i = 0; i < S*D*N; i++) { A[i] = 0.9f; B[i] = 0.1f; }
    for (int i = 0; i < S*N; i++) C[i] = 0.1f * (float)(i%5-2);

    simd_ssm_scan(x, A, B, C, y_simd, S, D, N);
    simd_ssm_scan_scalar(x, A, B, C, y_ref, S, D, N);

    int ok = 1;
    for (int i = 0; i < S*D; i++) {
        if (fabs(y_simd[i] - y_ref[i]) > 1e-4f) {
            printf("    [%d] simd=%.6f ref=%.6f\n", i, y_simd[i], y_ref[i]);
            ok = 0;
        }
    }
    CHECK(ok, "SSM scan: SIMD matches scalar");

    simd_aligned_free(x); simd_aligned_free(A);
    simd_aligned_free(B); simd_aligned_free(C);
    simd_aligned_free(y_simd); free(y_ref);
}

int main(void) {
    printf("\n=== SIMD Kernel Tests ===\n\n");
    test_backend_info();
    test_popcount();
    test_pack_unpack();
    test_binary_gemm();
    test_ssm_scan();
    printf("\n=== %d passed, %d failed ===\n\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
