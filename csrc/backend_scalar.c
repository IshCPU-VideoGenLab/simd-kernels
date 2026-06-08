/**
 * backend_scalar.c — Pure C scalar fallback.
 *
 * Runs on literally anything with a C compiler.
 * No SIMD intrinsics. Used for correctness testing and as fallback.
 */

#include "simd_common.h"
#include "simd_backend.h"

#ifndef SIMD_USE_AVX2
#ifndef SIMD_USE_NEON
/* Only compile scalar if no SIMD backend is active,
   OR if we need the scalar reference functions regardless */
#define COMPILE_SCALAR_IMPL 1
#endif
#endif

/* Always compile scalar references for testing */

int simd_popcount_scalar(const uint8_t* data, size_t len) {
    return popcount_array_scalar(data, len);
}

void simd_binary_gemm_scalar(
    const uint8_t* x_packed, const uint8_t* w_packed,
    float* y_out, int M, int N, int K,
    float alpha, float beta
) {
    const int K_bytes = K / 8;
    const float scale = alpha * beta;

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int xor_pop = 0;
            for (int k = 0; k < K_bytes; k++) {
                xor_pop += popcount_byte(
                    x_packed[m * K_bytes + k] ^ w_packed[n * K_bytes + k]
                );
            }
            y_out[m * N + n] = (float)(K - 2 * xor_pop) * scale;
        }
    }
}

void simd_ssm_scan_scalar(
    const float* x, const float* A_bar, const float* B_bar,
    const float* C, float* y_out,
    int seq_len, int d_inner, int d_state
) {
    float* h = (float*)calloc(d_inner * d_state, sizeof(float));
    if (!h) return;

    for (int t = 0; t < seq_len; t++) {
        for (int d = 0; d < d_inner; d++) {
            float x_val = x[t * d_inner + d];
            for (int s = 0; s < d_state; s++) {
                int idx = (t * d_inner + d) * d_state + s;
                h[d * d_state + s] = A_bar[idx] * h[d * d_state + s]
                                   + B_bar[idx] * x_val;
            }
            float y_val = 0.0f;
            for (int s = 0; s < d_state; s++) {
                y_val += C[t * d_state + s] * h[d * d_state + s];
            }
            y_out[t * d_inner + d] = y_val;
        }
    }
    free(h);
}
