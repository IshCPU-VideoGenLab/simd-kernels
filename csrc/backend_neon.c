/**
 * backend_neon.c — ARM NEON SIMD backend.
 *
 * Runs on: Apple M1/M2/M3/M4, AWS Graviton, Raspberry Pi 4+,
 * Qualcomm Snapdragon, Samsung Exynos — any ARMv8-A with NEON.
 *
 * Key advantage: vcntq_u8 gives us NATIVE popcount per byte.
 * No lookup table hack needed (unlike AVX2).
 */

#include "simd_common.h"
#include "simd_backend.h"

#ifdef SIMD_USE_NEON

/* ---- Popcount: native vcntq_u8 ---- */

int simd_popcount_neon(const uint8_t* data, size_t len) {
    int total = 0;
    size_t i = 0;

    /* Process 16 bytes at a time using NEON */
    for (; i + 16 <= len; i += 16) {
        /* Load 16 bytes */
        uint8x16_t v = vld1q_u8(data + i);

        /* vcntq_u8: popcount per byte — ONE instruction, native ARM */
        uint8x16_t counts = vcntq_u8(v);

        /* Horizontal sum: add all 16 byte counts */
        /* vaddlvq_u8: sum all bytes into a single uint16 */
        total += vaddlvq_u8(counts);
    }

    /* Handle remainder */
    for (; i < len; i++) {
        total += popcount_byte(data[i]);
    }

    return total;
}

/* ---- Binary GEMM via NEON ---- */

void simd_binary_gemm_neon(
    const uint8_t* x_packed,
    const uint8_t* w_packed,
    float* y_out,
    int M, int N, int K,
    float alpha, float beta
) {
    const int K_bytes = K / 8;
    const int K_neon = K_bytes / 16;  /* 128-bit NEON chunks */
    const float scale = alpha * beta;

    for (int m = 0; m < M; m++) {
        const uint8_t* x_row = x_packed + m * K_bytes;

        for (int n = 0; n < N; n++) {
            const uint8_t* w_row = w_packed + n * K_bytes;
            int xor_popcount = 0;

            /* Process 16 bytes (128 bits) at a time */
            for (int k = 0; k < K_neon; k++) {
                /* Load 128 bits of activations and weights */
                uint8x16_t xv = vld1q_u8(x_row + k * 16);
                uint8x16_t wv = vld1q_u8(w_row + k * 16);

                /* XOR: different bits → 1 */
                uint8x16_t xor_result = veorq_u8(xv, wv);

                /* Native popcount per byte */
                uint8x16_t counts = vcntq_u8(xor_result);

                /* Sum all byte counts */
                xor_popcount += vaddlvq_u8(counts);
            }

            /* Handle remaining bytes */
            for (int k = K_neon * 16; k < K_bytes; k++) {
                uint8_t xor_byte = x_row[k] ^ w_row[k];
                xor_popcount += popcount_byte(xor_byte);
            }

            /* Convert: matching = K - different, dot = K - 2*different */
            int dot_product = K - 2 * xor_popcount;
            y_out[m * N + n] = (float)dot_product * scale;
        }
    }
}

/* ---- SSM Scan via NEON ---- */

void simd_ssm_scan_neon(
    const float* x,
    const float* A_bar,
    const float* B_bar,
    const float* C,
    float* y_out,
    int seq_len, int d_inner, int d_state
) {
    float* h = (float*)simd_alloc_impl(d_inner * d_state * sizeof(float));
    if (!h) return;
    memset(h, 0, d_inner * d_state * sizeof(float));

    const int d_state_neon = d_state / 4;  /* NEON: 4 floats per vector */

    for (int t = 0; t < seq_len; t++) {
        for (int d = 0; d < d_inner; d++) {
            float x_val = x[t * d_inner + d];
            float32x4_t x_vec = vdupq_n_f32(x_val);

            float* h_row = h + d * d_state;
            const float* A_row = A_bar + (t * d_inner + d) * d_state;
            const float* B_row = B_bar + (t * d_inner + d) * d_state;

            /* h = A*h + B*x, vectorized in chunks of 4 floats */
            for (int s = 0; s < d_state_neon; s++) {
                int off = s * 4;
                float32x4_t hv = vld1q_f32(h_row + off);
                float32x4_t av = vld1q_f32(A_row + off);
                float32x4_t bv = vld1q_f32(B_row + off);

                /* h = A*h + B*x using fused multiply-add */
                float32x4_t ah = vmulq_f32(av, hv);
                float32x4_t new_h = vfmaq_f32(ah, bv, x_vec);  /* ah + bv*x */

                vst1q_f32(h_row + off, new_h);
            }

            /* Remainder */
            for (int s = d_state_neon * 4; s < d_state; s++) {
                h_row[s] = A_row[s] * h_row[s] + B_row[s] * x_val;
            }

            /* y[t][d] = dot(C[t], h[d]) */
            const float* C_row = C + t * d_state;
            float32x4_t sum_vec = vdupq_n_f32(0.0f);

            for (int s = 0; s < d_state_neon; s++) {
                int off = s * 4;
                float32x4_t cv = vld1q_f32(C_row + off);
                float32x4_t hv = vld1q_f32(h_row + off);
                sum_vec = vfmaq_f32(sum_vec, cv, hv);  /* sum += c*h */
            }

            /* Horizontal sum of 4 floats */
            float y_val = vaddvq_f32(sum_vec);

            for (int s = d_state_neon * 4; s < d_state; s++) {
                y_val += C_row[s] * h_row[s];
            }

            y_out[t * d_inner + d] = y_val;
        }
    }

    simd_free_impl(h);
}

#endif /* SIMD_USE_NEON */
