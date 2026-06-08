/**
 * backend_avx2.c — x86 AVX2 SIMD backend.
 *
 * Runs on: Intel Haswell+ (2013), AMD Zen+ (2018), including Pentium Gold.
 * Processes 256 bits (32 bytes) per instruction.
 * Popcount via nibble lookup table (no native VPOPCNT in AVX2).
 */

#include "simd_common.h"
#include "simd_backend.h"

#ifdef SIMD_USE_AVX2

/* Popcount lookup table for AVX2 (shared across functions) */
static inline __m256i avx2_lut(void) {
    return _mm256_setr_epi8(
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4
    );
}

static inline int avx2_popcnt256(__m256i v) {
    __m256i lut = avx2_lut();
    __m256i mask = _mm256_set1_epi8(0x0F);
    __m256i lo = _mm256_and_si256(v, mask);
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), mask);
    __m256i cnt = _mm256_add_epi8(
        _mm256_shuffle_epi8(lut, lo),
        _mm256_shuffle_epi8(lut, hi)
    );
    __m256i sad = _mm256_sad_epu8(cnt, _mm256_setzero_si256());
    return (int)(_mm256_extract_epi64(sad, 0) + _mm256_extract_epi64(sad, 1)
               + _mm256_extract_epi64(sad, 2) + _mm256_extract_epi64(sad, 3));
}

int simd_popcount_avx2(const uint8_t* data, size_t len) {
    int total = 0;
    size_t i = 0;
    for (; i + 32 <= len; i += 32) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(data + i));
        total += avx2_popcnt256(v);
    }
    for (; i < len; i++) total += popcount_byte(data[i]);
    return total;
}

void simd_binary_gemm_avx2(
    const uint8_t* x_packed, const uint8_t* w_packed,
    float* y_out, int M, int N, int K,
    float alpha, float beta
) {
    const int K_bytes = K / 8;
    const int K_avx = K_bytes / 32;
    const float scale = alpha * beta;

    for (int m = 0; m < M; m++) {
        const uint8_t* x_row = x_packed + m * K_bytes;
        for (int n = 0; n < N; n++) {
            const uint8_t* w_row = w_packed + n * K_bytes;
            int xor_pop = 0;

            for (int k = 0; k < K_avx; k++) {
                __m256i xv = _mm256_loadu_si256((const __m256i*)(x_row + k*32));
                __m256i wv = _mm256_loadu_si256((const __m256i*)(w_row + k*32));
                __m256i xor_r = _mm256_xor_si256(xv, wv);
                xor_pop += avx2_popcnt256(xor_r);
            }
            for (int k = K_avx * 32; k < K_bytes; k++) {
                xor_pop += popcount_byte(x_row[k] ^ w_row[k]);
            }

            y_out[m * N + n] = (float)(K - 2 * xor_pop) * scale;
        }
    }
}

void simd_ssm_scan_avx2(
    const float* x, const float* A_bar, const float* B_bar,
    const float* C, float* y_out,
    int seq_len, int d_inner, int d_state
) {
    float* h = (float*)simd_alloc_impl(d_inner * d_state * sizeof(float));
    if (!h) return;
    memset(h, 0, d_inner * d_state * sizeof(float));
    const int d_avx = d_state / 8;

    for (int t = 0; t < seq_len; t++) {
        for (int d = 0; d < d_inner; d++) {
            float x_val = x[t * d_inner + d];
            __m256 xv = _mm256_set1_ps(x_val);
            float* h_row = h + d * d_state;
            const float* A_row = A_bar + (t * d_inner + d) * d_state;
            const float* B_row = B_bar + (t * d_inner + d) * d_state;

            for (int s = 0; s < d_avx; s++) {
                int off = s * 8;
                __m256 hv = _mm256_load_ps(h_row + off);
                __m256 av = _mm256_loadu_ps(A_row + off);
                __m256 bv = _mm256_loadu_ps(B_row + off);
                __m256 new_h = _mm256_add_ps(
                    _mm256_mul_ps(av, hv),
                    _mm256_mul_ps(bv, xv)
                );
                _mm256_store_ps(h_row + off, new_h);
            }
            for (int s = d_avx * 8; s < d_state; s++) {
                h_row[s] = A_row[s] * h_row[s] + B_row[s] * x_val;
            }

            const float* C_row = C + t * d_state;
            __m256 sum = _mm256_setzero_ps();
            for (int s = 0; s < d_avx; s++) {
                int off = s * 8;
                sum = _mm256_fmadd_ps(
                    _mm256_loadu_ps(C_row + off),
                    _mm256_load_ps(h_row + off), sum
                );
            }
            __m128 hi128 = _mm256_extractf128_ps(sum, 1);
            __m128 lo128 = _mm256_castps256_ps128(sum);
            __m128 s128 = _mm_add_ps(lo128, hi128);
            s128 = _mm_hadd_ps(s128, s128);
            s128 = _mm_hadd_ps(s128, s128);
            float y_val = _mm_cvtss_f32(s128);

            for (int s = d_avx * 8; s < d_state; s++) {
                y_val += C_row[s] * h_row[s];
            }
            y_out[t * d_inner + d] = y_val;
        }
    }
    simd_free_impl(h);
}

#endif /* SIMD_USE_AVX2 */
