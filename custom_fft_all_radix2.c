#include "myfft.h"

#ifdef __aarch64__
#include <arm_neon.h>

// --- ARM NEON 최적화 버전 (Fused Bit-reversal + Radix-2) ---
#define GENERATE_CUSTOM_FFT_RADIX2_FUSED(N) \
extern int bitrev_##N[]; \
extern fftwf_complex twiddle_##N[]; \
void custom_fft_##N##_radix2_fused(const fftwf_complex *__restrict__ in, fftwf_complex *__restrict__ out) { \
    /* ------------------------------------------------------------------ */ \
    /* 🚀 [Stage 1: Fused Bit-Reversal + Radix-2]                       */ \
    /* in 배열의 bitrev 위치에서 바로 읽어와 Radix-2 (Twiddle=1) 연산 후 */ \
    /* out 배열에 순차적으로 저장합니다. (memcpy 및 Swap 제거)           */ \
    /* ------------------------------------------------------------------ */ \
    for (int k = 0; k < N; k += 2) { \
        int r0 = bitrev_##N[k]; \
        int r1 = bitrev_##N[k + 1]; \
        \
        float r0_r = in[r0][0]; float r0_i = in[r0][1]; \
        float r1_r = in[r1][0]; float r1_i = in[r1][1]; \
        \
        out[k][0]     = r0_r + r1_r; \
        out[k][1]     = r0_i + r1_i; \
        out[k + 1][0] = r0_r - r1_r; \
        out[k + 1][1] = r0_i - r1_i; \
    } \
    \
    /* ------------------------------------------------------------------ */ \
    /* [Stage 2 이상: Butterfly 연산] (step 2부터 out 배열에서 수행)      */ \
    /* ------------------------------------------------------------------ */ \
    for (int step = 2; step < N; step *= 2) { \
        const int jump = step * 2; \
        const int twiddle_step = N / jump; \
        if (step < 4) { \
            for (int i = 0; i < N; i += jump) { \
                for (int j = 0; j < step; j++) { \
                    int curr = i + j; \
                    int k = curr + step; \
                    float tr = twiddle_##N[j * twiddle_step][0]; \
                    float ti = twiddle_##N[j * twiddle_step][1]; \
                    float t_real = out[k][0] * tr - out[k][1] * ti; \
                    float t_imag = out[k][0] * ti + out[k][1] * tr; \
                    out[k][0] = out[curr][0] - t_real; \
                    out[k][1] = out[curr][1] - t_imag; \
                    out[curr][0] += t_real; \
                    out[curr][1] += t_imag; \
                } \
            } \
        } else { \
            for (int i = 0; i < N; i += jump) { \
                for (int j = 0; j < step; j += 4) { \
                    int curr = i + j; \
                    int k = curr + step; \
                    float32x4x2_t v_curr = vld2q_f32((const float*)&out[curr]); \
                    float32x4x2_t v_k    = vld2q_f32((const float*)&out[k]); \
                    int tj = j * twiddle_step; \
                    float tr_arr[4] = { \
                        twiddle_##N[tj][0], \
                        twiddle_##N[tj + twiddle_step][0], \
                        twiddle_##N[tj + 2*twiddle_step][0], \
                        twiddle_##N[tj + 3*twiddle_step][0] \
                    }; \
                    float ti_arr[4] = { \
                        twiddle_##N[tj][1], \
                        twiddle_##N[tj + twiddle_step][1], \
                        twiddle_##N[tj + 2*twiddle_step][1], \
                        twiddle_##N[tj + 3*twiddle_step][1] \
                    }; \
                    float32x4_t v_tr = vld1q_f32(tr_arr); \
                    float32x4_t v_ti = vld1q_f32(ti_arr); \
                    float32x4_t vt_real = vsubq_f32(vmulq_f32(v_k.val[0], v_tr), vmulq_f32(v_k.val[1], v_ti)); \
                    float32x4_t vt_imag = vaddq_f32(vmulq_f32(v_k.val[0], v_ti), vmulq_f32(v_k.val[1], v_tr)); \
                    float32x4x2_t res_k, res_curr; \
                    res_k.val[0] = vsubq_f32(v_curr.val[0], vt_real); \
                    res_k.val[1] = vsubq_f32(v_curr.val[1], vt_imag); \
                    vst2q_f32((float*)&out[k], res_k); \
                    res_curr.val[0] = vaddq_f32(v_curr.val[0], vt_real); \
                    res_curr.val[1] = vaddq_f32(v_curr.val[1], vt_imag); \
                    vst2q_f32((float*)&out[curr], res_curr); \
                } \
            } \
        } \
    } \
}

#else

// --- 스칼라 Fallback 버전 (비 ARM 환경용 Fused) ---
#define GENERATE_CUSTOM_FFT_RADIX2_FUSED(N) \
extern int bitrev_##N[]; \
extern fftwf_complex twiddle_##N[]; \
void custom_fft_##N##_radix2_fused(const fftwf_complex *__restrict__ in, fftwf_complex *__restrict__ out) { \
    /* Stage 1: Fused Bit-Reversal + Radix-2 */ \
    for (int k = 0; k < N; k += 2) { \
        int r0 = bitrev_##N[k]; \
        int r1 = bitrev_##N[k + 1]; \
        float r0_r = in[r0][0]; float r0_i = in[r0][1]; \
        float r1_r = in[r1][0]; float r1_i = in[r1][1]; \
        out[k][0]     = r0_r + r1_r; \
        out[k][1]     = r0_i + r1_i; \
        out[k + 1][0] = r0_r - r1_r; \
        out[k + 1][1] = r0_i - r1_i; \
    } \
    /* Stage 2 이상: Butterfly 연산 (step 2부터 시작) */ \
    for (int step = 2; step < N; step *= 2) { \
        const int jump = step * 2; \
        const int twiddle_step = N / jump; \
        for (int i = 0; i < N; i += jump) { \
            for (int j = 0; j < step; j++) { \
                int curr = i + j; \
                int k = curr + step; \
                float tr = twiddle_##N[j * twiddle_step][0]; \
                float ti = twiddle_##N[j * twiddle_step][1]; \
                float t_real = out[k][0] * tr - out[k][1] * ti; \
                float t_imag = out[k][0] * ti + out[k][1] * tr; \
                out[k][0] = out[curr][0] - t_real; \
                out[k][1] = out[curr][1] - t_imag; \
                out[curr][0] += t_real; \
                out[curr][1] += t_imag; \
            } \
        } \
    } \
}

#endif

// ====================================================================
// 함수 생성부
// ====================================================================
GENERATE_CUSTOM_FFT_RADIX2_FUSED(4096)
GENERATE_CUSTOM_FFT_RADIX2_FUSED(2048)
GENERATE_CUSTOM_FFT_RADIX2_FUSED(1024)
GENERATE_CUSTOM_FFT_RADIX2_FUSED(512)
GENERATE_CUSTOM_FFT_RADIX2_FUSED(256)
GENERATE_CUSTOM_FFT_RADIX2_FUSED(128)
GENERATE_CUSTOM_FFT_RADIX2_FUSED(64)
GENERATE_CUSTOM_FFT_RADIX2_FUSED(32)
GENERATE_CUSTOM_FFT_RADIX2_FUSED(16)