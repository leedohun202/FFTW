#include "myfft.h"

// ====================================================================
// --- [1] AArch64 (64비트) NEON 최적화 버전 ---
// ====================================================================
#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>

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
    /* Stage 2 이상: Butterfly 연산 */ \
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

// ====================================================================
// --- [2] 32비트 ARMv7 NEON 최적화 버전 (라즈베리파이 32비트 OS 등) ---
// ====================================================================
#elif defined(__arm__) && defined(__ARM_NEON)
#include <arm_neon.h>

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
    /* Stage 2 이상: Butterfly 연산 */ \
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
                    /* vmlsq / vmlaq 사용으로 32비트 파이프라인 최적화 */ \
                    float32x4_t vt_real = vmulq_f32(v_k.val[0], v_tr); \
                    vt_real = vmlsq_f32(vt_real, v_k.val[1], v_ti); \
                    float32x4_t vt_imag = vmulq_f32(v_k.val[0], v_ti); \
                    vt_imag = vmlaq_f32(vt_imag, v_k.val[1], v_tr); \
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

// =========================================================================
// ARMv7-R / In-Order FPU 전용 2-Way Interleaved Fused Radix-2 매크로
// =========================================================================
#define GENERATE_CUSTOM_FFT_RADIX2_FUSED(N) \
extern int bitrev_##N[]; \
extern fftwf_complex twiddle_##N[]; \
void custom_fft_##N##_radix2_fused(const fftwf_complex *__restrict__ in, fftwf_complex *__restrict__ out) { \
    /* -------------------------------------------------------------------- */ \
    /* 1. Fused Bit-Reversal + Stage 1 (Twiddle=1, 2-Way Interleaved)       */ \
    /* -------------------------------------------------------------------- */ \
    int k_bit = 0; \
    for (; k_bit + 2 < N; k_bit += 4) { \
        int r0_A = bitrev_##N[k_bit];     int r1_A = bitrev_##N[k_bit + 1]; \
        int r0_B = bitrev_##N[k_bit + 2]; int r1_B = bitrev_##N[k_bit + 3]; \
        \
        float a0_r = in[r0_A][0], a0_i = in[r0_A][1]; \
        float b0_r = in[r0_B][0], b0_i = in[r0_B][1]; \
        float a1_r = in[r1_A][0], a1_i = in[r1_A][1]; \
        float b1_r = in[r1_B][0], b1_i = in[r1_B][1]; \
        \
        out[k_bit][0]     = a0_r + a1_r; out[k_bit][1]     = a0_i + a1_i; \
        out[k_bit + 2][0] = b0_r + b1_r; out[k_bit + 2][1] = b0_i + b1_i; \
        out[k_bit + 1][0] = a0_r - a1_r; out[k_bit + 1][1] = a0_i - a1_i; \
        out[k_bit + 3][0] = b0_r - b1_r; out[k_bit + 3][1] = b0_i - b1_i; \
    } \
    for (; k_bit < N; k_bit += 2) { \
        int r0 = bitrev_##N[k_bit];   int r1 = bitrev_##N[k_bit + 1]; \
        float r0_r = in[r0][0]; float r0_i = in[r0][1]; \
        float r1_r = in[r1][0]; float r1_i = in[r1][1]; \
        out[k_bit][0]     = r0_r + r1_r; out[k_bit][1]     = r0_i + r1_i; \
        out[k_bit + 1][0] = r0_r - r1_r; out[k_bit + 1][1] = r0_i - r1_i; \
    } \
    \
    /* -------------------------------------------------------------------- */ \
    /* 2. Main Radix-2 Stages (step = 4부터 N까지)                          */ \
    /* -------------------------------------------------------------------- */ \
    for (int step = 4; step <= N; step <<= 1) { \
        int half = step >> 1; \
        int twiddle_stride = N / step; \
        for (int j = 0; j < half; j++) { \
            int tw_idx = j * twiddle_stride; \
            float c = twiddle_##N[tw_idx][0]; \
            float s = twiddle_##N[tw_idx][1]; \
            \
            int k = j; \
            /* 블록 간 2-Way Interleaving (step < N 일 때 실행) */ \
            for (; k + step < N; k += 2 * step) { \
                int i0_A = k,        i1_A = k + half; \
                int i0_B = k + step, i1_B = k + step + half; \
                \
                float a1_r = out[i1_A][0], a1_i = out[i1_A][1]; \
                float b1_r = out[i1_B][0], b1_i = out[i1_B][1]; \
                \
                float a_tr = a1_r * c - a1_i * s; \
                float b_tr = b1_r * c - b1_i * s; \
                float a_ti = a1_r * s + a1_i * c; \
                float b_ti = b1_r * s + b1_i * c; \
                \
                float a0_r = out[i0_A][0], a0_i = out[i0_A][1]; \
                float b0_r = out[i0_B][0], b0_i = out[i0_B][1]; \
                \
                out[i0_A][0] = a0_r + a_tr; out[i0_A][1] = a0_i + a_ti; \
                out[i0_B][0] = b0_r + b_tr; out[i0_B][1] = b0_i + b_ti; \
                out[i1_A][0] = a0_r - a_tr; out[i1_A][1] = a0_i - a_ti; \
                out[i1_B][0] = b0_r - b_tr; out[i1_B][1] = b0_i - b_ti; \
            } \
            /* 자투리 연산 및 step == N 단계 처리 (오타 수정됨) */ \
            for (; k < N; k += step) { \
                int i0 = k, i1 = k + half; \
                float r1 = out[i1][0], i1_val = out[i1][1]; \
                float tr = r1 * c - i1_val * s; \
                float ti = r1 * s + i1_val * c; \
                float u_r = out[i0][0], u_i = out[i0][1]; \
                out[i0][0] = u_r + tr; out[i0][1] = u_i + ti; /* ✅ out[i0][1] 허수부 정상 저장 */ \
                out[i1][0] = u_r - tr; out[i1][1] = u_i - ti; \
            } \
        } \
    } \
}

// ====================================================================
// --- [4] 스칼라 Fallback 버전 (armv7-r / Cortex-R / Non-NEON) ---
// ====================================================================
/*
#define GENERATE_CUSTOM_FFT_RADIX2_FUSED(N) \
extern int bitrev_##N[]; \
extern fftwf_complex twiddle_##N[]; \
void custom_fft_##N##_radix2_fused(const fftwf_complex *__restrict__ in, fftwf_complex *__restrict__ out) { \
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
*/

#endif

// ====================================================================
// 매크로 호출부: 실제 함수 생성
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