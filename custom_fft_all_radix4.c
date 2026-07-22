#include "myfft.h"

#ifdef __aarch64__
#include <arm_neon.h>

// =========================================================================
// [1] 순수 Radix-4 매크로 (4096, 1024, 256, 64, 16 용)
// =========================================================================
#define GENERATE_CUSTOM_FFT_RADIX4_PURE(N) \
extern int bitrev_##N[]; \
extern fftwf_complex twiddle_##N[]; \
void custom_fft_##N##_radix4(fftwf_complex *__restrict__ data) { \
    for (int k = 0; k < N; k++) { \
        int rev = bitrev_##N[k]; \
        if (k < rev) { \
            float temp_r = data[k][0]; data[k][0] = data[rev][0]; data[rev][0] = temp_r; \
            float temp_i = data[k][1]; data[k][1] = data[rev][1]; data[rev][1] = temp_i; \
        } \
    } \
    for (int step = 4; step <= N; step <<= 2) { \
        int n4 = step >> 2; \
        int twiddle_stride = N / step; \
        if (step == 4) { \
            for (int k = 0; k < N; k += 4) { \
                int i0 = k; int i1 = k + 1; int i2 = k + 2; int i3 = k + 3; \
                float r1_t = data[i2][0]; float i_1 = data[i2][1]; \
                float r2_t = data[i1][0]; float i_2 = data[i1][1]; \
                float r3_t = data[i3][0]; float i_3 = data[i3][1]; \
                float t_r0 = data[i0][0] + r2_t; float t_r1 = data[i0][0] - r2_t; \
                float t_r2 = r1_t + r3_t;        float t_r3 = r1_t - r3_t; \
                float t_i0 = data[i0][1] + i_2;  float t_i1 = data[i0][1] - i_2; \
                float t_i2 = i_1 + i_3;          float t_i3 = i_1 - i_3; \
                data[i0][0] = t_r0 + t_r2; data[i0][1] = t_i0 + t_i2; \
                data[i1][0] = t_r1 + t_i3; data[i1][1] = t_i1 - t_r3; \
                data[i2][0] = t_r0 - t_r2; data[i2][1] = t_i0 - t_i2; \
                data[i3][0] = t_r1 - t_i3; data[i3][1] = t_i1 + t_r3; \
            } \
        } else { \
            for (int j = 0; j < n4; j += 4) { \
                float c1_a[4], s1_a[4], c2_a[4], s2_a[4], c3_a[4], s3_a[4]; \
                for (int lane = 0; lane < 4; lane++) { \
                    int jj = j + lane; \
                    int idx1 = jj * twiddle_stride; int idx2 = idx1 * 2; int idx3 = idx1 * 3; \
                    c1_a[lane] = twiddle_##N[idx1][0]; s1_a[lane] = twiddle_##N[idx1][1]; \
                    c2_a[lane] = twiddle_##N[idx2][0]; s2_a[lane] = twiddle_##N[idx2][1]; \
                    if (idx3 >= (N / 2)) { \
                        c3_a[lane] = -twiddle_##N[idx3 - (N / 2)][0]; s3_a[lane] = -twiddle_##N[idx3 - (N / 2)][1]; \
                    } else { \
                        c3_a[lane] = twiddle_##N[idx3][0]; s3_a[lane] = twiddle_##N[idx3][1]; \
                    } \
                } \
                float32x4_t v_c1 = vld1q_f32(c1_a); float32x4_t v_s1 = vld1q_f32(s1_a); \
                float32x4_t v_c2 = vld1q_f32(c2_a); float32x4_t v_s2 = vld1q_f32(s2_a); \
                float32x4_t v_c3 = vld1q_f32(c3_a); float32x4_t v_s3 = vld1q_f32(s3_a); \
                for (int blk = 0; blk < N; blk += step) { \
                    int base = blk + j; \
                    float32x4x2_t in0 = vld2q_f32((const float*)&data[base]); \
                    float32x4_t v_r0 = in0.val[0]; float32x4_t v_i0 = in0.val[1]; \
                    float32x4x2_t in1 = vld2q_f32((const float*)&data[base + n4]); \
                    float32x4_t v_r1 = in1.val[0]; float32x4_t v_i1 = in1.val[1]; \
                    float32x4x2_t in2 = vld2q_f32((const float*)&data[base + 2 * n4]); \
                    float32x4_t v_r2 = in2.val[0]; float32x4_t v_i2 = in2.val[1]; \
                    float32x4x2_t in3 = vld2q_f32((const float*)&data[base + 3 * n4]); \
                    float32x4_t v_r3 = in3.val[0]; float32x4_t v_i3 = in3.val[1]; \
                    float32x4_t v_r1_t = vsubq_f32(vmulq_f32(v_r2, v_c1), vmulq_f32(v_i2, v_s1)); \
                    float32x4_t v_i1_t = vaddq_f32(vmulq_f32(v_r2, v_s1), vmulq_f32(v_i2, v_c1)); \
                    float32x4_t v_r2_t = vsubq_f32(vmulq_f32(v_r1, v_c2), vmulq_f32(v_i1, v_s2)); \
                    float32x4_t v_i2_t = vaddq_f32(vmulq_f32(v_r1, v_s2), vmulq_f32(v_i1, v_c2)); \
                    float32x4_t v_r3_t = vsubq_f32(vmulq_f32(v_r3, v_c3), vmulq_f32(v_i3, v_s3)); \
                    float32x4_t v_i3_t = vaddq_f32(vmulq_f32(v_r3, v_s3), vmulq_f32(v_i3, v_c3)); \
                    float32x4_t v_t_r0 = vaddq_f32(v_r0, v_r2_t); float32x4_t v_t_r1 = vsubq_f32(v_r0, v_r2_t); \
                    float32x4_t v_t_r2 = vaddq_f32(v_r1_t, v_r3_t); float32x4_t v_t_r3 = vsubq_f32(v_r1_t, v_r3_t); \
                    float32x4_t v_t_i0 = vaddq_f32(v_i0, v_i2_t); float32x4_t v_t_i1 = vsubq_f32(v_i0, v_i2_t); \
                    float32x4_t v_t_i2 = vaddq_f32(v_i1_t, v_i3_t); float32x4_t v_t_i3 = vsubq_f32(v_i1_t, v_i3_t); \
                    float32x4x2_t out0; out0.val[0] = vaddq_f32(v_t_r0, v_t_r2); out0.val[1] = vaddq_f32(v_t_i0, v_t_i2); \
                    vst2q_f32((float*)&data[base], out0); \
                    float32x4x2_t out1; out1.val[0] = vaddq_f32(v_t_r1, v_t_i3); out1.val[1] = vsubq_f32(v_t_i1, v_t_r3); \
                    vst2q_f32((float*)&data[base + n4], out1); \
                    float32x4x2_t out2; out2.val[0] = vsubq_f32(v_t_r0, v_t_r2); out2.val[1] = vsubq_f32(v_t_i0, v_t_i2); \
                    vst2q_f32((float*)&data[base + 2 * n4], out2); \
                    float32x4x2_t out3; out3.val[0] = vsubq_f32(v_t_r1, v_t_i3); out3.val[1] = vaddq_f32(v_t_i1, v_t_r3); \
                    vst2q_f32((float*)&data[base + 3 * n4], out3); \
                } \
            } \
        } \
    } \
}

// =========================================================================
// [2] 혼합 Radix-4 매크로 (2048, 512, 128, 32 용 - Fused Bit-reversal 적용)
// =========================================================================
#define GENERATE_CUSTOM_FFT_RADIX4_MIXED(N) \
extern int bitrev_##N[]; \
extern fftwf_complex twiddle_##N[]; \
void custom_fft_##N##_radix4_fused(const fftwf_complex *__restrict__ in, fftwf_complex *__restrict__ out) { \
    /* ------------------------------------------------------------------ */ \
    /* 🚀 [Stage 1: Fused Bit-Reversal + Radix-2]                       */ \
    /* memcpy와 in-place bitrev 루프를 완전 제거하고, in 버퍼의 bitrev  */ \
    /* 위치에서 바로 읽어와 Radix-2 연산 후 out 버퍼에 순차 저장합니다.  */ \
    /* ------------------------------------------------------------------ */ \
    for (int k = 0; k < N; k += 2) { \
        int r0 = bitrev_##N[k]; \
        int r1 = bitrev_##N[k + 1]; \
        \
        float r0_r = in[r0][0]; float r0_i = in[r0][1]; \
        float r1_r = in[r1][0]; float r1_i = in[r1][1]; \
        \
        /* Radix-2 Butterfly */ \
        out[k][0]     = r0_r + r1_r; \
        out[k][1]     = r0_i + r1_i; \
        out[k + 1][0] = r0_r - r1_r; \
        out[k + 1][1] = r0_i - r1_i; \
    } \
    \
    /* ------------------------------------------------------------------ */ \
    /* [Stage 2 이상: Radix-4 연산] (이후 단계는 out 버퍼 내에서 진행)  */ \
    /* ------------------------------------------------------------------ */ \
    for (int step = 8; step <= N; step <<= 2) { \
        int n4 = step >> 2; \
        int twiddle_stride = N / step; \
        if (step < 16) { \
            for (int j = 0; j < n4; j++) { \
                int idx1 = j * twiddle_stride; int idx2 = idx1 * 2; int idx3 = idx1 * 3; \
                float c1 = twiddle_##N[idx1][0]; float s1 = twiddle_##N[idx1][1]; \
                float c2 = twiddle_##N[idx2][0]; float s2 = twiddle_##N[idx2][1]; \
                float c3, s3; \
                if (idx3 >= (N / 2)) { \
                    c3 = -twiddle_##N[idx3 - (N / 2)][0]; s3 = -twiddle_##N[idx3 - (N / 2)][1]; \
                } else { \
                    c3 = twiddle_##N[idx3][0]; s3 = twiddle_##N[idx3][1]; \
                } \
                for (int k = j; k < N; k += step) { \
                    int i0 = k; int i1 = k + n4; int i2 = k + 2 * n4; int i3 = k + 3 * n4; \
                    float r1_t = out[i2][0] * c1 - out[i2][1] * s1; float i_1  = out[i2][0] * s1 + out[i2][1] * c1; \
                    float r2_t = out[i1][0] * c2 - out[i1][1] * s2; float i_2  = out[i1][0] * s2 + out[i1][1] * c2; \
                    float r3_t = out[i3][0] * c3 - out[i3][1] * s3; float i_3  = out[i3][0] * s3 + out[i3][1] * c3; \
                    float t_r0 = out[i0][0] + r2_t; float t_r1 = out[i0][0] - r2_t; \
                    float t_r2 = r1_t + r3_t;        float t_r3 = r1_t - r3_t; \
                    float t_i0 = out[i0][1] + i_2;  float t_i1 = out[i0][1] - i_2; \
                    float t_i2 = i_1 + i_3;          float t_i3 = i_1 - i_3; \
                    out[i0][0] = t_r0 + t_r2; out[i0][1] = t_i0 + t_i2; \
                    out[i1][0] = t_r1 + t_i3; out[i1][1] = t_i1 - t_r3; \
                    out[i2][0] = t_r0 - t_r2; out[i2][1] = t_i0 - t_i2; \
                    out[i3][0] = t_r1 - t_i3; out[i3][1] = t_i1 + t_r3; \
                } \
            } \
        } else { \
            for (int j = 0; j < n4; j += 4) { \
                float c1_a[4], s1_a[4], c2_a[4], s2_a[4], c3_a[4], s3_a[4]; \
                for (int lane = 0; lane < 4; lane++) { \
                    int jj = j + lane; \
                    int idx1 = jj * twiddle_stride; int idx2 = idx1 * 2; int idx3 = idx1 * 3; \
                    c1_a[lane] = twiddle_##N[idx1][0]; s1_a[lane] = twiddle_##N[idx1][1]; \
                    c2_a[lane] = twiddle_##N[idx2][0]; s2_a[lane] = twiddle_##N[idx2][1]; \
                    if (idx3 >= (N / 2)) { \
                        c3_a[lane] = -twiddle_##N[idx3 - (N / 2)][0]; s3_a[lane] = -twiddle_##N[idx3 - (N / 2)][1]; \
                    } else { \
                        c3_a[lane] = twiddle_##N[idx3][0]; s3_a[lane] = twiddle_##N[idx3][1]; \
                    } \
                } \
                float32x4_t v_c1 = vld1q_f32(c1_a); float32x4_t v_s1 = vld1q_f32(s1_a); \
                float32x4_t v_c2 = vld1q_f32(c2_a); float32x4_t v_s2 = vld1q_f32(s2_a); \
                float32x4_t v_c3 = vld1q_f32(c3_a); float32x4_t v_s3 = vld1q_f32(s3_a); \
                for (int blk = 0; blk < N; blk += step) { \
                    int base = blk + j; \
                    float32x4x2_t in0 = vld2q_f32((const float*)&out[base]); \
                    float32x4_t v_r0 = in0.val[0]; float32x4_t v_i0 = in0.val[1]; \
                    float32x4x2_t in1 = vld2q_f32((const float*)&out[base + n4]); \
                    float32x4_t v_r1 = in1.val[0]; float32x4_t v_i1 = in1.val[1]; \
                    float32x4x2_t in2 = vld2q_f32((const float*)&out[base + 2 * n4]); \
                    float32x4_t v_r2 = in2.val[0]; float32x4_t v_i2 = in2.val[1]; \
                    float32x4x2_t in3 = vld2q_f32((const float*)&out[base + 3 * n4]); \
                    float32x4_t v_r3 = in3.val[0]; float32x4_t v_i3 = in3.val[1]; \
                    float32x4_t v_r1_t = vsubq_f32(vmulq_f32(v_r2, v_c1), vmulq_f32(v_i2, v_s1)); \
                    float32x4_t v_i1_t = vaddq_f32(vmulq_f32(v_r2, v_s1), vmulq_f32(v_i2, v_c1)); \
                    float32x4_t v_r2_t = vsubq_f32(vmulq_f32(v_r1, v_c2), vmulq_f32(v_i1, v_s2)); \
                    float32x4_t v_i2_t = vaddq_f32(vmulq_f32(v_r1, v_s2), vmulq_f32(v_i1, v_c2)); \
                    float32x4_t v_r3_t = vsubq_f32(vmulq_f32(v_r3, v_c3), vmulq_f32(v_i3, v_s3)); \
                    float32x4_t v_i3_t = vaddq_f32(vmulq_f32(v_r3, v_s3), vmulq_f32(v_i3, v_c3)); \
                    float32x4_t v_t_r0 = vaddq_f32(v_r0, v_r2_t); float32x4_t v_t_r1 = vsubq_f32(v_r0, v_r2_t); \
                    float32x4_t v_t_r2 = vaddq_f32(v_r1_t, v_r3_t); float32x4_t v_t_r3 = vsubq_f32(v_r1_t, v_r3_t); \
                    float32x4_t v_t_i0 = vaddq_f32(v_i0, v_i2_t); float32x4_t v_t_i1 = vsubq_f32(v_i0, v_i2_t); \
                    float32x4_t v_t_i2 = vaddq_f32(v_i1_t, v_i3_t); float32x4_t v_t_i3 = vsubq_f32(v_i1_t, v_i3_t); \
                    float32x4x2_t out0; out0.val[0] = vaddq_f32(v_t_r0, v_t_r2); out0.val[1] = vaddq_f32(v_t_i0, v_t_i2); \
                    vst2q_f32((float*)&out[base], out0); \
                    float32x4x2_t out1; out1.val[0] = vaddq_f32(v_t_r1, v_t_i3); out1.val[1] = vsubq_f32(v_t_i1, v_t_r3); \
                    vst2q_f32((float*)&out[base + n4], out1); \
                    float32x4x2_t out2; out2.val[0] = vsubq_f32(v_t_r0, v_t_r2); out2.val[1] = vsubq_f32(v_t_i0, v_t_i2); \
                    vst2q_f32((float*)&out[base + 2 * n4], out2); \
                    float32x4x2_t out3; out3.val[0] = vsubq_f32(v_t_r1, v_t_i3); out3.val[1] = vaddq_f32(v_t_i1, v_t_r3); \
                    vst2q_f32((float*)&out[base + 3 * n4], out3); \
                } \
            } \
        } \
    } \
}

#else

// =========================================================================
// 비 ARM 환경 (x86 등 스칼라 Fallback) 매크로
// =========================================================================
#define GENERATE_CUSTOM_FFT_RADIX4_PURE(N) \
extern int bitrev_##N[]; \
extern fftwf_complex twiddle_##N[]; \
void custom_fft_##N##_radix4(fftwf_complex *__restrict__ data) { \
    for (int k = 0; k < N; k++) { \
        int rev = bitrev_##N[k]; \
        if (k < rev) { \
            float temp_r = data[k][0]; data[k][0] = data[rev][0]; data[rev][0] = temp_r; \
            float temp_i = data[k][1]; data[k][1] = data[rev][1]; data[rev][1] = temp_i; \
        } \
    } \
    for (int step = 4; step <= N; step <<= 2) { \
        int n4 = step >> 2; \
        int twiddle_stride = N / step; \
        if (step == 4) { \
            for (int k = 0; k < N; k += 4) { \
                int i0 = k; int i1 = k + 1; int i2 = k + 2; int i3 = k + 3; \
                float r1_t = data[i2][0]; float i_1 = data[i2][1]; \
                float r2_t = data[i1][0]; float i_2 = data[i1][1]; \
                float r3_t = data[i3][0]; float i_3 = data[i3][1]; \
                float t_r0 = data[i0][0] + r2_t; float t_r1 = data[i0][0] - r2_t; \
                float t_r2 = r1_t + r3_t;        float t_r3 = r1_t - r3_t; \
                float t_i0 = data[i0][1] + i_2;  float t_i1 = data[i0][1] - i_2; \
                float t_i2 = i_1 + i_3;          float t_i3 = i_1 - i_3; \
                data[i0][0] = t_r0 + t_r2; data[i0][1] = t_i0 + t_i2; \
                data[i1][0] = t_r1 + t_i3; data[i1][1] = t_i1 - t_r3; \
                data[i2][0] = t_r0 - t_r2; data[i2][1] = t_i0 - t_i2; \
                data[i3][0] = t_r1 - t_i3; data[i3][1] = t_i1 + t_r3; \
            } \
        } else { \
            for (int j = 0; j < n4; j++) { \
                int idx1 = j * twiddle_stride; int idx2 = idx1 * 2; int idx3 = idx1 * 3; \
                float c1 = twiddle_##N[idx1][0]; float s1 = twiddle_##N[idx1][1]; \
                float c2 = twiddle_##N[idx2][0]; float s2 = twiddle_##N[idx2][1]; \
                float c3, s3; \
                if (idx3 >= (N / 2)) { \
                    c3 = -twiddle_##N[idx3 - (N / 2)][0]; s3 = -twiddle_##N[idx3 - (N / 2)][1]; \
                } else { \
                    c3 = twiddle_##N[idx3][0]; s3 = twiddle_##N[idx3][1]; \
                } \
                for (int k = j; k < N; k += step) { \
                    int i0 = k; int i1 = k + n4; int i2 = k + 2 * n4; int i3 = k + 3 * n4; \
                    float r1_t = data[i2][0] * c1 - data[i2][1] * s1; float i_1  = data[i2][0] * s1 + data[i2][1] * c1; \
                    float r2_t = data[i1][0] * c2 - data[i1][1] * s2; float i_2  = data[i1][0] * s2 + data[i1][1] * c2; \
                    float r3_t = data[i3][0] * c3 - data[i3][1] * s3; float i_3  = data[i3][0] * s3 + data[i3][1] * c3; \
                    float t_r0 = data[i0][0] + r2_t; float t_r1 = data[i0][0] - r2_t; \
                    float t_r2 = r1_t + r3_t;        float t_r3 = r1_t - r3_t; \
                    float t_i0 = data[i0][1] + i_2;  float t_i1 = data[i0][1] - i_2; \
                    float t_i2 = i_1 + i_3;          float t_i3 = i_1 - i_3; \
                    data[i0][0] = t_r0 + t_r2; data[i0][1] = t_i0 + t_i2; \
                    data[i1][0] = t_r1 + t_i3; data[i1][1] = t_i1 - t_r3; \
                    data[i2][0] = t_r0 - t_r2; data[i2][1] = t_i0 - t_i2; \
                    data[i3][0] = t_r1 - t_i3; data[i3][1] = t_i1 + t_r3; \
                } \
            } \
        } \
    } \
}

#define GENERATE_CUSTOM_FFT_RADIX4_MIXED(N) \
extern int bitrev_##N[]; \
extern fftwf_complex twiddle_##N[]; \
void custom_fft_##N##_radix4(fftwf_complex *__restrict__ data) { \
    for (int k = 0; k < N; k++) { \
        int rev = bitrev_##N[k]; \
        if (k < rev) { \
            float temp_r = data[k][0]; data[k][0] = data[rev][0]; data[rev][0] = temp_r; \
            float temp_i = data[k][1]; data[k][1] = data[rev][1]; data[rev][1] = temp_i; \
        } \
    } \
    for (int k = 0; k < N; k += 2) { \
        float r0 = data[k][0];       float i0 = data[k][1]; \
        float r1 = data[k + 1][0];   float i1 = data[k + 1][1]; \
        data[k][0]     = r0 + r1;    data[k][1]     = i0 + i1; \
        data[k + 1][0] = r0 - r1;    data[k + 1][1] = i0 - i1; \
    } \
    for (int step = 8; step <= N; step <<= 2) { \
        int n4 = step >> 2; \
        int twiddle_stride = N / step; \
        for (int j = 0; j < n4; j++) { \
            int idx1 = j * twiddle_stride; int idx2 = idx1 * 2; int idx3 = idx1 * 3; \
            float c1 = twiddle_##N[idx1][0]; float s1 = twiddle_##N[idx1][1]; \
            float c2 = twiddle_##N[idx2][0]; float s2 = twiddle_##N[idx2][1]; \
            float c3, s3; \
            if (idx3 >= (N / 2)) { \
                c3 = -twiddle_##N[idx3 - (N / 2)][0]; s3 = -twiddle_##N[idx3 - (N / 2)][1]; \
            } else { \
                c3 = twiddle_##N[idx3][0]; s3 = twiddle_##N[idx3][1]; \
            } \
            for (int k = j; k < N; k += step) { \
                int i0 = k; int i1 = k + n4; int i2 = k + 2 * n4; int i3 = k + 3 * n4; \
                float r1_t = data[i2][0] * c1 - data[i2][1] * s1; float i_1  = data[i2][0] * s1 + data[i2][1] * c1; \
                float r2_t = data[i1][0] * c2 - data[i1][1] * s2; float i_2  = data[i1][0] * s2 + data[i1][1] * c2; \
                float r3_t = data[i3][0] * c3 - data[i3][1] * s3; float i_3  = data[i3][0] * s3 + data[i3][1] * c3; \
                float t_r0 = data[i0][0] + r2_t; float t_r1 = data[i0][0] - r2_t; \
                float t_r2 = r1_t + r3_t;        float t_r3 = r1_t - r3_t; \
                float t_i0 = data[i0][1] + i_2;  float t_i1 = data[i0][1] - i_2; \
                float t_i2 = i_1 + i_3;          float t_i3 = i_1 - i_3; \
                data[i0][0] = t_r0 + t_r2; data[i0][1] = t_i0 + t_i2; \
                data[i1][0] = t_r1 + t_i3; data[i1][1] = t_i1 - t_r3; \
                data[i2][0] = t_r0 - t_r2; data[i2][1] = t_i0 - t_i2; \
                data[i3][0] = t_r1 - t_i3; data[i3][1] = t_i1 + t_r3; \
            } \
        } \
    } \
}
#endif

// 순수 Radix-4 (4의 거듭제곱)
GENERATE_CUSTOM_FFT_RADIX4_PURE(4096)
GENERATE_CUSTOM_FFT_RADIX4_PURE(1024)
GENERATE_CUSTOM_FFT_RADIX4_PURE(256)
GENERATE_CUSTOM_FFT_RADIX4_PURE(64)
GENERATE_CUSTOM_FFT_RADIX4_PURE(16)

// 혼합 Radix-4 (Radix-2 x Radix-4)
GENERATE_CUSTOM_FFT_RADIX4_MIXED(2048)
GENERATE_CUSTOM_FFT_RADIX4_MIXED(512)
GENERATE_CUSTOM_FFT_RADIX4_MIXED(128)
GENERATE_CUSTOM_FFT_RADIX4_MIXED(32)