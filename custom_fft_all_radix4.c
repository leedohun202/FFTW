#include "myfft.h"
#include <stdint.h>
#include <math.h>

// =========================================================================
// [1] AArch64 (64비트) NEON 최적화 버전
// =========================================================================
#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>

// [순수 Radix-4 매크로]
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

// [혼합 Radix-4 매크로]
#define GENERATE_CUSTOM_FFT_RADIX4_MIXED(N) \
extern int bitrev_##N[]; \
extern fftwf_complex twiddle_##N[]; \
void custom_fft_##N##_radix4_fused(const fftwf_complex *__restrict__ in, fftwf_complex *__restrict__ out) { \
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

// =========================================================================
// [2] 32비트 ARMv7 NEON 최적화 버전 (vmlsq / vmlaq 파이프라인 적용)
// =========================================================================
#elif defined(__arm__) && defined(__ARM_NEON)
#include <arm_neon.h>

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
                    float32x4_t v_r1_t = vmulq_f32(v_r2, v_c1); v_r1_t = vmlsq_f32(v_r1_t, v_i2, v_s1); \
                    float32x4_t v_i1_t = vmulq_f32(v_r2, v_s1); v_i1_t = vmlaq_f32(v_i1_t, v_i2, v_c1); \
                    float32x4_t v_r2_t = vmulq_f32(v_r1, v_c2); v_r2_t = vmlsq_f32(v_r2_t, v_i1, v_s2); \
                    float32x4_t v_i2_t = vmulq_f32(v_r1, v_s2); v_i2_t = vmlaq_f32(v_i2_t, v_i1, v_c2); \
                    float32x4_t v_r3_t = vmulq_f32(v_r3, v_c3); v_r3_t = vmlsq_f32(v_r3_t, v_i3, v_s3); \
                    float32x4_t v_i3_t = vmulq_f32(v_r3, v_s3); v_i3_t = vmlaq_f32(v_i3_t, v_i3, v_c3); \
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

#define GENERATE_CUSTOM_FFT_RADIX4_MIXED(N) \
extern int bitrev_##N[]; \
extern fftwf_complex twiddle_##N[]; \
void custom_fft_##N##_radix4_fused(const fftwf_complex *__restrict__ in, fftwf_complex *__restrict__ out) { \
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
                    float32x4_t v_r1_t = vmulq_f32(v_r2, v_c1); v_r1_t = vmlsq_f32(v_r1_t, v_i2, v_s1); \
                    float32x4_t v_i1_t = vmulq_f32(v_r2, v_s1); v_i1_t = vmlaq_f32(v_i1_t, v_i2, v_c1); \
                    float32x4_t v_r2_t = vmulq_f32(v_r1, v_c2); v_r2_t = vmlsq_f32(v_r2_t, v_i1, v_s2); \
                    float32x4_t v_i2_t = vmulq_f32(v_r1, v_s2); v_i2_t = vmlaq_f32(v_i2_t, v_i1, v_c2); \
                    float32x4_t v_r3_t = vmulq_f32(v_r3, v_c3); v_r3_t = vmlsq_f32(v_r3_t, v_i3, v_s3); \
                    float32x4_t v_i3_t = vmulq_f32(v_r3, v_s3); v_i3_t = vmlaq_f32(v_i3_t, v_i3, v_c3); \
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
// [3] ARMv7-R / In-Order FPU 전용 2-Way Interleaved 스칼라 Fallback 버전
// =========================================================================
#define GENERATE_CUSTOM_FFT_RADIX4_PURE(N) \
extern int bitrev_##N[]; \
extern fftwf_complex twiddle_##N[]; \
void custom_fft_##N##_radix4(fftwf_complex *__restrict__ data) { \
    /* 1. Bit Reversal */ \
    for (int k = 0; k < N; k++) { \
        int rev = bitrev_##N[k]; \
        if (k < rev) { \
            float temp_r = data[k][0]; data[k][0] = data[rev][0]; data[rev][0] = temp_r; \
            float temp_i = data[k][1]; data[k][1] = data[rev][1]; data[rev][1] = temp_i; \
        } \
    } \
    /* 2. Radix-4 Stages */ \
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
                /* 🚀 [2-Way Interleaved Unrolling 메인 루프] */ \
                int k = j; \
                for (; k + step < N; k += 2 * step) { \
                    int k_A = k; int k_B = k + step; \
                    int i0_A = k_A, i1_A = k_A + n4, i2_A = k_A + 2*n4, i3_A = k_A + 3*n4; \
                    int i0_B = k_B, i1_B = k_B + n4, i2_B = k_B + 2*n4, i3_B = k_B + 3*n4; \
                    \
                    /* [단계 1] 메모리 로드 교차 */ \
                    float a0_r = data[i0_A][0], a0_i = data[i0_A][1]; \
                    float b0_r = data[i0_B][0], b0_i = data[i0_B][1]; \
                    float a1_r = data[i1_A][0], a1_i = data[i1_A][1]; \
                    float b1_r = data[i1_B][0], b1_i = data[i1_B][1]; \
                    float a2_r = data[i2_A][0], a2_i = data[i2_A][1]; \
                    float b2_r = data[i2_B][0], b2_i = data[i2_B][1]; \
                    float a3_r = data[i3_A][0], a3_i = data[i3_A][1]; \
                    float b3_r = data[i3_B][0], b3_i = data[i3_B][1]; \
                    \
                    /* [단계 2] 복소수 곱셈 인터리빙 (FPU Latency Hiding) */ \
                    float a1_tr = a2_r * c1 - a2_i * s1; \
                    float b1_tr = b2_r * c1 - b2_i * s1; /* A 곱셈 결과 대기 시간에 B 수행 */ \
                    float a1_ti = a2_r * s1 + a2_i * c1; \
                    float b1_ti = b2_r * s1 + b2_i * c1; \
                    \
                    float a2_tr = a1_r * c2 - a1_i * s2; \
                    float b2_tr = b1_r * c2 - b1_i * s2; \
                    float a2_ti = a1_r * s2 + a1_i * c2; \
                    float b2_ti = b1_r * s2 + b1_i * c2; \
                    \
                    float a3_tr = a3_r * c3 - a3_i * s3; \
                    float b3_tr = b3_r * c3 - b3_i * s3; \
                    float a3_ti = a3_r * s3 + a3_i * c3; \
                    float b3_ti = b3_r * s3 + b3_i * c3; \
                    \
                    /* [단계 3] 1차 덧셈/뺄셈 인터리빙 */ \
                    float a_tr0 = a0_r + a2_tr; float b_tr0 = b0_r + b2_tr; \
                    float a_tr1 = a0_r - a2_tr; float b_tr1 = b0_r - b2_tr; \
                    float a_tr2 = a1_tr + a3_tr; float b_tr2 = b1_tr + b3_tr; \
                    float a_tr3 = a1_tr - a3_tr; float b_tr3 = b1_tr - b3_tr; \
                    \
                    float a_ti0 = a0_i + a2_ti; float b_ti0 = b0_i + b2_ti; \
                    float a_ti1 = a0_i - a2_ti; float b_ti1 = b0_i - b2_ti; \
                    float a_ti2 = a1_ti + a3_ti; float b_ti2 = b1_ti + b3_ti; \
                    float a_ti3 = a1_ti - a3_ti; float b_ti3 = b1_ti - b3_ti; \
                    \
                    /* [단계 4] 최종 결과 저장 */ \
                    data[i0_A][0] = a_tr0 + a_tr2; data[i0_A][1] = a_ti0 + a_ti2; \
                    data[i0_B][0] = b_tr0 + b_tr2; data[i0_B][1] = b_ti0 + b_ti2; \
                    \
                    data[i1_A][0] = a_tr1 + a_ti3; data[i1_A][1] = a_ti1 - a_tr3; \
                    data[i1_B][0] = b_tr1 + b_ti3; data[i1_B][1] = b_ti1 - b_tr3; \
                    \
                    data[i2_A][0] = a_tr0 - a_tr2; data[i2_A][1] = a_ti0 - a_ti2; \
                    data[i2_B][0] = b_tr0 - b_tr2; data[i2_B][1] = b_ti0 - b_ti2; \
                    \
                    data[i3_A][0] = a_tr1 - a_ti3; data[i3_A][1] = a_ti1 + a_tr3; \
                    data[i3_B][0] = b_tr1 - b_ti3; data[i3_B][1] = b_ti1 + b_tr3; \
                } \
                /* 🚀 [자투리 처리 루프] 반복 횟수가 홀수일 때 안전하게 마무리 */ \
                for (; k < N; k += step) { \
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

// =========================================================================
// ARMv7-R / In-Order FPU 전용 2-Way Interleaved Fused Mixed-Radix 매크로
// =========================================================================
#define GENERATE_CUSTOM_FFT_RADIX4_MIXED(N) \
extern int bitrev_##N[]; \
extern fftwf_complex twiddle_##N[]; \
void custom_fft_##N##_radix4_fused(const fftwf_complex *__restrict__ in, fftwf_complex *__restrict__ out) { \
    /* -------------------------------------------------------------------- */ \
    /* 1. Fused Bit-Reversal + First Radix-2 Stage (2-Way Interleaved)      */ \
    /* -------------------------------------------------------------------- */ \
    int k_bit = 0; \
    for (; k_bit + 2 < N; k_bit += 4) { \
        /* A, B 두 개의 Radix-2 나비 연산 데이터 인덱스 로드 */ \
        int r0_A = bitrev_##N[k_bit];     int r1_A = bitrev_##N[k_bit + 1]; \
        int r0_B = bitrev_##N[k_bit + 2]; int r1_B = bitrev_##N[k_bit + 3]; \
        \
        float r0_A_r = in[r0_A][0], r0_A_i = in[r0_A][1]; \
        float r0_B_r = in[r0_B][0], r0_B_i = in[r0_B][1]; \
        float r1_A_r = in[r1_A][0], r1_A_i = in[r1_A][1]; \
        float r1_B_r = in[r1_B][0], r1_B_i = in[r1_B][1]; \
        \
        /* FPU Latency Hiding: A 연산과 B 연산의 덧셈/뺄셈 명령어를 엇갈려 배치 */ \
        out[k_bit][0]     = r0_A_r + r1_A_r; out[k_bit][1]     = r0_A_i + r1_A_i; \
        out[k_bit + 2][0] = r0_B_r + r1_B_r; out[k_bit + 2][1] = r0_B_i + r1_B_i; \
        out[k_bit + 1][0] = r0_A_r - r1_A_r; out[k_bit + 1][1] = r0_A_i - r1_A_i; \
        out[k_bit + 3][0] = r0_B_r - r1_B_r; out[k_bit + 3][1] = r0_B_i - r1_B_i; \
    } \
    /* 잔여 1개 세트 처리 (N이 4의 배수가 아닐 경우 대비 안전장치) */ \
    for (; k_bit < N; k_bit += 2) { \
        int r0 = bitrev_##N[k_bit];   int r1 = bitrev_##N[k_bit + 1]; \
        float r0_r = in[r0][0]; float r0_i = in[r0][1]; \
        float r1_r = in[r1][0]; float r1_i = in[r1][1]; \
        out[k_bit][0]     = r0_r + r1_r; out[k_bit][1]     = r0_i + r1_i; \
        out[k_bit + 1][0] = r0_r - r1_r; out[k_bit + 1][1] = r0_i - r1_i; \
    } \
    \
    /* -------------------------------------------------------------------- */ \
    /* 2. Radix-4 Stages (step = 8부터 N까지, 2-Way Interleaved In-Place)  */ \
    /* -------------------------------------------------------------------- */ \
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
            \
            /* 🚀 [Radix-4 : 2-Way Interleaved 메인 루프] */ \
            int k = j; \
            for (; k + step < N; k += 2 * step) { \
                int k_A = k, k_B = k + step; \
                int i0_A = k_A, i1_A = k_A + n4, i2_A = k_A + 2*n4, i3_A = k_A + 3*n4; \
                int i0_B = k_B, i1_B = k_B + n4, i2_B = k_B + 2*n4, i3_B = k_B + 3*n4; \
                \
                /* [메모리 로드 교차] */ \
                float a0_r = out[i0_A][0], a0_i = out[i0_A][1]; \
                float b0_r = out[i0_B][0], b0_i = out[i0_B][1]; \
                float a1_r = out[i1_A][0], a1_i = out[i1_A][1]; \
                float b1_r = out[i1_B][0], b1_i = out[i1_B][1]; \
                float a2_r = out[i2_A][0], a2_i = out[i2_A][1]; \
                float b2_r = out[i2_B][0], b2_i = out[i2_B][1]; \
                float a3_r = out[i3_A][0], a3_i = out[i3_A][1]; \
                float b3_r = out[i3_B][0], b3_i = out[i3_B][1]; \
                \
                /* [복소수 곱셈 인터리빙] A 연산과 B 연산을 번갈아 실행하여 Stall 제거 */ \
                float a1_tr = a2_r * c1 - a2_i * s1; float b1_tr = b2_r * c1 - b2_i * s1; \
                float a1_ti = a2_r * s1 + a2_i * c1; float b1_ti = b2_r * s1 + b2_i * c1; \
                float a2_tr = a1_r * c2 - a1_i * s2; float b2_tr = b1_r * c2 - b1_i * s2; \
                float a2_ti = a1_r * s2 + a1_i * c2; float b2_ti = b1_r * s2 + b1_i * c2; \
                float a3_tr = a3_r * c3 - a3_i * s3; float b3_tr = b3_r * c3 - b3_i * s3; \
                float a3_ti = a3_r * s3 + a3_i * c3; float b3_ti = b3_r * s3 + b3_i * c3; \
                \
                /* [1차 덧셈/뺄셈 인터리빙] */ \
                float a_tr0 = a0_r + a2_tr; float b_tr0 = b0_r + b2_tr; \
                float a_tr1 = a0_r - a2_tr; float b_tr1 = b0_r - b2_tr; \
                float a_tr2 = a1_tr + a3_tr; float b_tr2 = b1_tr + b3_tr; \
                float a_tr3 = a1_tr - a3_tr; float b_tr3 = b1_tr - b3_tr; \
                float a_ti0 = a0_i + a2_ti; float b_ti0 = b0_i + b2_ti; \
                float a_ti1 = a0_i - a2_ti; float b_ti1 = b0_i - b2_ti; \
                float a_ti2 = a1_ti + a3_ti; float b_ti2 = b1_ti + b3_ti; \
                float a_ti3 = a1_ti - a3_ti; float b_ti3 = b1_ti - b3_ti; \
                \
                /* [최종 결과 저장] */ \
                out[i0_A][0] = a_tr0 + a_tr2; out[i0_A][1] = a_ti0 + a_ti2; \
                out[i0_B][0] = b_tr0 + b_tr2; out[i0_B][1] = b_ti0 + b_ti2; \
                out[i1_A][0] = a_tr1 + a_ti3; out[i1_A][1] = a_ti1 - a_tr3; \
                out[i1_B][0] = b_tr1 + b_ti3; out[i1_B][1] = b_ti1 - b_tr3; \
                out[i2_A][0] = a_tr0 - a_tr2; out[i2_A][1] = a_ti0 - a_ti2; \
                out[i2_B][0] = b_tr0 - b_tr2; out[i2_B][1] = b_ti0 - b_ti2; \
                out[i3_A][0] = a_tr1 - a_ti3; out[i3_A][1] = a_ti1 + a_tr3; \
                out[i3_B][0] = b_tr1 - b_ti3; out[i3_B][1] = b_ti1 + b_tr3; \
            } \
            /* 🚀 [자투리 루프] 반복 횟수가 홀수일 때 안전하게 1개 처리 */ \
            for (; k < N; k += step) { \
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
    } \
}

// =========================================================================
// [4] 스칼라 Fallback 버전 (armv7-r / Cortex-R / Non-NEON)
// =========================================================================
/* 
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
void custom_fft_##N##_radix4_fused(const fftwf_complex *__restrict__ in, fftwf_complex *__restrict__ out) { \
    for (int k = 0; k < N; k += 2) { \
        int r0 = bitrev_##N[k]; \
        int r1 = bitrev_##N[k + 1]; \
        float r0_r = in[r0][0]; float r0_i = in[r0][1]; \
        float r1_r = in[r1][0]; float r1_i = in[r1][1]; \
        out[k][0]     = r0_r + r1_r; out[k][1]     = r0_i + r1_i; \
        out[k + 1][0] = r0_r - r1_r; out[k + 1][1] = r0_i - r1_i; \
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
    } \
}
*/
#endif

// =========================================================================
// [매크로 호출부: 실제 함수 생성]
// =========================================================================
GENERATE_CUSTOM_FFT_RADIX4_PURE(4096)
GENERATE_CUSTOM_FFT_RADIX4_PURE(1024)
GENERATE_CUSTOM_FFT_RADIX4_PURE(256)
GENERATE_CUSTOM_FFT_RADIX4_PURE(64)
GENERATE_CUSTOM_FFT_RADIX4_PURE(16)

GENERATE_CUSTOM_FFT_RADIX4_MIXED(2048)
GENERATE_CUSTOM_FFT_RADIX4_MIXED(512)
GENERATE_CUSTOM_FFT_RADIX4_MIXED(128)
GENERATE_CUSTOM_FFT_RADIX4_MIXED(32)