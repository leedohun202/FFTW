#include "myfft.h"

#ifdef __aarch64__
#include <arm_neon.h>
#endif

extern int bitrev_256[];
extern float twiddle_real_256[];
extern float twiddle_imag_256[];

void custom_fft_256_radix4(float *__restrict__ r, float *__restrict__ i) {
    const int N = 256;

    for (int k = 0; k < N; k++) {
        int rev = bitrev_256[k];
        if (k < rev) {
            float temp_r = r[k]; r[k] = r[rev]; r[rev] = temp_r;
            float temp_i = i[k]; i[k] = i[rev]; i[rev] = temp_i;
        }
    }

    for (int step = 4; step <= N; step <<= 2) {
        int n4 = step >> 2; 
        int twiddle_stride = N / step; 

        if (step == 4) {
            for (int k = 0; k < N; k += 4) {
                int i0 = k; int i1 = k + 1; int i2 = k + 2; int i3 = k + 3;

                float r1_t = r[i2]; float i_1 = i[i2];
                float r2_t = r[i1]; float i_2 = i[i1];
                float r3_t = r[i3]; float i_3 = i[i3];

                float t_r0 = r[i0] + r2_t; float t_r1 = r[i0] - r2_t;
                float t_r2 = r1_t + r3_t;  float t_r3 = r1_t - r3_t;
                float t_i0 = i[i0] + i_2;  float t_i1 = i[i0] - i_2;
                float t_i2 = i_1 + i_3;    float t_i3 = i_1 - i_3;

                r[i0] = t_r0 + t_r2; i[i0] = t_i0 + t_i2;
                r[i1] = t_r1 + t_i3; i[i1] = t_i1 - t_r3;
                r[i2] = t_r0 - t_r2; i[i2] = t_i0 - t_i2;
                r[i3] = t_r1 - t_i3; i[i3] = t_i1 + t_r3;
            }
        } 
        else {
#ifdef __aarch64__
            for (int j = 0; j < n4; j += 4) {
                float c1_a[4], s1_a[4], c2_a[4], s2_a[4], c3_a[4], s3_a[4];
                for (int lane = 0; lane < 4; lane++) {
                    int jj = j + lane;
                    int idx1 = jj * twiddle_stride; int idx2 = idx1 * 2; int idx3 = idx1 * 3;

                    c1_a[lane] = twiddle_real_256[idx1]; s1_a[lane] = twiddle_imag_256[idx1];
                    c2_a[lane] = twiddle_real_256[idx2]; s2_a[lane] = twiddle_imag_256[idx2];
                    if (idx3 >= (N / 2)) {
                        c3_a[lane] = -twiddle_real_256[idx3 - (N / 2)]; s3_a[lane] = -twiddle_imag_256[idx3 - (N / 2)];
                    } else {
                        c3_a[lane] = twiddle_real_256[idx3]; s3_a[lane] = twiddle_imag_256[idx3];
                    }
                }

                float32x4_t v_c1 = vld1q_f32(c1_a); float32x4_t v_s1 = vld1q_f32(s1_a);
                float32x4_t v_c2 = vld1q_f32(c2_a); float32x4_t v_s2 = vld1q_f32(s2_a);
                float32x4_t v_c3 = vld1q_f32(c3_a); float32x4_t v_s3 = vld1q_f32(s3_a);

                for (int blk = 0; blk < N; blk += step) {
                    int base = blk + j;
                    float32x4_t v_r0 = vld1q_f32(&r[base]);          float32x4_t v_i0 = vld1q_f32(&i[base]);
                    float32x4_t v_r1 = vld1q_f32(&r[base + n4]);     float32x4_t v_i1 = vld1q_f32(&i[base + n4]);
                    float32x4_t v_r2 = vld1q_f32(&r[base + 2 * n4]); float32x4_t v_i2 = vld1q_f32(&i[base + 2 * n4]);
                    float32x4_t v_r3 = vld1q_f32(&r[base + 3 * n4]); float32x4_t v_i3 = vld1q_f32(&i[base + 3 * n4]);

                    float32x4_t v_r1_t = vsubq_f32(vmulq_f32(v_r2, v_c1), vmulq_f32(v_i2, v_s1));
                    float32x4_t v_i1_t = vaddq_f32(vmulq_f32(v_r2, v_s1), vmulq_f32(v_i2, v_c1));
                    float32x4_t v_r2_t = vsubq_f32(vmulq_f32(v_r1, v_c2), vmulq_f32(v_i1, v_s2));
                    float32x4_t v_i2_t = vaddq_f32(vmulq_f32(v_r1, v_s2), vmulq_f32(v_i1, v_c2));
                    float32x4_t v_r3_t = vsubq_f32(vmulq_f32(v_r3, v_c3), vmulq_f32(v_i3, v_s3));
                    float32x4_t v_i3_t = vaddq_f32(vmulq_f32(v_r3, v_s3), vmulq_f32(v_i3, v_c3));

                    float32x4_t v_t_r0 = vaddq_f32(v_r0, v_r2_t); float32x4_t v_t_r1 = vsubq_f32(v_r0, v_r2_t);
                    float32x4_t v_t_r2 = vaddq_f32(v_r1_t, v_r3_t); float32x4_t v_t_r3 = vsubq_f32(v_r1_t, v_r3_t);
                    float32x4_t v_t_i0 = vaddq_f32(v_i0, v_i2_t); float32x4_t v_t_i1 = vsubq_f32(v_i0, v_i2_t);
                    float32x4_t v_t_i2 = vaddq_f32(v_i1_t, v_i3_t); float32x4_t v_t_i3 = vsubq_f32(v_i1_t, v_i3_t);

                    vst1q_f32(&r[base], vaddq_f32(v_t_r0, v_t_r2));             vst1q_f32(&i[base], vaddq_f32(v_t_i0, v_t_i2));
                    vst1q_f32(&r[base + n4], vaddq_f32(v_t_r1, v_t_i3));        vst1q_f32(&i[base + n4], vsubq_f32(v_t_i1, v_t_r3));
                    vst1q_f32(&r[base + 2 * n4], vsubq_f32(v_t_r0, v_t_r2));    vst1q_f32(&i[base + 2 * n4], vsubq_f32(v_t_i0, v_t_i2));
                    vst1q_f32(&r[base + 3 * n4], vsubq_f32(v_t_r1, v_t_i3));    vst1q_f32(&i[base + 3 * n4], vaddq_f32(v_t_i1, v_t_r3));
                }
            }
#else
            for (int j = 0; j < n4; j++) {
                int idx1 = j * twiddle_stride; int idx2 = idx1 * 2; int idx3 = idx1 * 3;
                float c1 = twiddle_real_256[idx1]; float s1 = twiddle_imag_256[idx1];
                float c2 = twiddle_real_256[idx2]; float s2 = twiddle_imag_256[idx2];
                float c3, s3;
                if (idx3 >= (N / 2)) {
                    c3 = -twiddle_real_256[idx3 - (N / 2)]; s3 = -twiddle_imag_256[idx3 - (N / 2)];
                } else {
                    c3 = twiddle_real_256[idx3]; s3 = twiddle_imag_256[idx3];
                }

                for (int k = j; k < N; k += step) {
                    int i0 = k; int i1 = k + n4; int i2 = k + 2 * n4; int i3 = k + 3 * n4;
                    float r1_t = r[i2] * c1 - i[i2] * s1; float i_1  = r[i2] * s1 + i[i2] * c1;
                    float r2_t = r[i1] * c2 - i[i1] * s2; float i_2  = r[i1] * s2 + i[i1] * c2;
                    float r3_t = r[i3] * c3 - i[i3] * s3; float i_3  = r[i3] * s3 + i[i3] * c3;
                    float t_r0 = r[i0] + r2_t; float t_r1 = r[i0] - r2_t;
                    float t_r2 = r1_t + r3_t;  float t_r3 = r1_t - r3_t;
                    float t_i0 = i[i0] + i_2;  float t_i1 = i[i0] - i_2;
                    float t_i2 = i_1 + i_3;    float t_i3 = i_1 - i_3;
                    r[i0] = t_r0 + t_r2; i[i0] = t_i0 + t_i2;
                    r[i1] = t_r1 + t_i3; i[i1] = t_i1 - t_r3;
                    r[i2] = t_r0 - t_r2; i[i2] = t_i0 - t_i2;
                    r[i3] = t_r1 - t_i3; i[i3] = t_i1 + t_r3;
                }
            }
#endif
        }
    }
}