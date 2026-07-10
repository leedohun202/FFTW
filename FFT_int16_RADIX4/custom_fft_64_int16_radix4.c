#include "radar_fft.h"
#include <stdint.h>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

extern int bitrev_64[];
extern int16_t twiddle_int16_real_64[];
extern int16_t twiddle_int16_imag_64[];

void custom_fft_64_int16_radix4(int16_t *__restrict__ r, int16_t *__restrict__ i) {
    const int N = 64;

    // 1. Bit-Reversal
    for (int k = 0; k < N; k++) {
        int rev = bitrev_64[k];
        if (k < rev) {
            int16_t temp_r = r[k]; r[k] = r[rev]; r[rev] = temp_r;
            int16_t temp_i = i[k]; i[k] = i[rev]; i[rev] = temp_i;
        }
    }

    // 2. 순수 Radix-4 스테이지 (step: 4, 16, 64)
    for (int step = 4; step <= N; step <<= 2) {
        int n4 = step >> 2; 
        int twiddle_stride = N / step; 

        if (step < 64) { // step 4, 16은 n4 < 8 이므로 C 처리
            for (int j = 0; j < n4; j++) {
                int idx1 = j * twiddle_stride; int idx2 = idx1 * 2; int idx3 = idx1 * 3;

                int32_t c1 = twiddle_int16_real_64[idx1]; int32_t s1 = twiddle_int16_imag_64[idx1];
                int32_t c2 = twiddle_int16_real_64[idx2]; int32_t s2 = twiddle_int16_imag_64[idx2];
                int32_t c3, s3;
                
                if (idx3 >= (N / 2)) {
                    c3 = -twiddle_int16_real_64[idx3 - (N / 2)]; s3 = -twiddle_int16_imag_64[idx3 - (N / 2)];
                } else {
                    c3 = twiddle_int16_real_64[idx3]; s3 = twiddle_int16_imag_64[idx3];
                }

                for (int k = j; k < N; k += step) {
                    int i0 = k, i1 = k + n4, i2 = k + 2 * n4, i3 = k + 3 * n4;

                    int32_t r1_in = r[i2], i1_in = i[i2]; 
                    int32_t r2_in = r[i1], i2_in = i[i1]; 
                    int32_t r3_in = r[i3], i3_in = i[i3];

                    int32_t r1_t = (r1_in * c1 - i1_in * s1) >> 15; int32_t i_1  = (r1_in * s1 + i1_in * c1) >> 15;
                    int32_t r2_t = (r2_in * c2 - i2_in * s2) >> 15; int32_t i_2  = (r2_in * s2 + i2_in * c2) >> 15;
                    int32_t r3_t = (r3_in * c3 - i3_in * s3) >> 15; int32_t i_3  = (r3_in * s3 + i3_in * c3) >> 15;

                    int32_t t_r0 = r[i0] + r2_t; int32_t t_r1 = r[i0] - r2_t;
                    int32_t t_r2 = r1_t + r3_t;  int32_t t_r3 = r1_t - r3_t;
                    int32_t t_i0 = i[i0] + i_2;  int32_t t_i1 = i[i0] - i_2;
                    int32_t t_i2 = i_1 + i_3;    int32_t t_i3 = i_1 - i_3;

                    r[i0] = (t_r0 + t_r2) >> 2; i[i0] = (t_i0 + t_i2) >> 2;
                    r[i1] = (t_r1 + t_i3) >> 2; i[i1] = (t_i1 - t_r3) >> 2;
                    r[i2] = (t_r0 - t_r2) >> 2; i[i2] = (t_i0 - t_i2) >> 2;
                    r[i3] = (t_r1 - t_i3) >> 2; i[i3] = (t_i1 + t_r3) >> 2;
                }
            }
        } 
        else {
#ifdef __aarch64__
            // 🚀 본격 NEON 구간 (step: 64)
            for (int j = 0; j < n4; j += 8) {
                int16_t c1_a[8], s1_a[8], c2_a[8], s2_a[8], c3_a[8], s3_a[8];
                for (int lane = 0; lane < 8; lane++) {
                    int jj = j + lane;
                    int idx1 = jj * twiddle_stride; int idx2 = idx1 * 2; int idx3 = idx1 * 3;

                    c1_a[lane] = twiddle_int16_real_64[idx1]; s1_a[lane] = twiddle_int16_imag_64[idx1];
                    c2_a[lane] = twiddle_int16_real_64[idx2]; s2_a[lane] = twiddle_int16_imag_64[idx2];
                    if (idx3 >= (N / 2)) {
                        c3_a[lane] = -twiddle_int16_real_64[idx3 - (N / 2)];
                        s3_a[lane] = -twiddle_int16_imag_64[idx3 - (N / 2)];
                    } else {
                        c3_a[lane] = twiddle_int16_real_64[idx3];
                        s3_a[lane] = twiddle_int16_imag_64[idx3];
                    }
                }
                int16x8_t vc1 = vld1q_s16(c1_a); int16x8_t vs1 = vld1q_s16(s1_a);
                int16x8_t vc2 = vld1q_s16(c2_a); int16x8_t vs2 = vld1q_s16(s2_a);
                int16x8_t vc3 = vld1q_s16(c3_a); int16x8_t vs3 = vld1q_s16(s3_a);

                for (int blk = 0; blk < N; blk += step) {
                    int base = blk + j;
                    int16x8_t vr0 = vld1q_s16(&r[base]);             int16x8_t vi0 = vld1q_s16(&i[base]);
                    int16x8_t vr1_in = vld1q_s16(&r[base + 2 * n4]); int16x8_t vi1_in = vld1q_s16(&i[base + 2 * n4]);
                    int16x8_t vr2_in = vld1q_s16(&r[base + n4]);     int16x8_t vi2_in = vld1q_s16(&i[base + n4]);
                    int16x8_t vr3_in = vld1q_s16(&r[base + 3 * n4]); int16x8_t vi3_in = vld1q_s16(&i[base + 3 * n4]);

                    int16x8_t vr1_t = vqsubq_s16(vqdmulhq_s16(vr1_in, vc1), vqdmulhq_s16(vi1_in, vs1));
                    int16x8_t vi_1  = vqaddq_s16(vqdmulhq_s16(vr1_in, vs1), vqdmulhq_s16(vi1_in, vc1));
                    int16x8_t vr2_t = vqsubq_s16(vqdmulhq_s16(vr2_in, vc2), vqdmulhq_s16(vi2_in, vs2));
                    int16x8_t vi_2  = vqaddq_s16(vqdmulhq_s16(vr2_in, vs2), vqdmulhq_s16(vi2_in, vc2));
                    int16x8_t vr3_t = vqsubq_s16(vqdmulhq_s16(vr3_in, vc3), vqdmulhq_s16(vi3_in, vs3));
                    int16x8_t vi_3  = vqaddq_s16(vqdmulhq_s16(vr3_in, vs3), vqdmulhq_s16(vi3_in, vc3));

                    int16x8_t tr0 = vqaddq_s16(vr0, vr2_t); int16x8_t tr1 = vqsubq_s16(vr0, vr2_t);
                    int16x8_t tr2 = vqaddq_s16(vr1_t, vr3_t); int16x8_t tr3 = vqsubq_s16(vr1_t, vr3_t);
                    int16x8_t ti0 = vqaddq_s16(vi0, vi_2);  int16x8_t ti1 = vqsubq_s16(vi0, vi_2);
                    int16x8_t ti2 = vqaddq_s16(vi_1, vi_3);   int16x8_t ti3 = vqsubq_s16(vi_1, vi_3);

                    vst1q_s16(&r[base], vshrq_n_s16(vqaddq_s16(tr0, tr2), 2));
                    vst1q_s16(&i[base], vshrq_n_s16(vqaddq_s16(ti0, ti2), 2));
                    vst1q_s16(&r[base + n4], vshrq_n_s16(vqaddq_s16(tr1, ti3), 2));
                    vst1q_s16(&i[base + n4], vshrq_n_s16(vqsubq_s16(ti1, tr3), 2));
                    vst1q_s16(&r[base + 2 * n4], vshrq_n_s16(vqsubq_s16(tr0, tr2), 2));
                    vst1q_s16(&i[base + 2 * n4], vshrq_n_s16(vqsubq_s16(ti0, ti2), 2));
                    vst1q_s16(&r[base + 3 * n4], vshrq_n_s16(vqsubq_s16(tr1, ti3), 2));
                    vst1q_s16(&i[base + 3 * n4], vshrq_n_s16(vqaddq_s16(ti1, tr3), 2));
                }
            }
#endif
        }
    }
}