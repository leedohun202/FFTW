#include "radar_fft.h"
#include <stdint.h>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

extern int bitrev_2048[];
extern int16_t twiddle_int16_real_2048[];
extern int16_t twiddle_int16_imag_2048[];

void custom_fft_2048_int16_radix4(int16_t *__restrict__ r, int16_t *__restrict__ i) {
    const int N = 2048;

    for (int k = 0; k < N; k++) {
        int rev = bitrev_2048[k];
        if (k < rev) {
            int16_t temp_r = r[k]; r[k] = r[rev]; r[rev] = temp_r;
            int16_t temp_i = i[k]; i[k] = i[rev]; i[rev] = temp_i;
        }
    }

    // 💥 [스칼라 구간 1] 반올림(+1) 복구: (val + 1) >> 1
    for (int k = 0; k < N; k += 2) {
        int16_t r0 = r[k];       int16_t i0 = i[k];
        int16_t r1 = r[k + 1];   int16_t i1 = i[k + 1];
        r[k]     = (r0 + r1 + 1) >> 1;    i[k]     = (i0 + i1 + 1) >> 1;
        r[k + 1] = (r0 - r1 + 1) >> 1;    i[k + 1] = (i0 - i1 + 1) >> 1;
    }

    for (int step = 8; step <= N; step <<= 2) {
        int n4 = step >> 2; 
        int twiddle_stride = N / step; 

        if (step == 8) {
            for (int j = 0; j < n4; j++) {
                int idx1 = j * twiddle_stride;
                int idx2 = idx1 * 2;
                int idx3 = idx1 * 3;

                int32_t c1 = twiddle_int16_real_2048[idx1]; int32_t s1 = twiddle_int16_imag_2048[idx1];
                int32_t c2 = twiddle_int16_real_2048[idx2]; int32_t s2 = twiddle_int16_imag_2048[idx2];
                int32_t c3, s3;
                
                if (idx3 >= (N / 2)) {
                    c3 = -twiddle_int16_real_2048[idx3 - (N / 2)];
                    s3 = -twiddle_int16_imag_2048[idx3 - (N / 2)];
                } else {
                    c3 = twiddle_int16_real_2048[idx3];
                    s3 = twiddle_int16_imag_2048[idx3];
                }

                for (int k = j; k < N; k += step) {
                    int i0 = k, i1 = k + n4, i2 = k + 2 * n4, i3 = k + 3 * n4;

                    int32_t r1_in = r[i2], i1_in = i[i2]; 
                    int32_t r2_in = r[i1], i2_in = i[i1]; 
                    int32_t r3_in = r[i3], i3_in = i[i3];

                    int32_t r1_t = (r1_in * c1 - i1_in * s1) >> 15;
                    int32_t i_1  = (r1_in * s1 + i1_in * c1) >> 15;
                    int32_t r2_t = (r2_in * c2 - i2_in * s2) >> 15;
                    int32_t i_2  = (r2_in * s2 + i2_in * c2) >> 15;
                    int32_t r3_t = (r3_in * c3 - i3_in * s3) >> 15;
                    int32_t i_3  = (r3_in * s3 + i3_in * c3) >> 15;

                    int32_t t_r0 = r[i0] + r2_t; int32_t t_r1 = r[i0] - r2_t;
                    int32_t t_r2 = r1_t + r3_t;  int32_t t_r3 = r1_t - r3_t;
                    int32_t t_i0 = i[i0] + i_2;  int32_t t_i1 = i[i0] - i_2;
                    int32_t t_i2 = i_1 + i_3;    int32_t t_i3 = i_1 - i_3;

                    // 💥 [스칼라 구간 2] 반올림(+2) 복구: (val + 2) >> 2
                    r[i0] = (t_r0 + t_r2 + 2) >> 2; i[i0] = (t_i0 + t_i2 + 2) >> 2;
                    r[i1] = (t_r1 + t_i3 + 2) >> 2; i[i1] = (t_i1 - t_r3 + 2) >> 2;
                    r[i2] = (t_r0 - t_r2 + 2) >> 2; i[i2] = (t_i0 - t_i2 + 2) >> 2;
                    r[i3] = (t_r1 - t_i3 + 2) >> 2; i[i3] = (t_i1 + t_r3 + 2) >> 2;
                }
            }
        } 
        else {
#ifdef __aarch64__
            for (int j = 0; j < n4; j += 8) {
                int16_t c1_a[8], s1_a[8], c2_a[8], s2_a[8], c3_a[8], s3_a[8];
                for (int lane = 0; lane < 8; lane++) {
                    int jj = j + lane;
                    int idx1 = jj * twiddle_stride;
                    int idx2 = idx1 * 2;
                    int idx3 = idx1 * 3;

                    c1_a[lane] = twiddle_int16_real_2048[idx1];
                    s1_a[lane] = twiddle_int16_imag_2048[idx1];
                    c2_a[lane] = twiddle_int16_real_2048[idx2];
                    s2_a[lane] = twiddle_int16_imag_2048[idx2];

                    if (idx3 >= (N / 2)) {
                        c3_a[lane] = -twiddle_int16_real_2048[idx3 - (N / 2)];
                        s3_a[lane] = -twiddle_int16_imag_2048[idx3 - (N / 2)];
                    } else {
                        c3_a[lane] = twiddle_int16_real_2048[idx3];
                        s3_a[lane] = twiddle_int16_imag_2048[idx3];
                    }
                }

                int16x8_t vc1 = vld1q_s16(c1_a); int16x8_t vs1 = vld1q_s16(s1_a);
                int16x8_t vc2 = vld1q_s16(c2_a); int16x8_t vs2 = vld1q_s16(s2_a);
                int16x8_t vc3 = vld1q_s16(c3_a); int16x8_t vs3 = vld1q_s16(s3_a);

                for (int blk = 0; blk < N; blk += step) {
                    int base = blk + j;
                    
                    int16x8_t vr0 = vld1q_s16(&r[base]);
                    int16x8_t vi0 = vld1q_s16(&i[base]);
                    
                    int16x8_t vr1_in = vld1q_s16(&r[base + 2 * n4]);
                    int16x8_t vi1_in = vld1q_s16(&i[base + 2 * n4]);
                    int16x8_t vr2_in = vld1q_s16(&r[base + n4]);
                    int16x8_t vi2_in = vld1q_s16(&i[base + n4]);
                    
                    int16x8_t vr3_in = vld1q_s16(&r[base + 3 * n4]);
                    int16x8_t vi3_in = vld1q_s16(&i[base + 3 * n4]);

                    int16x8_t vr1_t = vqsubq_s16(vqdmulhq_s16(vr1_in, vc1), vqdmulhq_s16(vi1_in, vs1));
                    int16x8_t vi_1  = vqaddq_s16(vqdmulhq_s16(vr1_in, vs1), vqdmulhq_s16(vi1_in, vc1));
                    int16x8_t vr2_t = vqsubq_s16(vqdmulhq_s16(vr2_in, vc2), vqdmulhq_s16(vi2_in, vs2));
                    int16x8_t vi_2  = vqaddq_s16(vqdmulhq_s16(vr2_in, vs2), vqdmulhq_s16(vi2_in, vc2));
                    int16x8_t vr3_t = vqsubq_s16(vqdmulhq_s16(vr3_in, vc3), vqdmulhq_s16(vi3_in, vs3));
                    int16x8_t vi_3  = vqaddq_s16(vqdmulhq_s16(vr3_in, vs3), vqdmulhq_s16(vi3_in, vc3));

                    // 💥 [NEON 구간 1] 하드웨어 반올림 시프트 적용 (vrshrq)
                    int16x8_t sr0 = vrshrq_n_s16(vr0, 1);
                    int16x8_t si0 = vrshrq_n_s16(vi0, 1);
                    int16x8_t sr1_t = vrshrq_n_s16(vr1_t, 1);
                    int16x8_t si_1  = vrshrq_n_s16(vi_1, 1);
                    int16x8_t sr2_t = vrshrq_n_s16(vr2_t, 1);
                    int16x8_t si_2  = vrshrq_n_s16(vi_2, 1);
                    int16x8_t sr3_t = vrshrq_n_s16(vr3_t, 1);
                    int16x8_t si_3  = vrshrq_n_s16(vi_3, 1);

                    int16x8_t tr0 = vaddq_s16(sr0, sr2_t);
                    int16x8_t tr1 = vsubq_s16(sr0, sr2_t);
                    int16x8_t tr2 = vaddq_s16(sr1_t, sr3_t);
                    int16x8_t tr3 = vsubq_s16(sr1_t, sr3_t);

                    int16x8_t ti0 = vaddq_s16(si0, si_2);
                    int16x8_t ti1 = vsubq_s16(si0, si_2);
                    int16x8_t ti2 = vaddq_s16(si_1, si_3);
                    int16x8_t ti3 = vsubq_s16(si_1, si_3);

                    // 💥 [NEON 구간 2] 하드웨어 반올림 시프트 적용 (vrshrq)
                    tr0 = vrshrq_n_s16(tr0, 1); tr1 = vrshrq_n_s16(tr1, 1);
                    tr2 = vrshrq_n_s16(tr2, 1); tr3 = vrshrq_n_s16(tr3, 1);
                    ti0 = vrshrq_n_s16(ti0, 1); ti1 = vrshrq_n_s16(ti1, 1);
                    ti2 = vrshrq_n_s16(ti2, 1); ti3 = vrshrq_n_s16(ti3, 1);

                    vst1q_s16(&r[base], vaddq_s16(tr0, tr2));
                    vst1q_s16(&i[base], vaddq_s16(ti0, ti2));

                    vst1q_s16(&r[base + n4], vaddq_s16(tr1, ti3));
                    vst1q_s16(&i[base + n4], vsubq_s16(ti1, tr3));

                    vst1q_s16(&r[base + 2 * n4], vsubq_s16(tr0, tr2));
                    vst1q_s16(&i[base + 2 * n4], vsubq_s16(ti0, ti2));

                    vst1q_s16(&r[base + 3 * n4], vsubq_s16(tr1, ti3));
                    vst1q_s16(&i[base + 3 * n4], vaddq_s16(ti1, tr3));
                }
            }
#endif
        }
    }
}