#include "radar_fft.h"

#ifdef __aarch64__
#include <arm_neon.h>
#endif

extern int bitrev_4096[];
extern int16_t twiddle_int16_real_4096[];
extern int16_t twiddle_int16_imag_4096[];

void custom_fft_4096_int16_radix8(int16_t *__restrict__ r, int16_t *__restrict__ i) {
    const int N = 4096;

    // [1단계] 고정소수점용 비트 리버설 
    for (int k = 0; k < N; k++) {
        int rev = bitrev_4096[k];
        if (k < rev) {
            int16_t temp_r = r[k]; r[k] = r[rev]; r[rev] = temp_r;
            int16_t temp_i = i[k]; i[k] = i[rev]; i[rev] = temp_i;
        }
    }

    // [2단계] Radix-8 고속 NEON 나비 연산 스테이지 (4번의 스텝)
    for (int step = 8; step <= N; step *= 8) {
        int n8 = step >> 3; 
        int twiddle_stride = N / step;

        for (int j = 0; j < n8; j += 8) {
            int16_t c1[8], s1[8], c2[8], s2[8], c3[8], s3[8];
            int16_t c4[8], s4[8], c5[8], s5[8], c6[8], s6[8], c7[8], s7[8];

            for (int lane = 0; lane < 8; lane++) {
                int jj = j + lane;
                int idx1 = jj * twiddle_stride;
                
                c1[lane] = twiddle_int16_real_4096[idx1];     s1[lane] = twiddle_int16_imag_4096[idx1];
                c2[lane] = twiddle_int16_real_4096[idx1 * 2]; s2[lane] = twiddle_int16_imag_4096[idx1 * 2];
                c3[lane] = twiddle_int16_real_4096[idx1 * 3]; s3[lane] = twiddle_int16_imag_4096[idx1 * 3];
                c4[lane] = twiddle_int16_real_4096[idx1 * 4]; s4[lane] = twiddle_int16_imag_4096[idx1 * 4];
                c5[lane] = twiddle_int16_real_4096[idx1 * 5]; s5[lane] = twiddle_int16_imag_4096[idx1 * 5];
                c6[lane] = twiddle_int16_real_4096[idx1 * 6]; s6[lane] = twiddle_int16_imag_4096[idx1 * 6];
                c7[lane] = twiddle_int16_real_4096[idx1 * 7]; s7[lane] = twiddle_int16_imag_4096[idx1 * 7];
            }

#ifdef __aarch64__
            int16x8_t v_c1 = vld1q_s16(c1); int16x8_t v_s1 = vld1q_s16(s1);
            int16x8_t v_c2 = vld1q_s16(c2); int16x8_t v_s2 = vld1q_s16(s2);
            int16x8_t v_c3 = vld1q_s16(c3); int16x8_t v_s3 = vld1q_s16(s3);
            int16x8_t v_c4 = vld1q_s16(c4); int16x8_t v_s4 = vld1q_s16(s4);
            int16x8_t v_c5 = vld1q_s16(c5); int16x8_t v_s5 = vld1q_s16(s5);
            int16x8_t v_c6 = vld1q_s16(c6); int16x8_t v_s6 = vld1q_s16(s6);
            int16x8_t v_c7 = vld1q_s16(c7); int16x8_t v_s7 = vld1q_s16(s7);

            for (int blk = 0; blk < N; blk += step) {
                int base = blk + j;

                int16x8_t v_r0 = vld1q_s16(&r[base]);         int16x8_t v_i0 = vld1q_s16(&i[base]);
                int16x8_t v_r1 = vld1q_s16(&r[base + n8]);     int16x8_t v_i1 = vld1q_s16(&i[base + n8]);
                int16x8_t v_r2 = vld1q_s16(&r[base + 2*n8]);   int16x8_t v_i2 = vld1q_s16(&i[base + 2*n8]);
                int16x8_t v_r3 = vld1q_s16(&r[base + 3*n8]);   int16x8_t v_i3 = vld1q_s16(&i[base + 3*n8]);
                int16x8_t v_r4 = vld1q_s16(&r[base + 4*n8]);   int16x8_t v_i4 = vld1q_s16(&i[base + 4*n8]);
                int16x8_t v_r5 = vld1q_s16(&r[base + 5*n8]);   int16x8_t v_i5 = vld1q_s16(&i[base + 5*n8]);
                int16x8_t v_r6 = vld1q_s16(&r[base + 6*n8]);   int16x8_t v_i6 = vld1q_s16(&i[base + 6*n8]);
                int16x8_t v_r7 = vld1q_s16(&r[base + 7*n8]);   int16x8_t v_i7 = vld1q_s16(&i[base + 7*n8]);

                int16x8_t v_r1_t = vsubq_s16(vqrdmulhq_s16(v_r1, v_c1), vqrdmulhq_s16(v_i1, v_s1));
                int16x8_t v_i1_t = vaddq_s16(vqrdmulhq_s16(v_i1, v_c1), vqrdmulhq_s16(v_r1, v_s1));
                int16x8_t v_r2_t = vsubq_s16(vqrdmulhq_s16(v_r2, v_c2), vqrdmulhq_s16(v_i2, v_s2));
                int16x8_t v_i2_t = vaddq_s16(vqrdmulhq_s16(v_i2, v_c2), vqrdmulhq_s16(v_r2, v_s2));
                int16x8_t v_r3_t = vsubq_s16(vqrdmulhq_s16(v_r3, v_c3), vqrdmulhq_s16(v_i3, v_s3));
                int16x8_t v_i3_t = vaddq_s16(vqrdmulhq_s16(v_i3, v_c3), vqrdmulhq_s16(v_r3, v_s3));
                int16x8_t v_r4_t = vsubq_s16(vqrdmulhq_s16(v_r4, v_c4), vqrdmulhq_s16(v_i4, v_s4));
                int16x8_t v_i4_t = vaddq_s16(vqrdmulhq_s16(v_i4, v_c4), vqrdmulhq_s16(v_r4, v_s4));
                int16x8_t v_r5_t = vsubq_s16(vqrdmulhq_s16(v_r5, v_c5), vqrdmulhq_s16(v_i5, v_s5));
                int16x8_t v_i5_t = vaddq_s16(vqrdmulhq_s16(v_i5, v_c5), vqrdmulhq_s16(v_r5, v_s5));
                int16x8_t v_r6_t = vsubq_s16(vqrdmulhq_s16(v_r6, v_c6), vqrdmulhq_s16(v_i6, v_s6));
                int16x8_t v_i6_t = vaddq_s16(vqrdmulhq_s16(v_i6, v_c6), vqrdmulhq_s16(v_r6, v_s6));
                int16x8_t v_r7_t = vsubq_s16(vqrdmulhq_s16(v_r7, v_c7), vqrdmulhq_s16(v_i7, v_s7));
                int16x8_t v_i7_t = vaddq_s16(vqrdmulhq_s16(v_i7, v_c7), vqrdmulhq_s16(v_r7, v_s7));

                int16x8_t r04_p = vaddq_s16(v_r0, v_r4_t); int16x8_t r04_m = vsubq_s16(v_r0, v_r4_t);
                int16x8_t r15_p = vaddq_s16(v_r1_t, v_r5_t); int16x8_t r15_m = vsubq_s16(v_r1_t, v_r5_t);
                int16x8_t r26_p = vaddq_s16(v_r2_t, v_r6_t); int16x8_t r26_m = vsubq_s16(v_r2_t, v_r6_t);
                int16x8_t r37_p = vaddq_s16(v_r3_t, v_r7_t); int16x8_t r37_m = vsubq_s16(v_r3_t, v_r7_t);

                int16x8_t i04_p = vaddq_s16(v_i0, v_i4_t); int16x8_t i04_m = vsubq_s16(v_i0, v_i4_t);
                int16x8_t i15_p = vaddq_s16(v_i1_t, v_i5_t); int16x8_t i15_m = vsubq_s16(v_i1_t, v_i5_t);
                int16x8_t i26_p = vaddq_s16(v_i2_t, v_i6_t); int16x8_t i26_m = vsubq_s16(v_i2_t, v_i6_t);
                int16x8_t i37_p = vaddq_s16(v_i3_t, v_i7_t); int16x8_t i37_m = vsubq_s16(v_i3_t, v_i7_t);

                int16x8_t v_out_r0 = vshrq_n_s16(vaddq_s16(vaddq_s16(r04_p, r26_p), vaddq_s16(r15_p, r37_p)), 3);
                int16x8_t v_out_i0 = vshrq_n_s16(vaddq_s16(vaddq_s16(i04_p, i26_p), vaddq_s16(i15_p, i37_p)), 3);
                int16x8_t v_out_r1 = vshrq_n_s16(vaddq_s16(vsubq_s16(r04_m, i26_m), vsubq_s16(r15_m, i37_m)), 3);
                int16x8_t v_out_i1 = vshrq_n_s16(vaddq_s16(vaddq_s16(i04_m, r26_m), vsubq_s16(i15_m, r37_m)), 3);
                int16x8_t v_out_r2 = vshrq_n_s16(vaddq_s16(vsubq_s16(r04_p, r26_p), vsubq_s16(i15_m, i37_m)), 3);
                int16x8_t v_out_i2 = vshrq_n_s16(vaddq_s16(vsubq_s16(i04_p, i26_p), vaddq_s16(r15_m, r37_m)), 3);
                int16x8_t v_out_r3 = vshrq_n_s16(vsubq_s16(vaddq_s16(r04_m, i26_m), vaddq_s16(r15_m, i37_m)), 3);
                int16x8_t v_out_i3 = vshrq_n_s16(vsubq_s16(vsubq_s16(i04_m, r26_m), vsubq_s16(i15_m, r37_m)), 3);
                int16x8_t v_out_r4 = vshrq_n_s16(vsubq_s16(vaddq_s16(r04_p, r26_p), vaddq_s16(r15_p, r37_p)), 3);
                int16x8_t v_out_i4 = vshrq_n_s16(vsubq_s16(vaddq_s16(i04_p, i26_p), vaddq_s16(i15_p, i37_p)), 3);
                int16x8_t v_out_r5 = vshrq_n_s16(vsubq_s16(vaddq_s16(r04_m, i26_m), vsubq_s16(r15_m, i37_m)), 3);
                int16x8_t v_out_i5 = vshrq_n_s16(vsubq_s16(vsubq_s16(i04_m, r26_m), vsubq_s16(i15_m, r37_m)), 3);
                int16x8_t v_out_r6 = vshrq_n_s16(vaddq_s16(vsubq_s16(r04_p, r26_p), vsubq_s16(r15_p, r37_p)), 3);
                int16x8_t v_out_i6 = vshrq_n_s16(vsubq_s16(vsubq_s16(i04_p, i26_p), vsubq_s16(r15_m, r37_m)), 3);
                int16x8_t v_out_r7 = vshrq_n_s16(vaddq_s16(vsubq_s16(r04_m, i26_m), vaddq_s16(r15_m, i37_m)), 3);
                int16x8_t v_out_i7 = vshrq_n_s16(vaddq_s16(vsubq_s16(i04_m, r26_m), vsubq_s16(i15_m, r37_m)), 3);

                vst1q_s16(&r[base], v_out_r0);       vst1q_s16(&i[base], v_out_i0);
                vst1q_s16(&r[base + n8], v_out_r1);   vst1q_s16(&i[base + n8], v_out_i1);
                vst1q_s16(&r[base + 2*n8], v_out_r2); vst1q_s16(&i[base + 2*n8], v_out_i2);
                vst1q_s16(&r[base + 3*n8], v_out_r3); vst1q_s16(&i[base + 3*n8], v_out_i3);
                vst1q_s16(&r[base + 4*n8], v_out_r4); vst1q_s16(&i[base + 4*n8], v_out_i4);
                vst1q_s16(&r[base + 5*n8], v_out_r5); vst1q_s16(&i[base + 5*n8], v_out_i5);
                vst1q_s16(&r[base + 6*n8], v_out_r6); vst1q_s16(&i[base + 6*n8], v_out_i6);
                vst1q_s16(&r[base + 7*n8], v_out_r7); vst1q_s16(&i[base + 7*n8], v_out_i7);
            }
#endif
        }
    }
}