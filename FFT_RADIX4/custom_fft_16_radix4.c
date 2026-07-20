#include "myfft.h"

extern int bitrev_16[];
extern float twiddle_real_16[];
extern float twiddle_imag_16[];

void custom_fft_16_radix4(float *__restrict__ r, float *__restrict__ i) {
    const int N = 16;

    for (int k = 0; k < N; k++) {
        int rev = bitrev_16[k];
        if (k < rev) {
            float temp_r = r[k]; r[k] = r[rev]; r[rev] = temp_r;
            float temp_i = i[k]; i[k] = i[rev]; i[rev] = temp_i;
        }
    }

    // 16은 4^2 이므로 step = 4, 16 두 스테이지만 수행 (스칼라 전용)
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
            for (int j = 0; j < n4; j++) {
                int idx1 = j * twiddle_stride; int idx2 = idx1 * 2; int idx3 = idx1 * 3;

                float c1 = twiddle_real_16[idx1]; float s1 = twiddle_imag_16[idx1];
                float c2 = twiddle_real_16[idx2]; float s2 = twiddle_imag_16[idx2];
                float c3, s3;
                
                if (idx3 >= (N / 2)) {
                    c3 = -twiddle_real_16[idx3 - (N / 2)]; s3 = -twiddle_imag_16[idx3 - (N / 2)];
                } else {
                    c3 = twiddle_real_16[idx3]; s3 = twiddle_imag_16[idx3];
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
        }
    }
}