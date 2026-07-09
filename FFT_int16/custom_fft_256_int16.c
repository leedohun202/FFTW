#include "radar_fft.h"

#ifdef __aarch64__
#include <arm_neon.h>
#endif

extern int bitrev_256[];
extern int16_t twiddle_int16_real_256[];
extern int16_t twiddle_int16_imag_256[];

void custom_fft_256_int16(int16_t *__restrict__ real, int16_t *__restrict__ imag) {
    const int N = 256;

    for (int i = 0; i < N; i++) { 
        int j = bitrev_256[i]; 
        if (i < j) { 
            int16_t tr = real[i]; real[i] = real[j]; real[j] = tr; 
            int16_t ti = imag[i]; imag[i] = imag[j]; imag[j] = ti; 
        } 
    }

    for (int step = 1; step < N; step *= 2) { 
        const int jump = step * 2; 
        const int twiddle_step = N / jump;
        
        if (step < 8) { 
            for (int i = 0; i < N; i += jump) { 
                for (int j = 0; j < step; j++) { 
                    int curr = i + j; 
                    int k = curr + step; 
                    
                    int32_t tr = twiddle_int16_real_256[j * twiddle_step]; 
                    int32_t ti = twiddle_int16_imag_256[j * twiddle_step]; 
                    
                    int32_t t_real = (real[k] * tr - imag[k] * ti + 16384) >> 15;
                    int32_t t_imag = (real[k] * ti + imag[k] * tr + 16384) >> 15; 
                    
                    int32_t out_k_r = (real[curr] - t_real) >> 1;
                    int32_t out_k_i = (imag[curr] - t_imag) >> 1;
                    int32_t out_c_r = (real[curr] + t_real) >> 1;
                    int32_t out_c_i = (imag[curr] + t_imag) >> 1;

                    real[k] = (int16_t)out_k_r;
                    imag[k] = (int16_t)out_k_i;
                    real[curr] = (int16_t)out_c_r;
                    imag[curr] = (int16_t)out_c_i;
                } 
            } 
        } 
        else { 
#ifdef __aarch64__
            for (int i = 0; i < N; i += jump) {
                for (int j = 0; j < step; j += 8) { 
                    int curr = i + j; 
                    int k = curr + step; 
                    
                    int16x8_t vr_curr = vld1q_s16(&real[curr]); 
                    int16x8_t vi_curr = vld1q_s16(&imag[curr]); 
                    int16x8_t vr_k    = vld1q_s16(&real[k]); 
                    int16x8_t vi_k    = vld1q_s16(&imag[k]); 
                    
                    int16_t tr_arr[8], ti_arr[8];
                    for (int lane = 0; lane < 8; lane++) {
                        tr_arr[lane] = twiddle_int16_real_256[(j + lane) * twiddle_step];
                        ti_arr[lane] = twiddle_int16_imag_256[(j + lane) * twiddle_step];
                    }
                    int16x8_t v_tr = vld1q_s16(tr_arr); 
                    int16x8_t v_ti = vld1q_s16(ti_arr); 
                    
                    int16x8_t vt_real = vsubq_s16(vqrdmulhq_s16(vr_k, v_tr), vqrdmulhq_s16(vi_k, v_ti)); 
                    int16x8_t vt_imag = vaddq_s16(vqrdmulhq_s16(vr_k, v_ti), vqrdmulhq_s16(vi_k, v_tr)); 
                    
                    int16x8_t v_out_k_r = vshrq_n_s16(vsubq_s16(vr_curr, vt_real), 1);
                    int16x8_t v_out_k_i = vshrq_n_s16(vsubq_s16(vi_curr, vt_imag), 1);
                    int16x8_t v_out_c_r = vshrq_n_s16(vaddq_s16(vr_curr, vt_real), 1);
                    int16x8_t v_out_c_i = vshrq_n_s16(vaddq_s16(vi_curr, vt_imag), 1);

                    vst1q_s16(&real[k], v_out_k_r); 
                    vst1q_s16(&imag[k], v_out_k_i); 
                    vst1q_s16(&real[curr], v_out_c_r); 
                    vst1q_s16(&imag[curr], v_out_c_i); 
                }
            }
#else
            for (int i = 0; i < N; i += jump) {
                for (int j = 0; j < step; j++) { 
                    int curr = i + j; 
                    int k = curr + step; 
                    
                    int32_t tr = twiddle_int16_real_256[j * twiddle_step]; 
                    int32_t ti = twiddle_int16_imag_256[j * twiddle_step]; 
                    
                    int32_t t_real = (real[k] * tr - imag[k] * ti + 16384) >> 15;
                    int32_t t_imag = (real[k] * ti + imag[k] * tr + 16384) >> 15; 
                    
                    int32_t out_k_r = (real[curr] - t_real) >> 1;
                    int32_t out_k_i = (imag[curr] - t_imag) >> 1;
                    int32_t out_c_r = (real[curr] + t_real) >> 1;
                    int32_t out_c_i = (imag[curr] + t_imag) >> 1;

                    real[k] = (int16_t)out_k_r; imag[k] = (int16_t)out_k_i;
                    real[curr] = (int16_t)out_c_r; imag[curr] = (int16_t)out_c_i;
                }
            }
#endif
        } 
    }
}