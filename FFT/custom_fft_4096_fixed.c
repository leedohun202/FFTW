#include "myfft.h"

#ifdef __aarch64__
#include <arm_neon.h>
#endif

extern int bitrev_4096[];
extern float twiddle_real_4096[];
extern float twiddle_imag_4096[];

void custom_fft_4096_fixed(float *__restrict__ real, float *__restrict__ imag) {
    const int N = 4096;
    
    for (int i = 0; i < N; i++) { 
        int j = bitrev_4096[i]; 
        if (i < j) { 
            float tr = real[i]; real[i] = real[j]; real[j] = tr; 
            float ti = imag[i]; imag[i] = imag[j]; imag[j] = ti; 
        } 
    }

    for (int step = 1; step < N; step *= 2) { 
        const int jump = step * 2; 
        const int twiddle_step = N / jump;
        
        if (step < 4) { 
            for (int i = 0; i < N; i += jump) { 
                for (int j = 0; j < step; j++) { 
                    int curr = i + j; 
                    int k = curr + step; 
                    
                    float tr = twiddle_real_4096[j * twiddle_step]; 
                    float ti = twiddle_imag_4096[j * twiddle_step]; 
                    
                    float t_real = real[k] * tr - imag[k] * ti; 
                    float t_imag = real[k] * ti + imag[k] * tr; 
                    
                    real[k] = real[curr] - t_real; 
                    imag[k] = imag[curr] - t_imag; 
                    real[curr] += t_real; 
                    imag[curr] += t_imag; 
                } 
            } 
        } 
        else { 
            for (int i = 0; i < N; i += jump) {
#ifdef __aarch64__
                for (int j = 0; j < step; j += 4) { 
                    int curr = i + j; 
                    int k = curr + step; 
                    
                    float32x4_t vr_curr = vld1q_f32(&real[curr]); 
                    float32x4_t vi_curr = vld1q_f32(&imag[curr]); 
                    float32x4_t vr_k = vld1q_f32(&real[k]); 
                    float32x4_t vi_k = vld1q_f32(&imag[k]); 
                    
                    int tj = j * twiddle_step;
                    float tr_arr[4] = { 
                        twiddle_real_4096[tj], 
                        twiddle_real_4096[tj + twiddle_step], 
                        twiddle_real_4096[tj + 2*twiddle_step], 
                        twiddle_real_4096[tj + 3*twiddle_step] 
                    }; 
                    float ti_arr[4] = { 
                        twiddle_imag_4096[tj], 
                        twiddle_imag_4096[tj + twiddle_step], 
                        twiddle_imag_4096[tj + 2*twiddle_step], 
                        twiddle_imag_4096[tj + 3*twiddle_step] 
                    }; 
                    
                    float32x4_t v_tr = vld1q_f32(tr_arr); 
                    float32x4_t v_ti = vld1q_f32(ti_arr); 
                    
                    float32x4_t vt_real = vsubq_f32(vmulq_f32(vr_k, v_tr), vmulq_f32(vi_k, v_ti)); 
                    float32x4_t vt_imag = vaddq_f32(vmulq_f32(vr_k, v_ti), vmulq_f32(vi_k, v_tr)); 
                    
                    vst1q_f32(&real[k], vsubq_f32(vr_curr, vt_real)); 
                    vst1q_f32(&imag[k], vsubq_f32(vi_curr, vt_imag)); 
                    vst1q_f32(&real[curr], vaddq_f32(vr_curr, vt_real)); 
                    vst1q_f32(&imag[curr], vaddq_f32(vi_curr, vt_imag)); 
                }
#else
                for (int j = 0; j < step; j++) {
                    int curr = i + j; 
                    int k = curr + step; 

                    float tr = twiddle_real_4096[j * twiddle_step]; 
                    float ti = twiddle_imag_4096[j * twiddle_step];
                    
                    float t_real = real[k] * tr - imag[k] * ti; 
                    float t_imag = real[k] * ti + imag[k] * tr;
                    
                    real[k] = real[curr] - t_real; 
                    imag[k] = imag[curr] - t_imag;
                    real[curr] += t_real; 
                    imag[curr] += t_imag;
                }
#endif
            } 
        } 
    }
}