#include "radar_fft.h"

/**
 * @brief 128 포인트 1D 고속 푸리에 변환 (고정 크기 최적화)
 * @param real 실수부 데이터 배열 포인터
 * @param imag 허수부 데이터 배열 포인터
 */
void custom_fft_128_fixed(float *__restrict__ real, float *__restrict__ imag) {
    
    // 1. Bit-Reversal
    for (int i = 0; i < 128; i++) { 
        int j = bitrev_128[i]; 
        if (i < j) { 
            float tr = real[i]; real[i] = real[j]; real[j] = tr; 
            float ti = imag[i]; imag[i] = imag[j]; imag[j] = ti; 
        } 
    }

    // 2. Butterfly 연산
    for (int step = 1; step < 128; step *= 2) { 
        const int jump = step * 2; 
        const int twiddle_step = 128 / jump;
        
        if (step < 4) { 
            for (int i = 0; i < 128; i += jump) { 
                for (int j = 0; j < step; j++) { 
                    int curr = i + j; 
                    int k = curr + step; 
                    
                    float tr = twiddle_real_128[j * twiddle_step]; 
                    float ti = twiddle_imag_128[j * twiddle_step]; 
                    
                    float t_real = real[k] * tr - imag[k] * ti; 
                    float t_imag = real[k] * ti + imag[k] * tr; 
                    
                    real[k] = real[curr] - t_real; 
                    imag[k] = imag[curr] - t_imag; 
                    real[curr] = real[curr] + t_real; 
                    imag[curr] = imag[curr] + t_imag; 
                } 
            } 
        } else { 
            for (int i = 0; i < 128; i += jump) {
#ifdef __aarch64__
                for (int j = 0; j < step; j += 4) { 
                    int curr = i + j; 
                    int k = curr + step; 
                    
                    float32x4_t vr_curr = vld1q_f32(&real[curr]); 
                    float32x4_t vi_curr = vld1q_f32(&imag[curr]); 
                    float32x4_t vr_k = vld1q_f32(&real[k]); 
                    float32x4_t vi_k = vld1q_f32(&imag[k]); 
                    
                    float tr_arr[4] = { 
                        twiddle_real_128[(j+0)*twiddle_step], 
                        twiddle_real_128[(j+1)*twiddle_step], 
                        twiddle_real_128[(j+2)*twiddle_step], 
                        twiddle_real_128[(j+3)*twiddle_step] 
                    }; 
                    float ti_arr[4] = { 
                        twiddle_imag_128[(j+0)*twiddle_step], 
                        twiddle_imag_128[(j+1)*twiddle_step], 
                        twiddle_imag_128[(j+2)*twiddle_step], 
                        twiddle_imag_128[(j+3)*twiddle_step] 
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
                    
                    float tr = twiddle_real_128[j * twiddle_step]; 
                    float ti = twiddle_imag_128[j * twiddle_step]; 
                    
                    float t_real = real[k] * tr - imag[k] * ti; 
                    float t_imag = real[k] * ti + imag[k] * tr; 
                    
                    real[k] = real[curr] - t_real; 
                    imag[k] = imag[curr] - t_imag; 
                    real[curr] = real[curr] + t_real; 
                    imag[curr] = imag[curr] + t_imag; 
                }
#endif
            } 
        } 
    }
}