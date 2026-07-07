#include "radar_fft.h"

/**
 * @brief 1024 포인트 1D 고속 푸리에 변환 (고정 크기 최적화)
 * @param real 실수부 데이터 배열 포인터 (입출력 겸용)
 * @param imag 허수부 데이터 배열 포인터 (입출력 겸용)
 * @details Range FFT 등에 주로 사용되는 1024 샘플용 전용 가속 함수입니다.
 */
void custom_fft_1024_fixed(float *__restrict__ real, float *__restrict__ imag) {
    
    for (int i = 0; i < 1024; i++) { 
        int j = bitrev_1024[i]; 
        if (i < j) { 
            float tr = real[i]; real[i] = real[j]; real[j] = tr; 
            float ti = imag[i]; imag[i] = imag[j]; imag[j] = ti; 
        } 
    }

    for (int step = 1; step < 1024; step *= 2) { 
        const int jump = step * 2; 
        const int twiddle_step = 1024 / jump;
        
        if (step < 4) { 
            for (int i = 0; i < 1024; i += jump) { 
                for (int j = 0; j < step; j++) { 
                    int curr = i + j; int k = curr + step; 
                    float tr = twiddle_real_1024[j * twiddle_step]; 
                    float ti = twiddle_imag_1024[j * twiddle_step]; 
                    float t_real = real[k] * tr - imag[k] * ti; 
                    float t_imag = real[k] * ti + imag[k] * tr; 
                    real[k] = real[curr] - t_real; imag[k] = imag[curr] - t_imag; 
                    real[curr] = real[curr] + t_real; imag[curr] = imag[curr] + t_imag; 
                } 
            } 
        } else { 
            for (int i = 0; i < 1024; i += jump) {
#ifdef __aarch64__
                for (int j = 0; j < step; j += 4) { 
                    int curr = i + j; int k = curr + step; 
                    
                    float32x4_t vr_curr = vld1q_f32(&real[curr]); 
                    float32x4_t vi_curr = vld1q_f32(&imag[curr]); 
                    float32x4_t vr_k = vld1q_f32(&real[k]); 
                    float32x4_t vi_k = vld1q_f32(&imag[k]); 
                    
                    // 🔥 레지스터 직결 로드 패킹
                    float32x4_t v_tr = vdupq_n_f32(0.0f);
                    v_tr = vsetq_lane_f32(twiddle_real_1024[(j+0)*twiddle_step], v_tr, 0);
                    v_tr = vsetq_lane_f32(twiddle_real_1024[(j+1)*twiddle_step], v_tr, 1);
                    v_tr = vsetq_lane_f32(twiddle_real_1024[(j+2)*twiddle_step], v_tr, 2);
                    v_tr = vsetq_lane_f32(twiddle_real_1024[(j+3)*twiddle_step], v_tr, 3);
                    
                    float32x4_t v_ti = vdupq_n_f32(0.0f);
                    v_ti = vsetq_lane_f32(twiddle_imag_1024[(j+0)*twiddle_step], v_ti, 0);
                    v_ti = vsetq_lane_f32(twiddle_imag_1024[(j+1)*twiddle_step], v_ti, 1);
                    v_ti = vsetq_lane_f32(twiddle_imag_1024[(j+2)*twiddle_step], v_ti, 2);
                    v_ti = vsetq_lane_f32(twiddle_imag_1024[(j+3)*twiddle_step], v_ti, 3);
                    
                    float32x4_t vt_real = vsubq_f32(vmulq_f32(vr_k, v_tr), vmulq_f32(vi_k, v_ti)); 
                    float32x4_t vt_imag = vaddq_f32(vmulq_f32(vr_k, v_ti), vmulq_f32(vi_k, v_tr)); 
                    
                    vst1q_f32(&real[k], vsubq_f32(vr_curr, vt_real)); 
                    vst1q_f32(&imag[k], vsubq_f32(vi_curr, vt_imag)); 
                    vst1q_f32(&real[curr], vaddq_f32(vr_curr, vt_real)); 
                    vst1q_f32(&imag[curr], vaddq_f32(vi_curr, vt_imag)); 
                }
#else
                for (int j = 0; j < step; j++) { 
                    int curr = i + j; int k = curr + step; 
                    float tr = twiddle_real_1024[j * twiddle_step]; float ti = twiddle_imag_1024[j * twiddle_step]; 
                    float t_real = real[k] * tr - imag[k] * ti; float t_imag = real[k] * ti + imag[k] * tr; 
                    real[k] = real[curr] - t_real; imag[k] = imag[curr] - t_imag; 
                    real[curr] = real[curr] + t_real; imag[curr] = imag[curr] + t_imag; 
                }
#endif
            } 
        } 
    }
}