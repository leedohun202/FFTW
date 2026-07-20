#include "myfft.h"

/**
 * @brief 512 포인트 1D 고속 푸리에 변환 (고정 크기 최적화)
 * @param real 실수부 데이터 배열 포인터 (입출력 겸용)
 * @param imag 허수부 데이터 배열 포인터 (입출력 겸용)
 * @details LUT를 이용한 비트 리버설 및 ARM NEON 벡터 연산(SIMD)을 통해 하드웨어 가속을 수행합니다.
 */
void custom_fft_512_fixed(float *__restrict__ real, float *__restrict__ imag) {
    
    // 1. Bit-Reversal 주소 매핑 (LUT 사용)
    for (int i = 0; i < 512; i++) { 
        int j = bitrev_512[i]; 
        // 절반만 스와핑하여 중복 교환 방지
        if (i < j) { 
            float tr = real[i]; 
            real[i] = real[j]; 
            real[j] = tr; 
            
            float ti = imag[i]; 
            imag[i] = imag[j]; 
            imag[j] = ti; 
        } 
    }

    // 2. Butterfly 연산 (Bottom-up 방식)
    for (int step = 1; step < 512; step *= 2) { 
        const int jump = step * 2; 
        const int twiddle_step = 512 / jump;
        
        // step이 4보다 작을 때는 벡터화 오버헤드가 더 크므로 일반 연산 수행
        if (step < 4) { 
            for (int i = 0; i < 512; i += jump) { 
                for (int j = 0; j < step; j++) { 
                    int curr = i + j; 
                    int k = curr + step; 
                    
                    float tr = twiddle_real_512[j * twiddle_step]; 
                    float ti = twiddle_imag_512[j * twiddle_step]; 
                    
                    float t_real = real[k] * tr - imag[k] * ti; 
                    float t_imag = real[k] * ti + imag[k] * tr; 
                    
                    real[k] = real[curr] - t_real; 
                    imag[k] = imag[curr] - t_imag; 
                    real[curr] = real[curr] + t_real; 
                    imag[curr] = imag[curr] + t_imag; 
                } 
            } 
        } 
        // step이 4 이상일 때부터 NEON 벡터 연산 수행
        else { 
            for (int i = 0; i < 512; i += jump) {
#ifdef __aarch64__
                // 4개의 데이터를 한 번에 처리 (SIMD)
                for (int j = 0; j < step; j += 4) { 
                    int curr = i + j; 
                    int k = curr + step; 
                    
                    // 메모리에서 4개씩 로드
                    float32x4_t vr_curr = vld1q_f32(&real[curr]); 
                    float32x4_t vi_curr = vld1q_f32(&imag[curr]); 
                    float32x4_t vr_k = vld1q_f32(&real[k]); 
                    float32x4_t vi_k = vld1q_f32(&imag[k]); 
                    
                    // Twiddle Factor 로드
                    float tr_arr[4] = { 
                        twiddle_real_512[(j+0)*twiddle_step], 
                        twiddle_real_512[(j+1)*twiddle_step], 
                        twiddle_real_512[(j+2)*twiddle_step], 
                        twiddle_real_512[(j+3)*twiddle_step] 
                    }; 
                    float ti_arr[4] = { 
                        twiddle_imag_512[(j+0)*twiddle_step], 
                        twiddle_imag_512[(j+1)*twiddle_step], 
                        twiddle_imag_512[(j+2)*twiddle_step], 
                        twiddle_imag_512[(j+3)*twiddle_step] 
                    }; 
                    
                    float32x4_t v_tr = vld1q_f32(tr_arr); 
                    float32x4_t v_ti = vld1q_f32(ti_arr); 
                    
                    // 복소수 곱셈 및 덧셈 (병렬)
                    float32x4_t vt_real = vsubq_f32(vmulq_f32(vr_k, v_tr), vmulq_f32(vi_k, v_ti)); 
                    float32x4_t vt_imag = vaddq_f32(vmulq_f32(vr_k, v_ti), vmulq_f32(vi_k, v_tr)); 
                    
                    // 결과 저장
                    vst1q_f32(&real[k], vsubq_f32(vr_curr, vt_real)); 
                    vst1q_f32(&imag[k], vsubq_f32(vi_curr, vt_imag)); 
                    vst1q_f32(&real[curr], vaddq_f32(vr_curr, vt_real)); 
                    vst1q_f32(&imag[curr], vaddq_f32(vi_curr, vt_imag)); 
                }
#else
                // 일반 ARM 코어가 아닐 경우의 Fallback 로직
                for (int j = 0; j < step; j++) { 
                    int curr = i + j; 
                    int k = curr + step; 
                    float tr = twiddle_real_512[j * twiddle_step]; 
                    float ti = twiddle_imag_512[j * twiddle_step]; 
                    
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