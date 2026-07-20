#include "myfft.h"

/**
 * @brief 2048 포인트 1D 고속 푸리에 변환 (고정 크기 최적화)
 * @param real 실수부 데이터 배열 포인터 (입출력 겸용, in-place)
 * @param imag 허수부 데이터 배열 포인터 (입출력 겸용, in-place)
 * @details 
 * - 역할: 2048 크기의 데이터에 대해 미리 계산된 LUT를 사용하여 고속 FFT를 수행합니다.
 * - 특징: __restrict__ 키워드를 통한 포인터 최적화 및 ARM NEON (SIMD) 병렬 연산이 적용되었습니다.
 */
void custom_fft_2048_fixed(float *__restrict__ real, float *__restrict__ imag) {
    
    // 1. Bit-Reversal 주소 매핑 (배열 요소를 짝수/홀수 인덱스 특성에 맞춰 재배치)
    for (int i = 0; i < 2048; i++) { 
        int j = bitrev_2048[i]; 
        if (i < j) { 
            float tr = real[i]; 
            real[i] = real[j]; 
            real[j] = tr; 
            
            float ti = imag[i]; 
            imag[i] = imag[j]; 
            imag[j] = ti; 
        } 
    }

    // 2. 나비 연산 (Butterfly Operation) - Bottom-up 방식으로 FFT 병합
    for (int step = 1; step < 2048; step *= 2) { 
        const int jump = step * 2; 
        const int twiddle_step = 2048 / jump;
        
        // 단계가 작을 때는 벡터화 오버헤드가 더 크므로 일반 스칼라 연산 수행
        if (step < 4) { 
            for (int i = 0; i < 2048; i += jump) { 
                for (int j = 0; j < step; j++) { 
                    int curr = i + j; 
                    int k = curr + step; 
                    
                    float tr = twiddle_real_2048[j * twiddle_step]; 
                    float ti = twiddle_imag_2048[j * twiddle_step]; 
                    
                    float t_real = real[k] * tr - imag[k] * ti; 
                    float t_imag = real[k] * ti + imag[k] * tr; 
                    
                    real[k] = real[curr] - t_real; 
                    imag[k] = imag[curr] - t_imag; 
                    real[curr] = real[curr] + t_real; 
                    imag[curr] = imag[curr] + t_imag; 
                } 
            } 
        } 
        // 단계가 4 이상일 때 ARM NEON SIMD 연산 활성화
        else { 
            for (int i = 0; i < 2048; i += jump) {
                float *r_curr_ptr = &real[i]; 
                float *i_curr_ptr = &imag[i];
                float *r_k_ptr = &real[i + step]; 
                float *i_k_ptr = &imag[i + step];

#ifdef __aarch64__
                // 한 번의 명령어로 4개의 복소수를 동시에 연산 (128-bit 벡터)
                for (int j = 0; j < step; j += 4) { 
                    float32x4_t vr_curr = vld1q_f32(r_curr_ptr + j); 
                    float32x4_t vi_curr = vld1q_f32(i_curr_ptr + j); 
                    float32x4_t vr_k = vld1q_f32(r_k_ptr + j); 
                    float32x4_t vi_k = vld1q_f32(i_k_ptr + j); 
                    
                    int tj = j * twiddle_step;
                    
                    // 4개의 Twiddle Factor 로드
                    float tr_arr[4] = { 
                        twiddle_real_2048[tj], 
                        twiddle_real_2048[tj + twiddle_step], 
                        twiddle_real_2048[tj + 2*twiddle_step], 
                        twiddle_real_2048[tj + 3*twiddle_step] 
                    }; 
                    float ti_arr[4] = { 
                        twiddle_imag_2048[tj], 
                        twiddle_imag_2048[tj + twiddle_step], 
                        twiddle_imag_2048[tj + 2*twiddle_step], 
                        twiddle_imag_2048[tj + 3*twiddle_step] 
                    }; 
                    
                    float32x4_t v_tr = vld1q_f32(tr_arr); 
                    float32x4_t v_ti = vld1q_f32(ti_arr); 
                    
                    // 병렬 복소수 곱셈 및 덧셈/뺄셈 수행
                    float32x4_t vt_real = vsubq_f32(vmulq_f32(vr_k, v_tr), vmulq_f32(vi_k, v_ti)); 
                    float32x4_t vt_imag = vaddq_f32(vmulq_f32(vr_k, v_ti), vmulq_f32(vi_k, v_tr)); 
                    
                    // 계산된 4개의 결과를 메모리에 다시 저장
                    vst1q_f32(r_k_ptr + j, vsubq_f32(vr_curr, vt_real)); 
                    vst1q_f32(i_k_ptr + j, vsubq_f32(vi_curr, vt_imag)); 
                    vst1q_f32(r_curr_ptr + j, vaddq_f32(vr_curr, vt_real)); 
                    vst1q_f32(i_curr_ptr + j, vaddq_f32(vi_curr, vt_imag)); 
                }
#else
                // 비 ARM 환경을 위한 Fallback (일반 연산)
                for (int j = 0; j < step; j++) {
                    float tr = twiddle_real_2048[j * twiddle_step]; 
                    float ti = twiddle_imag_2048[j * twiddle_step];
                    
                    float t_real = r_k_ptr[j] * tr - i_k_ptr[j] * ti; 
                    float t_imag = r_k_ptr[j] * ti + i_k_ptr[j] * tr;
                    
                    r_k_ptr[j] = r_curr_ptr[j] - t_real; 
                    i_k_ptr[j] = i_curr_ptr[j] - t_imag;
                    r_curr_ptr[j] = r_curr_ptr[j] + t_real; 
                    i_curr_ptr[j] = i_curr_ptr[j] + t_imag;
                }
#endif
            } 
        } 
    }
}