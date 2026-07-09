#include "radar_fft.h"

#ifdef __aarch64__
#include <arm_neon.h>
#endif

// 💡 2048 포인트 고정소수점 전용 전역 변수 참조
extern int bitrev_2048[];
extern int16_t twiddle_int16_real_2048[];
extern int16_t twiddle_int16_imag_2048[];

void custom_fft_2048_int16(int16_t *__restrict__ real, int16_t *__restrict__ imag) {
    const int N = 2048;

    // [1단계] Bit-Reversal (정수형 데이터 스와핑)
    for (int i = 0; i < N; i++) { 
        int j = bitrev_2048[i]; 
        if (i < j) { 
            int16_t tr = real[i]; real[i] = real[j]; real[j] = tr; 
            int16_t ti = imag[i]; imag[i] = imag[j]; imag[j] = ti; 
        } 
    }

    // [2단계] 고정소수점 Radix-2 버터플라이 연산
    for (int step = 1; step < N; step *= 2) { 
        const int jump = step * 2; 
        const int twiddle_step = N / jump;
        
        // -----------------------------------------------------------------
        // 공통 구역: step < 8 일 때는 8개 레이인을 꽉 채울 수 없으므로 스칼라 처리
        // -----------------------------------------------------------------
        if (step < 8) { 
            for (int i = 0; i < N; i += jump) { 
                for (int j = 0; j < step; j++) { 
                    int curr = i + j; 
                    int k = curr + step; 
                    
                    // Q15 포맷 회전 인자 로드
                    int32_t tr = twiddle_int16_real_2048[j * twiddle_step]; 
                    int32_t ti = twiddle_int16_imag_2048[j * twiddle_step]; 
                    
                    // 고정소수점 복소수 곱셈 및 반올림 처리 (32비트 임시 확장 후 Q15 복원)
                    int32_t t_real = (real[k] * tr - imag[k] * ti + 16384) >> 15;
                    int32_t t_imag = (real[k] * ti + imag[k] * tr + 16384) >> 15; 
                    
                    // 오버플로우 방지 1비트 스케일링 결합
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
        // -----------------------------------------------------------------
        // 분기 구역: step >= 8 부터 초고속 8-Lane NEON 파이프라인 가동! 🔥
        // -----------------------------------------------------------------
        else { 
#ifdef __aarch64__
            for (int i = 0; i < N; i += jump) {
                // 8개의 나비를 동시에 처리하기 위해 j가 8씩 전진합니다.
                for (int j = 0; j < step; j += 8) { 
                    int curr = i + j; 
                    int k = curr + step; 
                    
                    // 8-Lane 정수 벡터 데이터 연속 로드
                    int16x8_t vr_curr = vld1q_s16(&real[curr]); 
                    int16x8_t vi_curr = vld1q_s16(&imag[curr]); 
                    int16x8_t vr_k    = vld1q_s16(&real[k]); 
                    int16x8_t vi_k    = vld1q_s16(&imag[k]); 
                    
                    // 회전인자 벡터 팩 가공
                    int16_t tr_arr[8], ti_arr[8];
                    for (int lane = 0; lane < 8; lane++) {
                        tr_arr[lane] = twiddle_int16_real_2048[(j + lane) * twiddle_step];
                        ti_arr[lane] = twiddle_int16_imag_2048[(j + lane) * twiddle_step];
                    }
                    int16x8_t v_tr = vld1q_s16(tr_arr); 
                    int16x8_t v_ti = vld1q_s16(ti_arr); 
                    
                    // ⚡ [NEON 고정소수점 복소수 회전] Q15 포화 곱셈 집행 (AC-BD, AD+BC)
                    int16x8_t vt_real = vsubq_s16(vqrdmulhq_s16(vr_k, v_tr), vqrdmulhq_s16(vi_k, v_ti)); 
                    int16x8_t vt_imag = vaddq_s16(vqrdmulhq_s16(vr_k, v_ti), vqrdmulhq_s16(vi_k, v_tr)); 
                    
                    // ⚡ [NEON 가감산 및 1비트 스케일링] 8개 나비의 오버플로우를 1사이클 만에 동시 가공
                    int16x8_t v_out_k_r = vshrq_n_s16(vsubq_s16(vr_curr, vt_real), 1);
                    int16x8_t v_out_k_i = vshrq_n_s16(vsubq_s16(vi_curr, vt_imag), 1);
                    int16x8_t v_out_c_r = vshrq_n_s16(vaddq_s16(vr_curr, vt_real), 1);
                    int16x8_t v_out_c_i = vshrq_n_s16(vaddq_s16(vi_curr, vt_imag), 1);

                    // 연산 버퍼 제자리 박제 (In-place Vector Store)
                    vst1q_s16(&real[k], v_out_k_r); 
                    vst1q_s16(&imag[k], v_out_k_i); 
                    vst1q_s16(&real[curr], v_out_c_r); 
                    vst1q_s16(&imag[curr], v_out_c_i); 
                }
            }
#else
            // x86 Fallback 스칼라 코드
            for (int i = 0; i < N; i += jump) {
                for (int j = 0; j < step; j++) { 
                    int curr = i + j; 
                    int k = curr + step; 
                    
                    int32_t tr = twiddle_int16_real_2048[j * twiddle_step]; 
                    int32_t ti = twiddle_int16_imag_2048[j * twiddle_step]; 
                    
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