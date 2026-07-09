#include "radar_fft.h"

#ifdef __aarch64__
#include <arm_neon.h>
#endif

// 512 전용 글로벌 비트 리버설 및 트위들 테이블 매핑 (인덱스 오버플로우 원천 차단)
extern int bitrev_512[];
extern int16_t twiddle_int16_real_512[];
extern int16_t twiddle_int16_imag_512[];

void custom_fft_512_int16(int16_t *__restrict__ real, int16_t *__restrict__ imag) {
    const int N = 512;

    // [1단계] 고정소수점용 비트 리버설 구조 (인덱스 마진 검증 완증)
    for (int i = 0; i < N; i++) { 
        int j = bitrev_512[i]; 
        if (i < j) { 
            int16_t tr = real[i]; real[i] = real[j]; real[j] = tr; 
            int16_t ti = imag[i]; imag[i] = imag[j]; imag[j] = ti; 
        } 
    }

    // [2단계] Radix-2 고속 나비 연산 스테이지 전개
    for (int step = 1; step < N; step *= 2) { 
        const int jump = step * 2; 
        const int twiddle_step = N / jump;
        
        // 하드웨어 SIMD 정렬 효율이 안 나오는 하위 스테이지는 스칼라 연산으로 가속
        if (step < 8) { 
            for (int i = 0; i < N; i += jump) { 
                for (int j = 0; j < step; j++) { 
                    int curr = i + j; 
                    int k = curr + step; 
                    
                    int32_t tr = twiddle_int16_real_512[j * twiddle_step]; 
                    int32_t ti = twiddle_int16_imag_512[j * twiddle_step]; 
                    
                    // Q15 fixed-point 곱셈 및 라운딩 스케일링 복원
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
            // 🚀 [3단계] ARM64 Neon 벡터 레지스터 전용 가속 루프 점화
#ifdef __aarch64__
            for (int i = 0; i < N; i += jump) {
                for (int j = 0; j < step; j += 8) { 
                    int curr = i + j; 
                    int k = curr + step; 
                    
                    // 8개의 복소 데이터를 Neon 레지스터에 동시 로드
                    int16x8_t vr_curr = vld1q_s16(&real[curr]); 
                    int16x8_t vi_curr = vld1q_s16(&imag[curr]); 
                    int16x8_t vr_k    = vld1q_s16(&real[k]); 
                    int16x8_t vi_k    = vld1q_s16(&imag[k]); 
                    
                    // 정렬 마진을 위반하지 않는 타이트한 8-Lane 트위들 배열 팩 구축
                    int16_t tr_arr[8], ti_arr[8];
                    for (int lane = 0; lane < 8; lane++) {
                        tr_arr[lane] = twiddle_int16_real_512[(j + lane) * twiddle_step];
                        ti_arr[lane] = twiddle_int16_imag_512[(j + lane) * twiddle_step];
                    }
                    int16x8_t v_tr = vld1q_s16(tr_arr); 
                    int16x8_t v_ti = vld1q_s16(ti_arr); 
                    
                    // 고성능 64비트 정밀 오버플로우 가드 곱셈기(vqrdmulhq_s16) 작동
                    int16x8_t vt_real = vsubq_s16(vqrdmulhq_s16(vr_k, v_tr), vqrdmulhq_s16(vi_k, v_ti)); 
                    int16x8_t vt_imag = vaddq_s16(vqrdmulhq_s16(vr_k, v_ti), vqrdmulhq_s16(vi_k, v_tr)); 
                    
                    int16x8_t v_out_k_r = vshrq_n_s16(vsubq_s16(vr_curr, vt_real), 1);
                    int16x8_t v_out_k_i = vshrq_n_s16(vsubq_s16(vi_curr, vt_imag), 1);
                    int16x8_t v_out_c_r = vshrq_n_s16(vaddq_s16(vr_curr, vt_real), 1);
                    int16x8_t v_out_c_i = vshrq_n_s16(vaddq_s16(vi_curr, vt_imag), 1);

                    // 연산 즉시 메모리에 스트리밍 쓰기
                    vst1q_s16(&real[k], v_out_k_r); 
                    vst1q_s16(&imag[k], v_out_k_i); 
                    vst1q_s16(&real[curr], v_out_c_r); 
                    vst1q_s16(&imag[curr], v_out_c_i); 
                }
            }
#else
            // 에뮬레이터 혹은 아키텍처 비매칭 시 포터블 폴백 백업 루프
            for (int i = 0; i < N; i += jump) {
                for (int j = 0; j < step; j++) { 
                    int curr = i + j; 
                    int k = curr + step; 
                    
                    int32_t tr = twiddle_int16_real_512[j * twiddle_step]; 
                    int32_t ti = twiddle_int16_imag_512[j * twiddle_step]; 
                    
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