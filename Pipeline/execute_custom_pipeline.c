#include "radar_pipeline.h"

/**
 * @brief 자체 제작(Custom) 엔진을 사용한 전체 레이더 파이프라인 수행
 * @param cube_real 원본 데이터 실수부 (수신 버퍼)
 * @param cube_imag 원본 데이터 허수부 (수신 버퍼)
 * @param trans_cust_r 전치 연산용 실수부 임시 버퍼
 * @param trans_cust_i 전치 연산용 허수부 임시 버퍼
 * @param n_samples Range 축 샘플 수 (1024 또는 2048)
 */
void execute_custom_pipeline(float *__restrict__ cube_real, float *__restrict__ cube_imag, 
                             float *__restrict__ trans_cust_r, float *__restrict__ trans_cust_i, int n_samples) {
                             
    // 1단계: Range FFT 처리 (가로축 연산)
    #pragma omp parallel for collapse(2)
    for (int ant = 0; ant < N_ANTENNAS; ant++) {
        for (int chirp = 0; chirp < N_CHIRPS; chirp++) {
            int offset = ant * (N_CHIRPS * n_samples) + chirp * n_samples;
            
            // 크기에 따른 윈도우 스케일링 및 FFT 호출
            if (n_samples == 1024) {
                for(int i = 0; i < 1024; i++) { 
                    cube_real[offset+i] *= win_1024[i]; 
                    cube_imag[offset+i] *= win_1024[i]; 
                }
                custom_fft_1024_fixed(&cube_real[offset], &cube_imag[offset]);
            } else if (n_samples == 2048) {
                for(int i = 0; i < 2048; i++) { 
                    cube_real[offset+i] *= win_2048[i]; 
                    cube_imag[offset+i] *= win_2048[i]; 
                }
                custom_fft_2048_fixed(&cube_real[offset], &cube_imag[offset]);
            }
        }
    }

    // 2단계: 데이터 메모리 방향 뒤집기 (Transpose)
    transpose_radar_cube(cube_real, trans_cust_r, n_samples); 
    transpose_radar_cube(cube_imag, trans_cust_i, n_samples);

    // 3단계: Doppler FFT 처리 (세로축 연산)
    #pragma omp parallel for collapse(2)
    for (int ant = 0; ant < N_ANTENNAS; ant++) {
        for (int r = 0; r < n_samples; r++) {
            int offset = ant * (n_samples * N_CHIRPS) + r * N_CHIRPS;
            
            for (int chirp = 0; chirp < N_CHIRPS; chirp++) { 
                trans_cust_r[offset+chirp] *= win_512[chirp]; 
                trans_cust_i[offset+chirp] *= win_512[chirp]; 
            }
            custom_fft_512_fixed(&trans_cust_r[offset], &trans_cust_i[offset]);
        }
    }
}