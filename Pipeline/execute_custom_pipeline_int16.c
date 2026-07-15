#include "radar_pipeline.h"
#include "radar_fft.h"
#include "radar_config.h"
#include <stddef.h>
#include <stdlib.h> // abs() 함수 사용을 위해 추가

#ifdef _OPENMP
#include <omp.h>
#endif

// -------------------------------------------------------------
// [1] 기존 Int16 Radix-2 파이프라인 
// -------------------------------------------------------------
void execute_custom_pipeline_int16(int16_t *__restrict__ cube_real, int16_t *__restrict__ cube_imag, 
                                   int16_t *__restrict__ tmp_real, int16_t *__restrict__ tmp_imag, 
                                   int n_samples, int n_chirps) {
    
    // 💡 1단계: 1D Range-FFT
    #pragma omp parallel for collapse(2)
    for (int ant = 0; ant < N_ANTENNAS; ant++) {
        for (int c = 0; c < n_chirps; c++) {
            int offset = ant * (n_samples * n_chirps) + c * n_samples;
            
            // 💥 [패치] Range FFT는 아직 분리 전이므로 무조건 안전 모드(0) 고정
            if (n_samples == 2048) custom_fft_2048_int16(&cube_real[offset], &cube_imag[offset]);
            else if (n_samples == 1024) custom_fft_1024_int16(&cube_real[offset], &cube_imag[offset]);
            else if (n_samples == 512)  custom_fft_512_int16(&cube_real[offset], &cube_imag[offset], 0);
            else if (n_samples == 256)  custom_fft_256_int16(&cube_real[offset], &cube_imag[offset], 0);
            else if (n_samples == 128)  custom_fft_128_int16(&cube_real[offset], &cube_imag[offset]);
            else if (n_samples == 64)   custom_fft_64_int16(&cube_real[offset], &cube_imag[offset]);
        }
    }

    // 💡 2단계: Transpose & 윈도잉
    const int16_t *current_win = NULL;
    if (n_chirps == 512) current_win = win_int16_512;
    else if (n_chirps == 256) current_win = win_int16_256;
    else if (n_chirps == 128) current_win = win_int16_128;

    transpose_radar_cube_int16(cube_real, cube_imag, tmp_real, tmp_imag, n_samples, n_chirps, current_win); 
    
    // 💡 3단계: 2D Doppler-FFT
    #pragma omp parallel for collapse(2)
    for (int ant = 0; ant < N_ANTENNAS; ant++) {
        for (int r = 0; r < n_samples; r++) {
            int offset = ant * (n_samples * n_chirps) + r * n_chirps;
            
            // 💥 [패치] 도플러 스캔: 해당 거리 빈(Range Bin)의 최댓값 탐색
            int16_t max_val = 0;
            for (int c = 0; c < n_chirps; c++) {
                int16_t abs_r = abs(tmp_real[offset + c]);
                int16_t abs_i = abs(tmp_imag[offset + c]);
                if (abs_r > max_val) max_val = abs_r;
                if (abs_i > max_val) max_val = abs_i;
            }
            
            // 💥 [패치] 스케일링 생략 여부 결정 (오버플로우 안전 구역 판단)
            int skip_shift = (max_val < 8192) ? 1 : 0;

            if (n_chirps == 512)      custom_fft_512_int16(&tmp_real[offset], &tmp_imag[offset], skip_shift);
            else if (n_chirps == 256) custom_fft_256_int16(&tmp_real[offset], &tmp_imag[offset], skip_shift);
            else if (n_chirps == 128) custom_fft_128_int16(&tmp_real[offset], &tmp_imag[offset]);
            else if (n_chirps == 64)  custom_fft_64_int16(&tmp_real[offset], &tmp_imag[offset]);
        }
    }
}

// -------------------------------------------------------------
// 🔥 [2] 신규 Int16 Radix-4 궁극의 파이프라인
// -------------------------------------------------------------
void execute_custom_pipeline_int16_radix4(int16_t *__restrict__ cube_real, int16_t *__restrict__ cube_imag, 
                                          int16_t *__restrict__ tmp_real, int16_t *__restrict__ tmp_imag, 
                                          int n_samples, int n_chirps) {
    
    // 💡 1단계: 1D Range-FFT
    #pragma omp parallel for collapse(2)
    for (int ant = 0; ant < N_ANTENNAS; ant++) {
        for (int c = 0; c < n_chirps; c++) {
            int offset = ant * (n_samples * n_chirps) + c * n_samples;
            
            // 💥 [패치] Range FFT 무조건 안전 모드(0) 고정
            if (n_samples == 2048) custom_fft_2048_int16_radix4(&cube_real[offset], &cube_imag[offset]);
            else if (n_samples == 1024) custom_fft_1024_int16_radix4(&cube_real[offset], &cube_imag[offset]);
            else if (n_samples == 512)  custom_fft_512_int16_radix4(&cube_real[offset], &cube_imag[offset], 0);
            else if (n_samples == 256)  custom_fft_256_int16_radix4(&cube_real[offset], &cube_imag[offset], 0);
            else if (n_samples == 128)  custom_fft_128_int16_radix4(&cube_real[offset], &cube_imag[offset]);
            else if (n_samples == 64)   custom_fft_64_int16_radix4(&cube_real[offset], &cube_imag[offset]);
        }
    }

    // 💡 2단계: Transpose & 윈도잉
    const int16_t *current_win = NULL;
    if (n_chirps == 512) current_win = win_int16_512;
    else if (n_chirps == 256) current_win = win_int16_256;
    else if (n_chirps == 128) current_win = win_int16_128;

    transpose_radar_cube_int16(cube_real, cube_imag, tmp_real, tmp_imag, n_samples, n_chirps, current_win); 
    
    // 💡 3단계: 2D Doppler-FFT
    #pragma omp parallel for collapse(2)
    for (int ant = 0; ant < N_ANTENNAS; ant++) {
        for (int r = 0; r < n_samples; r++) {
            int offset = ant * (n_samples * n_chirps) + r * n_chirps;
            
            // 💥 [패치] 도플러 스캔: 해당 거리 빈(Range Bin)의 최댓값 탐색
            int16_t max_val = 0;
            for (int c = 0; c < n_chirps; c++) {
                int16_t abs_r = abs(tmp_real[offset + c]);
                int16_t abs_i = abs(tmp_imag[offset + c]);
                if (abs_r > max_val) max_val = abs_r;
                if (abs_i > max_val) max_val = abs_i;
            }
            
            // 💥 [패치] 스케일링 생략 여부 결정 (오버플로우 방어막)
            int skip_shift = (max_val < 8192) ? 1 : 0;

            if (n_chirps == 512)      custom_fft_512_int16_radix4(&tmp_real[offset], &tmp_imag[offset], skip_shift);
            else if (n_chirps == 256) custom_fft_256_int16_radix4(&tmp_real[offset], &tmp_imag[offset], skip_shift);
            else if (n_chirps == 128) custom_fft_128_int16_radix4(&tmp_real[offset], &tmp_imag[offset]);
            else if (n_chirps == 64)  custom_fft_64_int16_radix4(&tmp_real[offset], &tmp_imag[offset]);
        }
    }
}