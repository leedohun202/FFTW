#include "radar_pipeline.h"
#include "radar_fft.h"
#include "radar_config.h"
#include <stddef.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// -------------------------------------------------------------
// [1] 기존 Int16 Radix-2 파이프라인 (백업 및 비교용)
// -------------------------------------------------------------
void execute_custom_pipeline_int16(int16_t *__restrict__ cube_real, int16_t *__restrict__ cube_imag, 
                                   int16_t *__restrict__ tmp_real, int16_t *__restrict__ tmp_imag, 
                                   int n_samples, int n_chirps) {
    
    // 💡 1단계: 1D Range-FFT
    #pragma omp parallel for collapse(2)
    for (int ant = 0; ant < N_ANTENNAS; ant++) {
        for (int c = 0; c < n_chirps; c++) {
            int offset = ant * (n_samples * n_chirps) + c * n_samples;
            if (n_samples == 2048) custom_fft_2048_int16(&cube_real[offset], &cube_imag[offset]);
            else if (n_samples == 1024) custom_fft_1024_int16(&cube_real[offset], &cube_imag[offset]);
            else if (n_samples == 512)  custom_fft_512_int16(&cube_real[offset], &cube_imag[offset]);
            else if (n_samples == 256)  custom_fft_256_int16(&cube_real[offset], &cube_imag[offset]);
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
            if (n_chirps == 512)      custom_fft_512_int16(&tmp_real[offset], &tmp_imag[offset]);
            else if (n_chirps == 256) custom_fft_256_int16(&tmp_real[offset], &tmp_imag[offset]);
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
            if (n_samples == 2048) custom_fft_2048_int16_radix4(&cube_real[offset], &cube_imag[offset]);
            else if (n_samples == 1024) custom_fft_1024_int16_radix4(&cube_real[offset], &cube_imag[offset]);
            else if (n_samples == 512)  custom_fft_512_int16_radix4(&cube_real[offset], &cube_imag[offset]);
            else if (n_samples == 256)  custom_fft_256_int16_radix4(&cube_real[offset], &cube_imag[offset]);
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
            if (n_chirps == 512)      custom_fft_512_int16_radix4(&tmp_real[offset], &tmp_imag[offset]);
            else if (n_chirps == 256) custom_fft_256_int16_radix4(&tmp_real[offset], &tmp_imag[offset]);
            else if (n_chirps == 128) custom_fft_128_int16_radix4(&tmp_real[offset], &tmp_imag[offset]);
            else if (n_chirps == 64)  custom_fft_64_int16_radix4(&tmp_real[offset], &tmp_imag[offset]);
        }
    }
}