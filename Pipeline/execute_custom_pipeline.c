#include "radar_pipeline.h"
#include "radar_fft.h"
#include "radar_config.h"
#include <stddef.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// -------------------------------------------------------------
// [1] 기존 Float Radix-2 파이프라인
// -------------------------------------------------------------
void execute_custom_pipeline(float *__restrict__ cube_real, float *__restrict__ cube_imag, 
                             float *__restrict__ tmp_real, float *__restrict__ tmp_imag, 
                             int n_samples, int n_chirps) {
    
    const float *current_win = NULL;
    if (n_chirps == 512) current_win = win_512;
    else if (n_chirps == 256) current_win = win_256;
    else if (n_chirps == 128) current_win = win_128;

    transpose_radar_cube(cube_real, cube_imag, tmp_real, tmp_imag, n_samples, n_chirps, current_win); 
    
    #pragma omp parallel for collapse(2)
    for (int ant = 0; ant < N_ANTENNAS; ant++) {
        for (int r = 0; r < n_samples; r++) {
            int offset = ant * (n_samples * n_chirps) + r * n_chirps;
            if (n_chirps == 512)      custom_fft_512_fixed(&tmp_real[offset], &tmp_imag[offset]);
            else if (n_chirps == 256) custom_fft_256_fixed(&tmp_real[offset], &tmp_imag[offset]);
            else if (n_chirps == 128) custom_fft_128_fixed(&tmp_real[offset], &tmp_imag[offset]);
            else if (n_chirps == 64)  custom_fft_64_fixed(&tmp_real[offset], &tmp_imag[offset]);
        }
    }
}

// -------------------------------------------------------------
// 🚀 [2] 신규 Float Radix-4 파이프라인
// -------------------------------------------------------------
void execute_custom_pipeline_radix4(float *__restrict__ cube_real, float *__restrict__ cube_imag, 
                                    float *__restrict__ tmp_real, float *__restrict__ tmp_imag, 
                                    int n_samples, int n_chirps) {
    
    const float *current_win = NULL;
    if (n_chirps == 512) current_win = win_512;
    else if (n_chirps == 256) current_win = win_256;
    else if (n_chirps == 128) current_win = win_128;

    transpose_radar_cube(cube_real, cube_imag, tmp_real, tmp_imag, n_samples, n_chirps, current_win); 
    
    #pragma omp parallel for collapse(2)
    for (int ant = 0; ant < N_ANTENNAS; ant++) {
        for (int r = 0; r < n_samples; r++) {
            int offset = ant * (n_samples * n_chirps) + r * n_chirps;
            if (n_chirps == 512)      custom_fft_512_radix4(&tmp_real[offset], &tmp_imag[offset]);
            else if (n_chirps == 256) custom_fft_256_radix4(&tmp_real[offset], &tmp_imag[offset]);
            else if (n_chirps == 128) custom_fft_128_radix4(&tmp_real[offset], &tmp_imag[offset]);
            else if (n_chirps == 64)  custom_fft_64_radix4(&tmp_real[offset], &tmp_imag[offset]);
        }
    }
}