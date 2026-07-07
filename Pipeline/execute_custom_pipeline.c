#include "radar_pipeline.h"
#include <stdlib.h>
#include <string.h>

void execute_custom_pipeline(float *__restrict__ cube_real, float *__restrict__ cube_imag, int n_samples) {
    
    int total_elements = N_ANTENNAS * N_CHIRPS * n_samples;

    // [2단계] Doppler 가속을 위한 안전한 자기 내부 전치(Transpose)
    float *tmp_r = (float *)malloc(total_elements * sizeof(float));
    float *tmp_m = (float *)malloc(total_elements * sizeof(float));
    
    transpose_radar_cube(cube_real, tmp_r, n_samples); 
    transpose_radar_cube(cube_imag, tmp_m, n_samples);
    
    memcpy(cube_real, tmp_r, total_elements * sizeof(float));
    memcpy(cube_imag, tmp_m, total_elements * sizeof(float));
    
    free(tmp_r); free(tmp_m);

    // [3단계] 2D Doppler FFT (데이터는 이제 [ant][r][chirp] 포맷입니다)
    #pragma omp parallel for collapse(2)
    for (int ant = 0; ant < N_ANTENNAS; ant++) {
        for (int r = 0; r < n_samples; r++) {
            int offset = ant * (n_samples * N_CHIRPS) + r * N_CHIRPS;
            for (int chirp = 0; chirp < N_CHIRPS; chirp++) {
                cube_real[offset + chirp] *= win_512[chirp];
                cube_imag[offset + chirp] *= win_512[chirp];
            }
            custom_fft_512_fixed(&cube_real[offset], &cube_imag[offset]);
        }
    }
    // 💡 4단계(Angle)는 일괄 처리하지 않고 main에서 필요한 타겟만 솎아서 칩니다.
}