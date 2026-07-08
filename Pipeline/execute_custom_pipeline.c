#include "radar_pipeline.h"

/**
 * @brief Zero-Allocation 및 최적화된 2D Doppler FFT 파이프라인 집행 커널
 * @param cube_real   입력 데이터의 실수부 배열 포인터
 * @param cube_imag   입력 데이터의 허수부 배열 포인터
 * @param tmp_real    외부(run_benchmark)에서 미리 할당되어 넘어온 실수부 임시 버퍼
 * @param tmp_imag    외부(run_benchmark)에서 미리 할당되어 넘어온 허수부 임시 버퍼
 * @param n_samples   Range 축의 샘플 개수
 */
void execute_custom_pipeline(float *__restrict__ cube_real, float *__restrict__ cube_imag, 
                             float *__restrict__ tmp_real, float *__restrict__ tmp_imag, 
                             int n_samples, int n_chirps) {
    
    // 💡 1단계: 가변 처프 창함수를 판단하여 분리된 전치 함수 호출
    const float *current_win = (n_chirps == 512) ? win_512 : win_256;
    transpose_radar_cube(cube_real, cube_imag, tmp_real, tmp_imag, n_samples, n_chirps, current_win); 
    
    // 💡 2단계: 2D Doppler FFT 병렬 구동 (256처프면 Radix-4, 512처프면 기존 NEON 가동)
    #pragma omp parallel for collapse(2)
    for (int ant = 0; ant < N_ANTENNAS; ant++) {
        for (int r = 0; r < n_samples; r++) {
            int offset = ant * (n_samples * n_chirps) + r * n_chirps;
            if (n_chirps == 512) {
                custom_fft_512_fixed(&tmp_real[offset], &tmp_imag[offset]);
            } else {
                custom_fft_256_radix4(&tmp_real[offset], &tmp_imag[offset]);
            }
        }
    }
}