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
                             int n_samples) {
    
    // [1단계] 메모리 전치(Transpose)와 Doppler 창함수(Windowing)를 한 번에 처리 (Loop Fusion)
    // 이 과정이 끝나면 데이터는 [Antenna][Range][Chirp] 순서로 tmp 버퍼에 연속 정렬됩니다.
    transpose_radar_cube(cube_real, tmp_real, n_samples, win_512); 
    transpose_radar_cube(cube_imag, tmp_imag, n_samples, win_512);
    
    // [2단계] 2D Doppler FFT 병렬 구동
    // 안테나 축과 Range 샘플 축을 병합(Collapse)하여 라즈베리파이 5의 4개 코어에 부하를 분산합니다.
    #pragma omp parallel for collapse(2)
    for (int ant = 0; ant < N_ANTENNAS; ant++) {
        for (int r = 0; r < n_samples; r++) {
            // 전치된 배열 format에 맞는 물리적 오프셋 계산
            int offset = ant * (n_samples * N_CHIRPS) + r * N_CHIRPS;
            
            // 정렬된 메모리 포인터를 커스텀 ARM NEON 512 FFT 가속 함수에 직접 주입
            custom_fft_512_fixed(&tmp_real[offset], &tmp_imag[offset]);
        }
    }
}