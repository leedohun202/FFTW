#include "radar_pipeline.h"

/**
 * @brief 인라인 3D 레이더 후처리 파이프라인 엔진
 * @param cube_real 이미 실시간 Range-FFT가 완료되어 채워진 데이터 큐브 (실수부)
 * @param cube_imag 이미 실시간 Range-FFT가 완료되어 채워진 데이터 큐브 (허수부)
 * @details Range-FFT 단계가 인라인으로 생략되었으므로, 즉시 Transpose 및 Doppler/Angle 처리를 수행합니다.
 */
void execute_custom_pipeline(float *__restrict__ cube_real, float *__restrict__ cube_imag, 
                             float *__restrict__ trans_cust_r, float *__restrict__ trans_cust_i, int n_samples) {
                             
    // 🚀 [인라인 버프로 인해 기존 1단계 Range-FFT 루프 통째로 삭제 완료!]

    // [2단계] 캐시 최적화 타일링 전치 (Transpose)
    transpose_radar_cube(cube_real, trans_cust_r, n_samples); 
    transpose_radar_cube(cube_imag, trans_cust_i, n_samples);

    // [3단계] Doppler FFT (세로축 속도 분석)
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

    // [4단계] 3D Angle FFT (공간 각도 복원 엔진)
    #pragma omp parallel for collapse(2)
    for (int r = 0; r < n_samples; r++) {
        for (int chirp = 0; chirp < N_CHIRPS; chirp++) {
            float ang_r[64] = {0.0f};
            float ang_i[64] = {0.0f};
            
            for (int ant = 0; ant < N_ANTENNAS; ant++) {
                int src_offset = ant * (n_samples * N_CHIRPS) + r * N_CHIRPS + chirp;
                ang_r[ant] = trans_cust_r[src_offset];
                ang_i[ant] = trans_cust_i[src_offset];
            }
            
            custom_fft_64_fixed(ang_r, ang_i);
            
            for (int a_idx = 0; a_idx < N_ANGLE; a_idx++) {
                int dst_offset = a_idx * (n_samples * N_CHIRPS) + r * N_CHIRPS + chirp;
                trans_cust_r[dst_offset] = ang_r[a_idx];
                trans_cust_i[dst_offset] = ang_i[a_idx];
            }
        }
    }
}