/**
 * @brief 3D 레이더 큐브 메모리 전치 (Transpose) - 세그폴트 가드 완비 버전
 */
#include "radar_config.h"
#include <stdlib.h>

void transpose_radar_cube(const float *__restrict__ src_real, const float *__restrict__ src_imag,
                          float *__restrict__ dst_real, float *__restrict__ dst_imag, 
                          int n_samples, int n_chirps, const float *__restrict__ win) {

    // OpenMP 스레드들이 외부 3중 블록 루프를 쪼개어 병렬 처리
    #pragma omp parallel for collapse(3)
    for (int ant = 0; ant < N_ANTENNAS; ant++) {
        for (int c_blk = 0; c_blk < n_chirps; c_blk += TILE_SIZE) {
            for (int r_blk = 0; r_blk < n_samples; r_blk += TILE_SIZE) {
                
                for (int c = c_blk; c < c_blk + TILE_SIZE; c++) {
                    if (c >= n_chirps) break;
                    for (int r = r_blk; r < r_blk + TILE_SIZE; r++) {
                        if (r >= n_samples) break;
                        
                        int src_idx = ant * (n_chirps * n_samples) + c * n_samples + r;
                        int dst_idx = ant * (n_samples * n_chirps) + r * n_chirps + c;
                        
                        // 🛡️ [세그폴트 방지] win 포인터가 NULL이면 1.0(곱하나 마나)으로 처리, 있으면 배열 참조!
                        float w = (win == NULL) ? 1.0f : win[c];
                        
                        dst_real[dst_idx] = src_real[src_idx] * w;
                        dst_imag[dst_idx] = src_imag[src_idx] * w;
                    }
                }
                
            }
        }
    }
}