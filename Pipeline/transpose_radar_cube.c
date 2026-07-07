#include "radar_pipeline.h"

/**
 * @brief 3D 레이더 큐브 메모리 전치 (Transpose)
 * @param src 원본 데이터 배열 (Antenna -> Chirp -> Range 순)
 * @param dst 출력 데이터 배열 (Antenna -> Range -> Chirp 순)
 * @param n_samples Range 축의 샘플 개수
 * @details 16x16 타일링(Tiling) 기법을 사용해 라즈베리파이의 캐시 미스를 최소화하며, 
 * OpenMP를 이용해 3중 루프를 병렬 처리합니다.
 */
void transpose_radar_cube(const float *__restrict__ src, float *__restrict__ dst, 
                          int n_samples, const float *__restrict__ win) {
    #pragma omp parallel for collapse(3)
    for (int ant = 0; ant < N_ANTENNAS; ant++) {
        for (int c_blk = 0; c_blk < N_CHIRPS; c_blk += TILE_SIZE) {
            for (int r_blk = 0; r_blk < n_samples; r_blk += TILE_SIZE) {
                
                for (int c = c_blk; c < c_blk + TILE_SIZE; c++) {
                    for (int r = r_blk; r < r_blk + TILE_SIZE; r++) {
                        int src_idx = ant * (N_CHIRPS * n_samples) + c * n_samples + r;
                        int dst_idx = ant * (n_samples * N_CHIRPS) + r * N_CHIRPS + c;
                        
                        // 메모리에 쓸 때 Doppler 창함수(win_512)를 즉시 반영
                        dst[dst_idx] = src[src_idx] * win[c];
                    }
                }
                
            }
        }
    }
}