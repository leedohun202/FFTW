#include "radar_config.h"
#include <stdint.h>
#include <stddef.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * @brief int16 고정소수점 전용 3D 레이더 큐브 전치 및 창함수 적용 (블록 타일링 캐시 방어 버전)
 */
void transpose_radar_cube_int16(int16_t *__restrict__ in_real, int16_t *__restrict__ in_imag,
                                int16_t *__restrict__ out_real, int16_t *__restrict__ out_imag,
                                int n_samples, int n_chirps, const int16_t *__restrict__ win_int16) {
    
    // 🚀 Float 버전과 동일하게 외부 3중 블록 루프를 쪼개어 L1 캐시 히트율을 극대화합니다.
    #pragma omp parallel for collapse(3)
    for (int ant = 0; ant < N_ANTENNAS; ant++) {
        for (int c_blk = 0; c_blk < n_chirps; c_blk += TILE_SIZE) {
            for (int r_blk = 0; r_blk < n_samples; r_blk += TILE_SIZE) {
                
                for (int c = c_blk; c < c_blk + TILE_SIZE; c++) {
                    if (c >= n_chirps) break;
                    
                    // 윈도우 값을 바깥 루프에서 미리 읽어 레지스터에 상주
                    int32_t w = (win_int16 != NULL) ? win_int16[c] : 0;

                    for (int r = r_blk; r < r_blk + TILE_SIZE; r++) {
                        if (r >= n_samples) break;
                        
                        int in_idx = ant * (n_chirps * n_samples) + c * n_samples + r;
                        int out_idx = ant * (n_samples * n_chirps) + r * n_chirps + c;

                        if (win_int16 != NULL) {
                            // Q15 고정소수점 윈도잉 연산 (반올림 보정 +16384 포함)
                            int32_t val_r = ((int32_t)in_real[in_idx] * w + 16384) >> 15;
                            int32_t val_i = ((int32_t)in_imag[in_idx] * w + 16384) >> 15;
                            
                            out_real[out_idx] = (int16_t)val_r;
                            out_imag[out_idx] = (int16_t)val_i;
                        } else {
                            out_real[out_idx] = in_real[in_idx];
                            out_imag[out_idx] = in_imag[in_idx];
                        }
                    }
                }
                
            }
        }
    }
}