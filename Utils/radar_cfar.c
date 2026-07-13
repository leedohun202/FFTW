#include "radar_cfar.h"
#include <stdlib.h>
#include <stdio.h>

static void compute_power_map_float(const float *cube_r, const float *cube_i, float *pwr_map, int n_samples, int n_chirps) {
    int total = n_samples * n_chirps;
    #pragma omp parallel for
    for(int i = 0; i < total; i++) {
        pwr_map[i] = (cube_r[i] * cube_r[i]) + (cube_i[i] * cube_i[i]);
    }
}

static void compute_power_map_int16(const int16_t *cube_r, const int16_t *cube_i, float *pwr_map, int n_samples, int n_chirps) {
    int total = n_samples * n_chirps;
    #pragma omp parallel for
    for(int i = 0; i < total; i++) {
        float fr = (float)cube_r[i];
        float fi = (float)cube_i[i];
        pwr_map[i] = (fr * fr) + (fi * fi);
    }
}

// 💥 [수정] 복잡한 이중 루프와 잘못된 인덱스 공식을 제거하고, 직관적인 1D 루프로 변경
static void compute_power_map_fftw(const fftwf_complex *cube, float *pwr_map, int n_samples, int n_chirps) {
    int total = n_samples * n_chirps;
    
    // 이미 Transpose 되어 연속된 배열이므로, 순서대로 쭉 읽으면서 Power만 계산하면 끝입니다!
    #pragma omp parallel for
    for (int i = 0; i < total; i++) {
        pwr_map[i] = (cube[i][0] * cube[i][0]) + (cube[i][1] * cube[i][1]);
    }
}

static int execute_cfar_core(const float *pwr_map, int n_samples, int n_chirps, 
                             CFAR_Config cfg, CFAR_Target *detected_list, int max_targets) {
    int num_detected = 0;
    int margin_r = cfg.guard_r + cfg.train_r;
    int margin_d = cfg.guard_d + cfg.train_d;

    int win_r_size = 2 * margin_r + 1;
    int win_d_size = 2 * margin_d + 1;
    int guard_r_size = 2 * cfg.guard_r + 1;
    int guard_d_size = 2 * cfg.guard_d + 1;
    int num_training_cells = (win_r_size * win_d_size) - (guard_r_size * guard_d_size);

    for (int r = margin_r; r < n_samples - margin_r; r++) {
        for (int d = 0; d < n_chirps; d++) {
            
            float cut_power = pwr_map[r * n_chirps + d];
            float noise_sum = 0.0f;

            for (int j = -margin_r; j <= margin_r; j++) {
                for (int i = -margin_d; i <= margin_d; i++) {
                    if (abs(j) <= cfg.guard_r && abs(i) <= cfg.guard_d) {
                        continue; 
                    }
                    int target_d = d + i;
                    if (target_d < 0) {
                        target_d += n_chirps;
                    } else if (target_d >= n_chirps) {
                        target_d -= n_chirps;
                    }
                    noise_sum += pwr_map[(r + j) * n_chirps + target_d];
                }
            }

            float noise_avg = noise_sum / num_training_cells;
            float threshold = noise_avg * cfg.alpha;

            // 외부에서 주입된 최소 임계값으로 양자화 바닥노이즈 방어
            if (threshold < cfg.min_threshold) {
                threshold = cfg.min_threshold;
            }

            if (cut_power > threshold) {
                int is_local_max = 1;
                for(int j = -cfg.guard_r; j <= cfg.guard_r; j++) {
                    for(int i = -cfg.guard_d; i <= cfg.guard_d; i++) {
                        if (i == 0 && j == 0) continue;
                        
                        int check_d = d + i;
                        if (check_d < 0) check_d += n_chirps;
                        else if (check_d >= n_chirps) check_d -= n_chirps;
                        
                        if (pwr_map[(r + j) * n_chirps + check_d] > cut_power) {
                            is_local_max = 0; break;
                        }
                    }
                    if(!is_local_max) break;
                }

                if (is_local_max && num_detected < max_targets) {
                    detected_list[num_detected].r_idx = r;
                    detected_list[num_detected].d_idx = d;
                    detected_list[num_detected].power = cut_power;
                    detected_list[num_detected].noise_floor = noise_avg;
                    num_detected++;
                }
            }
        }
    }
    return num_detected; 
}

int run_2d_ca_cfar_float(const float *cube_r, const float *cube_i, float *pwr_map, 
                         int n_samples, int n_chirps, 
                         CFAR_Config cfg, CFAR_Target *detected_list, int max_targets) {
    compute_power_map_float(cube_r, cube_i, pwr_map, n_samples, n_chirps);
    return execute_cfar_core(pwr_map, n_samples, n_chirps, cfg, detected_list, max_targets);
}

int run_2d_ca_cfar_int16(const int16_t *cube_r, const int16_t *cube_i, float *pwr_map, 
                         int n_samples, int n_chirps, 
                         CFAR_Config cfg, CFAR_Target *detected_list, int max_targets) {
    compute_power_map_int16(cube_r, cube_i, pwr_map, n_samples, n_chirps);
    return execute_cfar_core(pwr_map, n_samples, n_chirps, cfg, detected_list, max_targets);
}

int run_2d_ca_cfar_fftw(const fftwf_complex *cube, float *pwr_map, 
                        int n_samples, int n_chirps, 
                        CFAR_Config cfg, CFAR_Target *detected_list, int max_targets) {
    compute_power_map_fftw(cube, pwr_map, n_samples, n_chirps);
    return execute_cfar_core(pwr_map, n_samples, n_chirps, cfg, detected_list, max_targets);
}