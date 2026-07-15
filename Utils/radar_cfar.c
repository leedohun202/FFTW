#include "radar_cfar.h"
#include "radar_config.h"
#include <stdlib.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __aarch64__
#include <arm_neon.h>
#endif

// ======================================================================
// [1] Power Map (전력망) 생성 함수들 (정적 헬퍼)
// ======================================================================

static void compute_power_map_float(const float *cube_r, const float *cube_i, float *pwr_map, int n_samples, int n_chirps) {
    int total = n_samples * n_chirps;
    #pragma omp parallel for
    for(int i = 0; i < total; i++) {
        pwr_map[i] = (cube_r[i] * cube_r[i]) + (cube_i[i] * cube_i[i]);
    }
}

static void compute_power_map_int16(const int16_t *cube_r, const int16_t *cube_i, float *pwr_map, int n_samples, int n_chirps) {
    int total = n_samples * n_chirps;
    const float scale_factor = 1.0f / (8192.0f * 8192.0f);

#ifdef __aarch64__
    #pragma omp parallel for
    for (int i = 0; i < total; i += 8) {
        // 1. 8개의 Int16 데이터를 동시 로드
        int16x8_t vr = vld1q_s16(&cube_r[i]);
        int16x8_t vi = vld1q_s16(&cube_i[i]);

        // 2. 하위 4개 연산: R^2 계산 (vmull) 후 I^2 누적 덧셈 (vmlal)
        int32x4_t pwr_low = vmull_s16(vget_low_s16(vr), vget_low_s16(vr));
        pwr_low = vmlal_s16(pwr_low, vget_low_s16(vi), vget_low_s16(vi));

        // 3. 상위 4개 연산: R^2 계산 (vmull) 후 I^2 누적 덧셈 (vmlal)
        int32x4_t pwr_high = vmull_high_s16(vr, vr);
        pwr_high = vmlal_high_s16(pwr_high, vi, vi);

        // 4. Int32 -> Float 하드웨어 동시 변환
        float32x4_t fpwr_low = vcvtq_f32_s32(pwr_low);
        float32x4_t fpwr_high = vcvtq_f32_s32(pwr_high);

        // 5. 스케일 팩터 곱셈
        fpwr_low = vmulq_n_f32(fpwr_low, scale_factor);
        fpwr_high = vmulq_n_f32(fpwr_high, scale_factor);

        // 6. 결과 저장
        vst1q_f32(&pwr_map[i], fpwr_low);
        vst1q_f32(&pwr_map[i + 4], fpwr_high);
    }
#else
    // 폴백(Fallback): 최적화된 스칼라 버전
    #pragma omp parallel for
    for(int i = 0; i < total; i++) {
        int32_t r32 = cube_r[i];
        int32_t i32 = cube_i[i];
        pwr_map[i] = (float)((r32 * r32) + (i32 * i32)) * scale_factor;
    }
#endif
}

static void compute_power_map_fftw(const fftwf_complex *cube, float *pwr_map, int n_samples, int n_chirps) {
    int total = n_samples * n_chirps;
    #pragma omp parallel for
    for (int i = 0; i < total; i++) {
        pwr_map[i] = (cube[i][0] * cube[i][0]) + (cube[i][1] * cube[i][1]);
    }
}

// ======================================================================
// [2] O(1) 초고속 2D CFAR를 위한 적분 영상 (Integral Image) 헬퍼
// ======================================================================

// 안전한 메모리 접근 (0 이하 인덱스는 0.0 반환)
static inline double get_integral_val(const double *S, int r, int d, int n_chirps) {
    if (r < 0 || d < 0) return 0.0;
    return S[r * n_chirps + d];
}

// 단 4번의 연산으로 사각형 구역(r1~r2, d1~d2)의 전체 파워 합 구하기
static inline double query_region(const double *S, int r1, int r2, int d1, int d2, int n_chirps) {
    double A = get_integral_val(S, r1 - 1, d1 - 1, n_chirps);
    double B = get_integral_val(S, r1 - 1, d2, n_chirps);
    double C = get_integral_val(S, r2, d1 - 1, n_chirps);
    double D = get_integral_val(S, r2, d2, n_chirps);
    return D - B - C + A;
}

// 도플러 축 순환(Wrap-around)을 지원하는 어댑티브 쿼리
static inline double get_cyclic_area(const double *S, int r1, int r2, int d1, int d2, int n_samples, int n_chirps) {
    if (d1 < 0) {
        // 왼쪽으로 삐져나간 경우: [오른쪽 끝단] + [0부터 현재까지] 2개 구역으로 분할 합산
        double sum1 = query_region(S, r1, r2, d1 + n_chirps, n_chirps - 1, n_chirps);
        double sum2 = query_region(S, r1, r2, 0, d2, n_chirps);
        return sum1 + sum2;
    } else if (d2 >= n_chirps) {
        // 오른쪽으로 삐져나간 경우: [현재부터 끝단] + [0부터 삐져나간 만큼] 2개 구역으로 분할 합산
        double sum1 = query_region(S, r1, r2, d1, n_chirps - 1, n_chirps);
        double sum2 = query_region(S, r1, r2, 0, d2 - n_chirps, n_chirps);
        return sum1 + sum2;
    } else {
        // 정상 구역
        return query_region(S, r1, r2, d1, d2, n_chirps);
    }
}

// ======================================================================
// [3] O(1) 시간 복잡도를 달성한 2D CFAR 코어 엔진
// ======================================================================

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

    // ---------------------------------------------------------
    // 💡 1단계: 누적합 지도 (Integral Image) 생성 (Double 정밀도 방어)
    // ---------------------------------------------------------
    double *S = (double*)malloc(n_samples * n_chirps * sizeof(double));
    
    for (int r = 0; r < n_samples; r++) {
        for (int d = 0; d < n_chirps; d++) {
            double val = (double)pwr_map[r * n_chirps + d];
            double up = (r > 0) ? S[(r - 1) * n_chirps + d] : 0.0;
            double left = (d > 0) ? S[r * n_chirps + (d - 1)] : 0.0;
            double up_left = (r > 0 && d > 0) ? S[(r - 1) * n_chirps + (d - 1)] : 0.0;
            
            S[r * n_chirps + d] = val + up + left - up_left;
        }
    }

    // ---------------------------------------------------------
    // 💡 2단계: O(1) 초고속 CFAR 스캔
    // ---------------------------------------------------------
    for (int r = margin_r; r < n_samples - margin_r; r++) {
        for (int d = 0; d < n_chirps; d++) {
            
            float cut_power = pwr_map[r * n_chirps + d];

            // 💥 마법의 O(1) 연산: 큰 창문 합에서 작은 가드 창문 합을 뺀다!
            double total_win_sum = get_cyclic_area(S, r - margin_r, r + margin_r, 
                                                   d - margin_d, d + margin_d, 
                                                   n_samples, n_chirps);
                                                   
            double guard_win_sum = get_cyclic_area(S, r - cfg.guard_r, r + cfg.guard_r, 
                                                   d - cfg.guard_d, d + cfg.guard_d, 
                                                   n_samples, n_chirps);
            
            float noise_sum = (float)(total_win_sum - guard_win_sum);
            float noise_avg = noise_sum / (float)num_training_cells;
            float threshold = noise_avg * cfg.alpha;

            if (threshold < cfg.min_threshold) {
                threshold = cfg.min_threshold;
            }

            // ---------------------------------------------------------
            // 💡 3단계: 임계값을 넘었을 때만 로컬 맥스(Local Max) 검사
            // ---------------------------------------------------------
            if (cut_power > threshold) {
                int is_local_max = 1;
                // 로컬 맥스 검사는 가드 영역 안에서만 빠르게 수행되므로 부하 최소화
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

    free(S); // 누적합 메모리 반환
    return num_detected; 
}

// ======================================================================
// [4] 공용 외부 호출 인터페이스 (Wrappers)
// ======================================================================

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