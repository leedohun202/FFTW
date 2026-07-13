#ifndef RADAR_CFAR_H
#define RADAR_CFAR_H

#include <stdint.h>
#include <fftw3.h>

// CFAR 환경 설정 구조체
typedef struct {
    int guard_r;       // Range 방향 가드 셀 개수 (한쪽 기준)
    int guard_d;       // Doppler 방향 가드 셀 개수 (한쪽 기준)
    int train_r;       // Range 방향 트레이닝 셀 개수 (한쪽 기준)
    int train_d;       // Doppler 방향 트레이닝 셀 개수 (한쪽 기준)
    float alpha;       // 문턱값 스케일링 팩터 (Threshold Factor)
    float min_threshold; // 동적 레인지 맞춤형 최소 노이즈 바닥값
} CFAR_Config;

// 탐지된 타겟의 좌표를 저장할 구조체
typedef struct {
    int r_idx;         // Range 인덱스
    int d_idx;         // Doppler 인덱스
    float power;       // 해당 타겟의 전력 크기
    float noise_floor; // 당시 주변 노이즈 평균치
} CFAR_Target;

// 💥 내부 동적 할당을 제거하고, pwr_map 버퍼를 인자로 받습니다.
int run_2d_ca_cfar_float(const float *cube_r, const float *cube_i, float *pwr_map, 
                         int n_samples, int n_chirps, 
                         CFAR_Config cfg, CFAR_Target *detected_list, int max_targets);

int run_2d_ca_cfar_int16(const int16_t *cube_r, const int16_t *cube_i, float *pwr_map, 
                         int n_samples, int n_chirps, 
                         CFAR_Config cfg, CFAR_Target *detected_list, int max_targets);

int run_2d_ca_cfar_fftw(const fftwf_complex *cube, float *pwr_map, 
                        int n_samples, int n_chirps, 
                        CFAR_Config cfg, CFAR_Target *detected_list, int max_targets);

#endif // RADAR_CFAR_H