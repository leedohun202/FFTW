#ifndef RADAR_PIPELINE_H
#define RADAR_PIPELINE_H

#include "radar_config.h"
#include "radar_fft.h"
#include <fftw3.h>

/**
 * @brief 3D 레이더 큐브 실수/허수 동시 전치 및 가변 처프 창함수 통합 가속 Kernel
 */
void transpose_radar_cube(const float *__restrict__ src_real, const float *__restrict__ src_imag,
                          float *__restrict__ dst_real, float *__restrict__ dst_imag, 
                          int n_samples, int n_chirps, const float *__restrict__ win);


void transpose_radar_cube_int16(int16_t *__restrict__ in_real, int16_t *__restrict__ in_imag,
                                int16_t *__restrict__ out_real, int16_t *__restrict__ out_imag,
                                int n_samples, int n_chirps, const int16_t *__restrict__ win_int16);


#endif // RADAR_PIPELINE_H