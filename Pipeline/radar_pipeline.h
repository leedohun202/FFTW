#ifndef RADAR_PIPELINE_H
#define RADAR_PIPELINE_H

#include "radar_config.h"
#include "radar_fft.h"
#include <fftw3.h>

/**
 * @brief 3D 레이더 큐브 메모리 전치 및 창함수 통합 가속 Kernel (Loop Fusion)
 * @param win 창함수 배열 포인터 추가 (N_CHIRPS 크기)
 */
void transpose_radar_cube(const float *__restrict__ src_real, const float *__restrict__ src_imag,
                          float *__restrict__ dst_real, float *__restrict__ dst_imag, 
                          int n_samples, const float *__restrict__ win);
/**
 * @brief Zero-Allocation 및 최적화된 2D Doppler FFT 파이프라인
 * @param tmp_real 외부에서 1회만 할당된 전치용 임시 실수부 버퍼
 * @param tmp_imag 외부에서 1회만 할당된 전치용 임시 허수부 버퍼
 */
void execute_custom_pipeline(float *__restrict__ cube_real, float *__restrict__ cube_imag, 
                             float *__restrict__ tmp_real, float *__restrict__ tmp_imag, 
                             int n_samples);

void execute_fftw_pipeline_optimized(fftwf_complex *cube_in, fftwf_complex *cube_out, 
                                     fftwf_plan p_doppler, int n_samples);

#endif // RADAR_PIPELINE_H