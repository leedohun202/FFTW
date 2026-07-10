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
/**
 * @brief 최적화된 Zero-Allocation 가변 2D Doppler FFT 파이프라인 집행 커널
 * @param n_chirps 현재 세션의 처프 개수 (256 또는 512)
 */
void execute_custom_pipeline(float *__restrict__ cube_real, float *__restrict__ cube_imag, 
                             float *__restrict__ tmp_real, float *__restrict__ tmp_imag, 
                             int n_samples, int n_chirps);

void execute_custom_pipeline_radix4(float *__restrict__ cube_real, float *__restrict__ cube_imag, 
                             float *__restrict__ tmp_real, float *__restrict__ tmp_imag, 
                             int n_samples, int n_chirps);
            
void execute_custom_pipeline_int16(int16_t *__restrict__ cube_real, int16_t *__restrict__ cube_imag, 
                                   int16_t *__restrict__ tmp_real, int16_t *__restrict__ tmp_imag, 
                                   int n_samples, int n_chirps);

void execute_custom_pipeline_int16_radix4(int16_t *__restrict__ cube_real, int16_t *__restrict__ cube_imag, 
                                   int16_t *__restrict__ tmp_real, int16_t *__restrict__ tmp_imag, 
                                   int n_samples, int n_chirps);
/**
 * @brief 대조군 FFTW3 전용 최적화 2D Doppler FFT 파이프라인
 */
void execute_fftw_pipeline_optimized(fftwf_complex *cube_in, fftwf_complex *cube_out, 
                                     fftwf_plan p_doppler, int n_samples);

#endif // RADAR_PIPELINE_H