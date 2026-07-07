#ifndef RADAR_PIPELINE_H
#define RADAR_PIPELINE_H

#include "radar_config.h"
#include "radar_fft.h"
#include <fftw3.h>

// 🎯 [다이어트 싱크 완료] 3D 확장 버퍼(trans)를 도려낸 3인자 함수 원형
void transpose_radar_cube(const float *__restrict__ src, float *__restrict__ dst, int n_samples);
void execute_custom_pipeline(float *__restrict__ cube_real, float *__restrict__ cube_imag, int n_samples);

void execute_fftw_pipeline_optimized(fftwf_complex *cube_in, fftwf_complex *cube_out, 
                                     fftwf_plan p_doppler, int n_samples);

#endif // RADAR_PIPELINE_H