#ifndef RADAR_PIPELINE_H
#define RADAR_PIPELINE_H

#include "radar_config.h"
#include "radar_fft.h"

void transpose_radar_cube(const float *__restrict__ src, float *__restrict__ dst, int n_samples);
void execute_custom_pipeline(float *__restrict__ cube_real, float *__restrict__ cube_imag, float *__restrict__ trans_cust_r, float *__restrict__ trans_cust_i, int n_samples);
void execute_fftw_pipeline_optimized(fftwf_complex* cube, fftwf_plan p_range, fftwf_plan p_doppler, int n_samples);

#endif // RADAR_PIPELINE_H