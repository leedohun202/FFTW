#ifndef RADAR_FFT_H
#define RADAR_FFT_H

#include "radar_config.h"

void custom_fft_2048_fixed(float *__restrict__ real, float *__restrict__ imag);
void custom_fft_1024_fixed(float *__restrict__ real, float *__restrict__ imag);
void custom_fft_512_fixed(float *__restrict__ real, float *__restrict__ imag);
void custom_fft_256_fixed(float *__restrict__ real, float *__restrict__ imag);
void custom_fft_128_fixed(float *__restrict__ real, float *__restrict__ imag);
void custom_fft_64_fixed(float *__restrict__ real, float *__restrict__ imag);

// 🔥 [신규 추가] 고속 Radix-4 FFT 커널 선언
void custom_fft_1024_radix4(float *__restrict__ real, float *__restrict__ imag);
void custom_fft_256_radix4(float *__restrict__ real, float *__restrict__ imag);
void custom_fft_64_radix4(float *__restrict__ real, float *__restrict__ imag);

#endif // RADAR_FFT_H