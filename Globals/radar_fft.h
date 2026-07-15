#ifndef RADAR_FFT_H
#define RADAR_FFT_H

#include "radar_config.h"
#include <stdint.h>

// =========================================================================
// 🔹 [GROUP 1] 순수 Float 기반 Radix-2 / NEON 가속 커널 (FFT/ 폴더)
// =========================================================================
void custom_fft_2048_fixed(float *__restrict__ real, float *__restrict__ imag);
void custom_fft_1024_fixed(float *__restrict__ real, float *__restrict__ imag);
void custom_fft_512_fixed(float *__restrict__ real, float *__restrict__ imag);
void custom_fft_256_fixed(float *__restrict__ real, float *__restrict__ imag);
void custom_fft_128_fixed(float *__restrict__ real, float *__restrict__ imag);
void custom_fft_64_fixed(float *__restrict__ real, float *__restrict__ imag);

// =========================================================================
// 🔹 [GROUP 2] 순수 Float 기반 Radix-4 / NEON 가속 커널 (FFT_RADIX4/ 폴더)
// =========================================================================
void custom_fft_2048_radix4(float *__restrict__ real, float *__restrict__ imag);
void custom_fft_1024_radix4(float *__restrict__ real, float *__restrict__ imag);
void custom_fft_512_radix4(float *__restrict__ real, float *__restrict__ imag);
void custom_fft_256_radix4(float *__restrict__ real, float *__restrict__ imag);
void custom_fft_128_radix4(float *__restrict__ real, float *__restrict__ imag);
void custom_fft_64_radix4(float *__restrict__ real, float *__restrict__ imag);

// =========================================================================
// 🔹 [GROUP 3] 하드코어 고정소수점 int16_t 전용 커널 (FFT_int16/ 폴더)
// =========================================================================
void custom_fft_2048_int16(int16_t *__restrict__ real, int16_t *__restrict__ imag);
void custom_fft_1024_int16(int16_t *__restrict__ real, int16_t *__restrict__ imag);
void custom_fft_512_int16(int16_t *__restrict__ real, int16_t *__restrict__ imag, int skip_shift);
void custom_fft_256_int16(int16_t *__restrict__ real, int16_t *__restrict__ imag, int skip_shift);
void custom_fft_128_int16(int16_t *__restrict__ real, int16_t *__restrict__ imag);
void custom_fft_64_int16(int16_t *__restrict__ real, int16_t *__restrict__ imag);

void custom_fft_64_int16_radix4(int16_t *__restrict__ real, int16_t *__restrict__ imag);
void custom_fft_128_int16_radix4(int16_t *__restrict__ real, int16_t *__restrict__ imag);
void custom_fft_256_int16_radix4(int16_t *__restrict__ real, int16_t *__restrict__ imag, int skip_shift);
void custom_fft_512_int16_radix4(int16_t *__restrict__ real, int16_t *__restrict__ imag, int skip_shift);
void custom_fft_1024_int16_radix4(int16_t *__restrict__ real, int16_t *__restrict__ imag);
void custom_fft_2048_int16_radix4(int16_t *__restrict__ real, int16_t *__restrict__ imag);


#endif // RADAR_FFT_H