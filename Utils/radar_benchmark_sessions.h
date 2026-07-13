#ifndef RADAR_BENCHMARK_SESSIONS_H
#define RADAR_BENCHMARK_SESSIONS_H

#include <stdint.h>

// 타겟 추출 결과를 담을 구조체
typedef struct {
    double est_R[3];
    double est_v[3];
    double est_a[3];
    double time_range;
    double time_doppler;
    double actual_ram_mb;
} BenchmarkResult;

// 5개 세션 함수 선언
void benchmark_session_fftw3(const float *lut_r, const float *lut_i, int n_samples, int n_chirps, BenchmarkResult *out, unsigned int fftw_flags);
void benchmark_session_float_r2(const float *lut_r, const float *lut_i, int n_samples, int n_chirps, BenchmarkResult *out);
void benchmark_session_float_r4(const float *lut_r, const float *lut_i, int n_samples, int n_chirps, BenchmarkResult *out);
void benchmark_session_int16_r2(const float *lut_r, const float *lut_i, int n_samples, int n_chirps, BenchmarkResult *out);
void benchmark_session_int16_r4(const float *lut_r, const float *lut_i, int n_samples, int n_chirps, BenchmarkResult *out);

#endif