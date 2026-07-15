#ifndef RADAR_BENCHMARK_SESSIONS_H
#define RADAR_BENCHMARK_SESSIONS_H

#include <stdint.h>
#include "radar_eval.h" // 💥 FinalObject 구조체 참조

#define MAX_OUTPUT_TARGETS 10 // 동적 출력을 위한 최대 허용 물체 수

// 💥 정답지 시대의 배열[3]을 버리고, 동적 객체 배열과 개수 카운터로 변경
typedef struct {
    FinalObject objects[MAX_OUTPUT_TARGETS]; 
    int num_targets;      // 레이더가 스스로 찾아낸 실제 물체 개수
    double time_range;
    double time_doppler;
    double time_angle;
    double actual_ram_mb;
} BenchmarkResult;

// 5개 세션 함수 선언 (그대로 유지)
void benchmark_session_fftw3(const float *lut_r, const float *lut_i, int n_samples, int n_chirps, BenchmarkResult *out, unsigned int fftw_flags);
void benchmark_session_float_r2(const float *lut_r, const float *lut_i, int n_samples, int n_chirps, BenchmarkResult *out);
void benchmark_session_float_r4(const float *lut_r, const float *lut_i, int n_samples, int n_chirps, BenchmarkResult *out);
void benchmark_session_int16_r2(const float *lut_r, const float *lut_i, int n_samples, int n_chirps, BenchmarkResult *out);
void benchmark_session_int16_r4(const float *lut_r, const float *lut_i, int n_samples, int n_chirps, BenchmarkResult *out);

#endif // RADAR_BENCHMARK_SESSIONS_H