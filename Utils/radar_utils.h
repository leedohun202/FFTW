#ifndef RADAR_UTILS_H
#define RADAR_UTILS_H

#include <stdint.h> // int16_t 타입을 위해 필수

#include "radar_config.h"
#include "radar_pipeline.h"

int get_current_ram_usage_kb();
double get_current_time_ms();
void init_resources();

// 🔥 [추가됨] 가변 처프 시뮬레이션을 위해 새로 분리된 코어 함수 선언
void run_benchmark_core(int n_samples, int n_chirps);

void run_benchmark_for_size(int n_samples, int n_chirps);

void run_individual_operation_benchmark(void);

double profile_fftw_operation(int size);
void profile_float_operation(int size, const char* name, void (*custom_func)(float*, float*), double fftw_us);
void profile_int16_operation(int size, const char* name, void (*custom_func)(int16_t*, int16_t*), double fftw_us);

void verify_twiddle_factors(const int16_t *real, const int16_t *imag, int N, const char *name);

#endif // RADAR_UTILS_H