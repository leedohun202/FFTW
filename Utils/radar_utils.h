#ifndef RADAR_UTILS_H
#define RADAR_UTILS_H

#include "radar_config.h"
#include "radar_pipeline.h"

int get_current_ram_usage_kb();
double get_current_time_ms();
void init_resources();

// 🔥 [추가됨] 가변 처프 시뮬레이션을 위해 새로 분리된 코어 함수 선언
void run_benchmark_core(int n_samples, int n_chirps);

void run_benchmark_for_size(int n_samples);
void profile_individual_operation(int size, void (*custom_func)(float*, float*));

#endif // RADAR_UTILS_H