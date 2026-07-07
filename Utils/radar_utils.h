#ifndef RADAR_UTILS_H
#define RADAR_UTILS_H

#include "radar_config.h"
#include "radar_pipeline.h"

int get_current_ram_usage_kb();
double get_current_time_ms();
void init_resources();
void run_benchmark_for_size(int n_samples);
void profile_individual_operation(int size, void (*custom_func)(float*, float*));

#endif // RADAR_UTILS_H