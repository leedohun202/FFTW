#ifndef RADAR_UTILS_H
#define RADAR_UTILS_H

#include <stdint.h> // int16_t 타입을 위해 필수

#include "radar_config.h"
#include "radar_pipeline.h"

int get_current_ram_usage_kb();
double get_current_time_ms();
void init_resources();

#endif // RADAR_UTILS_H