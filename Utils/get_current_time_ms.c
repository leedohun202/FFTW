#include "radar_utils.h"

double get_current_time_ms() {
    struct timeval tv; gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}