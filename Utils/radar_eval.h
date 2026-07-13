#ifndef RADAR_EVAL_H
#define RADAR_EVAL_H

#include "radar_cfar.h"
#include "radar_benchmark_sessions.h"

void extract_autonomous_objects(CFAR_Target *targets, int num_det, 
                                const float *tmp_r, const float *tmp_i, 
                                int n_samples, int n_chirps, BenchmarkResult *out, 
                                const double *true_R);

#endif // RADAR_EVAL_H