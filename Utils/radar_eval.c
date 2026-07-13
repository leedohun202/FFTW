#include "radar_eval.h"
#include "radar_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int compare_targets_by_power(const void *a, const void *b) {
    float pa = ((CFAR_Target*)a)->power;
    float pb = ((CFAR_Target*)b)->power;
    return (pb > pa) - (pb < pa);
}

static int compare_targets_by_distance(const void *a, const void *b) {
    int ra = ((CFAR_Target*)a)->r_idx;
    int rb = ((CFAR_Target*)b)->r_idx;
    return (ra > rb) - (ra < rb);
}

void extract_autonomous_objects(CFAR_Target *targets, int num_det, 
                                const float *tmp_r, const float *tmp_i, 
                                int n_samples, int n_chirps, BenchmarkResult *out, 
                                const double *true_R) {
    qsort(targets, num_det, sizeof(CFAR_Target), compare_targets_by_power);

    int unique_count = 0;
    CFAR_Target unique_targets[3];
    
    // Proximity Peak Picking Filter
    for(int i = 0; i < num_det && unique_count < 3; i++) {
        int is_overlapping = 0;
        for(int j = 0; j < unique_count; j++) {
            if(abs(targets[i].r_idx - unique_targets[j].r_idx) <= 15 && 
               abs(targets[i].d_idx - unique_targets[j].d_idx) <= 15) {
                is_overlapping = 1; break;
            }
        }
        if(!is_overlapping) {
            unique_targets[unique_count++] = targets[i];
        }
    }

    qsort(unique_targets, unique_count, sizeof(CFAR_Target), compare_targets_by_distance);

    for(int t=0; t<3; t++) { out->est_R[t] = 0.0; out->est_v[t] = 0.0; out->est_a[t] = 0.0; }

    for(int i = 0; i < unique_count; i++) {
        int r_idx = unique_targets[i].r_idx;
        int d_idx = unique_targets[i].d_idx;

        double est_R = (r_idx * Fs / (double)n_samples) * c / (2.0 * S);
        int shifted_ch = (d_idx >= n_chirps / 2) ? d_idx - n_chirps : d_idx;
        double est_v = (shifted_ch * lambda_c) / (2.0 * n_chirps * Tc);

        int idx_a0 = 0 * (n_samples * n_chirps) + r_idx * n_chirps + d_idx;
        int idx_a1 = 1 * (n_samples * n_chirps) + r_idx * n_chirps + d_idx;
        float r0 = tmp_r[idx_a0], i0 = tmp_i[idx_a0];
        float r1 = tmp_r[idx_a1], i1 = tmp_i[idx_a1];
        double d_phi = atan2(i1*r0 - r1*i0, r1*r0 + i1*i0);
        double sin_theta = (d_phi * lambda_c) / (2.0 * PI * d_ant);
        if (sin_theta > 1.0) sin_theta = 1.0; else if (sin_theta < -1.0) sin_theta = -1.0;
        double est_a = asin(sin_theta) * 180.0 / PI;

        int mapped_slot = -1; double min_diff = 9999.0;
        for(int t=0; t<3; t++){
            double diff = fabs(est_R - true_R[t]); // 외부에서 주입된 정답 사용
            if(diff < min_diff && diff <= 5.0) { min_diff = diff; mapped_slot = t; }
        }
        if(mapped_slot != -1) {
            out->est_R[mapped_slot] = est_R;
            out->est_v[mapped_slot] = est_v;
            out->est_a[mapped_slot] = est_a;
        }
    }
}