#include "radar_eval.h"
#include "radar_config.h"
#include "radar_fft.h" 
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

extern float doa_angle_lut_256[];

int extract_autonomous_objects(CFAR_Target *targets, int num_det, 
                               const float *tmp_r, const float *tmp_i, 
                               int n_samples, int n_chirps, 
                               FinalObject *out_objects, int max_output) {
    if (num_det == 0) return 0;

    int eps = 3; int min_pts = 1;   
    int cluster_id = 0;

    // 💥 [핵심 최적화] malloc/free 대신 스택(Stack) 메모리를 사용하는 VLA 선언
    // OS의 개입을 차단하여 클러스터링 연산 속도를 극단적으로 끌어올립니다.
    int labels[num_det];
    memset(labels, 0, num_det * sizeof(int));
    
    int neighbors[num_det];
    int queue[num_det];
    int curr_neighbors[num_det];

    // [1단계: DBSCAN Clustering Core - Zero Allocation 버전]
    for (int i = 0; i < num_det; i++) {
        if (labels[i] != 0) continue;
        
        int n_count = 0;
        for (int j = 0; j < num_det; j++) { 
            if (abs(targets[i].r_idx - targets[j].r_idx) <= eps && abs(targets[i].d_idx - targets[j].d_idx) <= eps) {
                neighbors[n_count++] = j; 
            }
        }
        
        if (n_count < min_pts) { 
            labels[i] = -1; 
            continue; 
        }
        
        cluster_id++; 
        labels[i] = cluster_id;
        
        int head = 0, tail = 0;
        for (int j = 0; j < n_count; j++) { 
            if (neighbors[j] != i) { 
                queue[tail++] = neighbors[j]; 
                labels[neighbors[j]] = cluster_id; 
            } 
        }
        
        while (head < tail) {
            int curr = queue[head++]; 
            int curr_n_count = 0;
            for (int j = 0; j < num_det; j++) { 
                if (abs(targets[curr].r_idx - targets[j].r_idx) <= eps && abs(targets[curr].d_idx - targets[j].d_idx) <= eps) { 
                    curr_neighbors[curr_n_count++] = j; 
                } 
            }
            if (curr_n_count >= min_pts) { 
                for (int j = 0; j < curr_n_count; j++) { 
                    int pt = curr_neighbors[j]; 
                    if (labels[pt] == 0 || labels[pt] == -1) { 
                        if (labels[pt] == 0) queue[tail++] = pt; 
                        labels[pt] = cluster_id; 
                    } 
                } 
            }
        }
    }

    // -------------------------------------------------------------------------------
    // 2단계: SNR 가드 해제 + Log-domain 포물선 보간 (초해상도 각도 복원)
    // -------------------------------------------------------------------------------
    int object_count = 0;
    const int angle_N = 256;
    
    for (int cid = 1; cid <= cluster_id && object_count < max_output; cid++) {
        int best_idx = -1; float max_power = -99999.0f;
        for (int i = 0; i < num_det; i++) { 
            if (labels[i] == cid) { 
                if (targets[i].power > max_power) { 
                    max_power = targets[i].power; best_idx = i; 
                } 
            } 
        }

        if (best_idx != -1) {
            float noise = (targets[best_idx].noise_floor > 1e-6f) ? targets[best_idx].noise_floor : 1e-6f;
            float snr_db = 10.0f * log10f(targets[best_idx].power / noise);
            
            // 💥 [SFDR 최적화] 16비트 정수 연산의 양자화 고스트 찌꺼기 완벽 차단
            if (snr_db < 24.0f) continue;

            int r_idx = targets[best_idx].r_idx;
            int d_idx = targets[best_idx].d_idx;

            float fft_in_r[256] __attribute__((aligned(64))) = {0.0f};
            float fft_in_i[256] __attribute__((aligned(64))) = {0.0f};
            float angle_pwr[256];

            float amp_scale = 256.0f; 

            for (int ant = 0; ant < N_ANTENNAS; ant++) {
                int src_idx = ant * (n_samples * n_chirps) + r_idx * n_chirps + d_idx;
                float win_factor = 1.0f; 
                
                if (N_ANTENNAS >= 8) {
                    win_factor = 0.54f - 0.46f * (float)cos(2.0 * PI * ant / (double)(N_ANTENNAS - 1));
                }

                fft_in_r[ant] = tmp_r[src_idx] * win_factor * amp_scale;
                fft_in_i[ant] = tmp_i[src_idx] * win_factor * amp_scale;
            } 

            custom_fft_256_fixed(fft_in_r, fft_in_i);

            float peak_limit = 0.0f;
            for (int k = 0; k < angle_N; k++) {
                angle_pwr[k] = fft_in_r[k] * fft_in_r[k] + fft_in_i[k] * fft_in_i[k];
                if (angle_pwr[k] > peak_limit) peak_limit = angle_pwr[k];
            }

            for (int k = 0; k < angle_N && object_count < max_output; k++) {
                int prev = (k - 1 + angle_N) % angle_N;
                int next = (k + 1) % angle_N;

                if (angle_pwr[k] > angle_pwr[prev] && angle_pwr[k] > angle_pwr[next] && angle_pwr[k] > (peak_limit * 0.2f)) {
                    
                    // 선형 전력을 로그(dB) 스케일로 변환하여 포물선 뼈대 구축
                    float db_k = 10.0f * log10f(angle_pwr[k] + 1e-6f);
                    float db_prev = 10.0f * log10f(angle_pwr[prev] + 1e-6f);
                    float db_next = 10.0f * log10f(angle_pwr[next] + 1e-6f);
                    
                    float denominator = db_prev - 2.0f * db_k + db_next;
                    float delta_k = 0.0f;
                    if (denominator != 0.0f) {
                        delta_k = 0.5f * (db_prev - db_next) / denominator;
                    }
                    
                    float k_true = (float)k + delta_k;
                    
                    float fft_bin = k_true;
                    if (fft_bin >= (float)angle_N / 2.0f) {
                        fft_bin -= (float)angle_N;
                    }
                    
                    double sin_theta = (fft_bin / (double)angle_N) * (lambda_c / d_ant);
                    if (sin_theta > 1.0) sin_theta = 1.0;
                    else if (sin_theta < -1.0) sin_theta = -1.0;
                    
                    float est_a = (float)(asin(sin_theta) * 180.0 / PI);
                    
                    if (est_a > 80.0f || est_a < -80.0f) continue;

                    double est_R = (r_idx * Fs / (double)n_samples) * c / (2.0 * S);
                    int shifted_ch = (d_idx >= n_chirps / 2) ? d_idx - n_chirps : d_idx;
                    double est_v = (shifted_ch * lambda_c) / (2.0 * n_chirps * Tc);

                    out_objects[object_count].distance = est_R;
                    out_objects[object_count].velocity = est_v;
                    out_objects[object_count].angle = (double)est_a;
                    out_objects[object_count].snr_db = snr_db;
                    object_count++;
                }
            }
        }
    }

    // 💥 VLA 스택 배열을 사용하므로 free(labels) 삭제 완료!
    return object_count; 
}