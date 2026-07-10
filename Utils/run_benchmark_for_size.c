#define _GNU_SOURCE

#include "radar_config.h"
#include "radar_utils.h"
#include "radar_benchmark_sessions.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void run_benchmark_core(int n_samples, int n_chirps) {
    int total_elements = N_ANTENNAS * n_chirps * n_samples;
    size_t alloc_size_float = total_elements * sizeof(float);
    size_t alignment = 64; 

    printf("\n 🛰️ [3D 가속 배틀] Range = %d | Chirps = %d\n", n_samples, n_chirps);
    printf(" ====================================================\n");

    // 공통 원본 데이터(LUT) 생성
    float *lut_r = NULL; float *lut_i = NULL;
    posix_memalign((void**)&lut_r, alignment, alloc_size_float);
    posix_memalign((void**)&lut_i, alignment, alloc_size_float);

    double true_R[] = {12.50, 25.20,  6.80}; 
    double true_v[] = {14.20, -5.50,  0.00}; 
    double true_a[] = {-15.0,  25.0,  0.00}; 
    int num_targets = 3;

    for (int ant = 0; ant < N_ANTENNAS; ant++) {
        for (int chirp = 0; chirp < n_chirps; chirp++) {
            int offset = ant * (n_chirps * n_samples) + chirp * n_samples;
            for (int n = 0; n < n_samples; n++) {
                double val_r = 0.0, val_i = 0.0;
                for (int t = 0; t < num_targets; t++) {
                    double f_R = (2.0 * S * true_R[t]) / c; 
                    double p_doppler = (4.0 * PI * true_v[t]) / lambda_c * Tc; 
                    double p_angle = (2.0 * PI * d_ant * sin(true_a[t] * M_PI / 180.0)) / lambda_c;
                    double phase = (2.0 * M_PI * f_R * ((double)n / Fs)) + (p_doppler * chirp) + (p_angle * ant);
                    double s, c_val;
                    sincos(phase, &s, &c_val);
                    val_r += c_val; val_i += s;
                }
                lut_r[offset + n] = (float)val_r; lut_i[offset + n] = (float)val_i;
            }
        }
    }

    // 🎯 5대 천왕 독립 저장 채널 선언
    BenchmarkResult res_fftw, res_f_r2, res_f_r4, res_i16_r2, res_i16_r4;

    // 🚀 모듈화된 세션 순차 실행 및 물리 지표 획득
    benchmark_session_fftw3(lut_r, lut_i, n_samples, n_chirps, &res_fftw);
    benchmark_session_float_r2(lut_r, lut_i, n_samples, n_chirps, &res_f_r2);
    benchmark_session_float_r4(lut_r, lut_i, n_samples, n_chirps, &res_f_r4);
    benchmark_session_int16_r2(lut_r, lut_i, n_samples, n_chirps, &res_i16_r2);
    benchmark_session_int16_r4(lut_r, lut_i, n_samples, n_chirps, &res_i16_r4);

    // ==========================================
    // 📺 종합 레이더 3D 5채널 정교 리포트 출력
    // ==========================================
    printf("  ---------------------------------------------------\n");
    printf("  [1] FFTW3       (Range / Doppler) : %6.2f ms / %6.2f ms\n", res_fftw.time_range, res_fftw.time_doppler);
    printf("  [2] Float R-2   (Range / Doppler) : %6.2f ms / %6.2f ms\n", res_f_r2.time_range, res_f_r2.time_doppler);
    printf("  [3] Float R-4   (Range / Doppler) : %6.2f ms / %6.2f ms\n", res_f_r4.time_range, res_f_r4.time_doppler);
    printf("  [4] Int16 R-2   (Range / Doppler) : %6.2f ms / %6.2f ms\n", res_i16_r2.time_range, res_i16_r2.time_doppler);
    printf("  [5] Int16 R-4   (Range / Doppler) : %6.2f ms / %6.2f ms\n", res_i16_r4.time_range, res_i16_r4.time_doppler);
    
    printf("  ---------------------------------------------------\n");
    printf(" 🎯 [알고리즘별 물리 타겟 역추적 정밀도 (Range / Velocity / Angle)]\n");
    for (int t = 0; t < num_targets; t++) {
        printf("  ---------------------------------------------------\n");
        printf("  📍 [Target %d] 참값: Range %6.3fm | Vel %6.3fm/s | Angle %6.3f°\n", t+1, true_R[t], true_v[t], true_a[t]);
        
        printf("   - FFTW3     : R %6.3fm (오차 %5.3f) | V %6.3fm/s (오차 %5.3f) | A %7.3f° (오차 %5.3f)\n", 
            res_fftw.est_R[t], fabs(res_fftw.est_R[t]-true_R[t]), res_fftw.est_v[t], fabs(res_fftw.est_v[t]-true_v[t]), res_fftw.est_a[t], fabs(res_fftw.est_a[t]-true_a[t]));
            
        printf("   - Float R-2 : R %6.3fm (오차 %5.3f) | V %6.3fm/s (오차 %5.3f) | A %7.3f° (오차 %5.3f)\n", 
            res_f_r2.est_R[t], fabs(res_f_r2.est_R[t]-true_R[t]), res_f_r2.est_v[t], fabs(res_f_r2.est_v[t]-true_v[t]), res_f_r2.est_a[t], fabs(res_f_r2.est_a[t]-true_a[t]));
            
        printf("   - Float R-4 : R %6.3fm (오차 %5.3f) | V %6.3fm/s (오차 %5.3f) | A %7.3f° (오차 %5.3f)\n", 
            res_f_r4.est_R[t], fabs(res_f_r4.est_R[t]-true_R[t]), res_f_r4.est_v[t], fabs(res_f_r4.est_v[t]-true_v[t]), res_f_r4.est_a[t], fabs(res_f_r4.est_a[t]-true_a[t]));
            
        printf("   - Int16 R-2 : R %6.3fm (오차 %5.3f) | V %6.3fm/s (오차 %5.3f) | A %7.3f° (오차 %5.3f)\n", 
            res_i16_r2.est_R[t], fabs(res_i16_r2.est_R[t]-true_R[t]), res_i16_r2.est_v[t], fabs(res_i16_r2.est_v[t]-true_v[t]), res_i16_r2.est_a[t], fabs(res_i16_r2.est_a[t]-true_a[t]));

        printf("   - Int16 R-4 : R %6.3fm (오차 %5.3f) | V %6.3fm/s (오차 %5.3f) | A %7.3f° (오차 %5.3f)\n", 
            res_i16_r4.est_R[t], fabs(res_i16_r4.est_R[t]-true_R[t]), res_i16_r4.est_v[t], fabs(res_i16_r4.est_v[t]-true_v[t]), res_i16_r4.est_a[t], fabs(res_i16_r4.est_a[t]-true_a[t]));
    }
    printf("  ---------------------------------------------------\n");
    
    printf(" 🖥️ [운영체제(OS) 물리 메모리 정밀 실측 리포트]\n");
    printf("  - 📈 OS 실측 FFTW3 할당 RAM       : %8.2f MB\n", res_fftw.actual_ram_mb);
    printf("  - 📈 OS 실측 Custom Float R-2 RAM : %8.2f MB\n", res_f_r2.actual_ram_mb);
    printf("  - 📈 OS 실측 Custom Float R-4 RAM : %8.2f MB\n", res_f_r4.actual_ram_mb);
    printf("  - 📈 OS 실측 Custom Int16 R-2 RAM : %8.2f MB\n", res_i16_r2.actual_ram_mb);
    printf("  - 📈 OS 실측 Custom Int16 R-4 RAM : %8.2f MB\n", res_i16_r4.actual_ram_mb);
    printf(" ====================================================\n\n");
    fflush(stdout);

    if (lut_r) { free(lut_r); } if (lut_i) { free(lut_i); }
}

void run_benchmark_for_size(int n_samples, int n_chirps) {
    run_benchmark_core(n_samples, n_chirps);
}