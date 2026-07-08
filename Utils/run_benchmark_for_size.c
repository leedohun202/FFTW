#define _GNU_SOURCE

#include "radar_config.h"
#include "radar_utils.h"
#include "radar_fft.h"
#include "radar_pipeline.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fftw3.h>

void run_benchmark_core(int n_samples, int n_chirps) {
    int total_elements = N_ANTENNAS * n_chirps * n_samples;
    size_t alloc_size = total_elements * sizeof(float);
    size_t alignment = 64; 

    printf("\n 🛰️ [3D 가속 배틀] Range = %d | Chirps = %d\n", n_samples, n_chirps);
    printf(" ====================================================\n");

    int ram_start_kb = get_current_ram_usage_kb();

    float *cust_cube_r = NULL; float *cust_cube_i = NULL;
    float *pipeline_tmp_r = NULL; float *pipeline_tmp_i = NULL;
    posix_memalign((void**)&cust_cube_r, alignment, alloc_size);
    posix_memalign((void**)&cust_cube_i, alignment, alloc_size);
    posix_memalign((void**)&pipeline_tmp_r, alignment, alloc_size);
    posix_memalign((void**)&pipeline_tmp_i, alignment, alloc_size);

    fftwf_complex *fftw_cube_in  = (fftwf_complex *)fftwf_malloc(total_elements * sizeof(fftwf_complex));
    fftwf_complex *fftw_cube_out = (fftwf_complex *)fftwf_malloc(total_elements * sizeof(fftwf_complex));

    double true_R[] = {12.50, 25.20,  6.80}; 
    double true_v[] = {14.20, -5.50,  0.00}; 
    double true_a[] = {-15.0, 25.0,   0.0}; 
    int num_targets = 3;

    unsigned int fftw_flags = FFTW_MEASURE;

    float *lut_r = NULL; float *lut_i = NULL;
    posix_memalign((void**)&lut_r, alignment, alloc_size);
    posix_memalign((void**)&lut_i, alignment, alloc_size);

    // 가상 타겟 데이터 주입
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

    // 1️⃣ [Custom] 1D Range-FFT
    double start_cust_range = get_current_time_ms();
    #pragma omp parallel for collapse(2)
    for (int ant = 0; ant < N_ANTENNAS; ant++) {
        for (int chirp = 0; chirp < n_chirps; chirp++) {
            int offset = ant * (n_chirps * n_samples) + chirp * n_samples;
            for (int n = 0; n < n_samples; n++) {
                float win = (n_samples == 1024) ? win_1024[n] : win_2048[n];
                cust_cube_r[offset + n] = lut_r[offset + n] * win;
                cust_cube_i[offset + n] = lut_i[offset + n] * win;
            }
            
            // 🚨 [수정 완료] 1024는 Radix-4, 2048은 도훈님의 기존 Fixed(Radix-2) 호출!
            if (n_samples == 1024) custom_fft_1024_radix4(&cust_cube_r[offset], &cust_cube_i[offset]);
            else                   custom_fft_2048_fixed(&cust_cube_r[offset], &cust_cube_i[offset]);
        }
    }
    double cust_range_ms = get_current_time_ms() - start_cust_range;

    // 2️⃣ [Custom] 2D Doppler-FFT 파이프라인
    double start_cust_doppler = get_current_time_ms();
    execute_custom_pipeline(cust_cube_r, cust_cube_i, pipeline_tmp_r, pipeline_tmp_i, n_samples, n_chirps);
    double cust_doppler_ms = get_current_time_ms() - start_cust_doppler;

    // 3️⃣ [FFTW3 대조군] 플랜 빌드 및 연산
    fftwf_plan p_range = fftwf_plan_dft_1d(n_samples, fftw_cube_in, fftw_cube_in, FFTW_FORWARD, fftw_flags);
    int d_rank = 1; int d_n[] = {n_chirps}; int d_howmany = N_ANTENNAS * n_samples;
    int d_idist = n_chirps, d_odist = n_chirps; int d_istride = 1, d_ostride = 1;
    fftwf_plan p_doppler = fftwf_plan_many_dft(d_rank, d_n, d_howmany, fftw_cube_in, NULL, d_istride, d_idist, fftw_cube_out, NULL, d_ostride, d_odist, FFTW_FORWARD, fftw_flags);

    double start_fftw = get_current_time_ms();
    #pragma omp parallel for collapse(2)
    for (int ant = 0; ant < N_ANTENNAS; ant++) {
        for (int chirp = 0; chirp < n_chirps; chirp++) {
            int offset = ant * (n_chirps * n_samples) + chirp * n_samples;
            for (int n = 0; n < n_samples; n++) {
                float win = (n_samples == 1024) ? win_1024[n] : win_2048[n];
                fftw_cube_in[offset + n][0] = lut_r[offset + n] * win;
                fftw_cube_in[offset + n][1] = lut_i[offset + n] * win;
            }
            fftwf_execute_dft(p_range, &fftw_cube_in[offset], &fftw_cube_in[offset]);
        }
    }
    double fftw_range_ms = get_current_time_ms() - start_fftw;

    double start_fftw_doppler = get_current_time_ms();
    execute_fftw_pipeline_optimized(fftw_cube_in, fftw_cube_out, p_doppler, n_samples);
    double fftw_doppler_ms = get_current_time_ms() - start_fftw_doppler;

    fftwf_destroy_plan(p_range); fftwf_destroy_plan(p_doppler);

    // 물리 값 정밀 검증
    printf(" 🎯 [물리 값 검증] -> ");
    int pass_count = 0;
    for (int t_idx = 0; t_idx < num_targets; t_idx++) {
        // 🚨 [수정 완료] 불필요했던 best_chirp 변수 선언 제거로 Warning 해결
        float max_2d_mag = -1.0f; int best_r = 0;
        int r_center = (int)(((2.0 * S * true_R[t_idx]) / c) * n_samples / Fs);
        int r_start = r_center - 15; if (r_start < 0) r_start = 0;
        int r_end = r_center + 15; if (r_end >= n_samples) r_end = n_samples - 1;
        
        for (int r = r_start; r <= r_end; r++) {
            for (int ch = 0; ch < n_chirps; ch++) {
                float sum_mag = 0.0f;
                for (int ant = 0; ant < N_ANTENNAS; ant++) {
                    int idx = ant * (n_samples * n_chirps) + r * n_chirps + ch;
                    sum_mag += pipeline_tmp_r[idx]*pipeline_tmp_r[idx] + pipeline_tmp_i[idx]*pipeline_tmp_i[idx];
                }
                if (sum_mag > max_2d_mag) { max_2d_mag = sum_mag; best_r = r; } // 여기서만 쓰임
            }
        }
        double est_Range = (best_r * Fs / n_samples) * c / (2.0 * S);
        if (fabs(est_Range - true_R[t_idx]) / true_R[t_idx] * 100.0 < 5.0) pass_count++;
    }
    printf("%d / 3개 타겟 정밀합격\n", pass_count);

    int ram_end_kb = get_current_ram_usage_kb();
    double current_session_ram_mb = (ram_end_kb - ram_start_kb) / 1024.0;

    printf("  ---------------------------------------------------\n");
    printf("  ▪️ [Custom] 1D Range-FFT            : %6.2f ms\n", cust_range_ms);
    printf("  ▪️ [Custom] 2D Doppler-FFT 파이프라인 : %6.2f ms\n", cust_doppler_ms);
    printf("  💻 [FFTW3]  1D Range-FFT 타임       : %6.2f ms\n", fftw_range_ms);
    printf("  💻 [FFTW3]  2D Doppler-FFT 타임     : %6.2f ms\n", fftw_doppler_ms);
    printf("  🔥 본 세션 실시간 메모리 순수 점유   : %.2f MB\n", current_session_ram_mb < 0 ? 0 : current_session_ram_mb);
    printf(" ====================================================\n");

    fftwf_free(cust_cube_r); fftwf_free(cust_cube_i);
    free(pipeline_tmp_r); free(pipeline_tmp_i);
    free(lut_r); free(lut_i);
    fftwf_free(fftw_cube_in); fftwf_free(fftw_cube_out);
}

void run_benchmark_for_size(int n_samples) {
    run_benchmark_core(n_samples, 256);
    run_benchmark_core(n_samples, 512);
}