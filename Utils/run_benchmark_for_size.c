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
#include <sys/time.h>
#include <malloc.h> // 👈 malloc_trim(0) 사용을 위한 헤더 추가

#ifdef _OPENMP
#include <omp.h>
#endif

static void flush_cache(void) {
    const int cache_size = 32 * 1024 * 1024; 
    volatile char *dummy = (char*)malloc(cache_size);
    if (dummy) {
        for (int i = 0; i < cache_size; i += 64) { dummy[i] = 1; }
        free((void*)dummy);
    }
}

static inline void call_custom_float_radix2(float *r, float *i, int n) {
    if (n == 2048)       { custom_fft_2048_fixed(r, i); }
    else if (n == 1024)  { custom_fft_1024_fixed(r, i); }
    else if (n == 512)   { custom_fft_512_fixed(r, i); }
    else if (n == 256)   { custom_fft_256_fixed(r, i); }
    else if (n == 128)   { custom_fft_128_fixed(r, i); }
    else if (n == 64)    { custom_fft_64_fixed(r, i); }
}

static inline void call_custom_float_radix4(float *r, float *i, int n) {
    if (n == 2048)       { custom_fft_2048_radix4(r, i); }
    else if (n == 1024)  { custom_fft_1024_radix4(r, i); }
    else if (n == 512)   { custom_fft_512_radix4(r, i); }
    else if (n == 256)   { custom_fft_256_radix4(r, i); }
    else if (n == 128)   { custom_fft_128_fixed(r, i); }   
    else if (n == 64)    { custom_fft_64_radix4(r, i); }
}

void run_benchmark_core(int n_samples, int n_chirps) {
    int total_elements = N_ANTENNAS * n_chirps * n_samples;
    size_t alloc_size_float = total_elements * sizeof(float);
    size_t alloc_size_int16 = total_elements * sizeof(int16_t);
    size_t alloc_size_int16_padded = (total_elements + 8) * sizeof(int16_t);
    size_t alignment = 64; 

    printf("\n 🛰️ [3D 가속 배틀] Range = %d | Chirps = %d\n", n_samples, n_chirps);
    printf(" ====================================================\n");

    // 공통 원본 데이터(LUT) 생성
    float *lut_r = NULL; float *lut_i = NULL;
    posix_memalign((void**)&lut_r, alignment, alloc_size_float);
    posix_memalign((void**)&lut_i, alignment, alloc_size_float);

    double true_R[] = {12.50, 25.20,  6.80}; 
    double true_v[] = {14.20, -5.50,  0.00}; 
    double true_a[] = {-15.0, 25.0,   0.0}; 
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

    double est_R_fftw[3], est_R_r2[3], est_R_r4[3], est_R_i16[3];

    // ==========================================
    // 1️⃣ [FFTW3 대조군 세션] - OS 메모리 실측
    // ==========================================
    malloc_trim(0);
    int base_mem_fftw = get_current_ram_usage_kb();

    fftwf_complex *fftw_in  = (fftwf_complex *)fftwf_malloc(total_elements * sizeof(fftwf_complex));
    fftwf_complex *fftw_out = (fftwf_complex *)fftwf_malloc(total_elements * sizeof(fftwf_complex));

    unsigned int fftw_flags = FFTW_MEASURE; 
#if defined(FFTW_MODE_ESTIMATE)
    fftw_flags = FFTW_ESTIMATE;
#elif defined(FFTW_MODE_PATIENT)
    fftw_flags = FFTW_PATIENT;
#endif

    fftwf_plan p_range = fftwf_plan_dft_1d(n_samples, fftw_in, fftw_in, FFTW_FORWARD, fftw_flags);
    int d_n[] = {n_chirps}; 
    fftwf_plan p_doppler = fftwf_plan_many_dft(1, d_n, N_ANTENNAS * n_samples, fftw_in, NULL, 1, n_chirps, fftw_out, NULL, 1, n_chirps, FFTW_FORWARD, fftw_flags);

    // 물리 메모리에 적재 (Page Fault 강제 유발)
    #pragma omp parallel for
    for (int i = 0; i < total_elements; i++) {
        fftw_in[i][0] = lut_r[i]; fftw_in[i][1] = lut_i[i];
        fftw_out[i][0] = 0; fftw_out[i][1] = 0;
    }

    int peak_mem_fftw = get_current_ram_usage_kb();
    double actual_mb_fftw = (peak_mem_fftw - base_mem_fftw) / 1024.0;

    flush_cache();
    double start_fftw_range = get_current_time_ms();
    if (p_range) {
        #pragma omp parallel for collapse(2)
        for (int ant = 0; ant < N_ANTENNAS; ant++) {
            for (int chirp = 0; chirp < n_chirps; chirp++) {
                int offset = ant * (n_chirps * n_samples) + chirp * n_samples;
                for (int n = 0; n < n_samples; n++) {
                    float win = (n_samples == 2048) ? win_2048[n] : win_1024[n];
                    fftw_in[offset + n][0] = lut_r[offset + n] * win;
                    fftw_in[offset + n][1] = lut_i[offset + n] * win;
                }
                fftwf_execute_dft(p_range, &fftw_in[offset], &fftw_in[offset]);
            }
        }
    }
    double time_fftw_range = get_current_time_ms() - start_fftw_range;

    double start_fftw_doppler = get_current_time_ms();
    if (p_doppler) { execute_fftw_pipeline_optimized(fftw_in, fftw_out, p_doppler, n_samples); }
    double time_fftw_doppler = get_current_time_ms() - start_fftw_doppler;

    // 타겟 역추적
    for (int t = 0; t < num_targets; t++) {
        float max_mag = -1.0f; int best_r = 0;
        int r_center = (int)(((2.0 * S * true_R[t]) / c) * n_samples / Fs);
        int r_start = (r_center - 15 < 0) ? 0 : r_center - 15;
        int r_end = (r_center + 15 >= n_samples) ? n_samples - 1 : r_center + 15;
        for (int r = r_start; r <= r_end; r++) {
            for (int ch = 0; ch < n_chirps; ch++) {
                float sum_mag = 0.0f;
                for (int ant = 0; ant < N_ANTENNAS; ant++) {
                    int idx = ant * (n_chirps * n_samples) + ch * n_samples + r; 
                    sum_mag += fftw_out[idx][0]*fftw_out[idx][0] + fftw_out[idx][1]*fftw_out[idx][1];
                }
                if (sum_mag > max_mag) { max_mag = sum_mag; best_r = r; }
            }
        }
        est_R_fftw[t] = (best_r * Fs / (double)n_samples) * c / (2.0 * S);
    }
    // 정확도 비교를 위해 fftw_out은 마지막에 free 합니다.
    if (p_range) { fftwf_destroy_plan(p_range); p_range = NULL; }
    if (p_doppler) { fftwf_destroy_plan(p_doppler); p_doppler = NULL; }

    // ==========================================
    // 2️⃣ [Custom Float 가속 세션] - OS 메모리 실측
    // ==========================================
    malloc_trim(0);
    int base_mem_float = get_current_ram_usage_kb();

    float *cust_cube_r = NULL; float *cust_cube_i = NULL;
    float *tmp_f_r = NULL; float *tmp_f_i = NULL;
    posix_memalign((void**)&cust_cube_r, alignment, alloc_size_float);
    posix_memalign((void**)&cust_cube_i, alignment, alloc_size_float);
    posix_memalign((void**)&tmp_f_r, alignment, alloc_size_float);
    posix_memalign((void**)&tmp_f_i, alignment, alloc_size_float);

    // 물리 메모리 강제 할당 (터치)
    #pragma omp parallel for
    for (int i = 0; i < total_elements; i++) {
        cust_cube_r[i] = lut_r[i]; cust_cube_i[i] = lut_i[i];
        tmp_f_r[i] = 0; tmp_f_i[i] = 0;
    }

    int peak_mem_float = get_current_ram_usage_kb();
    double actual_mb_float = (peak_mem_float - base_mem_float) / 1024.0;

    // --- R-2 연산 ---
    flush_cache();
    double start_r2_range = get_current_time_ms();
    #pragma omp parallel for collapse(2)
    for (int ant = 0; ant < N_ANTENNAS; ant++) {
        for (int ch = 0; ch < n_chirps; ch++) {
            int offset = ant * (n_samples * n_chirps) + ch * n_samples;
            for (int n = 0; n < n_samples; n++) {
                float win = (n_samples == 2048) ? win_2048[n] : win_1024[n];
                cust_cube_r[offset + n] = lut_r[offset + n] * win;
                cust_cube_i[offset + n] = lut_i[offset + n] * win;
            }
            call_custom_float_radix2(&cust_cube_r[offset], &cust_cube_i[offset], n_samples);
        }
    }
    double time_r2_range = get_current_time_ms() - start_r2_range;

    double start_r2_doppler = get_current_time_ms();
    execute_custom_pipeline(cust_cube_r, cust_cube_i, tmp_f_r, tmp_f_i, n_samples, n_chirps);
    double time_r2_doppler = get_current_time_ms() - start_r2_doppler;

    for (int t = 0; t < num_targets; t++) {
        float max_mag = -1.0f; int best_r = 0;
        int r_center = (int)(((2.0 * S * true_R[t]) / c) * n_samples / Fs);
        int r_start = (r_center - 15 < 0) ? 0 : r_center - 15;
        int r_end = (r_center + 15 >= n_samples) ? n_samples - 1 : r_center + 15;
        for (int r = r_start; r <= r_end; r++) {
            for (int ch = 0; ch < n_chirps; ch++) {
                float sum_mag = 0.0f;
                for (int ant = 0; ant < N_ANTENNAS; ant++) {
                    int idx = ant * (n_samples * n_chirps) + r * n_chirps + ch;
                    sum_mag += tmp_f_r[idx]*tmp_f_r[idx] + tmp_f_i[idx]*tmp_f_i[idx];
                }
                if (sum_mag > max_mag) { max_mag = sum_mag; best_r = r; }
            }
        }
        est_R_r2[t] = (best_r * Fs / (double)n_samples) * c / (2.0 * S);
    }

    // --- R-4 연산 ---
    flush_cache(); 
    double start_r4_range = get_current_time_ms();
    #pragma omp parallel for collapse(2)
    for (int ant = 0; ant < N_ANTENNAS; ant++) {
        for (int ch = 0; ch < n_chirps; ch++) {
            int offset = ant * (n_samples * n_chirps) + ch * n_samples;
            for (int n = 0; n < n_samples; n++) {
                float win = (n_samples == 2048) ? win_2048[n] : win_1024[n];
                cust_cube_r[offset + n] = lut_r[offset + n] * win;
                cust_cube_i[offset + n] = lut_i[offset + n] * win;
            }
            call_custom_float_radix4(&cust_cube_r[offset], &cust_cube_i[offset], n_samples);
        }
    }
    double time_r4_range = get_current_time_ms() - start_r4_range;

    double start_r4_doppler = get_current_time_ms();
    transpose_radar_cube(cust_cube_r, cust_cube_i, tmp_f_r, tmp_f_i, n_samples, n_chirps, NULL);
    #pragma omp parallel for collapse(2)
    for (int ant = 0; ant < N_ANTENNAS; ant++) {
        for (int r = 0; r < n_samples; r++) {
            int offset = ant * (n_samples * n_chirps) + r * n_chirps;
            call_custom_float_radix4(&tmp_f_r[offset], &tmp_f_i[offset], n_chirps);
        }
    }
    double time_r4_doppler = get_current_time_ms() - start_r4_doppler;

    for (int t = 0; t < num_targets; t++) {
        float max_mag = -1.0f; int best_r = 0;
        int r_center = (int)(((2.0 * S * true_R[t]) / c) * n_samples / Fs);
        int r_start = (r_center - 15 < 0) ? 0 : r_center - 15;
        int r_end = (r_center + 15 >= n_samples) ? n_samples - 1 : r_center + 15;
        for (int r = r_start; r <= r_end; r++) {
            for (int ch = 0; ch < n_chirps; ch++) {
                float sum_mag = 0.0f;
                for (int ant = 0; ant < N_ANTENNAS; ant++) {
                    int idx = ant * (n_samples * n_chirps) + r * n_chirps + ch;
                    sum_mag += tmp_f_r[idx]*tmp_f_r[idx] + tmp_f_i[idx]*tmp_f_i[idx];
                }
                if (sum_mag > max_mag) { max_mag = sum_mag; best_r = r; }
            }
        }
        est_R_r4[t] = (best_r * Fs / (double)n_samples) * c / (2.0 * S);
    }

    // ==========================================
    // 3️⃣ [Custom Int16 가속 세션] - OS 메모리 실측
    // ==========================================
    malloc_trim(0);
    int base_mem_i16 = get_current_ram_usage_kb();

    int16_t *i16_cube_r = NULL; int16_t *i16_cube_i = NULL;
    int16_t *tmp_i16_r = NULL; int16_t *tmp_i16_i = NULL;
    posix_memalign((void**)&i16_cube_r, alignment, alloc_size_int16);
    posix_memalign((void**)&i16_cube_i, alignment, alloc_size_int16);
    posix_memalign((void**)&tmp_i16_r, alignment, alloc_size_int16_padded);
    posix_memalign((void**)&tmp_i16_i, alignment, alloc_size_int16_padded);

    // 물리 메모리 강제 할당 (터치)
    #pragma omp parallel for
    for (int i = 0; i < total_elements; i++) {
        i16_cube_r[i] = (int16_t)(lut_r[i] * 2048.0f);
        i16_cube_i[i] = (int16_t)(lut_i[i] * 2048.0f);
        tmp_i16_r[i] = 0; tmp_i16_i[i] = 0;
    }

    int peak_mem_i16 = get_current_ram_usage_kb();
    double actual_mb_i16 = (peak_mem_i16 - base_mem_i16) / 1024.0;

    flush_cache(); 
    double start_i16_range = get_current_time_ms();
    #pragma omp parallel for collapse(2)
    for (int ant = 0; ant < N_ANTENNAS; ant++) {
        for (int c = 0; c < n_chirps; c++) {
            int offset = ant * (n_samples * n_chirps) + c * n_samples;
            if (n_samples == 2048)       { custom_fft_2048_int16(&i16_cube_r[offset], &i16_cube_i[offset]); }
            else if (n_samples == 1024)  { custom_fft_1024_int16(&i16_cube_r[offset], &i16_cube_i[offset]); }
            else if (n_samples == 512)   { custom_fft_512_int16_radix8(&i16_cube_r[offset], &i16_cube_i[offset]); } 
        }
    }
    double time_i16_range = get_current_time_ms() - start_i16_range;

    double start_i16_doppler = get_current_time_ms();
    const int16_t *current_win = NULL;
    if (n_chirps == 512)       { current_win = win_int16_512; }
    else if (n_chirps == 256)  { current_win = win_int16_256; }
    else if (n_chirps == 128)  { current_win = win_int16_128; }

    transpose_radar_cube_int16(i16_cube_r, i16_cube_i, tmp_i16_r, tmp_i16_i, n_samples, n_chirps, current_win); 
    
    #pragma omp parallel for collapse(2)
    for (int ant = 0; ant < N_ANTENNAS; ant++) {
        for (int r = 0; r < n_samples; r++) {
            int offset = ant * (n_samples * n_chirps) + r * n_chirps;
            if (n_chirps == 512)       { custom_fft_512_int16_radix8(&tmp_i16_r[offset], &tmp_i16_i[offset]); }
            else if (n_chirps == 256)  { custom_fft_256_int16(&tmp_i16_r[offset], &tmp_i16_i[offset]); }
            else if (n_chirps == 128)  { custom_fft_128_int16(&tmp_i16_r[offset], &tmp_i16_i[offset]); }
            else if (n_chirps == 64)   { custom_fft_64_int16_radix8(&tmp_i16_r[offset], &tmp_i16_i[offset]); }
        }
    }
    double time_i16_doppler = get_current_time_ms() - start_i16_doppler;

    for (int t = 0; t < num_targets; t++) {
        float max_mag = -1.0f; int best_r = 0;
        int r_center = (int)(((2.0 * S * true_R[t]) / c) * n_samples / Fs);
        int r_start = (r_center - 15 < 0) ? 0 : r_center - 15;
        int r_end = (r_center + 15 >= n_samples) ? n_samples - 1 : r_center + 15;
        for (int r = r_start; r <= r_end; r++) {
            for (int ch = 0; ch < n_chirps; ch++) {
                float sum_mag = 0.0f;
                for (int ant = 0; ant < N_ANTENNAS; ant++) {
                    int idx = ant * (n_samples * n_chirps) + r * n_chirps + ch;
                    sum_mag += (float)tmp_i16_r[idx]*(float)tmp_i16_r[idx] + (float)tmp_i16_i[idx]*(float)tmp_i16_i[idx];
                }
                if (sum_mag > max_mag) { max_mag = sum_mag; best_r = r; }
            }
        }
        est_R_i16[t] = (best_r * Fs / (double)n_samples) * c / (2.0 * S);
    }

    // ==========================================
    // 📺 종합 레이더 스코어보드 전광판 출력
    // ==========================================
    printf("  ---------------------------------------------------\n");
    printf("  [1] FFTW3       (Range / Doppler) : %6.2f ms / %6.2f ms\n", time_fftw_range, time_fftw_doppler);
    printf("  [2] Float R-2   (Range / Doppler) : %6.2f ms / %6.2f ms\n", time_r2_range, time_r2_doppler);
    printf("  [3] Float R-4   (Range / Doppler) : %6.2f ms / %6.2f ms\n", time_r4_range, time_r4_doppler);
    printf("  [4] Int16 Q15   (Range / Doppler) : %6.2f ms / %6.2f ms\n", time_i16_range, time_i16_doppler);
    
    printf("  ---------------------------------------------------\n");
    printf(" 🎯 [알고리즘별 물리 타겟 역추적 정밀도 (Range 추정 대조)]\n");
    for (int t = 0; t < num_targets; t++) {
        printf("  ---------------------------------------------------\n");
        printf("  📍 [Target %d] 실제 수치적 참값 거리: %8.3f m\n", t+1, true_R[t]);
        printf("   - FFTW3     : %8.3f m (오차: %6.3f m)\n", est_R_fftw[t], fabs(est_R_fftw[t] - true_R[t]));
        printf("   - Float R-2 : %8.3f m (오차: %6.3f m)\n", est_R_r2[t],   fabs(est_R_r2[t] - true_R[t]));
        printf("   - Float R-4 : %8.3f m (오차: %6.3f m)\n", est_R_r4[t],   fabs(est_R_r4[t] - true_R[t]));
        printf("   - Int16 Q15 : %8.3f m (오차: %6.3f m)\n", est_R_i16[t],  fabs(est_R_i16[t] - true_R[t]));
    }
    printf("  ---------------------------------------------------\n");
    
    printf(" 🖥️ [운영체제(OS) 물리 메모리 정밀 실측 리포트]\n");
    printf("  - 📈 OS 실측 FFTW3 할당 RAM       : %8.2f MB\n", actual_mb_fftw);
    printf("  - 📈 OS 실측 Custom Float 할당 RAM: %8.2f MB\n", actual_mb_float);
    printf("  - 📈 OS 실측 Custom Int16 할당 RAM: %8.2f MB\n", actual_mb_i16);
    printf("  ※ 알림: 델타(Peak-Base) 기법을 사용하여 각 알고리즘이 순수하게 운영체제에 요청한 VmRSS 증가량을 분리 측정한 결과입니다.\n");
    printf(" ====================================================\n\n");
    fflush(stdout);

    if (lut_r) { free(lut_r); lut_r = NULL; } 
    if (lut_i) { free(lut_i); lut_i = NULL; }
    if (cust_cube_r) { free(cust_cube_r); cust_cube_r = NULL; } 
    if (cust_cube_i) { free(cust_cube_i); cust_cube_i = NULL; }
    if (tmp_f_r) { free(tmp_f_r); tmp_f_r = NULL; } 
    if (tmp_f_i) { free(tmp_f_i); tmp_f_i = NULL; }
    if (i16_cube_r) { free(i16_cube_r); i16_cube_r = NULL; } 
    if (i16_cube_i) { free(i16_cube_i); i16_cube_i = NULL; }
    if (tmp_i16_r) { free(tmp_i16_r); tmp_i16_r = NULL; } 
    if (tmp_i16_i) { free(tmp_i16_i); tmp_i16_i = NULL; }
    if (fftw_in) { fftwf_free(fftw_in); fftw_in = NULL; } 
    if (fftw_out) { fftwf_free(fftw_out); fftw_out = NULL; }
}

void run_benchmark_for_size(int n_samples, int n_chirps) {
    run_benchmark_core(n_samples, n_chirps);
}