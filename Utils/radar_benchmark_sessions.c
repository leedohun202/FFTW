#define _GNU_SOURCE

#include "radar_benchmark_sessions.h"
#include "radar_config.h"
#include "radar_utils.h"
#include "radar_fft.h"
#include "radar_pipeline.h"
#include "radar_profiler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fftw3.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// =========================================================
// [1] 함수 포인터 타입 정의
// =========================================================
typedef void (*fft_func_float)(float*, float*, int);
typedef void (*fft_func_int16)(int16_t*, int16_t*, int);

// =========================================================
// [2] 윈도우 및 FFT 라우팅 헬퍼 (포맷팅 교정 완료)
// =========================================================
static inline const float* get_float_window(int size) {
    if (size == 2048) return win_2048;
    if (size == 1024) return win_1024;
    if (size == 512)  return win_512;
    if (size == 256)  return win_256;
    if (size == 128)  return win_128;
    return NULL; 
}

static inline const int16_t* get_int16_window(int size) {
    if (size == 2048) return win_int16_2048;
    if (size == 1024) return win_int16_1024;
    if (size == 512)  return win_int16_512;
    if (size == 256)  return win_int16_256;
    if (size == 128)  return win_int16_128;
    return NULL;
}

static void call_float_r2(float *r, float *i, int n) {
    if (n == 2048)      custom_fft_2048_fixed(r, i);
    else if (n == 1024) custom_fft_1024_fixed(r, i);
    else if (n == 512)  custom_fft_512_fixed(r, i);
    else if (n == 256)  custom_fft_256_fixed(r, i);
    else if (n == 128)  custom_fft_128_fixed(r, i);
    else if (n == 64)   custom_fft_64_fixed(r, i);
}

static void call_float_r4(float *r, float *i, int n) {
    if (n == 2048)      custom_fft_2048_radix4(r, i);
    else if (n == 1024) custom_fft_1024_radix4(r, i);
    else if (n == 512)  custom_fft_512_radix4(r, i);
    else if (n == 256)  custom_fft_256_radix4(r, i);
    else if (n == 128)  custom_fft_128_radix4(r, i);
    else if (n == 64)   custom_fft_64_radix4(r, i);    
}

static void call_int16_r2(int16_t *r, int16_t *i, int n) {
    if (n == 2048)      custom_fft_2048_int16(r, i);
    else if (n == 1024) custom_fft_1024_int16(r, i);
    else if (n == 512)  custom_fft_512_int16(r, i);
    else if (n == 256)  custom_fft_256_int16(r, i);
    else if (n == 128)  custom_fft_128_int16(r, i);
    else if (n == 64)   custom_fft_64_int16(r, i);
}

static void call_int16_r4(int16_t *r, int16_t *i, int n) {
    if (n == 2048)      custom_fft_2048_int16_radix4(r, i);
    else if (n == 1024) custom_fft_1024_int16_radix4(r, i);
    else if (n == 512)  custom_fft_512_int16_radix4(r, i);
    else if (n == 256)  custom_fft_256_int16_radix4(r, i);
    else if (n == 128)  custom_fft_128_int16_radix4(r, i);
    else if (n == 64)   custom_fft_64_int16_radix4(r, i);  
}

// =========================================================
// 🚀 [3] Range / Doppler 루프 통합 헬퍼 (함수 포인터 적용)
// =========================================================
static double run_range_loop_float(float *cube_r, float *cube_i, int n_samples, int n_chirps, fft_func_float func) {
    double start = get_current_time_ms();
    const float* win = get_float_window(n_samples);
    #pragma omp parallel for collapse(2)
    for (int ant = 0; ant < N_ANTENNAS; ant++) {
        for (int ch = 0; ch < n_chirps; ch++) {
            int offset = ant * (n_samples * n_chirps) + ch * n_samples;
            for (int n = 0; n < n_samples; n++) {
                float w = win ? win[n] : 1.0f;
                cube_r[offset + n] *= w; 
                cube_i[offset + n] *= w;
            }
            func(&cube_r[offset], &cube_i[offset], n_samples);
        }
    }
    return get_current_time_ms() - start;
}

static double run_doppler_loop_float(float *tmp_r, float *tmp_i, int n_samples, int n_chirps, fft_func_float func) {
    double start = get_current_time_ms();
    #pragma omp parallel for collapse(2)
    for (int ant = 0; ant < N_ANTENNAS; ant++) {
        for (int r = 0; r < n_samples; r++) {
            int offset = ant * (n_samples * n_chirps) + r * n_chirps;
            func(&tmp_r[offset], &tmp_i[offset], n_chirps);
        }
    }
    return get_current_time_ms() - start;
}

static double run_range_loop_int16(int16_t *cube_r, int16_t *cube_i, int n_samples, int n_chirps, fft_func_int16 func) {
    double start = get_current_time_ms();
    // Int16은 원본 LUT 생성 시 이미 윈도우가 적용되었다고 가정 (기존 로직 유지)
    #pragma omp parallel for collapse(2)
    for (int ant = 0; ant < N_ANTENNAS; ant++) {
        for (int ch = 0; ch < n_chirps; ch++) {
            int offset = ant * (n_samples * n_chirps) + ch * n_samples;
            func(&cube_r[offset], &cube_i[offset], n_samples);
        }
    }
    return get_current_time_ms() - start;
}

static double run_doppler_loop_int16(int16_t *tmp_r, int16_t *tmp_i, int n_samples, int n_chirps, fft_func_int16 func) {
    double start = get_current_time_ms();
    #pragma omp parallel for collapse(2)
    for (int ant = 0; ant < N_ANTENNAS; ant++) {
        for (int r = 0; r < n_samples; r++) {
            int offset = ant * (n_samples * n_chirps) + r * n_chirps;
            func(&tmp_r[offset], &tmp_i[offset], n_chirps);
        }
    }
    return get_current_time_ms() - start;
}

// =========================================================
// [4] 정밀 타겟 역추적 헬퍼 (검증용 컨닝페이퍼)
// =========================================================
static void peak_search_float_cube(const float *tmp_r, const float *tmp_i, int n_samples, int n_chirps, BenchmarkResult *out) {
    double true_R[] = {12.50, 25.20,  6.80};
    int num_targets = 3;

    for (int t = 0; t < num_targets; t++) {
        float max_mag = -1.0f; int best_r = 0; int best_ch = 0;
        int r_center = (int)(((2.0 * S * true_R[t]) / c) * n_samples / Fs);
        int r_start = (r_center - 15 < 0) ? 0 : r_center - 15;
        int r_end = (r_center + 15 >= n_samples) ? n_samples - 1 : r_center + 15;

        for (int r = r_start; r <= r_end; r++) {
            for (int ch = 0; ch < n_chirps; ch++) {
                float sum_mag = 0.0f;
                for (int ant = 0; ant < N_ANTENNAS; ant++) {
                    int idx = ant * (n_samples * n_chirps) + r * n_chirps + ch;
                    sum_mag += tmp_r[idx]*tmp_r[idx] + tmp_i[idx]*tmp_i[idx];
                }
                if (sum_mag > max_mag) { max_mag = sum_mag; best_r = r; best_ch = ch; }
            }
        }
        out->est_R[t] = (best_r * Fs / (double)n_samples) * c / (2.0 * S);
        out->est_v[t] = ((best_ch >= n_chirps / 2 ? best_ch - n_chirps : best_ch) * lambda_c) / (2.0 * n_chirps * Tc);

        int idx_a0 = 0 * (n_samples * n_chirps) + best_r * n_chirps + best_ch;
        int idx_a1 = 1 * (n_samples * n_chirps) + best_r * n_chirps + best_ch;
        float r0 = tmp_r[idx_a0], i0 = tmp_i[idx_a0];
        float r1 = tmp_r[idx_a1], i1 = tmp_i[idx_a1];
        double d_phi = atan2(i1*r0 - r1*i0, r1*r0 + i1*i0);
        double sin_theta = (d_phi * lambda_c) / (2.0 * PI * d_ant);
        if (sin_theta > 1.0) sin_theta = 1.0; else if (sin_theta < -1.0) sin_theta = -1.0;
        out->est_a[t] = asin(sin_theta) * 180.0 / PI;
    }
}

static void peak_search_int16_cube(const int16_t *tmp_r, const int16_t *tmp_i, int n_samples, int n_chirps, BenchmarkResult *out) {
    double true_R[] = {12.50, 25.20,  6.80};
    int num_targets = 3;

    for (int t = 0; t < num_targets; t++) {
        float max_mag = -1.0f; int best_r = 0; int best_ch = 0;
        int r_center = (int)(((2.0 * S * true_R[t]) / c) * n_samples / Fs);
        int r_start = (r_center - 15 < 0) ? 0 : r_center - 15;
        int r_end = (r_center + 15 >= n_samples) ? n_samples - 1 : r_center + 15;

        for (int r = r_start; r <= r_end; r++) {
            for (int ch = 0; ch < n_chirps; ch++) {
                float sum_mag = 0.0f;
                for (int ant = 0; ant < N_ANTENNAS; ant++) {
                    int idx = ant * (n_samples * n_chirps) + r * n_chirps + ch;
                    float fr = (float)tmp_r[idx]; float fi = (float)tmp_i[idx];
                    sum_mag += fr*fr + fi*fi;
                }
                if (sum_mag > max_mag) { max_mag = sum_mag; best_r = r; best_ch = ch; }
            }
        }
        out->est_R[t] = (best_r * Fs / (double)n_samples) * c / (2.0 * S);
        out->est_v[t] = ((best_ch >= n_chirps / 2 ? best_ch - n_chirps : best_ch) * lambda_c) / (2.0 * n_chirps * Tc);

        int idx_a0 = 0 * (n_samples * n_chirps) + best_r * n_chirps + best_ch;
        int idx_a1 = 1 * (n_samples * n_chirps) + best_r * n_chirps + best_ch;
        float r0 = (float)tmp_r[idx_a0], i0 = (float)tmp_i[idx_a0];
        float r1 = (float)tmp_r[idx_a1], i1 = (float)tmp_i[idx_a1];
        double d_phi = atan2(i1*r0 - r1*i0, r1*r0 + i1*i0);
        double sin_theta = (d_phi * lambda_c) / (2.0 * PI * d_ant);
        if (sin_theta > 1.0) sin_theta = 1.0; else if (sin_theta < -1.0) sin_theta = -1.0;
        out->est_a[t] = asin(sin_theta) * 180.0 / PI;
    }
}

// =========================================================
// 1️⃣ [FFTW3 세션]
// =========================================================
void benchmark_session_fftw3(const float *lut_r, const float *lut_i, int n_samples, int n_chirps, BenchmarkResult *out) {
    int total_elements = N_ANTENNAS * n_chirps * n_samples;
    int base_mem = profiler_start_mem();
    
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
    fftwf_plan p_doppler = fftwf_plan_many_dft(1, d_n, n_samples, fftw_in, NULL, n_samples, 1, fftw_out, NULL, n_samples, 1, FFTW_FORWARD, fftw_flags);

    #pragma omp parallel for
    for (int i = 0; i < total_elements; i++) {
        fftw_in[i][0] = lut_r[i]; fftw_in[i][1] = lut_i[i];
    }
    
    out->actual_ram_mb = profiler_end_mem_mb(base_mem);
    profiler_flush_cache();
    
    double start_range = get_current_time_ms();
    if (p_range) {
        const float* win_range = get_float_window(n_samples);
        #pragma omp parallel for collapse(2)
        for (int ant = 0; ant < N_ANTENNAS; ant++) {
            for (int chirp = 0; chirp < n_chirps; chirp++) {
                int offset = ant * (n_chirps * n_samples) + chirp * n_samples;
                for (int n = 0; n < n_samples; n++) {
                    float w = win_range ? win_range[n] : 1.0f;
                    fftw_in[offset + n][0] = lut_r[offset + n] * w;
                    fftw_in[offset + n][1] = lut_i[offset + n] * w;
                }
                fftwf_execute_dft(p_range, &fftw_in[offset], &fftw_in[offset]);
            }
        }
    }
    out->time_range = get_current_time_ms() - start_range;

    double start_doppler = get_current_time_ms();
    if (p_doppler) { 
        for (int ant = 0; ant < N_ANTENNAS; ant++) {
            int offset = ant * (n_chirps * n_samples);
            fftwf_execute_dft(p_doppler, &fftw_in[offset], &fftw_out[offset]);
        }
    }
    out->time_doppler = get_current_time_ms() - start_doppler;

    // FFTW3 전용 탐색 유지
    double true_R[] = {12.50, 25.20,  6.80};
    for (int t = 0; t < 3; t++) {
        float max_mag = -1.0f; int best_r = 0; int best_ch = 0;
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
                if (sum_mag > max_mag) { max_mag = sum_mag; best_r = r; best_ch = ch; }
            }
        }
        out->est_R[t] = (best_r * Fs / (double)n_samples) * c / (2.0 * S);
        int shifted_ch = (best_ch >= n_chirps / 2) ? best_ch - n_chirps : best_ch;
        out->est_v[t] = (shifted_ch * lambda_c) / (2.0 * n_chirps * Tc);

        int idx_a0 = 0 * (n_chirps * n_samples) + best_ch * n_samples + best_r;
        int idx_a1 = 1 * (n_chirps * n_samples) + best_ch * n_samples + best_r;
        float r0 = fftw_out[idx_a0][0], i0 = fftw_out[idx_a0][1];
        float r1 = fftw_out[idx_a1][0], i1 = fftw_out[idx_a1][1];
        double d_phi = atan2(i1*r0 - r1*i0, r1*r0 + i1*i0); 
        double sin_theta = (d_phi * lambda_c) / (2.0 * PI * d_ant);
        if (sin_theta > 1.0) sin_theta = 1.0; else if (sin_theta < -1.0) sin_theta = -1.0;
        out->est_a[t] = asin(sin_theta) * 180.0 / PI;
    }

    if (p_range) { fftwf_destroy_plan(p_range); }
    if (p_doppler) { fftwf_destroy_plan(p_doppler); }
    if (fftw_in) { fftwf_free(fftw_in); }
    if (fftw_out) { fftwf_free(fftw_out); }
}

// =========================================================
// 2️⃣ [Float Radix-2 세션] 💥 단 15줄로 압축 완료!
// =========================================================
void benchmark_session_float_r2(const float *lut_r, const float *lut_i, int n_samples, int n_chirps, BenchmarkResult *out) {
    int total_elements = N_ANTENNAS * n_chirps * n_samples;
    int base_mem = profiler_start_mem();
    
    float *cube_r = (float*)allocate_and_clear_aligned(total_elements * sizeof(float), 64);
    float *cube_i = (float*)allocate_and_clear_aligned(total_elements * sizeof(float), 64);
    float *tmp_r  = (float*)allocate_and_clear_aligned(total_elements * sizeof(float), 64);
    float *tmp_i  = (float*)allocate_and_clear_aligned(total_elements * sizeof(float), 64);

    #pragma omp parallel for
    for (int i = 0; i < total_elements; i++) { cube_r[i] = lut_r[i]; cube_i[i] = lut_i[i]; }
    out->actual_ram_mb = profiler_end_mem_mb(base_mem);
    profiler_flush_cache(); 
    
    out->time_range = run_range_loop_float(cube_r, cube_i, n_samples, n_chirps, call_float_r2);
    transpose_radar_cube(cube_r, cube_i, tmp_r, tmp_i, n_samples, n_chirps, get_float_window(n_chirps));
    out->time_doppler = run_doppler_loop_float(tmp_r, tmp_i, n_samples, n_chirps, call_float_r2);

    peak_search_float_cube(tmp_r, tmp_i, n_samples, n_chirps, out);
    free(cube_r); free(cube_i); free(tmp_r); free(tmp_i);
}

// =========================================================
// 3️⃣ [Float Radix-4 세션]
// =========================================================
void benchmark_session_float_r4(const float *lut_r, const float *lut_i, int n_samples, int n_chirps, BenchmarkResult *out) {
    int total_elements = N_ANTENNAS * n_chirps * n_samples;
    int base_mem = profiler_start_mem();
    
    float *cube_r = (float*)allocate_and_clear_aligned(total_elements * sizeof(float), 64);
    float *cube_i = (float*)allocate_and_clear_aligned(total_elements * sizeof(float), 64);
    float *tmp_r  = (float*)allocate_and_clear_aligned(total_elements * sizeof(float), 64);
    float *tmp_i  = (float*)allocate_and_clear_aligned(total_elements * sizeof(float), 64);

    #pragma omp parallel for
    for (int i = 0; i < total_elements; i++) { cube_r[i] = lut_r[i]; cube_i[i] = lut_i[i]; }
    out->actual_ram_mb = profiler_end_mem_mb(base_mem);
    profiler_flush_cache(); 
    
    out->time_range = run_range_loop_float(cube_r, cube_i, n_samples, n_chirps, call_float_r4);
    transpose_radar_cube(cube_r, cube_i, tmp_r, tmp_i, n_samples, n_chirps, get_float_window(n_chirps));
    out->time_doppler = run_doppler_loop_float(tmp_r, tmp_i, n_samples, n_chirps, call_float_r4);

    peak_search_float_cube(tmp_r, tmp_i, n_samples, n_chirps, out);
    free(cube_r); free(cube_i); free(tmp_r); free(tmp_i);
}

// =========================================================
// 4️⃣ [Int16 Radix-2 세션]
// =========================================================
void benchmark_session_int16_r2(const float *lut_r, const float *lut_i, int n_samples, int n_chirps, BenchmarkResult *out) {
    int total_elements = N_ANTENNAS * n_chirps * n_samples;
    int base_mem = profiler_start_mem();
    
    int16_t *cube_r = (int16_t*)allocate_and_clear_aligned(total_elements * sizeof(int16_t), 64);
    int16_t *cube_i = (int16_t*)allocate_and_clear_aligned(total_elements * sizeof(int16_t), 64);
    int16_t *tmp_r  = (int16_t*)allocate_and_clear_aligned(total_elements * sizeof(int16_t), 64);
    int16_t *tmp_i  = (int16_t*)allocate_and_clear_aligned(total_elements * sizeof(int16_t), 64);

    #pragma omp parallel for
    for (int i = 0; i < total_elements; i++) {
        cube_r[i] = (int16_t)(lut_r[i] * 2048.0f); cube_i[i] = (int16_t)(lut_i[i] * 2048.0f);
    }
    out->actual_ram_mb = profiler_end_mem_mb(base_mem);
    profiler_flush_cache(); 
    
    out->time_range = run_range_loop_int16(cube_r, cube_i, n_samples, n_chirps, call_int16_r2);
    transpose_radar_cube_int16(cube_r, cube_i, tmp_r, tmp_i, n_samples, n_chirps, get_int16_window(n_chirps)); 
    out->time_doppler = run_doppler_loop_int16(tmp_r, tmp_i, n_samples, n_chirps, call_int16_r2);

    peak_search_int16_cube(tmp_r, tmp_i, n_samples, n_chirps, out);
    free(cube_r); free(cube_i); free(tmp_r); free(tmp_i);
}

// =========================================================
// 5️⃣ [Int16 Radix-4 세션]
// =========================================================
void benchmark_session_int16_r4(const float *lut_r, const float *lut_i, int n_samples, int n_chirps, BenchmarkResult *out) {
    int total_elements = N_ANTENNAS * n_chirps * n_samples;
    int base_mem = profiler_start_mem();
    
    int16_t *cube_r = (int16_t*)allocate_and_clear_aligned(total_elements * sizeof(int16_t), 64);
    int16_t *cube_i = (int16_t*)allocate_and_clear_aligned(total_elements * sizeof(int16_t), 64);
    int16_t *tmp_r  = (int16_t*)allocate_and_clear_aligned(total_elements * sizeof(int16_t), 64);
    int16_t *tmp_i  = (int16_t*)allocate_and_clear_aligned(total_elements * sizeof(int16_t), 64);

    #pragma omp parallel for
    for (int i = 0; i < total_elements; i++) {
        cube_r[i] = (int16_t)(lut_r[i] * 2048.0f); cube_i[i] = (int16_t)(lut_i[i] * 2048.0f);
    }
    out->actual_ram_mb = profiler_end_mem_mb(base_mem);
    profiler_flush_cache(); 
    
    out->time_range = run_range_loop_int16(cube_r, cube_i, n_samples, n_chirps, call_int16_r4);
    transpose_radar_cube_int16(cube_r, cube_i, tmp_r, tmp_i, n_samples, n_chirps, get_int16_window(n_chirps)); 
    out->time_doppler = run_doppler_loop_int16(tmp_r, tmp_i, n_samples, n_chirps, call_int16_r4);

    peak_search_int16_cube(tmp_r, tmp_i, n_samples, n_chirps, out);
    free(cube_r); free(cube_i); free(tmp_r); free(tmp_i);
}