#define _GNU_SOURCE

#include "radar_benchmark_sessions.h"
#include "radar_config.h"
#include "radar_utils.h"
#include "radar_fft.h"
#include "radar_pipeline.h"
#include "radar_profiler.h"
#include "radar_cfar.h"
#include "radar_eval.h" 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fftw3.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

typedef void (*fft_func_float)(float*, float*, int);
typedef void (*fft_func_int16)(int16_t*, int16_t*, int);

// 글로벌 정답 데이터 세팅 (실제 환경에서는 외부에서 불러옴)
static const double mock_true_R[] = {12.50, 25.20,  6.80};

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
    const int16_t* win = get_int16_window(n_samples);
    #pragma omp parallel for collapse(2)
    for (int ant = 0; ant < N_ANTENNAS; ant++) {
        for (int ch = 0; ch < n_chirps; ch++) {
            int offset = ant * (n_samples * n_chirps) + ch * n_samples;
            for (int n = 0; n < n_samples; n++) {
                int32_t w = win ? win[n] : 32767;
                cube_r[offset + n] = (int16_t)(((int32_t)cube_r[offset + n] * w) >> 15);
                cube_i[offset + n] = (int16_t)(((int32_t)cube_i[offset + n] * w) >> 15);
            }
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

static void cfar_search_float_cube(const float *tmp_r, const float *tmp_i, int n_samples, int n_chirps, BenchmarkResult *out) {
    CFAR_Config cfg = { .guard_r = 4, .guard_d = 4, .train_r = 4, .train_d = 4, .alpha = 7.5f, .min_threshold = 1e-6f }; 
    CFAR_Target *targets = (CFAR_Target*)malloc(2000 * sizeof(CFAR_Target));
    float *pwr_map = (float*)malloc(n_samples * n_chirps * sizeof(float));

    int num_det = run_2d_ca_cfar_float(tmp_r, tmp_i, pwr_map, n_samples, n_chirps, cfg, targets, 2000);
    extract_autonomous_objects(targets, num_det, tmp_r, tmp_i, n_samples, n_chirps, out, mock_true_R);
    
    free(pwr_map); free(targets);
}

static void cfar_search_int16_cube(const int16_t *tmp_r, const int16_t *tmp_i, int n_samples, int n_chirps, BenchmarkResult *out) {
    CFAR_Config cfg = { .guard_r = 4, .guard_d = 4, .train_r = 4, .train_d = 4, .alpha = 4.5f, .min_threshold = 100.0f };
    CFAR_Target *targets = (CFAR_Target*)malloc(2000 * sizeof(CFAR_Target));
    float *pwr_map = (float*)malloc(n_samples * n_chirps * sizeof(float));

    int num_det = run_2d_ca_cfar_int16(tmp_r, tmp_i, pwr_map, n_samples, n_chirps, cfg, targets, 2000);

    int total = N_ANTENNAS * n_chirps * n_samples;
    float *f_tmp_r = (float*)malloc(total * sizeof(float));
    float *f_tmp_i = (float*)malloc(total * sizeof(float));
    #pragma omp parallel for
    for(int i = 0; i < total; i++){
        f_tmp_r[i] = (float)tmp_r[i]; f_tmp_i[i] = (float)tmp_i[i];
    }
    
    extract_autonomous_objects(targets, num_det, f_tmp_r, f_tmp_i, n_samples, n_chirps, out, mock_true_R);
    
    free(f_tmp_r); free(f_tmp_i); free(pwr_map); free(targets);
}

static void cfar_search_fftw_cube(const fftwf_complex *tmp, int n_samples, int n_chirps, BenchmarkResult *out) {
    CFAR_Config cfg = { .guard_r = 4, .guard_d = 4, .train_r = 4, .train_d = 4, .alpha = 7.5f, .min_threshold = 1e-6f }; 
    CFAR_Target *targets = (CFAR_Target*)malloc(2000 * sizeof(CFAR_Target));
    float *pwr_map = (float*)malloc(n_samples * n_chirps * sizeof(float));

    int num_det = run_2d_ca_cfar_fftw(tmp, pwr_map, n_samples, n_chirps, cfg, targets, 2000);

    int total_elements = N_ANTENNAS * n_chirps * n_samples;
    float *f_tmp_r = (float*)malloc(total_elements * sizeof(float));
    float *f_tmp_i = (float*)malloc(total_elements * sizeof(float));
    
    #pragma omp parallel for
    for (int i = 0; i < total_elements; i++) {
        f_tmp_r[i] = tmp[i][0];
        f_tmp_i[i] = tmp[i][1];
    }
    
    extract_autonomous_objects(targets, num_det, f_tmp_r, f_tmp_i, n_samples, n_chirps, out, mock_true_R);

    free(f_tmp_r); free(f_tmp_i); free(pwr_map); free(targets);
}

// ===================================================================================
// 💥 FFTW3 벤치마크
// ===================================================================================
void benchmark_session_fftw3(const float *lut_r, const float *lut_i, int n_samples, int n_chirps, BenchmarkResult *out, unsigned int fftw_flags) {
    int total_elements = N_ANTENNAS * n_chirps * n_samples;
    int base_mem = profiler_start_mem();
    
    // 1. 메모리 할당 (64MB 유지)
    fftwf_complex *cube = (fftwf_complex *)fftwf_malloc(total_elements * sizeof(fftwf_complex));
    fftwf_complex *tmp  = (fftwf_complex *)fftwf_malloc(total_elements * sizeof(fftwf_complex));

    // 💥 [핵심 수정] MEASURE의 정상적인 벤치마크를 위해 진짜 '노이즈 데이터'를 주입!
    // (Subnormal Number 예외로 인한 벤치마크 왜곡을 방지합니다)
    #pragma omp parallel for
    for (int i = 0; i < total_elements; i++) {
        cube[i][0] = lut_r[i]; 
        cube[i][1] = lut_i[i];
        // tmp 배열에도 노이즈를 넣어 Doppler 벤치마크 시 CPU 병목을 막아줍니다.
        tmp[i][0]  = lut_r[i]; 
        tmp[i][1]  = lut_i[i];
    }

#ifdef _OPENMP
    fftwf_init_threads(); // 스레드 엔진 초기화
    fftwf_plan_with_nthreads(omp_get_max_threads()); // 가용 코어(4개) 전면 할당!
#endif

    // 2. 일괄(Batch) 플랜 생성 
    // (이제 MEASURE 모드가 실제 노이즈 데이터를 기반으로 가장 빠르고 정상적인 NEON 커널을 채택합니다!)
    int n_r[] = {n_samples};
    fftwf_plan p_range = fftwf_plan_many_dft(1, n_r, N_ANTENNAS * n_chirps,
                                             cube, NULL, 1, n_samples,
                                             cube, NULL, 1, n_samples,
                                             FFTW_FORWARD, fftw_flags);

    int n_d[] = {n_chirps};
    fftwf_plan p_doppler = fftwf_plan_many_dft(1, n_d, N_ANTENNAS * n_samples,
                                               tmp, NULL, 1, n_chirps,
                                               tmp, NULL, 1, n_chirps,
                                               FFTW_FORWARD, fftw_flags);

    out->actual_ram_mb = profiler_end_mem_mb(base_mem);
    profiler_flush_cache();
    
    // ========================================================
    // [1] Range FFT 데이터 세팅 (MEASURE가 파괴한 배열을 다시 실제 데이터와 윈도우로 복구)
    // ========================================================
    double start_range = get_current_time_ms();
    const float* win_range = get_float_window(n_samples);
    
    #pragma omp parallel for collapse(2)
    for (int ant = 0; ant < N_ANTENNAS; ant++) {
        for (int ch = 0; ch < n_chirps; ch++) {
            int offset = ant * (n_chirps * n_samples) + ch * n_samples;
            for (int n = 0; n < n_samples; n++) {
                float w = win_range ? win_range[n] : 1.0f;
                // 다시 덮어씌우므로 안전합니다.
                cube[offset + n][0] = lut_r[offset + n] * w;
                cube[offset + n][1] = lut_i[offset + n] * w;
            }
        }
    }
    
    // 💥 가장 최적화된 진짜 플랜으로 일괄 연산!
    fftwf_execute(p_range);
    
    // (이후 코드는 동일하게 유지...)
    out->time_range = get_current_time_ms() - start_range;

    double start_doppler = get_current_time_ms();
    const float* win_doppler = get_float_window(n_chirps);
    int TILE = 16;
    
    #pragma omp parallel for collapse(3)
    for (int ant = 0; ant < N_ANTENNAS; ant++) {
        for (int r_blk = 0; r_blk < n_samples; r_blk += TILE) {
            for (int c_blk = 0; c_blk < n_chirps; c_blk += TILE) {
                for (int c = c_blk; c < c_blk + TILE && c < n_chirps; c++) {
                    float w = win_doppler ? win_doppler[c] : 1.0f;
                    for (int r = r_blk; r < r_blk + TILE && r < n_samples; r++) {
                        int src_idx = ant * (n_chirps * n_samples) + c * n_samples + r;
                        int dst_idx = ant * (n_samples * n_chirps) + r * n_chirps + c;
                        
                        tmp[dst_idx][0] = cube[src_idx][0] * w;
                        tmp[dst_idx][1] = cube[src_idx][1] * w;
                    }
                }
            }
        }
    }

    fftwf_execute(p_doppler);
    out->time_doppler = get_current_time_ms() - start_doppler;

    cfar_search_fftw_cube(tmp, n_samples, n_chirps, out);

    fftwf_destroy_plan(p_range);
    fftwf_destroy_plan(p_doppler);
    fftwf_free(cube); fftwf_free(tmp);
}

// -----------------------------------------------------------------------------------
// 나머지 세션 (Float R2, Float R4, Int16 R2, Int16 R4)
// -----------------------------------------------------------------------------------
void benchmark_session_float_r2(const float *lut_r, const float *lut_i, int n_samples, int n_chirps, BenchmarkResult *out) {
    int total_elements = N_ANTENNAS * n_chirps * n_samples;
    int base_mem = profiler_start_mem();
    float *cube_r = (float*)allocate_and_clear_aligned(total_elements * sizeof(float), 64);
    float *cube_i = (float*)allocate_and_clear_aligned(total_elements * sizeof(float), 64);
    float *tmp_r  = (float*)allocate_and_clear_aligned(total_elements * sizeof(float), 64);
    float *tmp_i  = (float*)allocate_and_clear_aligned(total_elements * sizeof(float), 64);
    #pragma omp parallel for
    for (int i = 0; i < total_elements; i++) { cube_r[i] = lut_r[i]; cube_i[i] = lut_i[i]; }
    out->actual_ram_mb = profiler_end_mem_mb(base_mem); profiler_flush_cache(); 
    out->time_range = run_range_loop_float(cube_r, cube_i, n_samples, n_chirps, call_float_r2);
    transpose_radar_cube(cube_r, cube_i, tmp_r, tmp_i, n_samples, n_chirps, get_float_window(n_chirps));
    out->time_doppler = run_doppler_loop_float(tmp_r, tmp_i, n_samples, n_chirps, call_float_r2);
    cfar_search_float_cube(tmp_r, tmp_i, n_samples, n_chirps, out);
    free(cube_r); free(cube_i); free(tmp_r); free(tmp_i);
}

void benchmark_session_float_r4(const float *lut_r, const float *lut_i, int n_samples, int n_chirps, BenchmarkResult *out) {
    int total_elements = N_ANTENNAS * n_chirps * n_samples;
    int base_mem = profiler_start_mem();
    float *cube_r = (float*)allocate_and_clear_aligned(total_elements * sizeof(float), 64);
    float *cube_i = (float*)allocate_and_clear_aligned(total_elements * sizeof(float), 64);
    float *tmp_r  = (float*)allocate_and_clear_aligned(total_elements * sizeof(float), 64);
    float *tmp_i  = (float*)allocate_and_clear_aligned(total_elements * sizeof(float), 64);
    #pragma omp parallel for
    for (int i = 0; i < total_elements; i++) { cube_r[i] = lut_r[i]; cube_i[i] = lut_i[i]; }
    out->actual_ram_mb = profiler_end_mem_mb(base_mem); profiler_flush_cache(); 
    out->time_range = run_range_loop_float(cube_r, cube_i, n_samples, n_chirps, call_float_r4);
    transpose_radar_cube(cube_r, cube_i, tmp_r, tmp_i, n_samples, n_chirps, get_float_window(n_chirps));
    out->time_doppler = run_doppler_loop_float(tmp_r, tmp_i, n_samples, n_chirps, call_float_r4);
    cfar_search_float_cube(tmp_r, tmp_i, n_samples, n_chirps, out);
    free(cube_r); free(cube_i); free(tmp_r); free(tmp_i);
}

void benchmark_session_int16_r2(const float *lut_r, const float *lut_i, int n_samples, int n_chirps, BenchmarkResult *out) {
    int total_elements = N_ANTENNAS * n_chirps * n_samples;
    int base_mem = profiler_start_mem();
    int16_t *cube_r = (int16_t*)allocate_and_clear_aligned(total_elements * sizeof(int16_t), 64);
    int16_t *cube_i = (int16_t*)allocate_and_clear_aligned(total_elements * sizeof(int16_t), 64);
    int16_t *tmp_r  = (int16_t*)allocate_and_clear_aligned(total_elements * sizeof(int16_t), 64);
    int16_t *tmp_i  = (int16_t*)allocate_and_clear_aligned(total_elements * sizeof(int16_t), 64);
    #pragma omp parallel for
    for (int i = 0; i < total_elements; i++) { cube_r[i] = (int16_t)(lut_r[i] * 2048.0f); cube_i[i] = (int16_t)(lut_i[i] * 2048.0f); }
    out->actual_ram_mb = profiler_end_mem_mb(base_mem); profiler_flush_cache(); 
    out->time_range = run_range_loop_int16(cube_r, cube_i, n_samples, n_chirps, call_int16_r2);
    transpose_radar_cube_int16(cube_r, cube_i, tmp_r, tmp_i, n_samples, n_chirps, get_int16_window(n_chirps)); 
    out->time_doppler = run_doppler_loop_int16(tmp_r, tmp_i, n_samples, n_chirps, call_int16_r2);
    cfar_search_int16_cube(tmp_r, tmp_i, n_samples, n_chirps, out);
    free(cube_r); free(cube_i); free(tmp_r); free(tmp_i);
}

void benchmark_session_int16_r4(const float *lut_r, const float *lut_i, int n_samples, int n_chirps, BenchmarkResult *out) {
    int total_elements = N_ANTENNAS * n_chirps * n_samples;
    int base_mem = profiler_start_mem();
    int16_t *cube_r = (int16_t*)allocate_and_clear_aligned(total_elements * sizeof(int16_t), 64);
    int16_t *cube_i = (int16_t*)allocate_and_clear_aligned(total_elements * sizeof(int16_t), 64);
    int16_t *tmp_r  = (int16_t*)allocate_and_clear_aligned(total_elements * sizeof(int16_t), 64);
    int16_t *tmp_i  = (int16_t*)allocate_and_clear_aligned(total_elements * sizeof(int16_t), 64);
    #pragma omp parallel for
    for (int i = 0; i < total_elements; i++) { cube_r[i] = (int16_t)(lut_r[i] * 2048.0f); cube_i[i] = (int16_t)(lut_i[i] * 2048.0f); }
    out->actual_ram_mb = profiler_end_mem_mb(base_mem); profiler_flush_cache(); 
    out->time_range = run_range_loop_int16(cube_r, cube_i, n_samples, n_chirps, call_int16_r4);
    transpose_radar_cube_int16(cube_r, cube_i, tmp_r, tmp_i, n_samples, n_chirps, get_int16_window(n_chirps)); 
    out->time_doppler = run_doppler_loop_int16(tmp_r, tmp_i, n_samples, n_chirps, call_int16_r4);
    cfar_search_int16_cube(tmp_r, tmp_i, n_samples, n_chirps, out);
    free(cube_r); free(cube_i); free(tmp_r); free(tmp_i);
}