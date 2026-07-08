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

void run_benchmark_for_size(int n_samples) {
    int total_elements = N_ANTENNAS * N_CHIRPS * n_samples;
    size_t alloc_size = total_elements * sizeof(float);
    size_t alignment = 64; // 🔥 ARM Cortex-A76 캐시 라인 크기 (64-byte)

    printf("\n====================================================\n");
#if defined(FFTW_MODE_ESTIMATE)
    printf(" 🛰️ [3D 궁극의 가속 엔진] Range = %d | ⚡ ESTIMATE 모드\n", n_samples);
#elif defined(FFTW_MODE_MEASURE)
    printf(" 🛰️ [3D 궁극의 가속 엔진] Range = %d | ⚖️ MEASURE 모드\n", n_samples);
#elif defined(FFTW_MODE_PATIENT)
    printf(" 🛰️ [3D 궁극의 가속 엔진] Range = %d | 🐢 PATIENT 모드\n", n_samples);
#endif
    printf("====================================================\n");

    int ram_start_kb = get_current_ram_usage_kb();

    // 🎯 [캐시 얼라인먼트 적용 할당]
    // malloc 대신 posix_memalign을 사용하여 시작 주소를 64바이트 경계에 완벽히 정렬합니다.
    float *cust_cube_r = NULL;
    float *cust_cube_i = NULL;
    float *pipeline_tmp_r = NULL;
    float *pipeline_tmp_i = NULL;
    
    if (posix_memalign((void**)&cust_cube_r, alignment, alloc_size) != 0) return;
    if (posix_memalign((void**)&cust_cube_i, alignment, alloc_size) != 0) return;
    if (posix_memalign((void**)&pipeline_tmp_r, alignment, alloc_size) != 0) return;
    if (posix_memalign((void**)&pipeline_tmp_i, alignment, alloc_size) != 0) return;

    // 대조군 FFTW3 전용 얼라인먼트 할당 (fftw_malloc은 자체적으로 하드웨어 정렬을 수행함)
    fftwf_complex *fftw_cube_in  = (fftwf_complex *)fftwf_malloc(total_elements * sizeof(fftwf_complex));
    fftwf_complex *fftw_cube_out = (fftwf_complex *)fftwf_malloc(total_elements * sizeof(fftwf_complex));
    memset(fftw_cube_in, 0, total_elements * sizeof(fftwf_complex));
    memset(fftw_cube_out, 0, total_elements * sizeof(fftwf_complex));

    double true_R[] = {12.50, 25.20,  6.80}; 
    double true_v[] = {14.20, -5.50,  0.00}; 
    double true_a[] = {-15.0, 25.0,   0.0}; 
    int num_targets = 3;

    unsigned int fftw_flags = FFTW_ESTIMATE;
#if defined(FFTW_MODE_MEASURE)
    fftw_flags = FFTW_MEASURE;
#elif defined(FFTW_MODE_PATIENT)
    fftw_flags = FFTW_PATIENT;
#endif

    float *lut_r = NULL;
    float *lut_i = NULL;
    if (posix_memalign((void**)&lut_r, alignment, alloc_size) != 0) return;
    if (posix_memalign((void**)&lut_i, alignment, alloc_size) != 0) return;

    for (int ant = 0; ant < N_ANTENNAS; ant++) {
        for (int chirp = 0; chirp < N_CHIRPS; chirp++) {
            int offset = ant * (N_CHIRPS * n_samples) + chirp * n_samples;
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
                lut_r[offset + n] = (float)val_r;
                lut_i[offset + n] = (float)val_i;
            }
        }
    }

    // [Custom] 1D Range-FFT (정렬된 메모리 덕분에 NEON 스트리밍 속도 향상 기대)
    double start_cust_range = get_current_time_ms();
    #pragma omp parallel for collapse(2)
    for (int ant = 0; ant < N_ANTENNAS; ant++) {
        for (int chirp = 0; chirp < N_CHIRPS; chirp++) {
            int offset = ant * (N_CHIRPS * n_samples) + chirp * n_samples;
            for (int n = 0; n < n_samples; n++) {
                float win = (n_samples == 1024) ? win_1024[n] : win_2048[n];
                cust_cube_r[offset + n] = lut_r[offset + n] * win;
                cust_cube_i[offset + n] = lut_i[offset + n] * win;
            }
            if (n_samples == 1024) custom_fft_1024_fixed(&cust_cube_r[offset], &cust_cube_i[offset]);
            else                   custom_fft_2048_fixed(&cust_cube_r[offset], &cust_cube_i[offset]);
        }
    }
    double cust_range_ms = get_current_time_ms() - start_cust_range;

    // [Custom] 2D Doppler-FFT 파이프라인
    double start_cust_doppler = get_current_time_ms();
    execute_custom_pipeline(cust_cube_r, cust_cube_i, pipeline_tmp_r, pipeline_tmp_i, n_samples);
    double cust_doppler_ms = get_current_time_ms() - start_cust_doppler;

    // [FFTW3 대조군]
    fftwf_plan p_range = fftwf_plan_dft_1d(n_samples, fftw_cube_in, fftw_cube_in, FFTW_FORWARD, fftw_flags);
    int d_rank = 1; int d_n[] = {N_CHIRPS}; int d_howmany = N_ANTENNAS * n_samples;
    int d_idist = N_CHIRPS, d_odist = N_CHIRPS; int d_istride = 1, d_ostride = 1;
    fftwf_plan p_doppler = fftwf_plan_many_dft(d_rank, d_n, d_howmany, fftw_cube_in, NULL, d_istride, d_idist, fftw_cube_out, NULL, d_ostride, d_odist, FFTW_FORWARD, fftw_flags);

    double start_fftw = get_current_time_ms();
    #pragma omp parallel for collapse(2)
    for (int ant = 0; ant < N_ANTENNAS; ant++) {
        for (int chirp = 0; chirp < N_CHIRPS; chirp++) {
            int offset = ant * (N_CHIRPS * n_samples) + chirp * n_samples;
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

    printf("\n 🎯 [물리 값 정밀 검증 결과] (구간별 정밀 프로파일링 모드)\n");
    printf(" -------------------------------------------------------------------------\n");
    
    double total_cust_angle_ms = 0.0;
    for (int t_idx = 0; t_idx < num_targets; t_idx++) {
        float max_2d_mag = -1.0f; int best_r = 0, best_chirp = 0;
        int r_center = (int)(((2.0 * S * true_R[t_idx]) / c) * n_samples / Fs);
        int r_start = r_center - 15; if (r_start < 0) r_start = 0;
        int r_end = r_center + 15; if (r_end >= n_samples) r_end = n_samples - 1;
        
        for (int r = r_start; r <= r_end; r++) {
            for (int ch = 0; ch < N_CHIRPS; ch++) {
                float sum_mag = 0.0f;
                for (int ant = 0; ant < N_ANTENNAS; ant++) {
                    int idx = ant * (n_samples * N_CHIRPS) + r * N_CHIRPS + ch;
                    sum_mag += pipeline_tmp_r[idx]*pipeline_tmp_r[idx] + pipeline_tmp_i[idx]*pipeline_tmp_i[idx];
                }
                if (sum_mag > max_2d_mag) { max_2d_mag = sum_mag; best_r = r; best_chirp = ch; }
            }
        }
        
        float ang_r[64] = {0.0f}; float ang_i[64] = {0.0f};
        for (int ant = 0; ant < N_ANTENNAS; ant++) {
            int idx = ant * (n_samples * N_CHIRPS) + best_r * N_CHIRPS + best_chirp;
            if (idx >= 0 && idx < total_elements) {
                ang_r[ant] = pipeline_tmp_r[idx]; ang_i[ant] = pipeline_tmp_i[idx];
            }
        }
        
        double start_angle = get_current_time_ms();
        custom_fft_64_fixed(ang_r, ang_i);
        total_cust_angle_ms += (get_current_time_ms() - start_angle);
        
        float max_ang_mag = -1.0f; int best_a = 0;
        for (int a = 0; a < N_ANGLE; a++) {
            float m = ang_r[a]*ang_r[a] + ang_i[a]*ang_i[a];
            if (m > max_ang_mag) { max_ang_mag = m; best_a = a; }
        }

        double est_Range = (best_r * Fs / n_samples) * c / (2.0 * S);
        double d_omega = (best_chirp >= N_CHIRPS / 2) ? (2.0 * M_PI * (best_chirp - N_CHIRPS) / N_CHIRPS) : (2.0 * M_PI * best_chirp / N_CHIRPS);
        double est_Velocity = (d_omega * lambda_c) / (4.0 * M_PI * Tc);
        double a_omega = (best_a >= N_ANGLE / 2) ? (2.0 * M_PI * (best_a - N_ANGLE) / N_ANGLE) : (2.0 * M_PI * best_a / N_ANGLE);
        double sin_val = (a_omega * lambda_c) / (2.0 * M_PI * d_ant);
        
        if (sin_val > 1.0)  sin_val = 1.0; 
        if (sin_val < -1.0) sin_val = -1.0;
        double est_Angle = asin(sin_val) * 180.0 / M_PI;

        double r_err = fabs(est_Range - true_R[t_idx]) / true_R[t_idx] * 100.0;
        const char* status = (r_err < 5.0) ? "정밀합격" : "❌ 불합격";

        printf("  타겟 %d | 주입값: R=%5.2fm, v=%6.2fm/s, a=%5.1f°\n", t_idx + 1, true_R[t_idx], true_v[t_idx], true_a[t_idx]);
        printf("         | DSP역산: R=%5.2fm, v=%6.2fm/s, a=%5.1f° ➡️  [%s]\n", est_Range, est_Velocity, est_Angle, status);
        printf(" -------------------------------------------------------------------------\n");
    }

    int ram_end_kb = get_current_ram_usage_kb();
    double current_session_ram_mb = (ram_end_kb - ram_start_kb) / 1024.0;
    if (current_session_ram_mb < 0) current_session_ram_mb = 0.0;

    printf(" 📊 [3D 궁극 가속 배틀 레포트 (캐시 얼라인먼트 패치 버전)]\n");
    printf("  =========================================================================\n");
    printf("  ▪️ [Custom] 1D Range-FFT 인라인 타임    : %6.2f ms\n", cust_range_ms);
    printf("  ▪️ [Custom] 2D Doppler-FFT 파이프라인   : %6.2f ms\n", cust_doppler_ms);
    printf("  🚀 [Custom] 3D 수평 Angle-FFT 순수 지연  : %6.2f ms\n", total_cust_angle_ms);
    printf("  -------------------------------------------------------------------------\n");
    printf("  💻 [FFTW3]  1D Range-FFT 매칭 타임       : %6.2f ms\n", fftw_range_ms);
    printf("  💻 [FFTW3]  2D Doppler-FFT 파이프라인     : %6.2f ms\n", fftw_doppler_ms);
    printf("  =========================================================================\n");
    printf("  🔥 본 세션 실시간 메모리 순수 점유량     : %.2f MB\n", current_session_ram_mb);
    printf("  =========================================================================\n\n");

    free(cust_cube_r); free(cust_cube_i);
    free(pipeline_tmp_r); free(pipeline_tmp_i);
    free(lut_r); free(lut_i);
    fftwf_free(fftw_cube_in); fftwf_free(fftw_cube_out);
}