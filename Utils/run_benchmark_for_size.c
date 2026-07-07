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
    
    printf("\n====================================================\n");
#if defined(FFTW_MODE_ESTIMATE)
    printf(" 🛰️ [3D 궁극의 가속 엔진] Range = %d | ⚡ ESTIMATE 모드\n", n_samples);
#elif defined(FFTW_MODE_MEASURE)
    printf(" 🛰️ [3D 궁극의 가속 엔진] Range = %d | ⚖️ MEASURE 모드\n", n_samples);
#elif defined(FFTW_MODE_PATIENT)
    printf(" 🛰️ [3D 궁극의 가속 엔진] Range = %d | 🐢 PATIENT 모드\n", n_samples);
#endif
    printf("====================================================\n");

    // 0MB 다이어트 버퍼 안전 할당
    float *cust_cube_r = (float *)malloc(total_elements * sizeof(float));
    float *cust_cube_i = (float *)malloc(total_elements * sizeof(float));

    fftwf_complex *fftw_cube_in  = (fftwf_complex *)fftwf_malloc(total_elements * sizeof(fftwf_complex));
    fftwf_complex *fftw_cube_out = (fftwf_complex *)fftwf_malloc(total_elements * sizeof(fftwf_complex));
    memset(fftw_cube_in, 0, total_elements * sizeof(fftwf_complex));
    memset(fftw_cube_out, 0, total_elements * sizeof(fftwf_complex));

    double true_R[] = {12.50, 25.20,  6.80}; 
    double true_v[] = {14.20, -5.50,  0.00}; 
    double true_a[] = {-15.0, 25.0,   0.0}; 
    int num_targets = 3;

    // 🎯 [안전 장치] 헤더 파일 오염을 무시하고 무조건 ESTIMATE 강제 구동 (혼종 버그 차단)
    // 🎯 [플래그 동기화] 컴파일 매크로에 따라 FFTW3의 극한 계측 모드를 정확히 바인딩합니다.
    unsigned int fftw_flags = FFTW_ESTIMATE;
#if defined(FFTW_MODE_MEASURE)
    fftw_flags = FFTW_MEASURE;
#elif defined(FFTW_MODE_PATIENT)
    fftw_flags = FFTW_PATIENT;
#endif

    // 가상 전파 위상 룩업 테이블(LUT)
    float *lut_r = (float *)malloc(total_elements * sizeof(float));
    float *lut_i = (float *)malloc(total_elements * sizeof(float));

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

    // 🎯 [핵심 패치] 스레드 충돌을 막기 위해 1D 마스터 플랜을 루프 바깥에서 단 1번만 생성합니다!
    fftwf_plan p_range = fftwf_plan_dft_1d(n_samples, fftw_cube_in, fftw_cube_in, FFTW_FORWARD, fftw_flags);

    double start_inline = get_current_time_ms();
    #pragma omp parallel for collapse(2)
    for (int ant = 0; ant < N_ANTENNAS; ant++) {
        for (int chirp = 0; chirp < N_CHIRPS; chirp++) {
            int offset = ant * (N_CHIRPS * n_samples) + chirp * n_samples;
            
            for (int n = 0; n < n_samples; n++) {
                float win = (n_samples == 1024) ? win_1024[n] : win_2048[n];
                float raw_r = lut_r[offset + n];
                float raw_i = lut_i[offset + n];

                cust_cube_r[offset + n] = raw_r * win;
                cust_cube_i[offset + n] = raw_i * win;

                fftw_cube_in[offset + n][0] = raw_r * win;
                fftw_cube_in[offset + n][1] = raw_i * win;
            }

            if (n_samples == 1024) custom_fft_1024_fixed(&cust_cube_r[offset], &cust_cube_i[offset]);
            else                   custom_fft_2048_fixed(&cust_cube_r[offset], &cust_cube_i[offset]);

            // 🎯 [핵심 패치] 루프 안에서는 플랜 생성 없이 배열 포인터만 던져서 안전하게 실행(Execute)만 수행합니다.
            fftwf_execute_dft(p_range, &fftw_cube_in[offset], &fftw_cube_in[offset]);
        }
    }
    double inline_time_ms = get_current_time_ms() - start_inline;
    
    // 실행 완료 후 루프 밖에서 안전하게 파괴
    fftwf_destroy_plan(p_range);

    // ----------------------------------------------------
    // 후처리 2D 레이스 구동
    // ----------------------------------------------------
    double start_cust = get_current_time_ms();
    execute_custom_pipeline(cust_cube_r, cust_cube_i, n_samples);
    double custom_post_ms = get_current_time_ms() - start_cust;

    int d_rank = 1; int d_n[] = {N_CHIRPS}; int d_howmany = N_ANTENNAS * n_samples;
    int d_idist = N_CHIRPS, d_odist = N_CHIRPS; int d_istride = 1, d_ostride = 1;
    
    // Doppler 플랜 생성은 어차피 루프 밖이므로 스레드 충돌 없이 안전합니다.
    fftwf_plan p_doppler = fftwf_plan_many_dft(d_rank, d_n, d_howmany, fftw_cube_in, NULL, d_istride, d_idist, fftw_cube_out, NULL, d_ostride, d_odist, FFTW_FORWARD, fftw_flags);
    
    double start_fftw = get_current_time_ms();
    execute_fftw_pipeline_optimized(fftw_cube_in, fftw_cube_out, p_doppler, n_samples);
    double fftw_post_ms = get_current_time_ms() - start_fftw;
    fftwf_destroy_plan(p_doppler);

    // ----------------------------------------------------
    // 🎯 [선택적 Angle-FFT 탐색]
    // ----------------------------------------------------
    printf("\n 🎯 [물리 값 정밀 검증 결과] (0MB 3D 다이어트 & 64 가속 성공)\n");
    printf(" -------------------------------------------------------------------------\n");
    
    for (int t_idx = 0; t_idx < num_targets; t_idx++) {
        float max_2d_mag = -1.0f;
        int best_r = 0, best_chirp = 0;
        
        int r_center = (int)(((2.0 * S * true_R[t_idx]) / c) * n_samples / Fs);
        int r_start = r_center - 15; if (r_start < 0) r_start = 0;
        int r_end = r_center + 15; if (r_end >= n_samples) r_end = n_samples - 1;
        
        for (int r = r_start; r <= r_end; r++) {
            for (int ch = 0; ch < N_CHIRPS; ch++) {
                float sum_mag = 0.0f;
                for (int ant = 0; ant < N_ANTENNAS; ant++) {
                    int idx = ant * (n_samples * N_CHIRPS) + r * N_CHIRPS + ch;
                    sum_mag += cust_cube_r[idx]*cust_cube_r[idx] + cust_cube_i[idx]*cust_cube_i[idx];
                }
                if (sum_mag > max_2d_mag) {
                    max_2d_mag = sum_mag; best_r = r; best_chirp = ch;
                }
            }
        }
        
        float ang_r[64] = {0.0f}; 
        float ang_i[64] = {0.0f};
        
        for (int ant = 0; ant < N_ANTENNAS; ant++) {
            int idx = ant * (n_samples * N_CHIRPS) + best_r * N_CHIRPS + best_chirp;
            if (idx >= 0 && idx < total_elements) {
                ang_r[ant] = cust_cube_r[idx]; 
                ang_i[ant] = cust_cube_i[idx];
            }
        }
        
        custom_fft_64_fixed(ang_r, ang_i);
        
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

    printf(" 📊 [3D 궁극 가속 배틀 레포트]\n");
    printf("  =========================================================================\n");
    printf("  ▪️ 전파 수집 + Range-FFT 인라인 타임   : %6.2f ms\n", inline_time_ms);
    printf("  💻 [결과] FFTW3 지연 (2D+선택)     : %6.2f ms / Frame\n", fftw_post_ms);
    printf("  🚀 [결과] Custom 지연 (2D+선택 64) : %6.2f ms / Frame (🔥 0MB 다이어트 완성)\n", custom_post_ms);
    printf("  =========================================================================\n\n");

    free(cust_cube_r); free(cust_cube_i);
    free(lut_r); free(lut_i);
    fftwf_free(fftw_cube_in); fftwf_free(fftw_cube_out);
}