#include "radar_utils.h"

void run_benchmark_for_size(int n_samples) {
    int total_elements = N_ANTENNAS * N_CHIRPS * n_samples;
    printf("\n====================================================\n");
    printf(" 🚀 [격리 벤치마크] Range Samples = %d 파이프라인\n", n_samples);
    printf("====================================================\n");

    float *raw_real = (float *)calloc(total_elements, sizeof(float));
    float *raw_imag = (float *)calloc(total_elements, sizeof(float));

    double target_R[] = {9.25, 22.40}; double target_v[] = {7.30, -15.80}; 
    for (int t_idx = 0; t_idx < 2; t_idx++) {
        double R = target_R[t_idx]; double v = target_v[t_idx]; double theta_rad = (-20.0 + t_idx * 55.0) * PI / 180.0;
        double f_R = (2.0 * S * R) / c; double phase_doppler = (4.0 * PI * v) / lambda_c * Tc; double phase_angle = (2.0 * PI * d_ant * sin(theta_rad)) / lambda_c;
        for (int ant = 0; ant < N_ANTENNAS; ant++) {
            for (int chirp = 0; chirp < N_CHIRPS; chirp++) {
                for (int n = 0; n < n_samples; n++) {
                    double phase = (2.0 * PI * f_R * ((double)n/Fs)) + (phase_doppler * chirp) + (phase_angle * ant);
                    int idx = ant * (N_CHIRPS * n_samples) + chirp * n_samples + n;
                    raw_real[idx] += (float)cos(phase); raw_imag[idx] += (float)sin(phase);
                }
            }
        }
    }

    malloc_trim(0); 
    int base_fftw_mem = get_current_ram_usage_kb();
    fftwf_complex *fftw_cube = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * total_elements);
    int n_range[] = {n_samples};
    fftwf_plan p_range = fftwf_plan_many_dft(1, n_range, N_ANTENNAS * N_CHIRPS, fftw_cube, NULL, 1, n_samples, fftw_cube, NULL, 1, n_samples, FFTW_FORWARD, FFTW_PATIENT);
    int n_doppler[] = {N_CHIRPS};
    fftwf_plan p_doppler = fftwf_plan_many_dft(1, n_doppler, n_samples, fftw_cube, NULL, n_samples, 1, fftw_cube, NULL, n_samples, 1, FFTW_FORWARD, FFTW_PATIENT);

    // 가동 과정에서 정밀 계산된 결과를 백업하기 위해 하단에 파일 쓰기 추가
    FILE *wf = fopen(FFTW_WISDOM_FILE, "w");
    if (wf != NULL) {
        fftwf_export_wisdom_to_file(wf);
        fclose(wf);
    }
    for(int i=0; i<total_elements; i++) { fftw_cube[i][0] = raw_real[i]; fftw_cube[i][1] = raw_imag[i]; }

    int peak_fftw_mem = get_current_ram_usage_kb();
    double actual_fftw_mb = (peak_fftw_mem - base_fftw_mem) / 1024.0;
    double start_fftw = get_current_time_ms();
    for(int i = 0; i < BENCH_RUNS; i++) {
        for(int j=0; j<total_elements; j++) { fftw_cube[j][0] = raw_real[j]; fftw_cube[j][1] = raw_imag[j]; }
        execute_fftw_pipeline_optimized(fftw_cube, p_range, p_doppler, n_samples);
    }
    double avg_fftw_ms = (get_current_time_ms() - start_fftw) / BENCH_RUNS;
    fftwf_destroy_plan(p_range); fftwf_destroy_plan(p_doppler); fftwf_free(fftw_cube);

    malloc_trim(0);
    int base_cust_mem = get_current_ram_usage_kb();
    float *cust_cube_r = (float *)malloc(total_elements * sizeof(float));
    float *cust_cube_i = (float *)malloc(total_elements * sizeof(float));
    float *trans_cust_r = (float *)malloc(total_elements * sizeof(float));
    float *trans_cust_i = (float *)malloc(total_elements * sizeof(float));
    for(int i=0; i<total_elements; i++) { cust_cube_r[i] = raw_real[i]; cust_cube_i[i] = raw_imag[i]; trans_cust_r[i] = 0; trans_cust_i[i] = 0; }

    int peak_cust_mem = get_current_ram_usage_kb();
    double actual_cust_mb = (peak_cust_mem - base_cust_mem) / 1024.0;
    double start_cust = get_current_time_ms();
    for(int i = 0; i < BENCH_RUNS; i++) {
        memcpy(cust_cube_r, raw_real, total_elements * sizeof(float));
        memcpy(cust_cube_i, raw_imag, total_elements * sizeof(float));
        execute_custom_pipeline(cust_cube_r, cust_cube_i, trans_cust_r, trans_cust_i, n_samples);
    }
    double avg_custom_ms = (get_current_time_ms() - start_cust) / BENCH_RUNS;

    free(cust_cube_r); free(cust_cube_i); free(trans_cust_r); free(trans_cust_i); free(raw_real); free(raw_imag);
    malloc_trim(0);

    printf(" [결과] 💻 FFTW3 Pipeline  : %6.2f ms / Frame (OS 실측 할당 RAM: %5.2f MB)\n", avg_fftw_ms, actual_fftw_mb);
    printf(" [결과] 🚀 Custom Pipeline : %6.2f ms / Frame (OS 실측 할당 RAM: %5.2f MB)\n", avg_custom_ms, actual_cust_mb);
}