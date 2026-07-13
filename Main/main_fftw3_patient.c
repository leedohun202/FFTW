#include "radar_pipeline.h"
#include "radar_benchmark_sessions.h"
#include "radar_config.h"
#include "radar_utils.h"
#include "radar_mock_data.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fftw3.h>
#include <omp.h>  // 💥 OpenMP 헤더 추가

void execute_standalone_run(int n_samples, int n_chirps) {
    int total_elements = N_ANTENNAS * n_chirps * n_samples;
    size_t alloc_size_float = total_elements * sizeof(float);

    float *lut_r = NULL; float *lut_i = NULL;
    posix_memalign((void**)&lut_r, 64, alloc_size_float);
    posix_memalign((void**)&lut_i, 64, alloc_size_float);

    generate_radar_mock_data(lut_r, lut_i, n_samples, n_chirps);

    BenchmarkResult res;
    memset(&res, 0, sizeof(BenchmarkResult));

    printf("\n 🛰️  [3D 독립 자율 인지 바이너리] FFTW3 (Patient) | Size = %d x %d\n", n_samples, n_chirps);
    benchmark_session_fftw3(lut_r, lut_i, n_samples, n_chirps, &res, FFTW_PATIENT);

    printf(" ---------------------------------------------------\n");
    printf(" ⏱️  연산 속도: Range FFT = %6.2f ms | Doppler FFT = %6.2f ms\n", res.time_range, res.time_doppler);
    printf(" 📈  물리 RAM 점유: %6.2f MB\n", res.actual_ram_mb);
    printf(" ---------------------------------------------------\n");
    printf(" 🎯  100%% 자율 탐지 포인트 클라우드 결과:\n");
    for(int t=0; t<3; t++) {
        if(res.est_R[t] > 0.0) {
            printf("   📍 [물체 %d] 거리: %6.3fm | 속도: %6.3fm/s | 각도: %7.3f°\n", t+1, res.est_R[t], res.est_v[t], res.est_a[t]);
        } else {
            printf("   ⚠️ [물체 %d] 탐지 실패 (Missed)\n", t+1);
        }
    }
    printf(" ====================================================\n");

    free(lut_r); free(lut_i);
}

int main() {
    init_resources();

    
    FILE *w_file_in = fopen("radar_wisdom.wisdom", "r");
    if (w_file_in != NULL) {
        if (fftwf_import_wisdom_from_file(w_file_in)) {
            printf("💾 [Wisdom] 기존 라즈베리파이 가속 설계도를 로드합니다.\n");
        }
        fclose(w_file_in);
    }


    execute_standalone_run(1024, 256);
    execute_standalone_run(1024, 512);
    execute_standalone_run(2048, 256);
    execute_standalone_run(2048, 512);

    
    FILE *w_file_out = fopen("radar_wisdom.wisdom", "w");
    if (w_file_out != NULL) {
        fftwf_export_wisdom_to_file(w_file_out);
        fclose(w_file_out);
        printf("\n💾 [Wisdom] 최적화 기계어 경로 학습이 완료되어 박제되었습니다.\n");
    }
#ifdef _OPENMP
    fftwf_cleanup_threads();
#endif
    fftwf_cleanup();

    
    return 0;
}