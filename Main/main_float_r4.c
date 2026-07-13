#include "radar_pipeline.h"
#include "radar_benchmark_sessions.h"
#include "radar_config.h"
#include "radar_utils.h"
#include "radar_mock_data.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>



void execute_standalone_run(int n_samples, int n_chirps) {
    int total_elements = N_ANTENNAS * n_chirps * n_samples;
    size_t alloc_size_float = total_elements * sizeof(float);

    float *lut_r = NULL; float *lut_i = NULL;
    posix_memalign((void**)&lut_r, 64, alloc_size_float);
    posix_memalign((void**)&lut_i, 64, alloc_size_float);

    generate_radar_mock_data(lut_r, lut_i, n_samples, n_chirps);

    BenchmarkResult res;
    memset(&res, 0, sizeof(BenchmarkResult));

    printf("\n 🛰️  [3D 독립 자율 인지 바이너리] Float Radix-4 | Size = %d x %d\n", n_samples, n_chirps);
    benchmark_session_float_r4(lut_r, lut_i, n_samples, n_chirps, &res);

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
    
    

    execute_standalone_run(1024, 256);
    execute_standalone_run(1024, 512);
    execute_standalone_run(2048, 256);
    execute_standalone_run(2048, 512);

    

    return 0;
}