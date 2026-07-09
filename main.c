#include "radar_utils.h"
#include "radar_config.h"
#include "radar_fft.h"
#include <stdio.h>
#include <stdlib.h>
#include <fftw3.h>

void run_benchmark_core(int n_samples, int n_chirps);

int main() {
    // 💡 1. 멀티스레드 환경 최우선 선언
    fftwf_init_threads();
    fftwf_plan_with_nthreads(4); 

    // 🎯 2. Wisdom 설계도 로드
#if defined(FFTW_MODE_PATIENT)
    FILE *w_file_in = fopen("radar_wisdom.wisdom", "r");
    if (w_file_in != NULL) {
        if (fftwf_import_wisdom_from_file(w_file_in)) {
            printf("💾 [Wisdom] 기존 라즈베리파이 가속 설계도를 로드합니다.\n");
        }
        fclose(w_file_in);
    } else {
        printf("🔬 [Wisdom] 최적화 설계도가 없습니다. 하드웨어 한계 측정 모드를 실행합니다.\n");
    }
#endif

    // 3. 자원 초기화 (모든 LUT 할당)
    init_resources();

    printf("====================================================\n");
    printf("  3D Radar Pipeline: Full Architecture Benchmark    \n");
    printf("====================================================\n\n");

    // 🚀 5. [메인 벤치마크] 불필요한 forget 제거, 4대 시나리오 순수 직렬 실행
    // 1024 / 2048 크기가 누적 학습되어 최종 Wisdom 파일에 완벽하게 박제됩니다.

    // [시나리오 1] Range = 1024 | Chirps = 256
    run_benchmark_core(1024, 256);

    // [시나리오 2] Range = 1024 | Chirps = 512
    run_benchmark_core(1024, 512);

    // [시나리오 3] Range = 2048 | Chirps = 256
    run_benchmark_core(2048, 256);

    // [시나리오 4] Range = 2048 | Chirps = 512
    run_benchmark_core(2048, 512);

    run_individual_operation_benchmark();

    // 🎯 6. Wisdom 설계도 영구 저장
#if defined(FFTW_MODE_PATIENT)
    FILE *w_file_out = fopen("radar_wisdom.wisdom", "w");
    if (w_file_out != NULL) {
        fftwf_export_wisdom_to_file(w_file_out);
        fclose(w_file_out);
        printf("\n💾 [Wisdom] 최적화 기계어 경로 학습이 완료되어 박제되었습니다.\n");
    }
#endif

    // 안전 종료
    fftwf_cleanup_threads();
    fftwf_cleanup();

    printf("🎉 [SUCCESS] 4가지 핵심 가속 세션을 에러 없이 완벽하게 완주했습니다!\n");
    return 0;
}