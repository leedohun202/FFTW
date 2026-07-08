#include <stdio.h>
#include <stdlib.h>
#include <fftw3.h>

#include "radar_config.h"
#include "radar_utils.h"
#include "radar_fft.h"
#include "radar_pipeline.h"

int main() {
    // 💡 1. 멀티스레드 환경을 '최우선'으로 선언하여 스레드 불일치 버그 원천 차단
    fftwf_init_threads();
    fftwf_plan_with_nthreads(4); 

    // 🎯 2. Wisdom 설계도 로드
#if defined(FFTW_MODE_PATIENT)
    FILE *w_file_in = fopen("radar_wisdom.wisdom", "r");
    if (w_file_in != NULL) {
        if (fftwf_import_wisdom_from_file(w_file_in)) {
            printf("💾 [Wisdom] 기존 라즈베리파이 5 (4스레드 멀티코어) 가속 설계도를 로드합니다.\n");
        }
        fclose(w_file_in);
    } else {
        printf("🔬 [Wisdom] 최적화 설계도가 없습니다. 하드웨어 한계 측정 모드를 실행합니다.\n");
        printf("    (수십 분의 극한 시뮬레이션이 진행될 수 있습니다...)\n");
    }
#endif

    // 자원 초기화
    init_resources();

    printf("====================================================\n");
    printf("  3D Radar Pipeline: Component-based Architecture   \n");
    printf("====================================================\n");

    // 1️⃣ 1024 샘플 포인트 세션 가동 (내부에서 256/512 처프 연속 수행)
    run_benchmark_for_size(1024);

    printf("\n----------------------------------------------------\n");
    printf(" 🔄 2048 샘플 세션으로 전환합니다... \n");
    printf("----------------------------------------------------\n");

    // 2️⃣ 2048 샘플 포인트 세션 가동 (내부에서 256/512 처프 연속 수행)
    run_benchmark_for_size(2048);


    // ----------------------------------------------------
    // [PART 4] 1D FFT 개별 연산 실행 속도 비교
    // ----------------------------------------------------
    printf("\n====================================================\n");
    printf(" [PART 4] 1D FFT 개별 연산 실행 속도 비교 (마이크로초 단위)\n");
    printf(" ----------------------------------------------------\n");
    
    // 2048 포인트 (Radix-2)
    profile_individual_operation(2048, custom_fft_2048_fixed);
    printf(" ----------------------------------------------------\n");

    // 1024 포인트 (Radix-2 vs Radix-4 진검승부)
    profile_individual_operation(1024, custom_fft_1024_fixed);
    profile_individual_operation(1024, custom_fft_1024_radix4);
    printf(" ----------------------------------------------------\n");

    // 512 포인트 (Radix-2)
    profile_individual_operation(512, custom_fft_512_fixed);
    printf(" ----------------------------------------------------\n");

    // 256 포인트 (Radix-2 vs Radix-4 진검승부)
    profile_individual_operation(256, custom_fft_256_fixed);
    profile_individual_operation(256, custom_fft_256_radix4);
    printf(" ----------------------------------------------------\n");

    // 128 포인트 (Radix-2)
    profile_individual_operation(128, custom_fft_128_fixed);
    printf(" ----------------------------------------------------\n");

    // 64 포인트 (Radix-2 vs Radix-4 진검승부)
    profile_individual_operation(64, custom_fft_64_fixed);
    profile_individual_operation(64, custom_fft_64_radix4);
    
    printf("====================================================\n");


    // 🎯 3. Wisdom 설계도 영구 저장
#if defined(FFTW_MODE_PATIENT)
    FILE *w_file_out = fopen("radar_wisdom.wisdom", "w");
    if (w_file_out != NULL) {
        fftwf_export_wisdom_to_file(w_file_out);
        fclose(w_file_out);
        printf("💾 [Wisdom] 라즈베리파이 5 최적화 기계어 경로 학습이 완료되어 박제되었습니다.\n");
    }
#endif

    // 안전 종료
    fftwf_cleanup_threads();
    fftwf_cleanup();

    return 0;
}