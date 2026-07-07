#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <fftw3.h>

#include "Globals/radar_config.h"
#include "Utils/radar_utils.h"
#include "FFT/radar_fft.h"

/**
 * @brief 레이더 신호 처리 벤치마크 메인 진입점
 * @return 정상 종료 시 0 반환
 */
int main() {
    // 🎯 [Wisdom 격리] 최적화 설계도 로드 (오직 PATIENT 모드에서만 허용)
#if defined(FFTW_MODE_PATIENT)
    FILE *w_file_in = fopen("radar_wisdom.wisdom", "r");
    if (w_file_in != NULL) {
        if (fftwf_import_wisdom_from_file(w_file_in)) {
            printf("💾 [Wisdom] 기존 라즈베리파이 5 가속 설계도를 로드합니다.\n");
        }
        fclose(w_file_in);
    } else {
        printf("🔬 [Wisdom] 최적화 설계도가 없습니다. 하드웨어 한계 측정 모드를 실행합니다.\n");
        printf("    (수십 분의 극한 시뮬레이션이 진행될 수 있습니다...)\n");
    }
#endif

    // 1. 시스템 초기화
    fftwf_init_threads();
    fftwf_plan_with_nthreads(4);
    
    // 사전 연산 테이블(LUT) 생성
    init_resources();
    
    printf("====================================================\n");
    printf("  3D Radar Pipeline: Component-based Architecture \n");
    printf("====================================================\n\n");

    // 2. 전체 파이프라인 벤치마크 연속 가동
    run_benchmark_for_size(1024);
    
    printf("\n----------------------------------------------------\n");
    printf(" 🔄 2048 샘플 세션으로 전환합니다... \n");
    printf("----------------------------------------------------\n\n");
    
    run_benchmark_for_size(2048);

    // 3. 1D FFT 개별 성능 프로파일링
    printf("\n====================================================\n");
    printf(" [PART 4] 1D FFT 개별 연산 실행 속도 비교 (마이크로초 단위)\n");
    printf("----------------------------------------------------\n");
    profile_individual_operation(2048, custom_fft_2048_fixed);
    profile_individual_operation(1024, custom_fft_1024_fixed);
    profile_individual_operation(512,  custom_fft_512_fixed);
    profile_individual_operation(256,  custom_fft_256_fixed);
    profile_individual_operation(128,  custom_fft_128_fixed);
    profile_individual_operation(64,   custom_fft_64_fixed);
    printf("====================================================\n");
    
    // 4. 시스템 메모리 점검 및 종료
    struct rusage usage; 
    getrusage(RUSAGE_SELF, &usage);
    printf(" 📊 프로세스 라이프사이클 Peak 물리 메모리(Max RSS) : %.2f MB\n", usage.ru_maxrss / 1024.0);
    printf("====================================================\n");

    // 🎯 [Wisdom 격리] 최적화 설계도 저장 (오직 PATIENT 모드에서만 허용)
#if defined(FFTW_MODE_PATIENT)
    FILE *w_file = fopen("radar_wisdom.wisdom", "w");
    if (w_file) {
        fftwf_export_wisdom_to_file(w_file);
        fclose(w_file);
        printf("💾 [Wisdom] 라즈베리파이 5 최적화 기계어 경로 학습이 완료되어 박제되었습니다.\n");
    }
#endif

    fftwf_cleanup_threads();
    return 0;
}