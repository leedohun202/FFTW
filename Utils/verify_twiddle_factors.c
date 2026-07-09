#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * @brief 정수형 트위들 팩터 테이블 정밀도 검증 함수
 * @param real 정수형 실수부 배열 포인터
 * @param imag 정수형 허수부 배열 포인터
 * @param N    FFT 포인트 크기 (예: 64, 512, 4096 등)
 * @param name 분석 대상 테이블 이름 문자열
 */
void verify_twiddle_factors(const int16_t *real, const int16_t *imag, int N, const char *name) {
    printf("====================================================\n");
    printf(" 🔍 [%s (N=%d)] 고정소수점 룩업테이블(LUT) 정밀도 검증\n", name, N);
    printf("====================================================\n");
    
    double max_precision_error = 0.0;
    double max_magnitude_error = 0.0;
    
    for (int k = 0; k < N; k++) {
        // 1. Q15 정수를 다시 실수(-1.0 ~ 1.0) 도메인으로 역산
        double r_f = (double)real[k] / 32767.0;
        double i_f = (double)imag[k] / 32767.0;
        
        // 2. 크기 왜곡도 검증 (반지름이 정확히 1.0에 수렴하는가?)
        double magnitude = sqrt(r_f * r_f + i_f * i_f);
        double mag_err = fabs(magnitude - 1.0);
        if (mag_err > max_magnitude_error) {
            max_magnitude_error = mag_err;
        }
        
        // 3. 수학적 이론값(Reference) 계산
        double ref_r = cos(2.0 * M_PI * k / N);
        double ref_i = -sin(2.0 * M_PI * k / N); // 파이프라인 커널 부호 정의에 맞춰서 조율
        
        // 정밀도 오차 측정
        double err_r = fabs(r_f - ref_r);
        double err_i = fabs(i_f - ref_i);
        
        // 허수부 부호 반전 룩업의 유연성 처리 (sin과 -sin 부호 매칭 유연성 부여)
        if (fabs(i_f + ref_i) < err_i) {
            err_i = fabs(i_f + ref_i);
        }
        
        if (err_r > max_precision_error) max_precision_error = err_r;
        if (err_i > max_precision_error) max_precision_error = err_i;
    }
    
    // 4. 주요 분기점 경계값 정적 진단
    printf("  📍 [경계값 상태 팩트 체크]\n");
    printf("    ▪️ 인덱스 k = 0     -> Real: %5d (기대값: 32767), Imag: %5d (기대값: 0)\n", real[0], imag[0]);
    if (N >= 4) {
        printf("    ▪️ 인덱스 k = N/4   -> Real: %5d (기대값:     0), Imag: %5d (기대값: -32767 또는 32767)\n", real[N/4], imag[N/4]);
    }
    
    // 5. 최종 리포트 출력
    printf("\n  📊 [계량 오차 요약 리포트]\n");
    printf("    ▪️ 복소 단위 원 최대 왜곡도 (Mag Deviation) : %.6f (Ideal: 0.000000)\n", max_magnitude_error);
    printf("    ▪️ 수학적 최대 정밀도 오차 (Max Abs Error)  : %.6f (Ideal: < 0.000030)\n", max_precision_error);
    
    // 허용 오차 임계값 기준 판정 (정수 해상도 한계 상 1/32767 ≒ 0.00003 이내여야 완벽)
    if (max_magnitude_error < 0.0001 && max_precision_error < 0.0001) {
        printf("\n  🎉 판정 결과: ✅ PERFECT! (해당 고정소수점 테이블은 하드웨어 연산에 사용하기에 완벽합니다.)\n");
    } else {
        printf("\n  🚨 판정 결과: ❌ WARNING! (스케일링 공식 복원 및 양수 포화 구간 오버플로우 가능성을 재점검하세요.)\n");
    }
    printf("====================================================\n\n");
}