#include "myfft.h"
#include <math.h>

#ifndef PI
#define PI 3.14159265358979323846
#endif

/**
 * @brief 프로그램 초기화 단계에서 메모리를 1회만 연산하여 LUT 테이블 생성
 * @details 
 * - 레이더 신호 처리에 필요한 모든 크기(4096 ~ 16)의 고정 배열을 사전에 계산합니다.
 */

// Bit-Reversal 및 Twiddle Factor 초기화를 동시에 수행하는 통합 매크로
#define INIT_RADAR_LUT(N, BITS) \
    /* 1. Bit-Reversal */ \
    for (int i = 0; i < (N); i++) { \
        int j = 0; \
        for (int b = 0; b < (BITS); b++) { \
            if (i & (1 << b)) j |= (1 << ((BITS - 1) - b)); \
        } \
        bitrev_##N[i] = j; \
    } \
    /* 2. Twiddle Factor (float 최적화 적용) */ \
    for (int i = 0; i < (N) / 2; i++) { \
        float angle = -2.0f * (float)PI * i / (float)(N); \
        twiddle_##N[i][0] = cosf(angle); \
        twiddle_##N[i][1] = sinf(angle); \
    }

void init_resources() {
    // =========================================================================
    // 자동 초기화 매크로 호출 (크기, 비트 수)
    // =========================================================================
    INIT_RADAR_LUT(4096, 12)
    INIT_RADAR_LUT(2048, 11)
    INIT_RADAR_LUT(1024, 10)
    INIT_RADAR_LUT(512,  9)
    INIT_RADAR_LUT(256,  8)
    INIT_RADAR_LUT(128,  7)
    INIT_RADAR_LUT(64,   6)
    INIT_RADAR_LUT(32,   5)
    INIT_RADAR_LUT(16,   4)
}

#undef INIT_RADAR_LUT