#ifndef RADAR_TEST_H
#define RADAR_TEST_H

#include <stdint.h>

// 모의 데이터에 동적으로 주입할 타겟 구조체
typedef struct {
    double r;      // 정답 거리 (m)
    double v;      // 정답 속도 (m/s)
    double angle;  // 정답 각도 (deg)
    float snr_db;  // 주입할 신호 세기 (dB)
} TestTarget;

// 시나리오별 테스트벤치 케이스 구조체
typedef struct {
    const char *case_name;
    int num_targets;
    TestTarget targets[5]; // 케이스당 최대 5개 타겟 지원
} TestCase;

// 전체 엣지케이스 데이터셋 선언
extern TestCase edge_cases[];
extern int num_edge_cases;

#endif // RADAR_TEST_H