#ifndef RADAR_MOCK_DATA_H
#define RADAR_MOCK_DATA_H

#include "radar_test.h" // 💥 TestCase 구조체 참조를 위해 추가

// 기존의 고정형 함수 선언
void generate_radar_mock_data(float *lut_r, float *lut_i, int n_samples, int n_chirps);

// 💥 [추가] 시나리오 기반 동적 테스트 데이터 생성 함수
void generate_radar_test_case_data(float *lut_r, float *lut_i, int n_samples, int n_chirps, TestCase tc);

#endif // RADAR_MOCK_DATA_H