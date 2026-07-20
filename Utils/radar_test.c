#include "radar_test.h"
#include <stddef.h>

TestCase edge_cases[] = {
    {
        .case_name = "1. Angular Resolution Test (Same R, Same V, Different Angle)",
        .num_targets = 2,
        .targets = {
            {.r = 15.0, .v = 5.0, .angle = -20.0, .snr_db = 55.0f},
            {.r = 15.0, .v = 5.0, .angle =  20.0, .snr_db = 55.0f} // 💥 각도만 다름
        }
    },
    {
        .case_name = "2. Near-Far Target Masking Test (Truck & Pedestrian)",
        .num_targets = 2,
        .targets = {
            {.r = 8.0,  .v = 0.0, .angle = 0.0, .snr_db = 70.0f},  // 💥 강한 반사체 (트럭)
            {.r = 9.5,  .v = 1.0, .angle = 5.0, .snr_db = 18.0f}   // 💥 근접한 약한 반사체 (보행자)
        }
    },
    {
        .case_name = "3. Multiple Fast Crossing Targets",
        .num_targets = 2,
        .targets = {
            {.r = 12.0, .v =  22.0, .angle = -15.0, .snr_db = 45.0f}, // 💥 접근 타겟
            {.r = 12.5, .v = -22.0, .angle =  15.0, .snr_db = 45.0f}  // 💥 멀어지는 타겟 (교차)
        }
    },
    {
        .case_name = "4. Pure Noise / Zero Target Environment",
        .num_targets = 0, // 💥 아무것도 없음 (오탐지 방어선 테스트)
    }
}

;int num_edge_cases = sizeof(edge_cases) / sizeof(TestCase);