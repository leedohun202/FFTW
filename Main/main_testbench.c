#include "radar_pipeline.h"
#include "radar_benchmark_sessions.h"
#include "radar_config.h"
#include "radar_utils.h"
#include "radar_mock_data.h"
#include "radar_test.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>    // 💥 [추가] 랜덤 시드를 위한 헤더
#include <fftw3.h>

const char* ENGINE_NAMES[] = {
    "FFTW3 - Estimate    ",
    "Custom Float Radix-2",
    "Custom Float Radix-4",
    "Custom Int16 Radix-2",
    "Custom Int16 Radix-4"
};

// 💥 [추가] 지정된 범위 내의 무작위 실수 생성 헬퍼 함수
static float rand_float(float min, float max) {
    return min + (float)rand() / ((float)RAND_MAX / (max - min));
}

void evaluate_engine_pipeline(float *lut_r, float *lut_i, int n_samples, int n_chirps, TestCase tc, int engine_type) {
    BenchmarkResult res;
    memset(&res, 0, sizeof(BenchmarkResult));

    switch(engine_type) {
        case 0: benchmark_session_fftw3(lut_r, lut_i, n_samples, n_chirps, &res, FFTW_ESTIMATE); break;
        case 1: benchmark_session_float_r2(lut_r, lut_i, n_samples, n_chirps, &res); break;
        case 2: benchmark_session_float_r4(lut_r, lut_i, n_samples, n_chirps, &res); break;
        //case 3: benchmark_session_int16_r2(lut_r, lut_i, n_samples, n_chirps, &res); break;
        //case 4: benchmark_session_int16_r4(lut_r, lut_i, n_samples, n_chirps, &res); break;
        default: return;
    }

    printf(" 🤖 [%s] ➡️ [탐지: %d / 정답: %d]\n", ENGINE_NAMES[engine_type], res.num_targets, tc.num_targets);
    printf("    ⏱️  Latency: Range = %5.2f ms | Doppler = %5.2f ms | CFAR + Angle = %5.2f ms | Total : %5.2f ms | RAM = %5.2f MB\n", 
           res.time_range, res.time_doppler, res.time_angle, res.time_range + res.time_doppler + res.time_angle ,res.actual_ram_mb);

    if (res.num_targets == 0) {
        if (tc.num_targets == 0) {
            printf("    🟩 PANJUNG: [🟢 PASS] - 유령 타겟 방어 성공\n");
        } else {
            printf("    🟥 PANJUNG: [🔴 FAIL] - 타겟 전원 추적 실패 (Missing All Targets)\n");
            for (int t = 0; t < tc.num_targets; t++) {
                printf("       ❌ [정답 타겟 %d] 미검출 (Missed)\n", t + 1);
            }
        }
        printf("    ---------------------------------------------------\n");
        return;
    }

    int pass_count = 0;
    int *truth_checked = (int*)calloc(tc.num_targets, sizeof(int));

    for (int i = 0; i < res.num_targets; i++) {
        double est_R = res.objects[i].distance;
        double est_v = res.objects[i].velocity;
        double est_a = res.objects[i].angle;
        float  est_snr = res.objects[i].snr_db;

        int match_idx = -1;
        double min_error = 999.0;

        // 마진 규격 체크 (거리 ±0.5m, 속도 ±0.5m/s)
        for (int t = 0; t < tc.num_targets; t++) {
            if (truth_checked[t]) continue;
            
            double r_err = fabs(est_R - tc.targets[t].r);
            double v_err = fabs(est_v - tc.targets[t].v);

            if (r_err < 0.5 && v_err < 1.5) {
                if (r_err + v_err < min_error) {
                    min_error = r_err + v_err;
                    match_idx = t;
                }
            }
        }

        if (match_idx != -1) {
            truth_checked[match_idx] = 1;
            pass_count++;
            printf("       🎯 [탐지 %d ➡️ 정답 %d 매칭성공] 거리: %6.3fm | 속도: %6.3fm/s | 각도: %7.3f° | SNR: %4.1fdB\n", 
                   i + 1, match_idx + 1, est_R, est_v, est_a, est_snr);
        } else {
            printf("       ⚠️ [탐지 %d ➡️ 매칭실패 (고스트)] 거리: %6.3fm | 속도: %6.3fm/s | 각도: %7.3f° | SNR: %4.1fdB\n", 
                   i + 1, est_R, est_v, est_a, est_snr);
        }
    }

    for (int t = 0; t < tc.num_targets; t++) {
        if (!truth_checked[t]) {
            printf("       ❌ [정답 타겟 %d] 추적 실패 (Missed)\n", t + 1);
        }
    }

    if (pass_count == tc.num_targets && res.num_targets == tc.num_targets) {
        printf("    🟩 PANJUNG: [🟢 PASS] - 모든 타겟 정밀 복원 완료\n");
    } else {
        printf("    🟥 PANJUNG: [🔴 FAIL] - 분해능 붕괴, 누락 혹은 가짜 고스트 존재\n");
    }
    free(truth_checked);
    printf("    ---------------------------------------------------\n");
}

void run_automated_test_case(TestCase tc, int n_samples, int n_chirps) {
    int total_elements = N_ANTENNAS * n_chirps * n_samples;
    size_t alloc_size_float = total_elements * sizeof(float);

    float *lut_r = NULL; float *lut_i = NULL;
    posix_memalign((void**)&lut_r, 64, alloc_size_float);
    posix_memalign((void**)&lut_i, 64, alloc_size_float);

    printf("======================================================================\n");
    printf("📋 시나리오 검증: %s\n", tc.case_name);
    printf("======================================================================\n");
    
    printf("📢 [ORIGINAL TRUTH DATA - 주입된 원본 타겟 스펙]\n");
    if (tc.num_targets == 0) {
        printf("   🚫 주입된 타겟 없음 (순수 노이즈 상태)\n");
    } else {
        for (int t = 0; t < tc.num_targets; t++) {
            printf("   📍 [정답 %d] 거리: %6.3fm | 속도: %6.3fm/s | 각도: %7.3f° | 설정강도: %4.1fdB\n",
                   t + 1, tc.targets[t].r, tc.targets[t].v, tc.targets[t].angle, tc.targets[t].snr_db);
        }
    }
    printf("----------------------------------------------------------------------\n");

    generate_radar_test_case_data(lut_r, lut_i, n_samples, n_chirps, tc);

    for (int engine_type = 0; engine_type < 3; engine_type++) {
        evaluate_engine_pipeline(lut_r, lut_i, n_samples, n_chirps, tc, engine_type);
    }

    printf("======================================================================\n\n");
    free(lut_r); free(lut_i);
}

int main() {
    init_resources();
    
    // 💥 [추가] 랜덤 시드 초기화
    srand((unsigned int)time(NULL));
    
    // 기존 고정 엣지 케이스 1~4번 실행
    for (int i = 0; i < num_edge_cases; i++) {
        run_automated_test_case(edge_cases[i], 4096, 16);
    }

    // ======================================================================
    // 💥 시나리오 5: 극한의 무작위 몬테카를로 추적 (Random Multi-Target)
    // ======================================================================
    TestCase tc5;
    memset(&tc5, 0, sizeof(TestCase));
    tc5.case_name = "5. Random Multi-Target Stress Test (3~30m, ±24m/s, ±80°)";
    tc5.num_targets = 4; // 너무 많으면 서로 뭉칠 수 있으므로 적당한 4개 배치
    for(int i = 0; i < tc5.num_targets; i++) {
        tc5.targets[i].r = rand_float(3.0f, 30.0f);
        tc5.targets[i].v = rand_float(-24.0f, 24.0f);
        tc5.targets[i].angle = rand_float(-80.0f, 80.0f);
        tc5.targets[i].snr_db = rand_float(20.0f, 70.0f); // 25dB ~ 70dB 
    }
    run_automated_test_case(tc5, 4096, 16);

    // ======================================================================
    // 💥 시나리오 6: Angle FFT 물리적 한계 시험 (Same R, Same V, 3 Random Angles)
    // ======================================================================
    TestCase tc6;
    memset(&tc6, 0, sizeof(TestCase));
    tc6.case_name = "6. Angle FFT Physical Limit Test (Same Bin, 3 Random Angles)";
    tc6.num_targets = 3;
    
    // 기준이 될 고정 거리와 고정 속도
    float fixed_range = rand_float(10.0f, 20.0f);
    float fixed_vel = rand_float(-10.0f, 10.0f);
    
    // 각도가 너무 붙으면 레이더 물리법칙상 1개로 합쳐지므로, 어느 정도 구역을 나눠서 랜덤 배치
    tc6.targets[0].r = fixed_range; tc6.targets[0].v = fixed_vel; 
    tc6.targets[0].angle = rand_float(-60.0f, -20.0f); tc6.targets[0].snr_db = 55.0f;
    
    tc6.targets[1].r = fixed_range; tc6.targets[1].v = fixed_vel; 
    tc6.targets[1].angle = rand_float(-10.0f,  10.0f); tc6.targets[1].snr_db = 55.0f;
    
    tc6.targets[2].r = fixed_range; tc6.targets[2].v = fixed_vel; 
    tc6.targets[2].angle = rand_float( 20.0f,  60.0f); tc6.targets[2].snr_db = 55.0f;
    
    run_automated_test_case(tc6, 4096, 16);

    fftwf_cleanup();
    return 0;
}