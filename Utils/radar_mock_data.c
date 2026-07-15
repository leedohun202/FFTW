#include "radar_mock_data.h"
#include "radar_config.h"
#include <math.h>
#include <stdlib.h>

// 기존 함수 (하위 호환성 유지)
void generate_radar_mock_data(float *lut_r, float *lut_i, int n_samples, int n_chirps) {
    // 낡은 고정 3대장 로직 (그대로 두셔도 안전합니다)
    TestCase default_case = {
        .num_targets = 3,
        .targets = {
            {.r = 6.8,  .v = 0.0,   .angle = 0.0,   .snr_db = 50.0},
            {.r = 12.5, .v = 14.2,  .angle = -15.0, .snr_db = 50.0},
            {.r = 25.2, .v = -5.5,  .angle = 25.0,  .snr_db = 50.0}
        }
    };
    generate_radar_test_case_data(lut_r, lut_i, n_samples, n_chirps, default_case);
}

// 💥 [신규 구현] 엣지케이스 동적 시그널 인젝터 엔진
void generate_radar_test_case_data(float *lut_r, float *lut_i, int n_samples, int n_chirps, TestCase tc) {
    int total_elements = N_ANTENNAS * n_chirps * n_samples;

    // 1. 순수 화이트 가우시안 노이즈 바닥 생성 (-40dB 기본 스케일)
    for (int i = 0; i < total_elements; i++) {
        lut_r[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
        lut_i[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
    }

    // 2. 시나리오 데이터셋에 명시된 타겟들을 순차적으로 신호 주입
    for (int t = 0; t < tc.num_targets; t++) {
        double R = tc.targets[t].r;
        double v = tc.targets[t].v;
        double angle_deg = tc.targets[t].angle;
        
        // SNR(dB) 지표를 바탕으로 원본 전압 진폭(Amplitude)으로 정밀 환산
        float amplitude = (float)pow(10.0, tc.targets[t].snr_db / 20.0) * 0.001f;

        double angle_rad = angle_deg * PI / 180.0;
        double sin_theta = sin(angle_rad);

        // 안테나 채널간의 물리적 도달 위상 지연 차이(AoA 수학의 정석)
        double phase_shift_per_antenna = (2.0 * PI * d_ant * sin_theta) / lambda_c;

        for (int ant = 0; ant < N_ANTENNAS; ant++) {
            double ant_phase_offset = ant * phase_shift_per_antenna;

            for (int ch = 0; ch < n_chirps; ch++) {
                // 도플러 주파수 (속도) 변조
                double doppler_phase = 2.0 * PI * (2.0 * v / lambda_c) * ch * Tc;

                for (int n = 0; n < n_samples; n++) {
                    // 비트 주파수 (거리) 변조
                    double beat_freq = (2.0 * S * R) / c;
                    double t_t = n / Fs;
                    double range_phase = 2.0 * PI * beat_freq * t_t;

                    // 최종 합성 전파 위상 통합
                    double total_phase = range_phase + doppler_phase + ant_phase_offset;

                    int idx = ant * (n_chirps * n_samples) + ch * n_samples + n;
                    lut_r[idx] += amplitude * (float)cos(total_phase);
                    lut_i[idx] += amplitude * (float)sin(total_phase);
                }
            }
        }
    }
}