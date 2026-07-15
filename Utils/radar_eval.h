#ifndef RADAR_EVAL_H
#define RADAR_EVAL_H

#include "radar_cfar.h"

// 정답지 없이, 레이더가 스스로 인지한 최종 물체 정보를 담을 구조체
typedef struct {
    double distance;    // 거리 (m)
    double velocity;    // 속도 (m/s)
    double angle;       // 각도 (deg)
    float snr_db;       // 신호 정밀도 지표 (SNR)
} FinalObject;

// 💥 정답지(true_R) 인자를 완전히 제거했습니다.
// 탐지된 물체의 총 개수를 반환하며, out_objects 배열에 결과를 동적으로 채웁니다.
int extract_autonomous_objects(CFAR_Target *targets, int num_det, 
                                const float *tmp_r, const float *tmp_i, 
                                int n_samples, int n_chirps, 
                                FinalObject *out_objects, int max_output);

#endif // RADAR_EVAL_H