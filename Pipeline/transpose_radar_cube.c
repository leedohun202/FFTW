/**
 * @brief 3D 레이더 큐브 메모리 전치 (Transpose) - ARM NEON 4x4 가속 + 세그폴트 가드 완비
 */
#include "radar_config.h"
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __aarch64__
#include <arm_neon.h>

// 🔥 ARM64 전용: 4x4 Float 레지스터 초고속 행렬 전치(Transpose) 매크로
// 32비트 단위 교환(vtrn_f32) 후 64비트 단위 교환(vtrn_u64)으로 4x4 전치 완성
#define TRANSPOSE4x4_F32(v0, v1, v2, v3) do { \
    float32x4_t q0 = vtrn1q_f32(v0, v1); \
    float32x4_t q1 = vtrn2q_f32(v0, v1); \
    float32x4_t q2 = vtrn1q_f32(v2, v3); \
    float32x4_t q3 = vtrn2q_f32(v2, v3); \
    v0 = vreinterpretq_f32_u64(vtrn1q_u64(vreinterpretq_u64_f32(q0), vreinterpretq_u64_f32(q2))); \
    v1 = vreinterpretq_f32_u64(vtrn1q_u64(vreinterpretq_u64_f32(q1), vreinterpretq_u64_f32(q3))); \
    v2 = vreinterpretq_f32_u64(vtrn2q_u64(vreinterpretq_u64_f32(q0), vreinterpretq_u64_f32(q2))); \
    v3 = vreinterpretq_f32_u64(vtrn2q_u64(vreinterpretq_u64_f32(q1), vreinterpretq_u64_f32(q3))); \
} while(0)
#endif

void transpose_radar_cube(const float *__restrict__ src_real, const float *__restrict__ src_imag,
                          float *__restrict__ dst_real, float *__restrict__ dst_imag, 
                          int n_samples, int n_chirps, const float *__restrict__ win) {

    // OpenMP 스레드들이 외부 3중 블록 루프를 쪼개어 병렬 처리
    #pragma omp parallel for collapse(3)
    for (int ant = 0; ant < N_ANTENNAS; ant++) {
        for (int c_blk = 0; c_blk < n_chirps; c_blk += TILE_SIZE) {
            for (int r_blk = 0; r_blk < n_samples; r_blk += TILE_SIZE) {
                
                // 💥 4x4 청크(Chunk) 단위로 타일 내부를 순회
                for (int c_chunk = c_blk; c_chunk < c_blk + TILE_SIZE; c_chunk += 4) {
                    if (c_chunk >= n_chirps) break;
                    for (int r_chunk = r_blk; r_chunk < r_blk + TILE_SIZE; r_chunk += 4) {
                        if (r_chunk >= n_samples) break;
                        
#ifdef __aarch64__
                        int src_base = ant * (n_chirps * n_samples);
                        int dst_base = ant * (n_samples * n_chirps);

                        // 1. 연속된 4개의 Float 데이터 로드 (Real & Imag)
                        float32x4_t vr0 = vld1q_f32(&src_real[src_base + (c_chunk + 0) * n_samples + r_chunk]);
                        float32x4_t vr1 = vld1q_f32(&src_real[src_base + (c_chunk + 1) * n_samples + r_chunk]);
                        float32x4_t vr2 = vld1q_f32(&src_real[src_base + (c_chunk + 2) * n_samples + r_chunk]);
                        float32x4_t vr3 = vld1q_f32(&src_real[src_base + (c_chunk + 3) * n_samples + r_chunk]);

                        float32x4_t vi0 = vld1q_f32(&src_imag[src_base + (c_chunk + 0) * n_samples + r_chunk]);
                        float32x4_t vi1 = vld1q_f32(&src_imag[src_base + (c_chunk + 1) * n_samples + r_chunk]);
                        float32x4_t vi2 = vld1q_f32(&src_imag[src_base + (c_chunk + 2) * n_samples + r_chunk]);
                        float32x4_t vi3 = vld1q_f32(&src_imag[src_base + (c_chunk + 3) * n_samples + r_chunk]);

                        // 2. 🛡️ Window 함수 벡터 곱셈 (Segfault 방지)
                        if (win != NULL) {
                            // vdupq_n_f32를 통해 스칼라 창함수 값을 4개 레인 전체로 브로드캐스트하여 곱함
                            vr0 = vmulq_f32(vr0, vdupq_n_f32(win[c_chunk + 0]));
                            vr1 = vmulq_f32(vr1, vdupq_n_f32(win[c_chunk + 1]));
                            vr2 = vmulq_f32(vr2, vdupq_n_f32(win[c_chunk + 2]));
                            vr3 = vmulq_f32(vr3, vdupq_n_f32(win[c_chunk + 3]));

                            vi0 = vmulq_f32(vi0, vdupq_n_f32(win[c_chunk + 0]));
                            vi1 = vmulq_f32(vi1, vdupq_n_f32(win[c_chunk + 1]));
                            vi2 = vmulq_f32(vi2, vdupq_n_f32(win[c_chunk + 2]));
                            vi3 = vmulq_f32(vi3, vdupq_n_f32(win[c_chunk + 3]));
                        }

                        // 3. 4x4 레지스터 행렬 전치!
                        TRANSPOSE4x4_F32(vr0, vr1, vr2, vr3);
                        TRANSPOSE4x4_F32(vi0, vi1, vi2, vi3);

                        // 4. 전치 완료된 결과물을 메모리에 연속 쓰기 (스트리밍)
                        vst1q_f32(&dst_real[dst_base + (r_chunk + 0) * n_chirps + c_chunk], vr0);
                        vst1q_f32(&dst_real[dst_base + (r_chunk + 1) * n_chirps + c_chunk], vr1);
                        vst1q_f32(&dst_real[dst_base + (r_chunk + 2) * n_chirps + c_chunk], vr2);
                        vst1q_f32(&dst_real[dst_base + (r_chunk + 3) * n_chirps + c_chunk], vr3);

                        vst1q_f32(&dst_imag[dst_base + (r_chunk + 0) * n_chirps + c_chunk], vi0);
                        vst1q_f32(&dst_imag[dst_base + (r_chunk + 1) * n_chirps + c_chunk], vi1);
                        vst1q_f32(&dst_imag[dst_base + (r_chunk + 2) * n_chirps + c_chunk], vi2);
                        vst1q_f32(&dst_imag[dst_base + (r_chunk + 3) * n_chirps + c_chunk], vi3);
#else
                        // 비 ARM 환경을 위한 기존 스칼라 폴백 (Fallback)
                        for (int c = c_chunk; c < c_chunk + 4; c++) {
                            if (c >= n_chirps) break;
                            float w = (win == NULL) ? 1.0f : win[c];
                            for (int r = r_chunk; r < r_chunk + 4; r++) {
                                if (r >= n_samples) break;
                                int src_idx = ant * (n_chirps * n_samples) + c * n_samples + r;
                                int dst_idx = ant * (n_samples * n_chirps) + r * n_chirps + c;
                                
                                dst_real[dst_idx] = src_real[src_idx] * w;
                                dst_imag[dst_idx] = src_imag[src_idx] * w;
                            }
                        }
#endif
                    }
                }
                
            }
        }
    }
}