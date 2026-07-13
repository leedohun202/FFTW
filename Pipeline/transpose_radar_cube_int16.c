#include "radar_config.h"
#include <stdint.h>
#include <stddef.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __aarch64__
#include <arm_neon.h>

// 🔥 ARM64 전용: 8x8 Int16 레지스터 초고속 행렬 전치(Transpose) 매크로
#define TRANSPOSE8x8_S16(v0, v1, v2, v3, v4, v5, v6, v7) do { \
    int16x8_t t0 = vtrn1q_s16(v0, v1); \
    int16x8_t t1 = vtrn2q_s16(v0, v1); \
    int16x8_t t2 = vtrn1q_s16(v2, v3); \
    int16x8_t t3 = vtrn2q_s16(v2, v3); \
    int16x8_t t4 = vtrn1q_s16(v4, v5); \
    int16x8_t t5 = vtrn2q_s16(v4, v5); \
    int16x8_t t6 = vtrn1q_s16(v6, v7); \
    int16x8_t t7 = vtrn2q_s16(v6, v7); \
    int32x4_t u0 = vtrn1q_s32(vreinterpretq_s32_s16(t0), vreinterpretq_s32_s16(t2)); \
    int32x4_t u1 = vtrn2q_s32(vreinterpretq_s32_s16(t0), vreinterpretq_s32_s16(t2)); \
    int32x4_t u2 = vtrn1q_s32(vreinterpretq_s32_s16(t1), vreinterpretq_s32_s16(t3)); \
    int32x4_t u3 = vtrn2q_s32(vreinterpretq_s32_s16(t1), vreinterpretq_s32_s16(t3)); \
    int32x4_t u4 = vtrn1q_s32(vreinterpretq_s32_s16(t4), vreinterpretq_s32_s16(t6)); \
    int32x4_t u5 = vtrn2q_s32(vreinterpretq_s32_s16(t4), vreinterpretq_s32_s16(t6)); \
    int32x4_t u6 = vtrn1q_s32(vreinterpretq_s32_s16(t5), vreinterpretq_s32_s16(t7)); \
    int32x4_t u7 = vtrn2q_s32(vreinterpretq_s32_s16(t5), vreinterpretq_s32_s16(t7)); \
    v0 = vreinterpretq_s16_s64(vtrn1q_s64(vreinterpretq_s64_s32(u0), vreinterpretq_s64_s32(u4))); \
    v1 = vreinterpretq_s16_s64(vtrn1q_s64(vreinterpretq_s64_s32(u2), vreinterpretq_s64_s32(u6))); \
    v2 = vreinterpretq_s16_s64(vtrn1q_s64(vreinterpretq_s64_s32(u1), vreinterpretq_s64_s32(u5))); \
    v3 = vreinterpretq_s16_s64(vtrn1q_s64(vreinterpretq_s64_s32(u3), vreinterpretq_s64_s32(u7))); \
    v4 = vreinterpretq_s16_s64(vtrn2q_s64(vreinterpretq_s64_s32(u0), vreinterpretq_s64_s32(u4))); \
    v5 = vreinterpretq_s16_s64(vtrn2q_s64(vreinterpretq_s64_s32(u2), vreinterpretq_s64_s32(u6))); \
    v6 = vreinterpretq_s16_s64(vtrn2q_s64(vreinterpretq_s64_s32(u1), vreinterpretq_s64_s32(u5))); \
    v7 = vreinterpretq_s16_s64(vtrn2q_s64(vreinterpretq_s64_s32(u3), vreinterpretq_s64_s32(u7))); \
} while(0)
#endif

/**
 * @brief int16 고정소수점 전용 3D 레이더 큐브 전치 및 창함수 적용 (NEON 8x8 레지스터 블록 가속)
 */
void transpose_radar_cube_int16(const int16_t *__restrict__ in_real, const int16_t *__restrict__ in_imag,
                                int16_t *__restrict__ out_real, int16_t *__restrict__ out_imag,
                                int n_samples, int n_chirps, const int16_t *__restrict__ win_int16) {
    
    #pragma omp parallel for collapse(3)
    for (int ant = 0; ant < N_ANTENNAS; ant++) {
        for (int c_blk = 0; c_blk < n_chirps; c_blk += TILE_SIZE_INT16) {
            for (int r_blk = 0; r_blk < n_samples; r_blk += TILE_SIZE_INT16) {
                
                // 32x32 타일 내부를 다시 8x8 NEON 블록으로 잘게 쪼개어 처리
                for (int c_chunk = c_blk; c_chunk < c_blk + TILE_SIZE_INT16; c_chunk += 8) {
                    for (int r_chunk = r_blk; r_chunk < r_blk + TILE_SIZE_INT16; r_chunk += 8) {
                        
#ifdef __aarch64__
                        int in_base  = ant * (n_chirps * n_samples);
                        int out_base = ant * (n_samples * n_chirps);

                        // 1. [메모리 로드] 8개의 연속된 샘플(r)을 8개의 처프(c) 행에 걸쳐 읽어옴
                        int16x8_t vr0 = vld1q_s16(&in_real[in_base + (c_chunk + 0) * n_samples + r_chunk]);
                        int16x8_t vr1 = vld1q_s16(&in_real[in_base + (c_chunk + 1) * n_samples + r_chunk]);
                        int16x8_t vr2 = vld1q_s16(&in_real[in_base + (c_chunk + 2) * n_samples + r_chunk]);
                        int16x8_t vr3 = vld1q_s16(&in_real[in_base + (c_chunk + 3) * n_samples + r_chunk]);
                        int16x8_t vr4 = vld1q_s16(&in_real[in_base + (c_chunk + 4) * n_samples + r_chunk]);
                        int16x8_t vr5 = vld1q_s16(&in_real[in_base + (c_chunk + 5) * n_samples + r_chunk]);
                        int16x8_t vr6 = vld1q_s16(&in_real[in_base + (c_chunk + 6) * n_samples + r_chunk]);
                        int16x8_t vr7 = vld1q_s16(&in_real[in_base + (c_chunk + 7) * n_samples + r_chunk]);

                        int16x8_t vi0 = vld1q_s16(&in_imag[in_base + (c_chunk + 0) * n_samples + r_chunk]);
                        int16x8_t vi1 = vld1q_s16(&in_imag[in_base + (c_chunk + 1) * n_samples + r_chunk]);
                        int16x8_t vi2 = vld1q_s16(&in_imag[in_base + (c_chunk + 2) * n_samples + r_chunk]);
                        int16x8_t vi3 = vld1q_s16(&in_imag[in_base + (c_chunk + 3) * n_samples + r_chunk]);
                        int16x8_t vi4 = vld1q_s16(&in_imag[in_base + (c_chunk + 4) * n_samples + r_chunk]);
                        int16x8_t vi5 = vld1q_s16(&in_imag[in_base + (c_chunk + 5) * n_samples + r_chunk]);
                        int16x8_t vi6 = vld1q_s16(&in_imag[in_base + (c_chunk + 6) * n_samples + r_chunk]);
                        int16x8_t vi7 = vld1q_s16(&in_imag[in_base + (c_chunk + 7) * n_samples + r_chunk]);

                        // 2. [초고속 윈도잉] vqdmulhq_s16: 곱셈 후 자동으로 >> 15 (반올림 포함) 처리하는 사기 명령어
                        if (win_int16 != NULL) {
                            vr0 = vqdmulhq_s16(vr0, vdupq_n_s16(win_int16[c_chunk + 0]));
                            vr1 = vqdmulhq_s16(vr1, vdupq_n_s16(win_int16[c_chunk + 1]));
                            vr2 = vqdmulhq_s16(vr2, vdupq_n_s16(win_int16[c_chunk + 2]));
                            vr3 = vqdmulhq_s16(vr3, vdupq_n_s16(win_int16[c_chunk + 3]));
                            vr4 = vqdmulhq_s16(vr4, vdupq_n_s16(win_int16[c_chunk + 4]));
                            vr5 = vqdmulhq_s16(vr5, vdupq_n_s16(win_int16[c_chunk + 5]));
                            vr6 = vqdmulhq_s16(vr6, vdupq_n_s16(win_int16[c_chunk + 6]));
                            vr7 = vqdmulhq_s16(vr7, vdupq_n_s16(win_int16[c_chunk + 7]));

                            vi0 = vqdmulhq_s16(vi0, vdupq_n_s16(win_int16[c_chunk + 0]));
                            vi1 = vqdmulhq_s16(vi1, vdupq_n_s16(win_int16[c_chunk + 1]));
                            vi2 = vqdmulhq_s16(vi2, vdupq_n_s16(win_int16[c_chunk + 2]));
                            vi3 = vqdmulhq_s16(vi3, vdupq_n_s16(win_int16[c_chunk + 3]));
                            vi4 = vqdmulhq_s16(vi4, vdupq_n_s16(win_int16[c_chunk + 4]));
                            vi5 = vqdmulhq_s16(vi5, vdupq_n_s16(win_int16[c_chunk + 5]));
                            vi6 = vqdmulhq_s16(vi6, vdupq_n_s16(win_int16[c_chunk + 6]));
                            vi7 = vqdmulhq_s16(vi7, vdupq_n_s16(win_int16[c_chunk + 7]));
                        }

                        // 3. [레지스터 전치] 8x8 블록의 행과 열을 뒤집음
                        TRANSPOSE8x8_S16(vr0, vr1, vr2, vr3, vr4, vr5, vr6, vr7);
                        TRANSPOSE8x8_S16(vi0, vi1, vi2, vi3, vi4, vi5, vi6, vi7);

                        // 4. [메모리 스토어] 전치된 상태이므로, 이제 r을 행으로 삼아 8개의 연속된 샘플(c)을 한 번에 저장!
                        vst1q_s16(&out_real[out_base + (r_chunk + 0) * n_chirps + c_chunk], vr0);
                        vst1q_s16(&out_real[out_base + (r_chunk + 1) * n_chirps + c_chunk], vr1);
                        vst1q_s16(&out_real[out_base + (r_chunk + 2) * n_chirps + c_chunk], vr2);
                        vst1q_s16(&out_real[out_base + (r_chunk + 3) * n_chirps + c_chunk], vr3);
                        vst1q_s16(&out_real[out_base + (r_chunk + 4) * n_chirps + c_chunk], vr4);
                        vst1q_s16(&out_real[out_base + (r_chunk + 5) * n_chirps + c_chunk], vr5);
                        vst1q_s16(&out_real[out_base + (r_chunk + 6) * n_chirps + c_chunk], vr6);
                        vst1q_s16(&out_real[out_base + (r_chunk + 7) * n_chirps + c_chunk], vr7);

                        vst1q_s16(&out_imag[out_base + (r_chunk + 0) * n_chirps + c_chunk], vi0);
                        vst1q_s16(&out_imag[out_base + (r_chunk + 1) * n_chirps + c_chunk], vi1);
                        vst1q_s16(&out_imag[out_base + (r_chunk + 2) * n_chirps + c_chunk], vi2);
                        vst1q_s16(&out_imag[out_base + (r_chunk + 3) * n_chirps + c_chunk], vi3);
                        vst1q_s16(&out_imag[out_base + (r_chunk + 4) * n_chirps + c_chunk], vi4);
                        vst1q_s16(&out_imag[out_base + (r_chunk + 5) * n_chirps + c_chunk], vi5);
                        vst1q_s16(&out_imag[out_base + (r_chunk + 6) * n_chirps + c_chunk], vi6);
                        vst1q_s16(&out_imag[out_base + (r_chunk + 7) * n_chirps + c_chunk], vi7);
#else
                        // 비 ARM 환경을 위한 스칼라 Fallback
                        for (int c = c_chunk; c < c_chunk + 8; c++) {
                            int32_t w = (win_int16 != NULL) ? win_int16[c] : 32767;
                            for (int r = r_chunk; r < r_chunk + 8; r++) {
                                int in_idx = ant * (n_chirps * n_samples) + c * n_samples + r;
                                int out_idx = ant * (n_samples * n_chirps) + r * n_chirps + c;

                                if (win_int16 != NULL) {
                                    out_real[out_idx] = (int16_t)(((int32_t)in_real[in_idx] * w + 16384) >> 15);
                                    out_imag[out_idx] = (int16_t)(((int32_t)in_imag[in_idx] * w + 16384) >> 15);
                                } else {
                                    out_real[out_idx] = in_real[in_idx];
                                    out_imag[out_idx] = in_imag[in_idx];
                                }
                            }
                        }
#endif
                    }
                }
                
            }
        }
    }
}