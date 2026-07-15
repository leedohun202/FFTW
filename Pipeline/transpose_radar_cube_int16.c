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

void transpose_radar_cube_int16(const int16_t *__restrict__ in_real, const int16_t *__restrict__ in_imag,
                                int16_t *__restrict__ out_real, int16_t *__restrict__ out_imag,
                                int n_samples, int n_chirps, const int16_t *__restrict__ win_int16) {
    
    #pragma omp parallel for collapse(3)
    for (int ant = 0; ant < N_ANTENNAS; ant++) {
        for (int c_blk = 0; c_blk < n_chirps; c_blk += TILE_SIZE_INT16) {
            for (int r_blk = 0; r_blk < n_samples; r_blk += TILE_SIZE_INT16) {
                
                for (int c_chunk = c_blk; c_chunk < c_blk + TILE_SIZE_INT16; c_chunk += 8) {
                    for (int r_chunk = r_blk; r_chunk < r_blk + TILE_SIZE_INT16; r_chunk += 8) {
                        
#ifdef __aarch64__
                        int in_base  = ant * (n_chirps * n_samples);
                        int out_base = ant * (n_samples * n_chirps);

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

                        TRANSPOSE8x8_S16(vr0, vr1, vr2, vr3, vr4, vr5, vr6, vr7);
                        TRANSPOSE8x8_S16(vi0, vi1, vi2, vi3, vi4, vi5, vi6, vi7);

                        // 💥 [핵심 패치] Inter-stage Gain (심폐소생술)
                        // Range FFT 과정에서 깎여나간 보행자 신호를 살리기 위해 8배(<< 3) 증폭합니다.
                        // vqshlq_n_s16는 포화 연산(Saturating)을 지원하여 오버플로우 시 32767로 깎아줍니다.
                        vr0 = vqshlq_n_s16(vr0, 3); vr1 = vqshlq_n_s16(vr1, 3);
                        vr2 = vqshlq_n_s16(vr2, 3); vr3 = vqshlq_n_s16(vr3, 3);
                        vr4 = vqshlq_n_s16(vr4, 3); vr5 = vqshlq_n_s16(vr5, 3);
                        vr6 = vqshlq_n_s16(vr6, 3); vr7 = vqshlq_n_s16(vr7, 3);

                        vi0 = vqshlq_n_s16(vi0, 3); vi1 = vqshlq_n_s16(vi1, 3);
                        vi2 = vqshlq_n_s16(vi2, 3); vi3 = vqshlq_n_s16(vi3, 3);
                        vi4 = vqshlq_n_s16(vi4, 3); vi5 = vqshlq_n_s16(vi5, 3);
                        vi6 = vqshlq_n_s16(vi6, 3); vi7 = vqshlq_n_s16(vi7, 3);

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
                        for (int c = c_chunk; c < c_chunk + 8; c++) {
                            int32_t w = (win_int16 != NULL) ? win_int16[c] : 32767;
                            for (int r = r_chunk; r < r_chunk + 8; r++) {
                                int in_idx = ant * (n_chirps * n_samples) + c * n_samples + r;
                                int out_idx = ant * (n_samples * n_chirps) + r * n_chirps + c;
                                int32_t temp_r, temp_i;

                                if (win_int16 != NULL) {
                                    temp_r = ((int32_t)in_real[in_idx] * w + 16384) >> 15;
                                    temp_i = ((int32_t)in_imag[in_idx] * w + 16384) >> 15;
                                } else {
                                    temp_r = in_real[in_idx];
                                    temp_i = in_imag[in_idx];
                                }

                                // 💥 [핵심 패치] 스칼라 환경 심폐소생술 (포화 연산 포함)
                                temp_r <<= 2; temp_i <<= 2;
                                if (temp_r > 32767) temp_r = 32767; else if (temp_r < -32768) temp_r = -32768;
                                if (temp_i > 32767) temp_i = 32767; else if (temp_i < -32768) temp_i = -32768;

                                out_real[out_idx] = (int16_t)temp_r;
                                out_imag[out_idx] = (int16_t)temp_i;
                            }
                        }
#endif
                    }
                }
                
            }
        }
    }
}