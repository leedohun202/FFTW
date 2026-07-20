#include "myfft.h"

/**
 * @brief 64 포인트 1D 고속 푸리에 변환 (극한의 루프 풀기 및 NEON 가속 버전)
 */
void custom_fft_64_fixed(float *__restrict__ real, float *__restrict__ imag) {
    
    // 1. Bit-Reversal (64포인트는 오버헤드가 작으므로 LUT 그대로 고수)
    for (int i = 0; i < 64; i++) { 
        int j = bitrev_64[i]; 
        if (i < j) { 
            float tr = real[i]; real[i] = real[j]; real[j] = tr; 
            float ti = imag[i]; imag[i] = imag[j]; imag[j] = ti; 
        } 
    }

    // 2-1. 나비 연산 초기 단계 (step < 4: 1단계와 2단계는 스칼라 연산)
    for (int step = 1; step < 4; step *= 2) { 
        const int jump = step * 2; 
        const int twiddle_step = 64 / jump;
        for (int i = 0; i < 64; i += jump) { 
            for (int j = 0; j < step; j++) { 
                int curr = i + j; int k = curr + step; 
                float tr = twiddle_real_64[j * twiddle_step]; 
                float ti = twiddle_imag_64[j * twiddle_step]; 
                float t_real = real[k] * tr - imag[k] * ti; 
                float t_imag = real[k] * ti + imag[k] * tr; 
                real[k] = real[curr] - t_real; imag[k] = imag[curr] - t_imag; 
                real[curr] = real[curr] + t_real; imag[curr] = imag[curr] + t_imag; 
            } 
        } 
    }

#ifdef __aarch64__
    // 2-2. ⚡ 극한의 가속 구간: 루프 구조를 완전히 파괴하고 평탄화 (Unrolling)
    // 매크로 함수를 정의하여 타이핑 오버헤드와 가독성을 동시에 잡습니다.
    #define NEON_BUTTERFLY_64(curr_idx, k_idx, t_idx) { \
        float32x4_t vr_curr = vld1q_f32(&real[curr_idx]); \
        float32x4_t vi_curr = vld1q_f32(&imag[curr_idx]); \
        float32x4_t vr_k    = vld1q_f32(&real[k_idx]); \
        float32x4_t vi_k    = vld1q_f32(&imag[k_idx]); \
        float32x4_t v_tr    = vld1q_f32(&twiddle_real_64[t_idx]); \
        float32x4_t v_ti    = vld1q_f32(&twiddle_imag_64[t_idx]); \
        float32x4_t vt_real = vsubq_f32(vmulq_f32(vr_k, v_tr), vmulq_f32(vi_k, v_ti)); \
        float32x4_t vt_imag = vaddq_f32(vmulq_f32(vr_k, v_ti), vmulq_f32(vi_k, v_tr)); \
        vst1q_f32(&real[k_idx], vsubq_f32(vr_curr, vt_real)); \
        vst1q_f32(&imag[k_idx], vsubq_f32(vi_curr, vt_imag)); \
        vst1q_f32(&real[curr_idx], vaddq_f32(vr_curr, vt_real)); \
        vst1q_f32(&imag[curr_idx], vaddq_f32(vi_curr, vt_imag)); \
    }

    #define NEON_BUTTERFLY_STRIDED_64(curr_idx, k_idx, t0, t1, t2, t3) { \
        float32x4_t vr_curr = vld1q_f32(&real[curr_idx]); \
        float32x4_t vi_curr = vld1q_f32(&imag[curr_idx]); \
        float32x4_t vr_k    = vld1q_f32(&real[k_idx]); \
        float32x4_t vi_k    = vld1q_f32(&imag[k_idx]); \
        float32x4_t v_tr = vdupq_n_f32(0.0f); \
        v_tr = vsetq_lane_f32(twiddle_real_64[t0], v_tr, 0); \
        v_tr = vsetq_lane_f32(twiddle_real_64[t1], v_tr, 1); \
        v_tr = vsetq_lane_f32(twiddle_real_64[t2], v_tr, 2); \
        v_tr = vsetq_lane_f32(twiddle_real_64[t3], v_tr, 3); \
        float32x4_t v_ti = vdupq_n_f32(0.0f); \
        v_ti = vsetq_lane_f32(twiddle_imag_64[t0], v_ti, 0); \
        v_ti = vsetq_lane_f32(twiddle_imag_64[t1], v_ti, 1); \
        v_ti = vsetq_lane_f32(twiddle_imag_64[t2], v_ti, 2); \
        v_ti = vsetq_lane_f32(twiddle_imag_64[t3], v_ti, 3); \
        float32x4_t vt_real = vsubq_f32(vmulq_f32(vr_k, v_tr), vmulq_f32(vi_k, v_ti)); \
        float32x4_t vt_imag = vaddq_f32(vmulq_f32(vr_k, v_ti), vmulq_f32(vi_k, v_tr)); \
        vst1q_f32(&real[k_idx], vsubq_f32(vr_curr, vt_real)); \
        vst1q_f32(&imag[k_idx], vsubq_f32(vi_curr, vt_imag)); \
        vst1q_f32(&real[curr_idx], vaddq_f32(vr_curr, vt_real)); \
        vst1q_f32(&imag[curr_idx], vaddq_f32(vi_curr, vt_imag)); \
    }

    // --- STAGE 3: step = 4 (jump = 8, twiddle_step = 8) ---
    // i = 0, 8, 16, 24, 32, 40, 48, 56 / j는 항상 0 한 번만 돎
    NEON_BUTTERFLY_STRIDED_64( 0,  4,  0,  8, 16, 24);
    NEON_BUTTERFLY_STRIDED_64( 8, 12,  0,  8, 16, 24);
    NEON_BUTTERFLY_STRIDED_64(16, 20,  0,  8, 16, 24);
    NEON_BUTTERFLY_STRIDED_64(24, 28,  0,  8, 16, 24);
    NEON_BUTTERFLY_STRIDED_64(32, 36,  0,  8, 16, 24);
    NEON_BUTTERFLY_STRIDED_64(40, 44,  0,  8, 16, 24);
    NEON_BUTTERFLY_STRIDED_64(48, 52,  0,  8, 16, 24);
    NEON_BUTTERFLY_STRIDED_64(56, 60,  0,  8, 16, 24);

    // --- STAGE 4: step = 8 (jump = 16, twiddle_step = 4) ---
    // i = 0, 16, 32, 48 / j = 0, 4
    NEON_BUTTERFLY_STRIDED_64( 0,  8,  0,  4,  8, 12); NEON_BUTTERFLY_STRIDED_64( 4, 12, 16, 20, 24, 28);
    NEON_BUTTERFLY_STRIDED_64(16, 24,  0,  4,  8, 12); NEON_BUTTERFLY_STRIDED_64(20, 28, 16, 20, 24, 28);
    NEON_BUTTERFLY_STRIDED_64(32, 40,  0,  4,  8, 12); NEON_BUTTERFLY_STRIDED_64(36, 44, 16, 20, 24, 28);
    NEON_BUTTERFLY_STRIDED_64(48, 56,  0,  4,  8, 12); NEON_BUTTERFLY_STRIDED_64(52, 60, 16, 20, 24, 28);

    // --- STAGE 5: step = 16 (jump = 32, twiddle_step = 2) ---
    // i = 0, 32 / j = 0, 4, 8, 12
    NEON_BUTTERFLY_STRIDED_64( 0, 16,  0,  2,  4,  6); NEON_BUTTERFLY_STRIDED_64( 4, 20,  8, 10, 12, 14);
    NEON_BUTTERFLY_STRIDED_64( 8, 24, 16, 18, 20, 22); NEON_BUTTERFLY_STRIDED_64(12, 28, 24, 26, 28, 30);
    
    NEON_BUTTERFLY_STRIDED_64(32, 48,  0,  2,  4,  6); NEON_BUTTERFLY_STRIDED_64(36, 52,  8, 10, 12, 14);
    NEON_BUTTERFLY_STRIDED_64(40, 56, 16, 18, 20, 22); NEON_BUTTERFLY_STRIDED_64(44, 60, 24, 26, 28, 30);

    // --- STAGE 6: step = 32 (jump = 64, twiddle_step = 1) ---
    // i = 0 (끝) / j = 0, 4, 8, 12, 16, 20, 24, 28
    // 💡 twiddle_step이 1이므로 초고속 연속 벡터 로드(NEON_BUTTERFLY_64) 작동!
    NEON_BUTTERFLY_64( 0, 32,  0); NEON_BUTTERFLY_64( 4, 36,  4);
    NEON_BUTTERFLY_64( 8, 40,  8); NEON_BUTTERFLY_64(12, 44, 12);
    NEON_BUTTERFLY_64(16, 48, 16); NEON_BUTTERFLY_64(20, 52, 20);
    NEON_BUTTERFLY_64(24, 56, 24); NEON_BUTTERFLY_64(28, 60, 28);

    #undef NEON_BUTTERFLY_64
    #undef NEON_BUTTERFLY_STRIDED_64
#else
    // 비 ARM 환경을 위한 Fallback (일반 루프 구조 유지)
    for (int step = 4; step < 64; step *= 2) {
        const int jump = step * 2; const int twiddle_step = 64 / jump;
        for (int i = 0; i < 64; i += jump) {
            for (int j = 0; j < step; j++) {
                int curr = i + j; int k = curr + step;
                float tr = twiddle_real_64[j * twiddle_step]; float ti = twiddle_imag_64[j * twiddle_step];
                float t_real = real[k] * tr - imag[k] * ti; float t_imag = real[k] * ti + imag[k] * tr;
                real[k] = real[curr] - t_real; imag[k] = imag[curr] - t_imag;
                real[curr] = real[curr] + t_real; imag[curr] = imag[curr] + t_imag;
            }
        }
    }
#endif
}