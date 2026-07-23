/*
 * myfft.c — FFTW 대체 커스텀 FFT (Custom Float Radix-4 Interleaved 버전)
 */
#include "myfft.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================ plan 내부 표현 ============================= */

typedef enum { KIND_C2C = 0, KIND_R2C } myfft_kind;

struct myfft_plan_s {
    myfft_kind kind;
    int        n;
    int        sign;

    int        howmany;
    int        istride, idist;
    int        ostride, odist;

    void* in_bound;
    fftwf_complex* out_bound;

    fftwf_complex* scratch_cpx;
    fftwf_complex* tmp_cpx;
};

/* ============================== 메모리 ================================== */
float* fftwf_alloc_real(size_t n)            { return (float*)malloc(n * sizeof(float)); }
fftwf_complex* fftwf_alloc_complex(size_t n) { return (fftwf_complex*)malloc(n * sizeof(fftwf_complex)); }
void* fftwf_malloc(size_t bytes)            { return malloc(bytes); }
void           fftwf_free(void* p)           { free(p); }

/* ============================== 유틸 =================================== */
int   myfft_is_pow2(size_t n)   { return n && ((n & (n - 1)) == 0); }
size_t myfft_next_pow2(size_t n) { size_t p = 1; while (p < n) p <<= 1; return p; }

// Fused 커널 디스패처 (32, 128, 512, 2048)
static void dispatch_neon_fft_r4_fused(const fftwf_complex *in, fftwf_complex *out, int n) {
    switch (n) {
        case 2048: custom_fft_2048_radix4_fused(in, out); break;
        case 512:  custom_fft_512_radix4_fused(in, out); break;
        case 128:  custom_fft_128_radix4_fused(in, out); break;
        case 32:   custom_fft_32_radix4_fused(in, out); break;
        default: break;
    }
}

// PURE 커널 디스패처 (16, 64, 256, 1024, 4096)
static void dispatch_neon_fft_r4_pure(fftwf_complex *data, int n) {
    switch (n) {
        case 4096: custom_fft_4096_radix4(data); break;
        case 1024: custom_fft_1024_radix4(data); break;
        case 256:  custom_fft_256_radix4(data); break;
        case 64:   custom_fft_64_radix4(data); break;
        case 16:   custom_fft_16_radix4(data); break;
        default: break;
    }
}

/* ===================== 플랜 공통 할당 ================================== */
static myfft_plan alloc_plan(myfft_kind kind, int n, int sign,
                              int howmany, int istride, int idist,
                              int ostride, int odist,
                              void* in_bound, fftwf_complex* out_bound) {
    if (n > 4096 || !myfft_is_pow2((size_t)n)) {
        fprintf(stderr, "[myfft] ERROR: N=%d is unsupported (must be pow2 and <= 4096).\n", n);
        return NULL;
    }

    if (kind == KIND_C2C && ostride != 1) {
        fprintf(stderr, "[myfft] ERROR: C2C plan currently strictly requires ostride == 1.\n");
        return NULL;
    }

    myfft_plan p = (myfft_plan)calloc(1, sizeof(struct myfft_plan_s));
    if (!p) return NULL;
    
    p->kind = kind; p->n = n; p->sign = sign; p->howmany = howmany;
    p->istride = istride; p->idist = idist;
    p->ostride = ostride; p->odist = odist;
    p->in_bound = in_bound; p->out_bound = out_bound;

    if (kind == KIND_R2C) {
        if (posix_memalign((void**)&p->scratch_cpx, 64, n * sizeof(fftwf_complex)) != 0) {
            free(p); return NULL;
        }
    } else {
        p->scratch_cpx = NULL; 
    }

    if (posix_memalign((void**)&p->tmp_cpx, 64, n * sizeof(fftwf_complex)) != 0) {
        if (p->scratch_cpx) free(p->scratch_cpx);
        free(p); return NULL;
    }

    return p;
}

/* ============================== 플랜 API ================================ */
fftwf_plan fftwf_plan_dft_1d(int n, fftwf_complex* in, fftwf_complex* out, int sign, unsigned flags) {
    (void)flags;
    return alloc_plan(KIND_C2C, n, sign, 1, 1, n, 1, n, (void*)in, out);
}

fftwf_plan fftwf_plan_many_dft(int rank, const int* n, int howmany,
                               fftwf_complex* in,  const int* inembed, int istride, int idist,
                               fftwf_complex* out, const int* onembed, int ostride, int odist,
                               int sign, unsigned flags) {
    (void)rank; (void)inembed; (void)onembed; (void)flags;
    return alloc_plan(KIND_C2C, n[0], sign, howmany, istride, idist, ostride, odist, (void*)in, out);
}

fftwf_plan fftwf_plan_many_dft_r2c(int rank, const int* n, int howmany,
                                   float* in,  const int* inembed, int istride, int idist,
                                   fftwf_complex* out, const int* onembed, int ostride, int odist,
                                   unsigned flags) {
    (void)rank; (void)inembed; (void)onembed; (void)flags;
    return alloc_plan(KIND_R2C, n[0], FFTW_FORWARD, howmany, istride, idist, ostride, odist, (void*)in, out);
}

void fftwf_destroy_plan(fftwf_plan p) {
    if (!p) return;
    if (p->scratch_cpx) free(p->scratch_cpx);
    if (p->tmp_cpx)     free(p->tmp_cpx);
    free(p);
}

// =========================================================================
// ⚡ N=16 Direct NEON Ultra-Fast Path (AArch64 / ARMv7 NEON 공용)
// =========================================================================
#if defined(__ARM_NEON)
static inline void custom_fft_16_radix4_direct_fast(const fftwf_complex *in, fftwf_complex *out, int istride) {
    out[0][0]  = in[0  * istride][0]; out[0][1]  = in[0  * istride][1];
    out[1][0]  = in[4  * istride][0]; out[1][1]  = in[4  * istride][1];
    out[2][0]  = in[8  * istride][0]; out[2][1]  = in[8  * istride][1];
    out[3][0]  = in[12 * istride][0]; out[3][1]  = in[12 * istride][1];

    out[4][0]  = in[1  * istride][0]; out[4][1]  = in[1  * istride][1];
    out[5][0]  = in[5  * istride][0]; out[5][1]  = in[5  * istride][1];
    out[6][0]  = in[9  * istride][0]; out[6][1]  = in[9  * istride][1];
    out[7][0]  = in[13 * istride][0]; out[7][1]  = in[13 * istride][1];

    out[8][0]  = in[2  * istride][0]; out[8][1]  = in[2  * istride][1];
    out[9][0]  = in[6  * istride][0]; out[9][1]  = in[6  * istride][1];
    out[10][0] = in[10 * istride][0]; out[10][1] = in[10 * istride][1];
    out[11][0] = in[14 * istride][0]; out[11][1] = in[14 * istride][1];

    out[12][0] = in[3  * istride][0]; out[12][1] = in[3  * istride][1];
    out[13][0] = in[7  * istride][0]; out[13][1] = in[7  * istride][1];
    out[14][0] = in[11 * istride][0]; out[14][1] = in[11 * istride][1];
    out[15][0] = in[15 * istride][0]; out[15][1] = in[15 * istride][1];

    for (int k = 0; k < 16; k += 4) {
        float r0 = out[k][0],   i0 = out[k][1];
        float r1 = out[k+1][0], i1 = out[k+1][1];
        float r2 = out[k+2][0], i2 = out[k+2][1];
        float r3 = out[k+3][0], i3 = out[k+3][1];

        float s02_r = r0 + r2, s02_i = i0 + i2;
        float d02_r = r0 - r2, d02_i = i0 - i2;
        float s13_r = r1 + r3, s13_i = i1 + i3;
        float d13_r = r1 - r3, d13_i = i1 - i3;

        out[k][0]   = s02_r + s13_r;
        out[k][1]   = s02_i + s13_i;

        out[k+1][0] = d02_r + d13_i;
        out[k+1][1] = d02_i - d13_r;

        out[k+2][0] = s02_r - s13_r;
        out[k+2][1] = s02_i - s13_i;

        out[k+3][0] = d02_r - d13_i;
        out[k+3][1] = d02_i + d13_r;
    }

    static const float c1_raw[4] = { 1.0f, 0.9238795325f, 0.7071067812f, 0.3826834323f };
    static const float s1_raw[4] = { 0.0f, 0.3826834323f, 0.7071067812f, 0.9238795325f };
    static const float c2_raw[4] = { 1.0f, 0.7071067812f, 0.0f,          -0.7071067812f };
    static const float s2_raw[4] = { 0.0f, 0.7071067812f, 1.0f,           0.7071067812f };
    static const float c3_raw[4] = { 1.0f, 0.3826834323f, -0.7071067812f,-0.9238795325f };
    static const float s3_raw[4] = { 0.0f, 0.9238795325f,  0.7071067812f, -0.3826834323f };

    float32x4_t v_c1 = vld1q_f32(c1_raw); float32x4_t v_s1 = vld1q_f32(s1_raw);
    float32x4_t v_c2 = vld1q_f32(c2_raw); float32x4_t v_s2 = vld1q_f32(s2_raw);
    float32x4_t v_c3 = vld1q_f32(c3_raw); float32x4_t v_s3 = vld1q_f32(s3_raw);

    float32x4x2_t in0 = vld2q_f32((const float*)&out[0]);
    float32x4_t v_r0 = in0.val[0]; float32x4_t v_i0 = in0.val[1];
    float32x4x2_t in1 = vld2q_f32((const float*)&out[4]);
    float32x4_t v_r1 = in1.val[0]; float32x4_t v_i1 = in1.val[1];
    float32x4x2_t in2 = vld2q_f32((const float*)&out[8]);
    float32x4_t v_r2 = in2.val[0]; float32x4_t v_i2 = in2.val[1];
    float32x4x2_t in3 = vld2q_f32((const float*)&out[12]);
    float32x4_t v_r3 = in3.val[0]; float32x4_t v_i3 = in3.val[1];

    float32x4_t v_r1_t = vaddq_f32(vmulq_f32(v_r1, v_c1), vmulq_f32(v_i1, v_s1));
    float32x4_t v_i1_t = vsubq_f32(vmulq_f32(v_i1, v_c1), vmulq_f32(v_r1, v_s1));

    float32x4_t v_r2_t = vaddq_f32(vmulq_f32(v_r2, v_c2), vmulq_f32(v_i2, v_s2));
    float32x4_t v_i2_t = vsubq_f32(vmulq_f32(v_i2, v_c2), vmulq_f32(v_r2, v_s2));

    float32x4_t v_r3_t = vaddq_f32(vmulq_f32(v_r3, v_c3), vmulq_f32(v_i3, v_s3));
    float32x4_t v_i3_t = vsubq_f32(vmulq_f32(v_i3, v_c3), vmulq_f32(v_r3, v_s3));

    float32x4_t v_t_r0 = vaddq_f32(v_r0, v_r2_t); float32x4_t v_t_r1 = vsubq_f32(v_r0, v_r2_t);
    float32x4_t v_t_r2 = vaddq_f32(v_r1_t, v_r3_t); float32x4_t v_t_r3 = vsubq_f32(v_r1_t, v_r3_t);
    float32x4_t v_t_i0 = vaddq_f32(v_i0, v_i2_t); float32x4_t v_t_i1 = vsubq_f32(v_i0, v_i2_t);
    float32x4_t v_t_i2 = vaddq_f32(v_i1_t, v_i3_t); float32x4_t v_t_i3 = vsubq_f32(v_i1_t, v_i3_t);

    float32x4x2_t out0; out0.val[0] = vaddq_f32(v_t_r0, v_t_r2); out0.val[1] = vaddq_f32(v_t_i0, v_t_i2);
    vst2q_f32((float*)&out[0], out0);
    float32x4x2_t out1; out1.val[0] = vaddq_f32(v_t_r1, v_t_i3); out1.val[1] = vsubq_f32(v_t_i1, v_t_r3);
    vst2q_f32((float*)&out[4], out1);
    float32x4x2_t out2; out2.val[0] = vsubq_f32(v_t_r0, v_t_r2); out2.val[1] = vsubq_f32(v_t_i0, v_t_i2);
    vst2q_f32((float*)&out[8], out2);
    float32x4x2_t out3; out3.val[0] = vsubq_f32(v_t_r1, v_t_i3); out3.val[1] = vaddq_f32(v_t_i1, v_t_r3);
    vst2q_f32((float*)&out[12], out3);
}
#endif

/* ========================================================================= */
/* [1] C2C 배치 실행 커널                                                     */
/* ========================================================================= */
static void run_c2c_batched(const myfft_plan p, const void* in_v, fftwf_complex* out) {
    const fftwf_complex* in = (const fftwf_complex*)in_v;
    const int n = p->n;

    for (int b = 0; b < p->howmany; ++b) {
        fftwf_complex* out_base = &out[b * p->odist];
        const fftwf_complex* in_base = &in[b * p->idist];

#if defined(__ARM_NEON)
        if (n == 16 && p->sign == FFTW_FORWARD && p->ostride == 1 && p->istride > 1) {
            custom_fft_16_radix4_direct_fast(in_base, out_base, p->istride);
            continue;
        }
#endif

        if (n == 2048 || n == 512 || n == 128 || n == 32) {
            if (p->sign == FFTW_BACKWARD) {
                for (int k = 0; k < n; ++k) {
                    p->tmp_cpx[k][0] = in_base[k * p->istride][0];
                    p->tmp_cpx[k][1] = -in_base[k * p->istride][1];
                }
                dispatch_neon_fft_r4_fused(p->tmp_cpx, out_base, n);
            } else if (in_base != out_base) {
                dispatch_neon_fft_r4_fused(in_base, out_base, n);
            } else {
                memcpy(p->tmp_cpx, in_base, n * sizeof(fftwf_complex));
                dispatch_neon_fft_r4_fused(p->tmp_cpx, out_base, n);
            }
        }
        else {
            if (p->istride == 1 && p->ostride == 1 && p->sign == FFTW_FORWARD) {
                memcpy(out_base, in_base, n * sizeof(fftwf_complex));
            } else {
                for (int k = 0; k < n; ++k) {
                    out_base[k * p->ostride][0] = in_base[k * p->istride][0];
                    out_base[k * p->ostride][1] = (p->sign == FFTW_BACKWARD) ? 
                                                  -in_base[k * p->istride][1] : in_base[k * p->istride][1];
                }
            }
            dispatch_neon_fft_r4_pure(out_base, n);
        }

        if (p->sign == FFTW_BACKWARD) {
            for (int k = 0; k < n; ++k) {
                out_base[k][1] = -out_base[k][1];
            }
        }
    }
}

static const fftwf_complex* get_twiddle_ptr(int n) {
    switch (n) {
        case 4096: return (const fftwf_complex*)twiddle_4096;
        case 2048: return (const fftwf_complex*)twiddle_2048;
        case 1024: return (const fftwf_complex*)twiddle_1024;
        case 512:  return (const fftwf_complex*)twiddle_512;
        case 256:  return (const fftwf_complex*)twiddle_256;
        case 128:  return (const fftwf_complex*)twiddle_128;
        case 64:   return (const fftwf_complex*)twiddle_64;
        case 32:   return (const fftwf_complex*)twiddle_32;
        case 16:   return (const fftwf_complex*)twiddle_16;
        default:   return NULL;
    }
}

static inline void unpack_r2c_half_spectrum(fftwf_complex* z, fftwf_complex* out_base, const fftwf_complex* tw, int n2) {
    float z0r = z[0][0], z0i = z[0][1];
    out_base[0][0]  = z0r + z0i; out_base[0][1]  = 0.0f;
    out_base[n2][0] = z0r - z0i; out_base[n2][1] = 0.0f;
    
    for (int k = 1; k <= n2 / 2; ++k) {
        int m = n2 - k;
        float zkr = z[k][0], zki = z[k][1];
        float zmr = z[m][0], zmi = z[m][1];
        
        float akr = 0.5f * (zkr + zmr);
        float aki = 0.5f * (zki - zmi);
        float bkr = 0.5f * (zkr - zmr);
        float bki = 0.5f * (zki + zmi);
        
        float twr = tw[k][0];
        float twi = tw[k][1];
        
        float ckr = -(twr * bki + twi * bkr);
        float cki =  (twr * bkr - twi * bki);
        
        out_base[k][0] = akr - ckr;
        out_base[k][1] = aki - cki;
        
        out_base[m][0] = akr + ckr;
        out_base[m][1] = -(aki + cki);
    }
}

/* ========================================================================= */
/* [2] R2C 배치 실행 커널                                                     */
/* ========================================================================= */
static void run_r2c_batched(const myfft_plan p, const float* in, fftwf_complex* out) {
    const int n = p->n;
    const int n2 = n / 2;
    const int nout = n2 + 1;
    const fftwf_complex* tw = get_twiddle_ptr(n);

    for (int b = 0; b < p->howmany; ++b) {
        const float* in_base = &in[b * p->idist];
        fftwf_complex* out_base = (p->ostride == 1) ? &out[b * p->odist] : p->scratch_cpx;
        
        if (n2 == 2048 || n2 == 512 || n2 == 128 || n2 == 32) {
            if (p->istride == 1) {
                dispatch_neon_fft_r4_fused((const fftwf_complex*)in_base, p->scratch_cpx, n2);
            } else {
                for (int k = 0; k < n2; ++k) {
                    p->scratch_cpx[k][0] = in_base[(2 * k) * p->istride];
                    p->scratch_cpx[k][1] = in_base[(2 * k + 1) * p->istride];
                }
                memcpy(p->tmp_cpx, p->scratch_cpx, n2 * sizeof(fftwf_complex));
                dispatch_neon_fft_r4_fused(p->tmp_cpx, p->scratch_cpx, n2);
            }
        } else {
            if (p->istride == 1) {
                memcpy(p->scratch_cpx, in_base, n * sizeof(float));
            } else {
                for (int k = 0; k < n2; ++k) {
                    p->scratch_cpx[k][0] = in_base[(2 * k) * p->istride];
                    p->scratch_cpx[k][1] = in_base[(2 * k + 1) * p->istride];
                }
            }
            dispatch_neon_fft_r4_pure(p->scratch_cpx, n2);
        }
        
        unpack_r2c_half_spectrum(p->scratch_cpx, out_base, tw, n2);

        if (p->ostride != 1) {
            for (int k = 0; k < nout; ++k) {
                out[b * p->odist + k * p->ostride][0] = out_base[k][0];
                out[b * p->odist + k * p->ostride][1] = out_base[k][1];
            }
        }
    }
}

/* ============================== 실행 API =============================== */
void fftwf_execute(const fftwf_plan p) {
    if (!p) return;
    if (p->kind == KIND_R2C) run_r2c_batched(p, (const float*)p->in_bound, p->out_bound);
    else                     run_c2c_batched(p, p->in_bound, p->out_bound);
}

void fftwf_execute_dft(const fftwf_plan p, fftwf_complex* in, fftwf_complex* out) {
    if (!p) return;
    run_c2c_batched(p, (const void*)in, out);
}

/* =========================== 스레딩 (no-op & Hijacking) ================ */
int fftwf_init_threads(void) { 
    init_resources(); 
    return 1; 
}
void fftwf_plan_with_nthreads(int n) { (void)n; }
void fftwf_cleanup_threads(void)     { }
void fftwf_cleanup(void)             { }