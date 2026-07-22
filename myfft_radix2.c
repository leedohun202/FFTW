/*
 * myfft_radix2.c — FFTW 대체 커스텀 FFT (Radix-2 Fused 버전 전용 백엔드)
 * -----------------------------------------------------------------------------
 * - FFTW3의 인터페이스를 완벽하게 모방 (Shim API)
 * - Interleaved(AoS) 레이아웃 적용 (fftwf_complex = float[2])
 * - 모든 크기(16 ~ 4096)에 대해 memcpy 없는 Fused Bit-reversal 커널 호출
 * -----------------------------------------------------------------------------
 */
#include "myfft.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

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

    /* R2C 변환 시 N 크기의 복원 연산을 위해 필요한 Interleaved 단일 버퍼 (C2C는 NULL) */
    fftwf_complex* scratch_cpx;
    
    /* 🚀 스택 오버플로우 방지용 힙(Heap) 할당 임시 버퍼 */
    fftwf_complex* tmp_cpx; 
};

/* ============================== 메모리 ================================== */
float* fftwf_alloc_real(size_t n)    { return (float*)malloc(n * sizeof(float)); }
fftwf_complex* fftwf_alloc_complex(size_t n) { return (fftwf_complex*)malloc(n * sizeof(fftwf_complex)); }
void* fftwf_malloc(size_t bytes)    { return malloc(bytes); }
void           fftwf_free(void* p)           { free(p); }

/* ============================== 유틸 =================================== */
int    myfft_is_pow2(size_t n)   { return n && ((n & (n - 1)) == 0); }
size_t myfft_next_pow2(size_t n) { size_t p = 1; while (p < n) p <<= 1; return p; }

// Fused Radix-2 커널 디스패처 (모든 크기 통합)
static void dispatch_fft_r2_fused(const fftwf_complex *in, fftwf_complex *out, int n) {
    switch (n) {
        case 4096: custom_fft_4096_radix2_fused(in, out); break;
        case 2048: custom_fft_2048_radix2_fused(in, out); break;
        case 1024: custom_fft_1024_radix2_fused(in, out); break;
        case 512:  custom_fft_512_radix2_fused(in, out); break;
        case 256:  custom_fft_256_radix2_fused(in, out); break;
        case 128:  custom_fft_128_radix2_fused(in, out); break;
        case 64:   custom_fft_64_radix2_fused(in, out); break;
        case 32:   custom_fft_32_radix2_fused(in, out); break;
        case 16:   custom_fft_16_radix2_fused(in, out); break;
        default: break;
    }
}

/* ===================== 플랜 공통 할당 ================================== */
static myfft_plan alloc_plan(myfft_kind kind, int n, int sign,
                             int howmany, int istride, int idist,
                             int ostride, int odist,
                             void* in_bound, fftwf_complex* out_bound) {
    // 🚀 [High 3 방어] 지원 상한선(4096) 초과 및 2의 승수 검사 (Hard Fail)
    if (n > 4096 || !myfft_is_pow2((size_t)n)) {
        fprintf(stderr, "[myfft] ERROR: N=%d is unsupported (must be pow2 and <= 4096).\n", n);
        return NULL;
    }

    // 🚀 [Medium 4 방어] C2C 플랜에서 ostride != 1 인 경우 지원 거부
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

    /* R2C의 경우 연산을 위한 단일 Scratch 버퍼 할당 */
    if (kind == KIND_R2C) {
        if (posix_memalign((void**)&p->scratch_cpx, 64, n * sizeof(fftwf_complex)) != 0) {
            free(p); return NULL;
        }
    } else {
        p->scratch_cpx = NULL; 
    }

    /* 🚀 [High 3 방어] 루프 내 스택 오버플로우를 막기 위한 힙 버퍼 할당 */
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
    if (p->tmp_cpx)     free(p->tmp_cpx); // 🚀 할당 해제 추가
    free(p);
}

// =========================================================================
// [1] C2C 배치 실행 커널 (Fused Radix-2 전용)
// =========================================================================
static void run_c2c_batched(const myfft_plan p, const void* in_v, fftwf_complex* out) {
    const fftwf_complex* in = (const fftwf_complex*)in_v;
    const int n = p->n;

    for (int b = 0; b < p->howmany; ++b) {
        fftwf_complex* out_base = &out[b * p->odist];
        const fftwf_complex* in_base = &in[b * p->idist];

        if (p->sign == FFTW_BACKWARD) {
            // 🛠️ IFFT 패턴: Pre-conjugate (입력 허수부 반전)
            for (int k = 0; k < n; ++k) {
                p->tmp_cpx[k][0] = in_base[k * p->istride][0];
                p->tmp_cpx[k][1] = -in_base[k * p->istride][1];
            }
            dispatch_fft_r2_fused(p->tmp_cpx, out_base, n);
        } else if (in_base == out_base || p->istride != 1) {
            // ⚠️ In-place 연산이거나 stride가 1이 아닌 경우 데이터 덮어쓰기 방지
            for (int k = 0; k < n; ++k) {
                p->tmp_cpx[k][0] = in_base[k * p->istride][0];
                p->tmp_cpx[k][1] = in_base[k * p->istride][1];
            }
            dispatch_fft_r2_fused(p->tmp_cpx, out_base, n);
        } else {
            // 🚀 Out-of-place (Test 1 등): 제로 카피 다이렉트 실행
            dispatch_fft_r2_fused(in_base, out_base, n);
        }

        // 🛠️ IFFT Post-conjugate 처리
        if (p->sign == FFTW_BACKWARD) {
            for (int k = 0; k < n; ++k) {
                // 🚀 [Medium 4 반영] 커널의 연속 기록 계약(ostride=1)과 완벽하게 일치시킴
                out_base[k][1] = -out_base[k][1];
            }
        }
    }
}

/* Twiddle Factor 포인터를 런타임에 동적으로 가져오기 위한 라우터 */
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

/* R2C Unpack (공액 대칭 분리) 공통 헬퍼 함수 */
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

// =========================================================================
// [2] R2C 배치 실행 커널
// =========================================================================
static void run_r2c_batched(const myfft_plan p, const float* in, fftwf_complex* out) {
    const int n = p->n;
    const int n2 = n / 2;             /* N/2 크기의 C2C FFT 수행 */
    const int nout = n2 + 1;          /* 반스펙트럼 출력 크기 */
    const fftwf_complex* tw = get_twiddle_ptr(n);

    for (int b = 0; b < p->howmany; ++b) {
        const float* in_base = &in[b * p->idist];
        fftwf_complex* out_base = (p->ostride == 1) ? &out[b * p->odist] : p->scratch_cpx;
        
        /* 1 & 2. N/2 C2C FFT 실행 */
        if (p->istride == 1) {
            dispatch_fft_r2_fused((const fftwf_complex*)in_base, p->scratch_cpx, n2);
        } else {
            for (int k = 0; k < n2; ++k) {
                p->tmp_cpx[k][0] = in_base[(2 * k) * p->istride];
                p->tmp_cpx[k][1] = in_base[(2 * k + 1) * p->istride];
            }
            dispatch_fft_r2_fused(p->tmp_cpx, p->scratch_cpx, n2);
        }
        
        unpack_r2c_half_spectrum(p->scratch_cpx, out_base, tw, n2);

        /* ostride != 1 일 때 Scatter */
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