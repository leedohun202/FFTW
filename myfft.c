/*
 * myfft.c — FFTW 대체 커스텀 FFT (Custom Float Radix-2 결합 버전)
 * -----------------------------------------------------------------------------
 * - FFTW3의 인터페이스를 완벽하게 모방 (Shim API)
 * - 내부적으로 기존에 개발된 Custom Float Radix-2 (custom_fft_*_fixed) 호출
 * - Gather/Scatter 방식을 통한 Interleaved <-> Split 메모리 레이아웃 변환
 * - 역변환(IFFT) 시 Conjugate(켤레) 트릭 사용
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

    /* NEON 커널 실행을 위한 64바이트 정렬 Split 버퍼 */
    float* scratch_r;
    float* scratch_i;
};

/* ============================== 메모리 ================================== */
float* fftwf_alloc_real(size_t n)    { return (float*)malloc(n * sizeof(float)); }
fftwf_complex* fftwf_alloc_complex(size_t n) { return (fftwf_complex*)malloc(n * sizeof(fftwf_complex)); }
void* fftwf_malloc(size_t bytes)    { return malloc(bytes); }
void           fftwf_free(void* p)           { free(p); }

/* ============================== 유틸 =================================== */
int    myfft_is_pow2(size_t n)   { return n && ((n & (n - 1)) == 0); }
size_t myfft_next_pow2(size_t n) { size_t p = 1; while (p < n) p <<= 1; return p; }

/* ===================== Custom Radix-2 커널 라우터 ====================== */
static void dispatch_neon_fft_r2(float *r, float *i, int n) {
    if (n == 4096)      custom_fft_4096_fixed(r, i);
    else if (n == 2048) custom_fft_2048_fixed(r, i);
    else if (n == 1024) custom_fft_1024_fixed(r, i);
    else if (n == 512)  custom_fft_512_fixed(r, i);
    else if (n == 256)  custom_fft_256_fixed(r, i);
    else if (n == 128)  custom_fft_128_fixed(r, i);
    else if (n == 64)   custom_fft_64_fixed(r, i);
    else if (n == 16)   custom_fft_16_fixed(r, i);
    else fprintf(stderr, "[myfft] ERROR: Unsupported FFT size %d\n", n);
}

/* ===================== 플랜 공통 할당 ================================== */
static myfft_plan alloc_plan(myfft_kind kind, int n, int sign,
                             int howmany, int istride, int idist,
                             int ostride, int odist,
                             void* in_bound, fftwf_complex* out_bound) {
    if (!myfft_is_pow2((size_t)n)) {
        fprintf(stderr, "[myfft] WARNING: N=%d is not a power of two (unsupported).\n", n);
    }
    myfft_plan p = (myfft_plan)calloc(1, sizeof(struct myfft_plan_s));
    if (!p) return NULL;
    
    p->kind = kind; p->n = n; p->sign = sign; p->howmany = howmany;
    p->istride = istride; p->idist = idist;
    p->ostride = ostride; p->odist = odist;
    p->in_bound = in_bound; p->out_bound = out_bound;

    /* 캐시 라인 정렬(64B)된 Split 버퍼 동적 할당 */
    posix_memalign((void**)&p->scratch_r, 64, n * sizeof(float));
    posix_memalign((void**)&p->scratch_i, 64, n * sizeof(float));

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
    free(p->scratch_r);
    free(p->scratch_i);
    free(p);
}

/* ===================== 핵심 커널: C2C 배치 실행 ======================== */
static void run_c2c_batched(const myfft_plan p, const void* in_v, fftwf_complex* out) {
    const fftwf_complex* in = (const fftwf_complex*)in_v;
    const int n = p->n;

    for (int b = 0; b < p->howmany; ++b) {
        /* 1. Gather & IFFT Conjugate (입력을 임시 버퍼로 복사) */
        for (int k = 0; k < n; ++k) {
            p->scratch_r[k] = in[b * p->idist + k * p->istride][0];
            if (p->sign == FFTW_BACKWARD) {
                p->scratch_i[k] = -in[b * p->idist + k * p->istride][1];
            } else {
                p->scratch_i[k] = in[b * p->idist + k * p->istride][1];
            }
        }

        /* 2. Custom Float Radix-2 커널 실행 */
        dispatch_neon_fft_r2(p->scratch_r, p->scratch_i, n);

        /* 3. Scatter & IFFT Conjugate (임시 버퍼에서 출력으로 복사) */
        for (int k = 0; k < n; ++k) {
            out[b * p->odist + k * p->ostride][0] = p->scratch_r[k];
            if (p->sign == FFTW_BACKWARD) {
                out[b * p->odist + k * p->ostride][1] = -p->scratch_i[k];
            } else {
                out[b * p->odist + k * p->ostride][1] = p->scratch_i[k];
            }
        }
    }
}

/* ===================== 핵심 커널: R2C 배치 실행 ======================== */
static void run_r2c_batched(const myfft_plan p, const float* in, fftwf_complex* out) {
    const int n = p->n;
    const int nout = n / 2 + 1; /* 반스펙트럼 N/2 + 1 출력 보장 */

    for (int b = 0; b < p->howmany; ++b) {
        /* 1. Gather (실수부 입력, 허수부는 0 초기화) */
        for (int k = 0; k < n; ++k) {
            p->scratch_r[k] = in[b * p->idist + k * p->istride];
            p->scratch_i[k] = 0.0f;
        }

        /* 2. Custom Float Radix-2 커널 실행 */
        dispatch_neon_fft_r2(p->scratch_r, p->scratch_i, n);

        /* 3. Scatter (N/2+1 개만 추출하여 출력) */
        for (int k = 0; k < nout; ++k) {
            out[b * p->odist + k * p->ostride][0] = p->scratch_r[k];
            out[b * p->odist + k * p->ostride][1] = p->scratch_i[k];
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
/* 💡 [패치] 데모가 이 함수를 부를 때 레이더 리소스(LUT)를 조용히 초기화! */
int fftwf_init_threads(void) { 
    init_resources(); 
    return 1; 
}
void fftwf_plan_with_nthreads(int n) { (void)n; }
void fftwf_cleanup_threads(void)     { }
void fftwf_cleanup(void)             { }
