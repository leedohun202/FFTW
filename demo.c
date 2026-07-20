/*
 * demo.c — FFTW-API 최소 실행 데모 (레이더 알고리즘 제외)
 * -----------------------------------------------------------------------------
 * 레이더 파이프라인이 쓰는 5가지 FFT "패턴"만 뽑아, 간단한 입력을 생성하고
 * FFT 를 돌려 결과를 출력한다. 각 테스트는 스스로 정답을 검증(PASS/FAIL)한다.
 *
 * 백엔드는 fft_backend.h 가 결정:
 *   make        → 커스텀 myfft (무의존, 바로 실행)
 *   make fftw   → 실제 libfftw3f (골든 레퍼런스, FFTW 설치 필요)
 *
 * 동료 활용법: 두 백엔드로 각각 빌드→실행하여 출력이 일치하는지 대조하면,
 *             커스텀 FFT 구현의 정확성을 검증할 수 있다.
 * -----------------------------------------------------------------------------
 */
#include "fft_backend.h"

#include <stdio.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do {                                  \
    if (cond) { printf("    [PASS] %s\n", msg); ++g_pass; }     \
    else      { printf("    [FAIL] %s\n", msg); ++g_fail; }     \
} while (0)

static int argmax_mag(const fftwf_complex* a, int n) {
    int bi = 0; float bm = -1.0f;
    for (int i = 0; i < n; ++i) {
        float m = a[i][0]*a[i][0] + a[i][1]*a[i][1];
        if (m > bm) { bm = m; bi = i; }
    }
    return bi;
}
static void print_mag(const char* label, const fftwf_complex* a, int n) {
    printf("    %-14s", label);
    for (int i = 0; i < n; ++i)
        printf("%6.2f", sqrtf(a[i][0]*a[i][0] + a[i][1]*a[i][1]));
    printf("\n");
}

/* ------------------------------------------------------------------ */
/* 1) c2c forward 1D  (angle FFT 패턴)                                 */
/*    입력: bin 2 의 복소 톤 → 출력 스펙트럼은 bin 2 에서 피크          */
/* ------------------------------------------------------------------ */
static void test_c2c_forward(void) {
    const int N = 4096, tone = 2;
    printf("[1] c2c forward 1D  (N=%d, angle-FFT 패턴)\n", N);
    fftwf_complex* in  = fftwf_alloc_complex(N);
    fftwf_complex* out = fftwf_alloc_complex(N);
    fftwf_plan p = fftwf_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    for (int k = 0; k < N; ++k) {                       /* 입력 생성: e^{+i 2π·tone·k/N} */
        in[k][0] = cosf(2.0f*(float)M_PI*tone*k/N);
        in[k][1] = sinf(2.0f*(float)M_PI*tone*k/N);
    }
    fftwf_execute(p);
    print_mag("|X[k]| =", out, N);
    CHECK(argmax_mag(out, N) == tone, "peak at expected bin (2)");

    fftwf_destroy_plan(p); fftwf_free(in); fftwf_free(out);
}

/* ------------------------------------------------------------------ */
/* 2) c2c backward + 수동 /N  (iFFT 패턴)                              */
/*    forward 결과를 역변환하고 N 으로 나눠 원신호 복원                 */
/* ------------------------------------------------------------------ */
static void test_ifft_roundtrip(void) {
    const int N = 4096;
    printf("[2] c2c backward + 수동 /N  (N=%d, iFFT 패턴 — 비정규화 규약)\n", N);
    fftwf_complex* x  = fftwf_alloc_complex(N);
    fftwf_complex* X  = fftwf_alloc_complex(N);
    fftwf_complex* xr = fftwf_alloc_complex(N);
    fftwf_plan pf = fftwf_plan_dft_1d(N, x, X,  FFTW_FORWARD,  FFTW_ESTIMATE);
    fftwf_plan pb = fftwf_plan_dft_1d(N, X, xr, FFTW_BACKWARD, FFTW_ESTIMATE);

    for (int k = 0; k < N; ++k) { x[k][0] = (float)(k + 1); x[k][1] = 0.0f; }  /* 램프 */
    fftwf_execute(pf);
    fftwf_execute(pb);
    for (int k = 0; k < N; ++k) { xr[k][0] /= N; xr[k][1] /= N; }              /* ← 수동 정규화 */

    float maxerr = 0.0f;
    for (int k = 0; k < N; ++k) {
        float e = fabsf(xr[k][0] - x[k][0]) + fabsf(xr[k][1] - x[k][1]);
        if (e > maxerr) maxerr = e;
    }
    printf("    복원 최대오차 = %.2e\n", maxerr);
    CHECK(maxerr < 1e-3f, "IFFT(FFT(x))/N == x");

    fftwf_destroy_plan(pf); fftwf_destroy_plan(pb);
    fftwf_free(x); fftwf_free(X); fftwf_free(xr);
}

/* ------------------------------------------------------------------ */
/* 3) r2c forward  (range FFT 패턴)                                    */
/*    실수 코사인(bin 3) 입력 → N/2+1 반스펙트럼, bin 3 에서 피크       */
/* ------------------------------------------------------------------ */
static void test_r2c(void) {
    const int N = 4096, tone = 3, NOUT = N/2 + 1;
    printf("[3] r2c forward  (N=%d → %d bins, range-FFT 패턴 — 반스펙트럼)\n", N, NOUT);
    float*         in  = fftwf_alloc_real(N);
    fftwf_complex* out = fftwf_alloc_complex(NOUT);
    int nn[1] = { N };
    fftwf_plan p = fftwf_plan_many_dft_r2c(1, nn, 1,
                                           in,  NULL, 1, N,
                                           out, NULL, 1, NOUT, FFTW_ESTIMATE);
    for (int k = 0; k < N; ++k) in[k] = cosf(2.0f*(float)M_PI*tone*k/N);
    fftwf_execute(p);
    print_mag("|R[k]| =", out, NOUT);
    CHECK(argmax_mag(out, NOUT) == tone, "peak at expected bin (3)");

    fftwf_destroy_plan(p); fftwf_free(in); fftwf_free(out);
}

/* ------------------------------------------------------------------ */
/* 4) 배치 strided c2c + execute_dft  (doppler FFT 패턴)               */
/*    입력 레이아웃: in[b + howmany*k]  (istride=howmany, idist=1)      */
/*    → 열 방향으로 읽어 batch b 는 bin b 의 톤. 출력은 배치별 연속.     */
/* ------------------------------------------------------------------ */
static void test_batched_doppler(void) {
    const int N = 4096, HOW = 4;
    printf("[4] 배치 strided c2c + execute_dft  (N=%d, howmany=%d, doppler 패턴)\n", N, HOW);
    fftwf_complex* in   = fftwf_alloc_complex(N * HOW);
    fftwf_complex* outA = fftwf_alloc_complex(N * HOW);
    fftwf_complex* outB = fftwf_alloc_complex(N * HOW);
    int nn[1] = { N };
    /* istride=HOW, idist=1 (열 방향 입력) / ostride=1, odist=N (배치별 연속 출력) */
    fftwf_plan p = fftwf_plan_many_dft(1, nn, HOW,
                                       in,   NULL, HOW, 1,
                                       outA, NULL, 1,   N,
                                       FFTW_FORWARD, FFTW_ESTIMATE);
    for (int b = 0; b < HOW; ++b)
        for (int k = 0; k < N; ++k) {
            in[b + HOW*k][0] = cosf(2.0f*(float)M_PI*b*k/N);   /* batch b: bin b 톤 */
            in[b + HOW*k][1] = sinf(2.0f*(float)M_PI*b*k/N);
        }

    fftwf_execute_dft(p, in, outA);       /* new-array 실행 #1 */
    fftwf_execute_dft(p, in, outB);       /* new-array 실행 #2 (plan 재사용) */

    int ok = 1;
    for (int b = 0; b < HOW; ++b) {
        char lbl[24]; snprintf(lbl, sizeof(lbl), "batch %d |X|=", b);
        print_mag(lbl, outA + b*N, N);
        if (argmax_mag(outA + b*N, N) != b) ok = 0;
    }
    CHECK(ok, "each batch peaks at its own bin (0..3)");
    /* 두 new-array 실행 결과 동일한지(plan 재바인딩 정합성) */
    float d = 0.0f;
    for (int i = 0; i < N*HOW; ++i) { d += fabsf(outA[i][0]-outB[i][0]) + fabsf(outA[i][1]-outB[i][1]); }
    CHECK(d < 1e-3f, "execute_dft rebinding consistent (outA == outB)");

    fftwf_destroy_plan(p); fftwf_free(in); fftwf_free(outA); fftwf_free(outB);
}

/* ------------------------------------------------------------------ */
/* 5) in-place c2c  (zpFFT 패턴)                                       */
/*    입력 버퍼 == 출력 버퍼. out-of-place 결과와 일치해야 함.          */
/* ------------------------------------------------------------------ */
static void test_inplace(void) {
    const int N = 4096, tone = 5;
    printf("[5] in-place c2c  (N=%d, zpFFT 패턴 — 입력버퍼=출력버퍼)\n", N);
    fftwf_complex* buf = fftwf_alloc_complex(N);   /* in-place */
    fftwf_complex* ref_in  = fftwf_alloc_complex(N);
    fftwf_complex* ref_out = fftwf_alloc_complex(N);
    for (int k = 0; k < N; ++k) {
        float re = cosf(2.0f*(float)M_PI*tone*k/N), im = sinf(2.0f*(float)M_PI*tone*k/N);
        buf[k][0] = re; buf[k][1] = im; ref_in[k][0] = re; ref_in[k][1] = im;
    }
    fftwf_plan pin  = fftwf_plan_dft_1d(N, buf,    buf,     FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_plan pref = fftwf_plan_dft_1d(N, ref_in, ref_out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(pin);
    fftwf_execute(pref);

    float d = 0.0f;
    for (int k = 0; k < N; ++k) d += fabsf(buf[k][0]-ref_out[k][0]) + fabsf(buf[k][1]-ref_out[k][1]);
    printf("    in-place vs out-of-place 차이 = %.2e\n", d);
    CHECK(argmax_mag(buf, N) == tone, "peak at expected bin (5)");
    CHECK(d < 1e-3f, "in-place == out-of-place");

    fftwf_destroy_plan(pin); fftwf_destroy_plan(pref);
    fftwf_free(buf); fftwf_free(ref_in); fftwf_free(ref_out);
}

int main(void) {
#ifdef USE_FFTW
    printf("=== FFT demo  [backend: FFTW (libfftw3f)] ===\n\n");
#else
    printf("=== FFT demo  [backend: myfft (custom radix-2)] ===\n\n");
#endif
    fftwf_init_threads();
    fftwf_plan_with_nthreads(1);

    test_c2c_forward();     printf("\n");
    test_ifft_roundtrip();  printf("\n");
    test_r2c();             printf("\n");
    test_batched_doppler(); printf("\n");
    test_inplace();         printf("\n");

    fftwf_cleanup_threads();
    fftwf_cleanup();

    printf("=== 결과: %d PASS, %d FAIL ===\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
