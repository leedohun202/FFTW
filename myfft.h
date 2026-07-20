/*
 * myfft.h — FFTW(libfftw3f) 대체용 커스텀 FFT shim & 레이더 하드웨어 가속 통합 헤더
 * -----------------------------------------------------------------------------
 * 목적: 
 * 1. 소스의 #include <fftw3.h> 를 #include "myfft.h" 로만 바꾸어 FFTW 종속성을 제거.
 * 2. 레이더 파이프라인의 전역 물리 상수, LUT, NEON 가속 커널 선언을 통합 관리.
 * -----------------------------------------------------------------------------
 */
#ifndef MYFFT_H
#define MYFFT_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h> 
#include <sys/time.h>
#include <sys/resource.h> 
#include <omp.h>
#include <malloc.h>
#include <unistd.h>
#include <stddef.h>   /* size_t */

#ifdef __aarch64__
#include <arm_neon.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * [PART 1] 레이더 파이프라인 전역 설정 (from radar_config.h)
 * ========================================================================= */

// 레이더 파이프라인 상수
#define PI 3.14159265358979323846  
#define N_ANTENNAS  8  
#define TILE_SIZE   16    

// 전역 물리 상수
extern const double c; 
extern const double fc; 
extern const double B; 
extern const double Tc; 
extern const double Fs; 
extern const double S; 
extern const double lambda_c; 
extern const double d_ant;

// 전역 룩업 테이블(LUT) 및 윈도우 배열 선언
extern int bitrev_4096[4096]; extern int bitrev_2048[2048]; extern int bitrev_1024[1024]; 
extern int bitrev_512[512];   extern int bitrev_256[256];   extern int bitrev_128[128];   
extern int bitrev_64[64];     extern int bitrev_16[16];

// FLOAT 전용 Twiddle Factor 및 Window 테이블 (나비 연산 절반 크기 N/2)
extern float twiddle_real_4096[2048]; extern float twiddle_imag_4096[2048];
extern float twiddle_real_2048[1024]; extern float twiddle_imag_2048[1024];
extern float twiddle_real_1024[512];  extern float twiddle_imag_1024[512];
extern float twiddle_real_512[256];   extern float twiddle_imag_512[256];
extern float twiddle_real_256[128];   extern float twiddle_imag_256[128];
extern float twiddle_real_128[64];    extern float twiddle_imag_128[64];
extern float twiddle_real_64[32];     extern float twiddle_imag_64[32];
extern float twiddle_real_16[8];      extern float twiddle_imag_16[8];

extern float win_4096[4096];
extern float win_2048[2048];
extern float win_1024[1024];
extern float win_512[512];
extern float win_256[256];
extern float win_128[128];
extern float win_64[64];
extern float win_16[16];

// 리소스 초기화 함수
void init_resources();

/* =========================================================================
 * [PART 2] 하드웨어 가속 FFT 커널 선언 (from radar_fft.h)
 * ========================================================================= */

// 🔹 [GROUP 1] 순수 Float 기반 Radix-2 / NEON 가속 커널
void custom_fft_4096_fixed(float *__restrict__ real, float *__restrict__ imag);
void custom_fft_2048_fixed(float *__restrict__ real, float *__restrict__ imag);
void custom_fft_1024_fixed(float *__restrict__ real, float *__restrict__ imag);
void custom_fft_512_fixed(float *__restrict__ real, float *__restrict__ imag);
void custom_fft_256_fixed(float *__restrict__ real, float *__restrict__ imag);
void custom_fft_128_fixed(float *__restrict__ real, float *__restrict__ imag);
void custom_fft_64_fixed(float *__restrict__ real, float *__restrict__ imag);
void custom_fft_16_fixed(float *__restrict__ real, float *__restrict__ imag);

// 🔹 [GROUP 2] 순수 Float 기반 Radix-4 / NEON 가속 커널
void custom_fft_4096_radix4(float *__restrict__ real, float *__restrict__ imag);
void custom_fft_2048_radix4(float *__restrict__ real, float *__restrict__ imag);
void custom_fft_1024_radix4(float *__restrict__ real, float *__restrict__ imag);
void custom_fft_512_radix4(float *__restrict__ real, float *__restrict__ imag);
void custom_fft_256_radix4(float *__restrict__ real, float *__restrict__ imag);
void custom_fft_128_radix4(float *__restrict__ real, float *__restrict__ imag);
void custom_fft_64_radix4(float *__restrict__ real, float *__restrict__ imag);
void custom_fft_16_radix4(float *__restrict__ real, float *__restrict__ imag);


/* =========================================================================
 * [PART 3] FFTW3 호환 Shim API (from myfft.h)
 * ========================================================================= */

/* FFTW와 동일한 메모리 레이아웃(하드 계약). */
typedef float myfft_complex[2];
typedef myfft_complex fftwf_complex;      /* 호출부 호환용 별칭 */

/* 불투명 plan 핸들. */
typedef struct myfft_plan_s* myfft_plan;
typedef myfft_plan fftwf_plan;            /* 호출부 호환용 별칭 */

/* 상수 */
#define FFTW_FORWARD  (-1)   /* 순변환 */
#define FFTW_BACKWARD (+1)   /* 역변환(비정규화 — 1/N 미적용) */

/* FFTW 플래너 전략 플래그. */
#define FFTW_ESTIMATE   (0x01U)
#define FFTW_MEASURE    (0x00U)
#define FFTW_PATIENT    (0x20U)

/* 메모리 할당/해제 */
float* fftwf_alloc_real(size_t n);           /* float n개 */
fftwf_complex* fftwf_alloc_complex(size_t n);        /* 복소 n개 */
void* fftwf_malloc(size_t bytes);
void           fftwf_free(void* p);

/* 플랜 생성 */
fftwf_plan fftwf_plan_dft_1d(int n, fftwf_complex* in, fftwf_complex* out,
                             int sign, unsigned flags);

fftwf_plan fftwf_plan_many_dft_r2c(int rank, const int* n, int howmany,
                                   float* in,  const int* inembed, int istride, int idist,
                                   fftwf_complex* out, const int* onembed, int ostride, int odist,
                                   unsigned flags);

fftwf_plan fftwf_plan_many_dft(int rank, const int* n, int howmany,
                               fftwf_complex* in,  const int* inembed, int istride, int idist,
                               fftwf_complex* out, const int* onembed, int ostride, int odist,
                               int sign, unsigned flags);

/* 실행 */
void fftwf_execute(const fftwf_plan p);
void fftwf_execute_dft(const fftwf_plan p, fftwf_complex* in, fftwf_complex* out);

/* 정리 */
void fftwf_destroy_plan(fftwf_plan p);

/* 스레딩 (no-op) */
int  fftwf_init_threads(void);
void fftwf_plan_with_nthreads(int nthreads);
void fftwf_cleanup_threads(void);
void fftwf_cleanup(void);

/* 유틸(선택) */
int          myfft_is_pow2(size_t n);
size_t       myfft_next_pow2(size_t n);

#ifdef __cplusplus
}
#endif

#endif /* MYFFT_H */