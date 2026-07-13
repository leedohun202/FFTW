#ifndef RADAR_CONFIG_H
#define RADAR_CONFIG_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h> 
#include <sys/time.h>
#include <sys/resource.h> 
#include <fftw3.h> 
#include <omp.h>
#include <malloc.h>
#include <unistd.h>
#include <stdint.h>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

// 레이더 파이프라인 상수
#define PI 3.14159265358979323846
#define N_CHIRPS    512   
#define N_ANTENNAS  4     
#define N_ANGLE     64    
#define MAX_DETECTED 10
#define BENCH_RUNS  10 
#define TILE_SIZE   16    
// 🎯 [핵심 패치] 64바이트 캐시 라인 완벽 정렬을 위한 Int16 전용 32x32 타일
#define TILE_SIZE_INT16 32
#define FFTW_WISDOM_FILE "radar_fftw_pi5.wisdom"

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
extern int bitrev_2048[2048]; extern int bitrev_1024[1024]; extern int bitrev_512[512]; 
extern int bitrev_256[256];   extern int bitrev_128[128];   extern int bitrev_64[64];

// =========================================================================
// [3] FLOAT 전용 Twiddle Factor 및 Window 테이블
// =========================================================================
// (나비 연산 절반 크기 N/2 최적화 적용)
extern float twiddle_real_2048[1024]; extern float twiddle_imag_2048[1024];
extern float twiddle_real_1024[512];  extern float twiddle_imag_1024[512];
extern float twiddle_real_512[256];   extern float twiddle_imag_512[256];
extern float twiddle_real_256[128];   extern float twiddle_imag_256[128];
extern float twiddle_real_128[64];    extern float twiddle_imag_128[64];
extern float twiddle_real_64[32];     extern float twiddle_imag_64[32];


extern float win_2048[2048];
extern float win_1024[1024];
extern float win_512[512];
extern float win_256[256];
extern float win_128[128];

// =========================================================================
// [4] 🔥 INT16 전용 Twiddle Factor 테이블 (Q15 포맷)
// =========================================================================
// (Radix-8 분기문 제로화 가속을 위해 풀사이즈 N으로 할당)
extern int16_t twiddle_int16_real_2048[2048]; extern int16_t twiddle_int16_imag_2048[2048];
extern int16_t twiddle_int16_real_1024[1024]; extern int16_t twiddle_int16_imag_1024[1024];
extern int16_t twiddle_int16_real_512[512];   extern int16_t twiddle_int16_imag_512[512];
extern int16_t twiddle_int16_real_256[256];   extern int16_t twiddle_int16_imag_256[256];
extern int16_t twiddle_int16_real_128[128];   extern int16_t twiddle_int16_imag_128[128];
extern int16_t twiddle_int16_real_64[64];     extern int16_t twiddle_int16_imag_64[64];

// =========================================================================
// [5] 🔥 INT16 전용 Hanning Window 테이블 (Q15 포맷)
// =========================================================================
extern int16_t win_int16_2048[2048];
extern int16_t win_int16_1024[1024];
extern int16_t win_int16_512[512];
extern int16_t win_int16_256[256];
extern int16_t win_int16_128[128];

#endif // RADAR_CONFIG_H