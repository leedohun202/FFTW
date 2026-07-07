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

extern float twiddle_real_2048[1024]; extern float twiddle_imag_2048[1024];
extern float twiddle_real_1024[512];  extern float twiddle_imag_1024[512];
extern float twiddle_real_512[256];   extern float twiddle_imag_512[256];
extern float twiddle_real_256[128];   extern float twiddle_imag_256[128];
extern float twiddle_real_128[64];    extern float twiddle_imag_128[64];
extern float twiddle_real_64[32];     extern float twiddle_imag_64[32];

extern float win_2048[2048]; extern float win_1024[1024]; extern float win_512[512]; 
extern float win_256[256];   extern float win_128[128];

#endif // RADAR_CONFIG_H