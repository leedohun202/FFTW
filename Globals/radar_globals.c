#include "radar_config.h"
#include <stdint.h> // int16_t 자료형 사용을 위한 필수 헤더

// =========================================================================
// [1] 물리 상수 (기존 유지)
// =========================================================================
const double c = 3e8; 
const double fc = 77e9; 
const double B = 8e9; 
const double Tc = 40e-6; 
const double Fs = 40e6; 
const double S = 8e9 / 40e-6; 
const double lambda_c = 3e8 / 77e9; 
const double d_ant = (3e8 / 77e9) / 2.0;

// =========================================================================
// [2] Bit-Reversal 인덱스 테이블 (정수형/실수형 공용 사용)
// =========================================================================
int bitrev_4096[4096];
int bitrev_2048[2048]; 
int bitrev_1024[1024]; 
int bitrev_512[512]; 
int bitrev_256[256];   
int bitrev_128[128];   
int bitrev_64[64];
int bitrev_16[16];


// =========================================================================
// [3] FLOAT 전용 Twiddle Factor 및 Window 테이블 (기존 유지)
// =========================================================================
// (나비 연산 절반 크기 N/2 최적화 적용)
float twiddle_real_4096[2048]; float twiddle_imag_4096[2048];
float twiddle_real_2048[1024]; float twiddle_imag_2048[1024];
float twiddle_real_1024[512];  float twiddle_imag_1024[512];
float twiddle_real_512[256];   float twiddle_imag_512[256];
float twiddle_real_256[128];   float twiddle_imag_256[128];
float twiddle_real_128[64];    float twiddle_imag_128[64];
float twiddle_real_64[32];     float twiddle_imag_64[32];
float twiddle_real_16[8];     float twiddle_imag_16[8];

float win_4096[4096]; float win_2048[2048]; float win_1024[1024]; float win_512[512]; 
float win_256[256];   float win_128[128]; float win_16[16];

float doa_angle_lut_256[256];

// =========================================================================
// [4] 🔥 INT16 전용 Twiddle Factor 테이블 (신규 추가, Q15 포맷)
// =========================================================================
// (Radix-8 분기문 제로화(Zero-branch) 가속을 위해 풀사이즈 N으로 할당)
int16_t twiddle_int16_real_2048[2048]; int16_t twiddle_int16_imag_2048[2048];
int16_t twiddle_int16_real_1024[1024]; int16_t twiddle_int16_imag_1024[1024];
int16_t twiddle_int16_real_512[512];   int16_t twiddle_int16_imag_512[512];
int16_t twiddle_int16_real_256[256];   int16_t twiddle_int16_imag_256[256];
int16_t twiddle_int16_real_128[128];   int16_t twiddle_int16_imag_128[128];
int16_t twiddle_int16_real_64[64];     int16_t twiddle_int16_imag_64[64];

// =========================================================================
// [5] 🔥 INT16 전용 Hanning Window 테이블 (신규 추가, Q15 포맷)
// =========================================================================
// (ADC 원시 데이터에 바로 정수 곱셈을 때리기 위한 배열)
int16_t win_int16_2048[2048]; 
int16_t win_int16_1024[1024]; 
int16_t win_int16_512[512]; 
int16_t win_int16_256[256];   
int16_t win_int16_128[128];