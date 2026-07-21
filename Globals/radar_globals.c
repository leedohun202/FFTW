#include "myfft.h"

// =========================================================================
// [1] Bit-Reversal 인덱스 테이블 (정수형/실수형 공용 사용)
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
// [2] FLOAT 전용 Twiddle Factor 및 Window 테이블s
// =========================================================================
// (Butterfly 연산 절반 크기 N/2 최적화 적용)
float twiddle_real_4096[2048]; float twiddle_imag_4096[2048];
float twiddle_real_2048[1024]; float twiddle_imag_2048[1024];
float twiddle_real_1024[512];  float twiddle_imag_1024[512];
float twiddle_real_512[256];   float twiddle_imag_512[256];
float twiddle_real_256[128];   float twiddle_imag_256[128];
float twiddle_real_128[64];    float twiddle_imag_128[64];
float twiddle_real_64[32];     float twiddle_imag_64[32];
float twiddle_real_16[8];     float twiddle_imag_16[8];


// window 함수(기존 코드에 window 함수가 존재하면 삭제해도 무방. 현재 demo에서는 사용하지 않음)
/*
float win_4096[4096]; float win_2048[2048]; float win_1024[1024]; float win_512[512]; 
float win_256[256];   float win_128[128]; float win_16[16];
*/