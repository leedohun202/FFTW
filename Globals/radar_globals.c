#include "myfft.h"

// =========================================================================
// [1] Bit-Reversal 인덱스 테이블 (정수형/실수형 공용 사용 - 변경 없음)
// =========================================================================
int bitrev_4096[4096];
int bitrev_2048[2048]; 
int bitrev_1024[1024]; 
int bitrev_512[512]; 
int bitrev_256[256];   
int bitrev_128[128];   
int bitrev_64[64];
int bitrev_32[32];
int bitrev_16[16];

// =========================================================================
// [2] 복소수(Interleaved) Twiddle Factor 테이블
// =========================================================================
// (Butterfly 연산 절반 크기 N/2 최적화 적용, fftwf_complex 형태)
fftwf_complex twiddle_4096[2048];
fftwf_complex twiddle_2048[1024];
fftwf_complex twiddle_1024[512];
fftwf_complex twiddle_512[256];
fftwf_complex twiddle_256[128];
fftwf_complex twiddle_128[64];
fftwf_complex twiddle_64[32];
fftwf_complex twiddle_32[16];
fftwf_complex twiddle_16[8];
