#include "radar_utils.h"
#include <math.h>
#include <stdint.h>

#ifndef PI
#define PI 3.14159265358979323846
#endif

/**
 * @brief 프로그램 초기화 단계에서 메모리를 1회만 연산하여 LUT 테이블 생성
 * @details 
 * - 레이더 신호 처리에 필요한 모든 크기(4096 ~ 64)의 고정 배열을 사전에 계산합니다.
 * - 1. Bit-Reversal 주소 테이블 (메모리 스와핑용)
 * - 2. Float Twiddle Factor (N/2 크기) & Hanning Window
 * - 3. Int16 Twiddle Factor (분기 제거를 위해 전체 N 크기, Q15 포맷) & Int16 Window
 */
void init_resources() {

    // =========================================================================
    // [1] Bit-Reversal 주소 테이블 초기화
    // =========================================================================
    
    // 4096 크기 (12 bits)
    for (int i = 0; i < 4096; i++) { 
        int j = 0; 
        for (int b = 0; b < 12; b++) {
            if (i & (1 << b)) j |= (1 << (11 - b)); 
        }
        bitrev_4096[i] = j; 
    }

    // 2048 크기 (11 bits)
    for (int i = 0; i < 2048; i++) { 
        int j = 0; 
        for (int b = 0; b < 11; b++) {
            if (i & (1 << b)) j |= (1 << (10 - b)); 
        }
        bitrev_2048[i] = j; 
    }
    
    // 1024 크기 (10 bits)
    for (int i = 0; i < 1024; i++) { 
        int j = 0; 
        for (int b = 0; b < 10; b++) {
            if (i & (1 << b)) j |= (1 << (9 - b));  
        }
        bitrev_1024[i] = j;  
    }

    // 512 크기 (9 bits)
    for (int i = 0; i < 512; i++) { 
        int j = 0; 
        for (int b = 0; b < 9; b++) {
            if (i & (1 << b)) j |= (1 << (8 - b));  
        }
        bitrev_512[i] = j;  
    }

    // 256 크기 (8 bits)
    for (int i = 0; i < 256; i++) { 
        int j = 0; 
        for (int b = 0; b < 8; b++) {
            if (i & (1 << b)) j |= (1 << (7 - b));  
        }
        bitrev_256[i] = j;  
    }

    // 128 크기 (7 bits)
    for (int i = 0; i < 128; i++) { 
        int j = 0; 
        for (int b = 0; b < 7; b++) {
            if (i & (1 << b)) j |= (1 << (6 - b));  
        }
        bitrev_128[i] = j;  
    }

    // 64 크기 (6 bits)
    for (int i = 0; i < 64; i++) { 
        int j = 0; 
        for (int b = 0; b < 6; b++) {
            if (i & (1 << b)) j |= (1 << (5 - b));  
        }
        bitrev_64[i] = j;   
    }

    // 16 크기 (4 bits)
    for (int i = 0; i < 16; i++) { 
        int j = 0; 
        for (int b = 0; b < 4; b++) {
            if (i & (1 << b)) j |= (1 << (3 - b));  
        }
        bitrev_16[i] = j;   
    }

    // =========================================================================
    // [3] 🔥 INT16 고정소수점(Q15 포맷) 전용 테이블 초기화
    // =========================================================================
    // 주의: Radix-8 분기 제거를 위해 N/2가 아닌 풀사이즈(N)로 생성합니다.
    

    // 2048 크기
    for (int i = 0; i < 2048; i++) {
        twiddle_int16_real_2048[i] = (int16_t)round(cos(-2.0 * PI * i / 2048) * 32767.0);
        twiddle_int16_imag_2048[i] = (int16_t)round(sin(-2.0 * PI * i / 2048) * 32767.0);
        win_int16_2048[i] = (int16_t)round((0.5 * (1.0 - cos(2.0 * PI * i / (2048 - 1)))) * 32767.0);
    }

    // 1024 크기
    for (int i = 0; i < 1024; i++) {
        twiddle_int16_real_1024[i] = (int16_t)round(cos(-2.0 * PI * i / 1024) * 32767.0);
        twiddle_int16_imag_1024[i] = (int16_t)round(sin(-2.0 * PI * i / 1024) * 32767.0);
        win_int16_1024[i] = (int16_t)round((0.5 * (1.0 - cos(2.0 * PI * i / (1024 - 1)))) * 32767.0);
    }

    // 512 크기
    for (int i = 0; i < 512; i++) {
        twiddle_int16_real_512[i] = (int16_t)round(cos(-2.0 * PI * i / 512) * 32767.0);
        twiddle_int16_imag_512[i] = (int16_t)round(sin(-2.0 * PI * i / 512) * 32767.0);
        win_int16_512[i] = (int16_t)round((0.5 * (1.0 - cos(2.0 * PI * i / (512 - 1)))) * 32767.0);
    }

    // 256 크기
    for (int i = 0; i < 256; i++) {
        twiddle_int16_real_256[i] = (int16_t)round(cos(-2.0 * PI * i / 256) * 32767.0);
        twiddle_int16_imag_256[i] = (int16_t)round(sin(-2.0 * PI * i / 256) * 32767.0);
        win_int16_256[i] = (int16_t)round((0.5 * (1.0 - cos(2.0 * PI * i / (256 - 1)))) * 32767.0);
    }

    // 128 크기
    for (int i = 0; i < 128; i++) {
        twiddle_int16_real_128[i] = (int16_t)round(cos(-2.0 * PI * i / 128) * 32767.0);
        twiddle_int16_imag_128[i] = (int16_t)round(sin(-2.0 * PI * i / 128) * 32767.0);
        win_int16_128[i] = (int16_t)round((0.5 * (1.0 - cos(2.0 * PI * i / (128 - 1)))) * 32767.0);
    }

    // 64 크기 (윈도우 불필요)
    for (int i = 0; i < 64; i++) {
        twiddle_int16_real_64[i] = (int16_t)round(cos(-2.0 * PI * i / 64) * 32767.0);
        twiddle_int16_imag_64[i] = (int16_t)round(sin(-2.0 * PI * i / 64) * 32767.0);
    }

    // =========================================================================
    // [4] 🔥 256포인트 초해상도 DoA 각도 복원용 LUT 초기화 (신규 추가)
    // =========================================================================
    for (int k = 0; k < 256; k++) {
        // FFT Bin 인덱스를 부호 있는 공간 주파수 영역으로 언래핑
        int fft_bin = (k >= 256 / 2) ? k - 256 : k;
        
        // 공간 주파수 $2\pi \cdot \frac{d}{\lambda} \sin(\theta)$ 역산
        double sin_theta = (fft_bin / 256.0) * (lambda_c / d_ant);
        
        if (sin_theta > 1.0) sin_theta = 1.0;
        else if (sin_theta < -1.0) sin_theta = -1.0;
        
        // 최종 각도 물리 좌표를 전역 마스터 테이블에 바인딩
        doa_angle_lut_256[k] = (float)(asin(sin_theta) * 180.0 / PI);
    }
}