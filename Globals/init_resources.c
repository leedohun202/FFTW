#include "myfft.h"
#include <math.h>

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
    // [2] FLOAT 삼각함수 (Twiddle Factor) 및 윈도우 초기화
    // =========================================================================

    for (int i = 0; i < 2048; i++) { // 4096 (Twiddle: 2048, Win: 4096)
        twiddle_real_4096[i] = (float)cos(-2.0 * PI * i / 4096); 
        twiddle_imag_4096[i] = (float)sin(-2.0 * PI * i / 4096); 
    }
    for (int i = 0; i < 4096; i++) win_4096[i] = 0.5f * (1.0f - (float)cos(2.0 * PI * i / (4096 - 1)));
    
    for (int i = 0; i < 1024; i++) { // 2048 (Twiddle: 1024, Win: 2048)
        twiddle_real_2048[i] = (float)cos(-2.0 * PI * i / 2048); 
        twiddle_imag_2048[i] = (float)sin(-2.0 * PI * i / 2048); 
    }
    for (int i = 0; i < 2048; i++) win_2048[i] = 0.5f * (1.0f - (float)cos(2.0 * PI * i / (2048 - 1)));
    
    for (int i = 0; i < 512; i++) { // 1024 (Twiddle: 512, Win: 1024)
        twiddle_real_1024[i] = (float)cos(-2.0 * PI * i / 1024); 
        twiddle_imag_1024[i] = (float)sin(-2.0 * PI * i / 1024); 
    }
    for (int i = 0; i < 1024; i++) win_1024[i] = 0.5f * (1.0f - (float)cos(2.0 * PI * i / (1024 - 1)));

    for (int i = 0; i < 256; i++) { // 512 (Twiddle: 256, Win: 512)
        twiddle_real_512[i]  = (float)cos(-2.0 * PI * i / 512);  
        twiddle_imag_512[i]  = (float)sin(-2.0 * PI * i / 512); 
    }
    for (int i = 0; i < 512; i++) win_512[i]  = 0.5f * (1.0f - (float)cos(2.0 * PI * i / (512 - 1)));

    for (int i = 0; i < 128; i++) { // 256 (Twiddle: 128, Win: 256)
        twiddle_real_256[i]  = (float)cos(-2.0 * PI * i / 256);  
        twiddle_imag_256[i]  = (float)sin(-2.0 * PI * i / 256); 
    }
    for (int i = 0; i < 256; i++) win_256[i]  = 0.5f * (1.0f - (float)cos(2.0 * PI * i / (256 - 1)));

    for (int i = 0; i < 64; i++) { // 128 (Twiddle: 64, Win: 128)
        twiddle_real_128[i]  = (float)cos(-2.0 * PI * i / 128);  
        twiddle_imag_128[i]  = (float)sin(-2.0 * PI * i / 128); 
    }
    for (int i = 0; i < 128; i++) win_128[i]  = 0.5f * (1.0f - (float)cos(2.0 * PI * i / (128 - 1)));

    for (int i = 0; i < 32; i++) { // 64 (Twiddle: 32)
        twiddle_real_64[i]   = (float)cos(-2.0 * PI * i / 64);   
        twiddle_imag_64[i]   = (float)sin(-2.0 * PI * i / 64); 
    }

    for (int i = 0; i < 8; i++) { // 16 (Twiddle: 8)
        twiddle_real_16[i]   = (float)cos(-2.0 * PI * i / 16);   
        twiddle_imag_16[i]   = (float)sin(-2.0 * PI * i / 16); 
    }

    for (int i = 0; i < 16; i++) win_16[i]  = 0.5f * (1.0f - (float)cos(2.0 * PI * i / (16 - 1)));



    
    
}