#include "radar_utils.h"

/**
 * @brief 프로그램 초기화 단계에서 메모리를 1회만 연산하여 LUT 테이블 생성
 * @details 
 * - 레이더 신호 처리에 필요한 모든 크기(2048 ~ 64)의 고정 배열을 사전에 계산합니다.
 * - 1. Bit-Reversal 주소 테이블 (메모리 스와핑용)
 * - 2. Twiddle Factor (FFT 회전 인자용 삼각함수 테이블)
 * - 3. Hanning Window (주파수 누출 방지용 윈도우 테이블)
 */
void init_resources() {
    // 🎯 [수정 완료] Wisdom 로드 로직은 main.c로 완전히 이관 및 격리되었으므로 삭제했습니다.

    // =========================================================================
    // [1] Bit-Reversal 주소 테이블 초기화
    // =========================================================================
    
    // 2048 크기 (11 bits)
    for (int i = 0; i < 2048; i++) { 
        int j = 0; 
        for (int b = 0; b < 11; b++) {
            if (i & (1 << b)) {
                j |= (1 << (10 - b)); 
            }
        }
        bitrev_2048[i] = j; 
    }
    
    // 1024 크기 (10 bits)
    for (int i = 0; i < 1024; i++) { 
        int j = 0; 
        for (int b = 0; b < 10; b++) {
            if (i & (1 << b)) {
                j |= (1 << (9 - b));  
            }
        }
        bitrev_1024[i] = j;  
    }

    // 512 크기 (9 bits)
    for (int i = 0; i < 512; i++) { 
        int j = 0; 
        for (int b = 0; b < 9; b++) {
            if (i & (1 << b)) {
                j |= (1 << (8 - b));  
            }
        }
        bitrev_512[i] = j;  
    }

    // 256 크기 (8 bits)
    for (int i = 0; i < 256; i++) { 
        int j = 0; 
        for (int b = 0; b < 8; b++) {
            if (i & (1 << b)) {
                j |= (1 << (7 - b));  
            }
        }
        bitrev_256[i] = j;  
    }

    // 128 크기 (7 bits)
    for (int i = 0; i < 128; i++) { 
        int j = 0; 
        for (int b = 0; b < 7; b++) {
            if (i & (1 << b)) {
                j |= (1 << (6 - b));  
            }
        }
        bitrev_128[i] = j;  
    }

    // 64 크기 (6 bits)
    for (int i = 0; i < 64; i++) { 
        int j = 0; 
        for (int b = 0; b < 6; b++) {
            if (i & (1 << b)) {
                j |= (1 << (5 - b));  
            }
        }
        bitrev_64[i] = j;   
    }


    // =========================================================================
    // [2] 삼각함수 (Twiddle Factor) 테이블 초기화
    // =========================================================================
    // 나비 연산(Butterfly)에 필요한 절반 크기(N/2)의 복소수 회전 인자를 계산합니다.
    
    // 2048 크기
    for (int i = 0; i < 1024; i++) { 
        twiddle_real_2048[i] = (float)cos(-2.0 * PI * i / 2048); 
        twiddle_imag_2048[i] = (float)sin(-2.0 * PI * i / 2048); 
    }
    
    // 1024 크기
    for (int i = 0; i < 512; i++) { 
        twiddle_real_1024[i] = (float)cos(-2.0 * PI * i / 1024); 
        twiddle_imag_1024[i] = (float)sin(-2.0 * PI * i / 1024); 
    }

    // 512 크기
    for (int i = 0; i < 256; i++) { 
        twiddle_real_512[i]  = (float)cos(-2.0 * PI * i / 512);  
        twiddle_imag_512[i]  = (float)sin(-2.0 * PI * i / 512); 
    }

    // 256 크기
    for (int i = 0; i < 128; i++) { 
        twiddle_real_256[i]  = (float)cos(-2.0 * PI * i / 256);  
        twiddle_imag_256[i]  = (float)sin(-2.0 * PI * i / 256); 
    }

    // 128 크기
    for (int i = 0; i < 64; i++) { 
        twiddle_real_128[i]  = (float)cos(-2.0 * PI * i / 128);  
        twiddle_imag_128[i]  = (float)sin(-2.0 * PI * i / 128); 
    }

    // 64 크기
    for (int i = 0; i < 32; i++) { 
        twiddle_real_64[i]   = (float)cos(-2.0 * PI * i / 64);   
        twiddle_imag_64[i]   = (float)sin(-2.0 * PI * i / 64); 
    }


    // =========================================================================
    // [3] 해닝 윈도우 (Hanning Window) 테이블 초기화
    // =========================================================================
    // 양 끝단을 부드럽게 깎아주어 주파수 영역에서의 Sidelobe 누출을 방지합니다.
    
    // 2048 크기
    for (int i = 0; i < 2048; i++) {
        win_2048[i] = 0.5f * (1.0f - (float)cos(2.0 * PI * i / (2048 - 1)));
    }
        
    // 1024 크기
    for (int i = 0; i < 1024; i++) {
        win_1024[i] = 0.5f * (1.0f - (float)cos(2.0 * PI * i / (1024 - 1)));
    }
    
    // 512 크기
    for (int i = 0; i < 512; i++) {
        win_512[i]  = 0.5f * (1.0f - (float)cos(2.0 * PI * i / (512 - 1)));
    }
    
    // 256 크기
    for (int i = 0; i < 256; i++) {
        win_256[i]  = 0.5f * (1.0f - (float)cos(2.0 * PI * i / (256 - 1)));
    }
    
    // 128 크기
    for (int i = 0; i < 128; i++) {
        win_128[i]  = 0.5f * (1.0f - (float)cos(2.0 * PI * i / (128 - 1)));
    }
}