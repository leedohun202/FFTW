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
    // -------------------------------------------------------------------------
    // [FFTW3 Wisdom 무기 장착]
    // -------------------------------------------------------------------------
    FILE *wf = fopen(FFTW_WISDOM_FILE, "r");
    if (wf != NULL) {
        // 기존에 정밀 최적화된 설계도 파일이 있다면 0.001초 만에 즉시 로드
        printf(" 💾 [Wisdom] 기존 라즈베리파이 5 가속 설계도를 로드합니다.\n");
        fftwf_import_wisdom_from_file(wf);
        fclose(wf);
    } else {
        // 설계도 파일이 없다면 최초 1회 정밀 학습(Patient) 모드 활성화 알림
        printf(" 🔬 [Wisdom] 최적화 설계도가 없습니다. 하드웨어 한계 측정 모드를 실행합니다.\n");
        printf("    (최초 1회만 실행되며, 수초에서 수분의 시간이 소요될 수 있습니다...)\n");
    }
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