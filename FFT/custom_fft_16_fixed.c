#include "myfft.h"

extern int bitrev_16[];
extern float twiddle_real_16[];
extern float twiddle_imag_16[];

void custom_fft_16_fixed(float *__restrict__ real, float *__restrict__ imag) {
    const int N = 16;

    // 1. Bit-Reversal
    for (int i = 0; i < N; i++) { 
        int j = bitrev_16[i]; 
        if (i < j) { 
            float tr = real[i]; real[i] = real[j]; real[j] = tr; 
            float ti = imag[i]; imag[i] = imag[j]; imag[j] = ti; 
        } 
    }

    // 2. Butterfly 연산 (스칼라 전용)
    for (int step = 1; step < N; step *= 2) { 
        const int jump = step * 2; 
        const int twiddle_step = N / jump;
        
        for (int i = 0; i < N; i += jump) { 
            for (int j = 0; j < step; j++) { 
                int curr = i + j; 
                int k = curr + step; 
                
                float tr = twiddle_real_16[j * twiddle_step]; 
                float ti = twiddle_imag_16[j * twiddle_step]; 
                
                float t_real = real[k] * tr - imag[k] * ti; 
                float t_imag = real[k] * ti + imag[k] * tr; 
                
                real[k] = real[curr] - t_real; 
                imag[k] = imag[curr] - t_imag; 
                real[curr] += t_real; 
                imag[curr] += t_imag; 
            } 
        } 
    }
}