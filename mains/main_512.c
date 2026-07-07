#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h> 
#include <sys/time.h>
#include <sys/resource.h> 
#include <fftw3.h> 
#include <omp.h>
#include <malloc.h>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

#define PI 3.14159265358979323846

// [유지됨] N_CHIRPS 512 확장 (데이터 폭증)
#define N_CHIRPS    512   
#define N_ANTENNAS  4     
#define N_ANGLE     64    
#define MAX_DETECTED 10
#define BENCH_RUNS  10 
#define TILE_SIZE   16    

const double c = 3e8; const double fc = 77e9; const double B = 8e9; const double Tc = 40e-6; const double Fs = 40e6; const double S = 8e9 / 40e-6; const double lambda_c = 3e8 / 77e9; const double d_ant = (3e8 / 77e9) / 2.0;

int bitrev_2048[2048]; int bitrev_1024[1024]; int bitrev_512[512]; 
int bitrev_256[256];   int bitrev_128[128];   int bitrev_64[64];

float twiddle_real_2048[1024]; float twiddle_imag_2048[1024];
float twiddle_real_1024[512];  float twiddle_imag_1024[512];
float twiddle_real_512[256];   float twiddle_imag_512[256];
float twiddle_real_256[128];   float twiddle_imag_256[128];
float twiddle_real_128[64];    float twiddle_imag_128[64];
float twiddle_real_64[32];     float twiddle_imag_64[32];

float win_2048[2048]; float win_1024[1024]; float win_512[512]; float win_256[256]; float win_128[128];

int get_current_ram_usage_kb() {
    FILE* file = fopen("/proc/self/status", "r");
    char line[128]; int vmrss = 0;
    if (file != NULL) {
        while (fgets(line, 128, file) != NULL) {
            if (strncmp(line, "VmRSS:", 6) == 0) { sscanf(line, "VmRSS: %d kB", &vmrss); break; }
        }
        fclose(file);
    }
    return vmrss;
}

void init_resources() {
    for (int i = 0; i < 2048; i++) { int j = 0; for (int b = 0; b < 11; b++) if (i & (1 << b)) j |= (1 << (10 - b)); bitrev_2048[i] = j; }
    for (int i = 0; i < 1024; i++) { int j = 0; for (int b = 0; b < 10; b++) if (i & (1 << b)) j |= (1 << (9 - b));  bitrev_1024[i] = j; }
    for (int i = 0; i < 512; i++)  { int j = 0; for (int b = 0; b < 9; b++)  if (i & (1 << b)) j |= (1 << (8 - b));  bitrev_512[i] = j;  }
    for (int i = 0; i < 256; i++)  { int j = 0; for (int b = 0; b < 8; b++)  if (i & (1 << b)) j |= (1 << (7 - b));  bitrev_256[i] = j;  }
    for (int i = 0; i < 128; i++)  { int j = 0; for (int b = 0; b < 7; b++)  if (i & (1 << b)) j |= (1 << (6 - b));  bitrev_128[i] = j;  }
    for (int i = 0; i < 64; i++)   { int j = 0; for (int b = 0; b < 6; b++)  if (i & (1 << b)) j |= (1 << (5 - b));  bitrev_64[i] = j;   }

    for (int i = 0; i < 1024; i++) { twiddle_real_2048[i] = (float)cos(-2.0 * PI * i / 2048); twiddle_imag_2048[i] = (float)sin(-2.0 * PI * i / 2048); }
    for (int i = 0; i < 512; i++)  { twiddle_real_1024[i] = (float)cos(-2.0 * PI * i / 1024); twiddle_imag_1024[i] = (float)sin(-2.0 * PI * i / 1024); }
    for (int i = 0; i < 256; i++)  { twiddle_real_512[i]  = (float)cos(-2.0 * PI * i / 512);  twiddle_imag_512[i]  = (float)sin(-2.0 * PI * i / 512); }
    for (int i = 0; i < 128; i++)  { twiddle_real_256[i]  = (float)cos(-2.0 * PI * i / 256);  twiddle_imag_256[i]  = (float)sin(-2.0 * PI * i / 256); }
    for (int i = 0; i < 64; i++)   { twiddle_real_128[i]  = (float)cos(-2.0 * PI * i / 128);  twiddle_imag_128[i]  = (float)sin(-2.0 * PI * i / 128); }
    for (int i = 0; i < 32; i++)   { twiddle_real_64[i]   = (float)cos(-2.0 * PI * i / 64);   twiddle_imag_64[i]   = (float)sin(-2.0 * PI * i / 64); }

    for (int i = 0; i < 2048; i++) win_2048[i] = 0.5f * (1.0f - (float)cos(2.0 * PI * i / (2048 - 1)));
    for (int i = 0; i < 1024; i++) win_1024[i] = 0.5f * (1.0f - (float)cos(2.0 * PI * i / (1024 - 1)));
    for (int i = 0; i < 512; i++)  win_512[i]  = 0.5f * (1.0f - (float)cos(2.0 * PI * i / (512 - 1)));
    for (int i = 0; i < 256; i++)  win_256[i]  = 0.5f * (1.0f - (float)cos(2.0 * PI * i / (256 - 1)));
    for (int i = 0; i < 128; i++)  win_128[i]  = 0.5f * (1.0f - (float)cos(2.0 * PI * i / (128 - 1)));
}

double get_current_time_ms() {
    struct timeval tv; gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

void custom_fft_2048_fixed(float *__restrict__ real, float *__restrict__ imag) {
    for (int i = 0; i < 2048; i++) { int j = bitrev_2048[i]; if (i < j) { float tr = real[i]; real[i] = real[j]; real[j] = tr; float ti = imag[i]; imag[i] = imag[j]; imag[j] = ti; } }
    for (int step = 1; step < 2048; step *= 2) { 
        const int jump = step * 2; const int twiddle_step = 2048 / jump;
        if (step < 4) { 
            for (int i = 0; i < 2048; i += jump) { 
                for (int j = 0; j < step; j++) { 
                    int curr = i + j; int k = curr + step; 
                    float tr = twiddle_real_2048[j * twiddle_step]; float ti = twiddle_imag_2048[j * twiddle_step]; 
                    float t_real = real[k] * tr - imag[k] * ti; float t_imag = real[k] * ti + imag[k] * tr; 
                    real[k] = real[curr] - t_real; imag[k] = imag[curr] - t_imag; 
                    real[curr] = real[curr] + t_real; imag[curr] = imag[curr] + t_imag; 
                } 
            } 
        } else { 
            for (int i = 0; i < 2048; i += jump) {
                float *r_curr_ptr = &real[i]; float *i_curr_ptr = &imag[i];
                float *r_k_ptr = &real[i + step]; float *i_k_ptr = &imag[i + step];
#ifdef __aarch64__
                for (int j = 0; j < step; j += 4) { 
                    float32x4_t vr_curr = vld1q_f32(r_curr_ptr + j); float32x4_t vi_curr = vld1q_f32(i_curr_ptr + j); 
                    float32x4_t vr_k = vld1q_f32(r_k_ptr + j); float32x4_t vi_k = vld1q_f32(i_k_ptr + j); 
                    int tj = j * twiddle_step;
                    float tr_arr[4] = { twiddle_real_2048[tj], twiddle_real_2048[tj + twiddle_step], twiddle_real_2048[tj + 2*twiddle_step], twiddle_real_2048[tj + 3*twiddle_step] }; 
                    float ti_arr[4] = { twiddle_imag_2048[tj], twiddle_imag_2048[tj + twiddle_step], twiddle_imag_2048[tj + 2*twiddle_step], twiddle_imag_2048[tj + 3*twiddle_step] }; 
                    float32x4_t v_tr = vld1q_f32(tr_arr); float32x4_t v_ti = vld1q_f32(ti_arr); 
                    float32x4_t vt_real = vsubq_f32(vmulq_f32(vr_k, v_tr), vmulq_f32(vi_k, v_ti)); 
                    float32x4_t vt_imag = vaddq_f32(vmulq_f32(vr_k, v_ti), vmulq_f32(vi_k, v_tr)); 
                    vst1q_f32(r_k_ptr + j, vsubq_f32(vr_curr, vt_real)); vst1q_f32(i_k_ptr + j, vsubq_f32(vi_curr, vt_imag)); 
                    vst1q_f32(r_curr_ptr + j, vaddq_f32(vr_curr, vt_real)); vst1q_f32(i_curr_ptr + j, vaddq_f32(vi_curr, vt_imag)); 
                }
#else
                for (int j = 0; j < step; j++) {
                    float tr = twiddle_real_2048[j * twiddle_step]; float ti = twiddle_imag_2048[j * twiddle_step];
                    float t_real = r_k_ptr[j] * tr - i_k_ptr[j] * ti; float t_imag = r_k_ptr[j] * ti + i_k_ptr[j] * tr;
                    r_k_ptr[j] = r_curr_ptr[j] - t_real; i_k_ptr[j] = i_curr_ptr[j] - t_imag;
                    r_curr_ptr[j] = r_curr_ptr[j] + t_real; i_curr_ptr[j] = i_curr_ptr[j] + t_imag;
                }
#endif
            } 
        } 
    }
}

void custom_fft_1024_fixed(float *__restrict__ real, float *__restrict__ imag) {
    for (int i = 0; i < 1024; i++) { int j = bitrev_1024[i]; if (i < j) { float tr = real[i]; real[i] = real[j]; real[j] = tr; float ti = imag[i]; imag[i] = imag[j]; imag[j] = ti; } }
    for (int step = 1; step < 1024; step *= 2) { const int jump = step * 2; const int twiddle_step = 1024 / jump;
        if (step < 4) { for (int i = 0; i < 1024; i += jump) { for (int j = 0; j < step; j++) { int curr = i + j; int k = curr + step; float tr = twiddle_real_1024[j * twiddle_step]; float ti = twiddle_imag_1024[j * twiddle_step]; float t_real = real[k] * tr - imag[k] * ti; float t_imag = real[k] * ti + imag[k] * tr; real[k] = real[curr] - t_real; imag[k] = imag[curr] - t_imag; real[curr] = real[curr] + t_real; imag[curr] = imag[curr] + t_imag; } } }
        else { for (int i = 0; i < 1024; i += jump) {
#ifdef __aarch64__
                for (int j = 0; j < step; j += 4) { int curr = i + j; int k = curr + step; float32x4_t vr_curr = vld1q_f32(&real[curr]); float32x4_t vi_curr = vld1q_f32(&imag[curr]); float32x4_t vr_k = vld1q_f32(&real[k]); float32x4_t vi_k = vld1q_f32(&imag[k]); float tr_arr[4] = { twiddle_real_1024[(j+0)*twiddle_step], twiddle_real_1024[(j+1)*twiddle_step], twiddle_real_1024[(j+2)*twiddle_step], twiddle_real_1024[(j+3)*twiddle_step] }; float ti_arr[4] = { twiddle_imag_1024[(j+0)*twiddle_step], twiddle_imag_1024[(j+1)*twiddle_step], twiddle_imag_1024[(j+2)*twiddle_step], twiddle_imag_1024[(j+3)*twiddle_step] }; float32x4_t v_tr = vld1q_f32(tr_arr); float32x4_t v_ti = vld1q_f32(ti_arr); float32x4_t vt_real = vsubq_f32(vmulq_f32(vr_k, v_tr), vmulq_f32(vi_k, v_ti)); float32x4_t vt_imag = vaddq_f32(vmulq_f32(vr_k, v_ti), vmulq_f32(vi_k, v_tr)); vst1q_f32(&real[k], vsubq_f32(vr_curr, vt_real)); vst1q_f32(&imag[k], vsubq_f32(vi_curr, vt_imag)); vst1q_f32(&real[curr], vaddq_f32(vr_curr, vt_real)); vst1q_f32(&imag[curr], vaddq_f32(vi_curr, vt_imag)); }
#else
                for (int j = 0; j < step; j++) { int curr = i + j; int k = curr + step; float tr = twiddle_real_1024[j * twiddle_step]; float ti = twiddle_imag_1024[j * twiddle_step]; float t_real = real[k] * tr - imag[k] * ti; float t_imag = real[k] * ti + imag[k] * tr; real[k] = real[curr] - t_real; imag[k] = imag[curr] - t_imag; real[curr] = real[curr] + t_real; imag[curr] = imag[curr] + t_imag; }
#endif
    } } }
}

void custom_fft_512_fixed(float *__restrict__ real, float *__restrict__ imag) {
    for (int i = 0; i < 512; i++) { int j = bitrev_512[i]; if (i < j) { float tr = real[i]; real[i] = real[j]; real[j] = tr; float ti = imag[i]; imag[i] = imag[j]; imag[j] = ti; } }
    for (int step = 1; step < 512; step *= 2) { const int jump = step * 2; const int twiddle_step = 512 / jump;
        if (step < 4) { for (int i = 0; i < 512; i += jump) { for (int j = 0; j < step; j++) { int curr = i + j; int k = curr + step; float tr = twiddle_real_512[j * twiddle_step]; float ti = twiddle_imag_512[j * twiddle_step]; float t_real = real[k] * tr - imag[k] * ti; float t_imag = real[k] * ti + imag[k] * tr; real[k] = real[curr] - t_real; imag[k] = imag[curr] - t_imag; real[curr] = real[curr] + t_real; imag[curr] = imag[curr] + t_imag; } } }
        else { for (int i = 0; i < 512; i += jump) {
#ifdef __aarch64__
                for (int j = 0; j < step; j += 4) { int curr = i + j; int k = curr + step; float32x4_t vr_curr = vld1q_f32(&real[curr]); float32x4_t vi_curr = vld1q_f32(&imag[curr]); float32x4_t vr_k = vld1q_f32(&real[k]); float32x4_t vi_k = vld1q_f32(&imag[k]); float tr_arr[4] = { twiddle_real_512[(j+0)*twiddle_step], twiddle_real_512[(j+1)*twiddle_step], twiddle_real_512[(j+2)*twiddle_step], twiddle_real_512[(j+3)*twiddle_step] }; float ti_arr[4] = { twiddle_imag_512[(j+0)*twiddle_step], twiddle_imag_512[(j+1)*twiddle_step], twiddle_imag_512[(j+2)*twiddle_step], twiddle_imag_512[(j+3)*twiddle_step] }; float32x4_t v_tr = vld1q_f32(tr_arr); float32x4_t v_ti = vld1q_f32(ti_arr); float32x4_t vt_real = vsubq_f32(vmulq_f32(vr_k, v_tr), vmulq_f32(vi_k, v_ti)); float32x4_t vt_imag = vaddq_f32(vmulq_f32(vr_k, v_ti), vmulq_f32(vi_k, v_tr)); vst1q_f32(&real[k], vsubq_f32(vr_curr, vt_real)); vst1q_f32(&imag[k], vsubq_f32(vi_curr, vt_imag)); vst1q_f32(&real[curr], vaddq_f32(vr_curr, vt_real)); vst1q_f32(&imag[curr], vaddq_f32(vi_curr, vt_imag)); }
#else
                for (int j = 0; j < step; j++) { int curr = i + j; int k = curr + step; float tr = twiddle_real_512[j * twiddle_step]; float ti = twiddle_imag_512[j * twiddle_step]; float t_real = real[k] * tr - imag[k] * ti; float t_imag = real[k] * ti + imag[k] * tr; real[k] = real[curr] - t_real; imag[k] = imag[curr] - t_imag; real[curr] = real[curr] + t_real; imag[curr] = imag[curr] + t_imag; }
#endif
    } } }
}

void custom_fft_256_fixed(float *__restrict__ real, float *__restrict__ imag) {
    for (int i = 0; i < 256; i++) { int j = bitrev_256[i]; if (i < j) { float tr = real[i]; real[i] = real[j]; real[j] = tr; float ti = imag[i]; imag[i] = imag[j]; imag[j] = ti; } }
    for (int step = 1; step < 256; step *= 2) { const int jump = step * 2; const int twiddle_step = 256 / jump;
        if (step < 4) { for (int i = 0; i < 256; i += jump) { for (int j = 0; j < step; j++) { int curr = i + j; int k = curr + step; float tr = twiddle_real_256[j * twiddle_step]; float ti = twiddle_imag_256[j * twiddle_step]; float t_real = real[k] * tr - imag[k] * ti; float t_imag = real[k] * ti + imag[k] * tr; real[k] = real[curr] - t_real; imag[k] = imag[curr] - t_imag; real[curr] = real[curr] + t_real; imag[curr] = imag[curr] + t_imag; } } }
        else { for (int i = 0; i < 256; i += jump) {
#ifdef __aarch64__
                for (int j = 0; j < step; j += 4) { int curr = i + j; int k = curr + step; float32x4_t vr_curr = vld1q_f32(&real[curr]); float32x4_t vi_curr = vld1q_f32(&imag[curr]); float32x4_t vr_k = vld1q_f32(&real[k]); float32x4_t vi_k = vld1q_f32(&imag[k]); float tr_arr[4] = { twiddle_real_256[(j+0)*twiddle_step], twiddle_real_256[(j+1)*twiddle_step], twiddle_real_256[(j+2)*twiddle_step], twiddle_real_256[(j+3)*twiddle_step] }; float ti_arr[4] = { twiddle_imag_256[(j+0)*twiddle_step], twiddle_imag_256[(j+1)*twiddle_step], twiddle_imag_256[(j+2)*twiddle_step], twiddle_imag_256[(j+3)*twiddle_step] }; float32x4_t v_tr = vld1q_f32(tr_arr); float32x4_t v_ti = vld1q_f32(ti_arr); float32x4_t vt_real = vsubq_f32(vmulq_f32(vr_k, v_tr), vmulq_f32(vi_k, v_ti)); float32x4_t vt_imag = vaddq_f32(vmulq_f32(vr_k, v_ti), vmulq_f32(vi_k, v_tr)); vst1q_f32(&real[k], vsubq_f32(vr_curr, vt_real)); vst1q_f32(&imag[k], vsubq_f32(vi_curr, vt_imag)); vst1q_f32(&real[curr], vaddq_f32(vr_curr, vt_real)); vst1q_f32(&imag[curr], vaddq_f32(vi_curr, vt_imag)); }
#else
                for (int j = 0; j < step; j++) { int curr = i + j; int k = curr + step; float tr = twiddle_real_256[j * twiddle_step]; float ti = twiddle_imag_256[j * twiddle_step]; float t_real = real[k] * tr - imag[k] * ti; float t_imag = real[k] * ti + imag[k] * tr; real[k] = real[curr] - t_real; imag[k] = imag[curr] - t_imag; real[curr] = real[curr] + t_real; imag[curr] = imag[curr] + t_imag; }
#endif
    } } }
}

void custom_fft_128_fixed(float *__restrict__ real, float *__restrict__ imag) {
    for (int i = 0; i < 128; i++) { int j = bitrev_128[i]; if (i < j) { float tr = real[i]; real[i] = real[j]; real[j] = tr; float ti = imag[i]; imag[i] = imag[j]; imag[j] = ti; } }
    for (int step = 1; step < 128; step *= 2) { const int jump = step * 2; const int twiddle_step = 128 / jump;
        if (step < 4) { for (int i = 0; i < 128; i += jump) { for (int j = 0; j < step; j++) { int curr = i + j; int k = curr + step; float tr = twiddle_real_128[j * twiddle_step]; float ti = twiddle_imag_128[j * twiddle_step]; float t_real = real[k] * tr - imag[k] * ti; float t_imag = real[k] * ti + imag[k] * tr; real[k] = real[curr] - t_real; imag[k] = imag[curr] - t_imag; real[curr] = real[curr] + t_real; imag[curr] = imag[curr] + t_imag; } } }
        else { for (int i = 0; i < 128; i += jump) {
#ifdef __aarch64__
                for (int j = 0; j < step; j += 4) { int curr = i + j; int k = curr + step; float32x4_t vr_curr = vld1q_f32(&real[curr]); float32x4_t vi_curr = vld1q_f32(&imag[curr]); float32x4_t vr_k = vld1q_f32(&real[k]); float32x4_t vi_k = vld1q_f32(&imag[k]); float tr_arr[4] = { twiddle_real_128[(j+0)*twiddle_step], twiddle_real_128[(j+1)*twiddle_step], twiddle_real_128[(j+2)*twiddle_step], twiddle_real_128[(j+3)*twiddle_step] }; float ti_arr[4] = { twiddle_imag_128[(j+0)*twiddle_step], twiddle_imag_128[(j+1)*twiddle_step], twiddle_imag_128[(j+2)*twiddle_step], twiddle_imag_128[(j+3)*twiddle_step] }; float32x4_t v_tr = vld1q_f32(tr_arr); float32x4_t v_ti = vld1q_f32(ti_arr); float32x4_t vt_real = vsubq_f32(vmulq_f32(vr_k, v_tr), vmulq_f32(vi_k, v_ti)); float32x4_t vt_imag = vaddq_f32(vmulq_f32(vr_k, v_ti), vmulq_f32(vi_k, v_tr)); vst1q_f32(&real[k], vsubq_f32(vr_curr, vt_real)); vst1q_f32(&imag[k], vsubq_f32(vi_curr, vt_imag)); vst1q_f32(&real[curr], vaddq_f32(vr_curr, vt_real)); vst1q_f32(&imag[curr], vaddq_f32(vi_curr, vt_imag)); }
#else
                for (int j = 0; j < step; j++) { int curr = i + j; int k = curr + step; float tr = twiddle_real_128[j * twiddle_step]; float ti = twiddle_imag_128[j * twiddle_step]; float t_real = real[k] * tr - imag[k] * ti; float t_imag = real[k] * ti + imag[k] * tr; real[k] = real[curr] - t_real; imag[k] = imag[curr] - t_imag; real[curr] = real[curr] + t_real; imag[curr] = imag[curr] + t_imag; }
#endif
    } } }
}

void custom_fft_64_fixed(float *__restrict__ real, float *__restrict__ imag) {
    for (int i = 0; i < 64; i++) { int j = bitrev_64[i]; if (i < j) { float tr = real[i]; real[i] = real[j]; real[j] = tr; float ti = imag[i]; imag[i] = imag[j]; imag[j] = ti; } }
    for (int step = 1; step < 64; step *= 2) { const int jump = step * 2; const int twiddle_step = 64 / jump;
        if (step < 4) { for (int i = 0; i < 64; i += jump) { for (int j = 0; j < step; j++) { int curr = i + j; int k = curr + step; float tr = twiddle_real_64[j * twiddle_step]; float ti = twiddle_imag_64[j * twiddle_step]; float t_real = real[k] * tr - imag[k] * ti; float t_imag = real[k] * ti + imag[k] * tr; real[k] = real[curr] - t_real; imag[k] = imag[curr] - t_imag; real[curr] = real[curr] + t_real; imag[curr] = imag[curr] + t_imag; } } }
        else { for (int i = 0; i < 64; i += jump) {
#ifdef __aarch64__
                for (int j = 0; j < step; j += 4) { int curr = i + j; int k = curr + step; float32x4_t vr_curr = vld1q_f32(&real[curr]); float32x4_t vi_curr = vld1q_f32(&imag[curr]); float32x4_t vr_k = vld1q_f32(&real[k]); float32x4_t vi_k = vld1q_f32(&imag[k]); float tr_arr[4] = { twiddle_real_64[(j+0)*twiddle_step], twiddle_real_64[(j+1)*twiddle_step], twiddle_real_64[(j+2)*twiddle_step], twiddle_real_64[(j+3)*twiddle_step] }; float ti_arr[4] = { twiddle_imag_64[(j+0)*twiddle_step], twiddle_imag_64[(j+1)*twiddle_step], twiddle_imag_64[(j+2)*twiddle_step], twiddle_imag_64[(j+3)*twiddle_step] }; float32x4_t v_tr = vld1q_f32(tr_arr); float32x4_t v_ti = vld1q_f32(ti_arr); float32x4_t vt_real = vsubq_f32(vmulq_f32(vr_k, v_tr), vmulq_f32(vi_k, v_ti)); float32x4_t vt_imag = vaddq_f32(vmulq_f32(vr_k, v_ti), vmulq_f32(vi_k, v_tr)); vst1q_f32(&real[k], vsubq_f32(vr_curr, vt_real)); vst1q_f32(&imag[k], vsubq_f32(vi_curr, vt_imag)); vst1q_f32(&real[curr], vaddq_f32(vr_curr, vt_real)); vst1q_f32(&imag[curr], vaddq_f32(vi_curr, vt_imag)); }
#else
                for (int j = 0; j < step; j++) { int curr = i + j; int k = curr + step; float tr = twiddle_real_64[j * twiddle_step]; float ti = twiddle_imag_64[j * twiddle_step]; float t_real = real[k] * tr - imag[k] * ti; float t_imag = real[k] * ti + imag[k] * tr; real[k] = real[curr] - t_real; imag[k] = imag[curr] - t_imag; real[curr] = real[curr] + t_real; imag[curr] = imag[curr] + t_imag; }
#endif
    } } }
}

void transpose_radar_cube(const float *__restrict__ src, float *__restrict__ dst, int n_samples) {
    #pragma omp parallel for collapse(3)
    for (int ant = 0; ant < N_ANTENNAS; ant++) {
        for (int c_blk = 0; c_blk < N_CHIRPS; c_blk += TILE_SIZE) {
            for (int r_blk = 0; r_blk < n_samples; r_blk += TILE_SIZE) {
                for (int c = c_blk; c < c_blk + TILE_SIZE; c++) {
                    for (int r = r_blk; r < r_blk + TILE_SIZE; r++) {
                        int src_idx = ant * (N_CHIRPS * n_samples) + c * n_samples + r;
                        int dst_idx = ant * (n_samples * N_CHIRPS) + r * N_CHIRPS + c;
                        dst[dst_idx] = src[src_idx];
                    }
                }
            }
        }
    }
}

void execute_custom_pipeline(float *__restrict__ cube_real, float *__restrict__ cube_imag, float *__restrict__ trans_cust_r, float *__restrict__ trans_cust_i, int n_samples) {
    #pragma omp parallel for collapse(2)
    for (int ant = 0; ant < N_ANTENNAS; ant++) {
        for (int chirp = 0; chirp < N_CHIRPS; chirp++) {
            int offset = ant * (N_CHIRPS * n_samples) + chirp * n_samples;
            if (n_samples == 1024) {
                for(int i = 0; i < 1024; i++) { cube_real[offset+i] *= win_1024[i]; cube_imag[offset+i] *= win_1024[i]; }
                custom_fft_1024_fixed(&cube_real[offset], &cube_imag[offset]);
            } else if (n_samples == 2048) {
                for(int i = 0; i < 2048; i++) { cube_real[offset+i] *= win_2048[i]; cube_imag[offset+i] *= win_2048[i]; }
                custom_fft_2048_fixed(&cube_real[offset], &cube_imag[offset]);
            }
        }
    }

    transpose_radar_cube(cube_real, trans_cust_r, n_samples); 
    transpose_radar_cube(cube_imag, trans_cust_i, n_samples);

    #pragma omp parallel for collapse(2)
    for (int ant = 0; ant < N_ANTENNAS; ant++) {
        for (int r = 0; r < n_samples; r++) {
            int offset = ant * (n_samples * N_CHIRPS) + r * N_CHIRPS;
            for (int chirp = 0; chirp < N_CHIRPS; chirp++) { 
                trans_cust_r[offset+chirp] *= win_512[chirp]; 
                trans_cust_i[offset+chirp] *= win_512[chirp]; 
            }
            custom_fft_512_fixed(&trans_cust_r[offset], &trans_cust_i[offset]);
        }
    }
}

void execute_fftw_pipeline_optimized(fftwf_complex* cube, fftwf_plan p_range, fftwf_plan p_doppler, int n_samples) {
    int total_elements = N_ANTENNAS * N_CHIRPS * n_samples;
    for (int idx = 0; idx < total_elements; idx++) {
        int i = idx % n_samples;
        if (n_samples == 1024) { cube[idx][0] *= win_1024[i]; cube[idx][1] *= win_1024[i]; }
        else if (n_samples == 2048) { cube[idx][0] *= win_2048[i]; cube[idx][1] *= win_2048[i]; }
    }
    fftwf_execute_dft(p_range, cube, cube);
    for (int idx = 0; idx < total_elements; idx++) {
        int chirp = (idx / n_samples) % N_CHIRPS;
        cube[idx][0] *= win_512[chirp]; 
        cube[idx][1] *= win_512[chirp];
    }
    for (int ant = 0; ant < N_ANTENNAS; ant++) {
        int offset = ant * (N_CHIRPS * n_samples);
        fftwf_execute_dft(p_doppler, &cube[offset], &cube[offset]); 
    }
}

void run_benchmark_for_size(int n_samples) {
    int total_elements = N_ANTENNAS * N_CHIRPS * n_samples;
    
    printf("\n====================================================\n");
    printf(" 🚀 [격리 벤치마크] Range Samples = %d 파이프라인\n", n_samples);
    printf("====================================================\n");

    float *raw_real = (float *)calloc(total_elements, sizeof(float));
    float *raw_imag = (float *)calloc(total_elements, sizeof(float));

    double target_R[] = {9.25, 22.40}; double target_v[] = {7.30, -15.80}; 
    for (int t_idx = 0; t_idx < 2; t_idx++) {
        double R = target_R[t_idx]; double v = target_v[t_idx]; double theta_rad = (-20.0 + t_idx * 55.0) * PI / 180.0;
        double f_R = (2.0 * S * R) / c; double phase_doppler = (4.0 * PI * v) / lambda_c * Tc; double phase_angle = (2.0 * PI * d_ant * sin(theta_rad)) / lambda_c;
        for (int ant = 0; ant < N_ANTENNAS; ant++) {
            for (int chirp = 0; chirp < N_CHIRPS; chirp++) {
                for (int n = 0; n < n_samples; n++) {
                    double phase = (2.0 * PI * f_R * ((double)n/Fs)) + (phase_doppler * chirp) + (phase_angle * ant);
                    int idx = ant * (N_CHIRPS * n_samples) + chirp * n_samples + n;
                    raw_real[idx] += (float)cos(phase); raw_imag[idx] += (float)sin(phase);
                }
            }
        }
    }

    // ---------------------------------------------------------
    // [1 라운드] FFTW3 계측
    // ---------------------------------------------------------
    malloc_trim(0); 
    int base_fftw_mem = get_current_ram_usage_kb();
    
    fftwf_complex *fftw_cube = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * total_elements);
    int n_range[] = {n_samples};
    fftwf_plan p_range = fftwf_plan_many_dft(1, n_range, N_ANTENNAS * N_CHIRPS, fftw_cube, NULL, 1, n_samples, fftw_cube, NULL, 1, n_samples, FFTW_FORWARD, FFTW_MEASURE);
    int n_doppler[] = {N_CHIRPS};
    fftwf_plan p_doppler = fftwf_plan_many_dft(1, n_doppler, n_samples, fftw_cube, NULL, n_samples, 1, fftw_cube, NULL, n_samples, 1, FFTW_FORWARD, FFTW_MEASURE);

    for(int i=0; i<total_elements; i++) { fftw_cube[i][0] = raw_real[i]; fftw_cube[i][1] = raw_imag[i]; }

    int peak_fftw_mem = get_current_ram_usage_kb();
    double actual_fftw_mb = (peak_fftw_mem - base_fftw_mem) / 1024.0;

    double start_fftw = get_current_time_ms();
    for(int i = 0; i < BENCH_RUNS; i++) {
        for(int j=0; j<total_elements; j++) { fftw_cube[j][0] = raw_real[j]; fftw_cube[j][1] = raw_imag[j]; }
        execute_fftw_pipeline_optimized(fftw_cube, p_range, p_doppler, n_samples);
    }
    double avg_fftw_ms = (get_current_time_ms() - start_fftw) / BENCH_RUNS;
    fftwf_destroy_plan(p_range); fftwf_destroy_plan(p_doppler); fftwf_free(fftw_cube);

    // ---------------------------------------------------------
    // [2 라운드] Custom 계측
    // ---------------------------------------------------------
    malloc_trim(0);
    int base_cust_mem = get_current_ram_usage_kb();

    float *cust_cube_r = (float *)malloc(total_elements * sizeof(float));
    float *cust_cube_i = (float *)malloc(total_elements * sizeof(float));
    float *trans_cust_r = (float *)malloc(total_elements * sizeof(float));
    float *trans_cust_i = (float *)malloc(total_elements * sizeof(float));

    for(int i=0; i<total_elements; i++) { cust_cube_r[i] = raw_real[i]; cust_cube_i[i] = raw_imag[i]; trans_cust_r[i] = 0; trans_cust_i[i] = 0; }

    int peak_cust_mem = get_current_ram_usage_kb();
    double actual_cust_mb = (peak_cust_mem - base_cust_mem) / 1024.0;

    double start_cust = get_current_time_ms();
    for(int i = 0; i < BENCH_RUNS; i++) {
        memcpy(cust_cube_r, raw_real, total_elements * sizeof(float));
        memcpy(cust_cube_i, raw_imag, total_elements * sizeof(float));
        execute_custom_pipeline(cust_cube_r, cust_cube_i, trans_cust_r, trans_cust_i, n_samples);
    }
    double avg_custom_ms = (get_current_time_ms() - start_cust) / BENCH_RUNS;

    free(cust_cube_r); free(cust_cube_i); free(trans_cust_r); free(trans_cust_i); free(raw_real); free(raw_imag);
    malloc_trim(0);

    printf(" [결과] 💻 FFTW3 Pipeline  : %6.2f ms / Frame (OS 실측 할당 RAM: %5.2f MB)\n", avg_fftw_ms, actual_fftw_mb);
    printf(" [결과] 🚀 Custom Pipeline : %6.2f ms / Frame (OS 실측 할당 RAM: %5.2f MB)\n", avg_custom_ms, actual_cust_mb);
}

void profile_individual_operation(int size, void (*custom_func)(float*, float*)) {
    float *r = (float*)calloc(size, sizeof(float));
    float *i = (float*)calloc(size, sizeof(float));
    fftwf_complex *f_in = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * size);
    fftwf_plan p = fftwf_plan_dft_1d(size, f_in, f_in, FFTW_FORWARD, FFTW_MEASURE);
    
    const int iterations = 1000; struct timeval start, end;
    
    gettimeofday(&start, NULL);
    for(int k=0; k<iterations; k++) custom_func(r, i);
    gettimeofday(&end, NULL);
    double custom_us = ((end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec)) / iterations;

    gettimeofday(&start, NULL);
    for(int k=0; k<iterations; k++) fftwf_execute(p);
    gettimeofday(&end, NULL);
    double fftw_us = ((end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec)) / iterations;

    printf(" 🔹 %4d 포인트 1D FFT 연산 -> [Custom]: %7.2f µs  |  [FFTW3]: %7.2f µs (비율: %.2fx)\n", size, custom_us, fftw_us, custom_us / fftw_us);
    fftwf_destroy_plan(p); free(r); free(i); fftwf_free(f_in);
}

int main() {
    fftwf_init_threads();
    fftwf_plan_with_nthreads(4);
    init_resources();
    
    printf("====================================================\n");
    printf("  3D Radar Pipeline: Ultimate Accuracy & Speed Test \n");
    printf("====================================================\n\n");

    run_benchmark_for_size(1024);
    run_benchmark_for_size(2048);

    printf("\n====================================================\n");
    printf(" [PART 4] 1D FFT 개별 연산 실행 속도 비교 (마이크로초 단위)\n");
    printf("----------------------------------------------------\n");
    profile_individual_operation(2048, custom_fft_2048_fixed);
    profile_individual_operation(1024, custom_fft_1024_fixed);
    profile_individual_operation(512,  custom_fft_512_fixed);
    profile_individual_operation(256,  custom_fft_256_fixed);
    profile_individual_operation(128,  custom_fft_128_fixed);
    profile_individual_operation(64,   custom_fft_64_fixed);
    printf("====================================================\n");
    
    struct rusage usage; getrusage(RUSAGE_SELF, &usage);
    printf(" 📊 프로세스 라이프사이클 Peak 물리 메모리(Max RSS) : %.2f MB\n", usage.ru_maxrss / 1024.0);
    printf("====================================================\n");

    fftwf_cleanup_threads();
    return 0;
}