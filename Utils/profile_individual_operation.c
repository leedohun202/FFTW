#include "radar_config.h"
#include "radar_utils.h"
#include "radar_fft.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>
#include <fftw3.h>

// ---------------------------------------------------------
// [0] FFTW3 단독 프로파일러 (기준 속도 측정 및 반환)
// ---------------------------------------------------------
double profile_fftw_operation(int size) {
    fftwf_complex *f_in = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * size);
    
    unsigned int fftw_flags = FFTW_ESTIMATE;
#if defined(FFTW_MODE_MEASURE)
    fftw_flags = FFTW_MEASURE;
#elif defined(FFTW_MODE_PATIENT)
    fftw_flags = FFTW_PATIENT;
#endif

    fftwf_plan p = fftwf_plan_dft_1d(size, f_in, f_in, FFTW_FORWARD, fftw_flags);
    
#if defined(FFTW_MODE_PATIENT)
    FILE *wf = fopen("radar_wisdom.wisdom", "w");
    if (wf != NULL) {
        fftwf_export_wisdom_to_file(wf);
        fclose(wf);
    }
#endif

    const int iterations = 1000; struct timeval start, end;
    
    gettimeofday(&start, NULL);
    for(int k=0; k<iterations; k++) fftwf_execute(p);
    gettimeofday(&end, NULL);
    double fftw_us = ((end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec)) / iterations;

    printf("  🔹 %4d 포인트 [%-13s] -> 기준 속도: %7.2f µs\n", size, "FFTW3 Baseline", fftw_us);
    
    fftwf_destroy_plan(p); 
    fftwf_free(f_in);
    
    return fftw_us;
}

// ---------------------------------------------------------
// [1] Float 커널 전용 프로파일러
// ---------------------------------------------------------
void profile_float_operation(int size, const char* name, void (*custom_func)(float*, float*), double fftw_us) {
    float *r = NULL, *i = NULL;
    posix_memalign((void**)&r, 64, size * sizeof(float));
    posix_memalign((void**)&i, 64, size * sizeof(float));

    const int iterations = 1000; struct timeval start, end;
    
    gettimeofday(&start, NULL);
    for(int k=0; k<iterations; k++) custom_func(r, i);
    gettimeofday(&end, NULL);
    double custom_us = ((end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec)) / iterations;

    printf("  🔸 %4d 포인트 [%-13s] -> Custom: %7.2f µs (FFTW3 대비 %5.2fx)\n", 
           size, name, custom_us, custom_us / fftw_us);
    
    free(r); free(i);
}

// ---------------------------------------------------------
// [2] Int16 커널 전용 프로파일러 (+8 패딩 가드룸 장착)
// ---------------------------------------------------------
void profile_int16_operation(int size, const char* name, void (*custom_func)(int16_t*, int16_t*), double fftw_us) {
    int16_t *r = NULL, *i = NULL;
    posix_memalign((void**)&r, 64, size * sizeof(int16_t));
    posix_memalign((void**)&i, 64, size * sizeof(int16_t));
    
    const int iterations = 1000; struct timeval start, end;
    
    gettimeofday(&start, NULL);
    for(int k=0; k<iterations; k++) custom_func(r, i);
    gettimeofday(&end, NULL);
    double custom_us = ((end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec)) / iterations;

    printf("  🔸 %4d 포인트 [%-13s] -> Custom: %7.2f µs (FFTW3 대비 %5.2fx)\n", 
           size, name, custom_us, custom_us / fftw_us);
    
    free(r); free(i);
}

// ---------------------------------------------------------
// [3] 마이크로초 벤치마크 런처 (5채널 확장 및 오타 교정 완료)
// ---------------------------------------------------------
void run_individual_operation_benchmark(void) {
    printf("\n====================================================\n");
    printf(" [PART 4] 1D FFT 개별 연산 실행 속도 비교 (마이크로초 단위)\n");
    printf("====================================================\n");
    
    double ref_us;

    // 🔹 2048 포인트
    ref_us = profile_fftw_operation(2048);
    profile_float_operation(2048, "Float R-2", custom_fft_2048_fixed, ref_us);
    profile_float_operation(2048, "Float R-4", custom_fft_2048_radix4, ref_us);
    profile_int16_operation(2048, "Int16 R-2", custom_fft_2048_int16, ref_us);
    profile_int16_operation(2048, "Int16 R-4", custom_fft_2048_int16_radix4, ref_us); // 🔥 신규 R-4 연결
    printf(" ----------------------------------------------------\n");

    // 🔹 1024 포인트
    ref_us = profile_fftw_operation(1024);
    profile_float_operation(1024, "Float R-2", custom_fft_1024_fixed,  ref_us);
    profile_float_operation(1024, "Float R-4", custom_fft_1024_radix4, ref_us);
    profile_int16_operation(1024, "Int16 R-2", custom_fft_1024_int16,  ref_us);
    profile_int16_operation(1024, "Int16 R-4", custom_fft_1024_int16_radix4, ref_us); // 🔥 신규 R-4 연결
    printf(" ----------------------------------------------------\n");

    // 🔹 512 포인트
    ref_us = profile_fftw_operation(512);
    profile_float_operation(512, "Float R-2", custom_fft_512_fixed, ref_us);
    profile_float_operation(512, "Float R-4", custom_fft_512_radix4, ref_us); // 🟢 오타 교정됨 (fixed -> radix4)
    profile_int16_operation(512, "Int16 R-2", custom_fft_512_int16, ref_us); // (기존 R-8 이름 유지)
    profile_int16_operation(512, "Int16 R-4", custom_fft_512_int16_radix4, ref_us); // 🔥 신규 R-4 연결
    printf(" ----------------------------------------------------\n");

    // 🔹 256 포인트
    ref_us = profile_fftw_operation(256);
    profile_float_operation(256, "Float R-2", custom_fft_256_fixed,  ref_us);
    profile_float_operation(256, "Float R-4", custom_fft_256_radix4, ref_us);
    profile_int16_operation(256, "Int16 R-2", custom_fft_256_int16,  ref_us);
    profile_int16_operation(256, "Int16 R-4", custom_fft_256_int16_radix4, ref_us); // 🔥 신규 R-4 연결
    printf(" ----------------------------------------------------\n");

    // 🔹 128 포인트 (Radix-4 미지원 사이즈 -> R-2만 출력)
    ref_us = profile_fftw_operation(128);
    profile_float_operation(128, "Float R-2", custom_fft_128_fixed, ref_us);
    profile_float_operation(128, "Float R-4", custom_fft_128_radix4, ref_us);
    profile_int16_operation(128, "Int16 R-2", custom_fft_128_int16, ref_us);
    profile_int16_operation(128, "Int16 R-4", custom_fft_128_int16_radix4, ref_us); // 🔥 신규 R-4 연결
    printf(" ----------------------------------------------------\n");

    // 🔹 64 포인트
    ref_us = profile_fftw_operation(64);
    profile_float_operation(64, "Float R-2", custom_fft_64_fixed, ref_us);
    profile_float_operation(64, "Float R-4", custom_fft_64_radix4, ref_us);
    profile_int16_operation(64, "Int16 R-2", custom_fft_64_int16, ref_us);
     profile_int16_operation(64, "Int16 R-4", custom_fft_64_int16_radix4, ref_us); // 🔥 신규 R-4 연결
    
    printf("====================================================\n\n");
}