#include "radar_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <fftw3.h>

// (다른 유틸 함수들이 있다면 이어서 작성...)

void profile_individual_operation(int size, void (*custom_func)(float*, float*)) {
    float *r = (float*)calloc(size, sizeof(float));
    float *i = (float*)calloc(size, sizeof(float));
    fftwf_complex *f_in = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * size);
    
    // 🎯 [패치 1] 무식한 PATIENT 하드코딩 제거 및 전역 매크로 동기화
    unsigned int fftw_flags = FFTW_ESTIMATE;
#if defined(FFTW_MODE_MEASURE)
    fftw_flags = FFTW_MEASURE;
#elif defined(FFTW_MODE_PATIENT)
    fftw_flags = FFTW_PATIENT;
#endif

    fftwf_plan p = fftwf_plan_dft_1d(size, f_in, f_in, FFTW_FORWARD, fftw_flags);
    
    // 🎯 [패치 2] Wisdom 백업 로직을 PATIENT 모드 전용으로 철저히 격리
#if defined(FFTW_MODE_PATIENT)
    FILE *wf = fopen("radar_wisdom.wisdom", "w"); // 또는 FFTW_WISDOM_FILE 매크로 사용
    if (wf != NULL) {
        fftwf_export_wisdom_to_file(wf);
        fclose(wf);
    }
#endif

    const int iterations = 1000; struct timeval start, end;
    
    gettimeofday(&start, NULL);
    for(int k=0; k<iterations; k++) custom_func(r, i);
    gettimeofday(&end, NULL);
    double custom_us = ((end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec)) / iterations;

    gettimeofday(&start, NULL);
    for(int k=0; k<iterations; k++) fftwf_execute(p);
    gettimeofday(&end, NULL);
    double fftw_us = ((end.tv_sec - start.tv_sec) * 1000000.0 + (end.tv_usec - start.tv_usec)) / iterations;

    printf("  🔹 %4d 포인트 1D FFT 연산 -> [Custom]: %7.2f µs  |  [FFTW3]: %7.2f µs (비율: %.2fx)\n", size, custom_us, fftw_us, custom_us / fftw_us);
    fftwf_destroy_plan(p); free(r); free(i); fftwf_free(f_in);
}