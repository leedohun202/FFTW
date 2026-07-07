#include "radar_utils.h"

void profile_individual_operation(int size, void (*custom_func)(float*, float*)) {
    float *r = (float*)calloc(size, sizeof(float));
    float *i = (float*)calloc(size, sizeof(float));
    fftwf_complex *f_in = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * size);
    fftwf_plan p = fftwf_plan_dft_1d(size, f_in, f_in, FFTW_FORWARD, FFTW_PATIENT);
    
    // 만약 기존에 세이브 파일(.wisdom)이 없었다면, 이 타이밍에 정밀 학습된 
    // 최고 성능의 기계어 조각(Wisdom)을 디스크 파일로 즉시 백업 및 영구 봉인합니다.
    FILE *wf = fopen(FFTW_WISDOM_FILE, "w");
    if (wf != NULL) {
        fftwf_export_wisdom_to_file(wf);
        fclose(wf);
    }


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