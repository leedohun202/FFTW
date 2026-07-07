#include "radar_pipeline.h"
#include <fftw3.h>

/**
 * @brief 💻 대조군 FFTW3 2D Doppler 일괄 집행 커널
 */
void execute_fftw_pipeline_optimized(fftwf_complex *cube_in, fftwf_complex *cube_out, 
                                     fftwf_plan p_doppler, int n_samples) {
    // 거대 64포인트 하드웨어 계측 계획을 전면 취소하고, 깔끔하게 Doppler 플랜만 가동합니다.
    fftwf_execute(p_doppler);
}