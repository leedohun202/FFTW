#include "radar_pipeline.h"

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