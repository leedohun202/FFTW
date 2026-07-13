#include "radar_mock_data.h"
#include "radar_config.h"
#include <stdlib.h>
#include <math.h>

void generate_radar_mock_data(float *lut_r, float *lut_i, int n_samples, int n_chirps) {
    double true_R[] = {12.50, 25.20,  6.80}; 
    double true_v[] = {14.20, -5.50,  0.00}; 
    double true_a[] = {-15.0,  25.0,  0.00}; 
    int num_targets = 3;

    double snr_db = 15.0; 
    double noise_power = pow(10.0, -snr_db / 10.0);
    double noise_stddev = sqrt(noise_power / 2.0); 

    srand(42); 

    for (int ant = 0; ant < N_ANTENNAS; ant++) {
        for (int chirp = 0; chirp < n_chirps; chirp++) {
            int offset = ant * (n_chirps * n_samples) + chirp * n_samples;
            for (int n = 0; n < n_samples; n++) {
                double val_r = 0.0, val_i = 0.0;
                
                for (int t = 0; t < num_targets; t++) {
                    double f_R = (2.0 * S * true_R[t]) / c; 
                    double p_doppler = (4.0 * PI * true_v[t]) / lambda_c * Tc; 
                    double p_angle = (2.0 * PI * d_ant * sin(true_a[t] * M_PI / 180.0)) / lambda_c;
                    double phase = (2.0 * M_PI * f_R * ((double)n / Fs)) + (p_doppler * chirp) + (p_angle * ant);
                    val_r += cos(phase);
                    val_i += sin(phase);
                }

                double u1 = (rand() + 1.0) / (RAND_MAX + 1.0); 
                double u2 = (rand() + 1.0) / (RAND_MAX + 1.0);
                double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2); 
                double z1 = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2); 

                val_r += noise_stddev * z0;
                val_i += noise_stddev * z1;

                lut_r[offset + n] = (float)val_r; 
                lut_i[offset + n] = (float)val_i;
            }
        }
    }
}