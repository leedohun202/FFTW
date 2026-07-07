#include "radar_config.h"

const double c = 3e8; const double fc = 77e9; const double B = 8e9; 
const double Tc = 40e-6; const double Fs = 40e6; const double S = 8e9 / 40e-6; 
const double lambda_c = 3e8 / 77e9; const double d_ant = (3e8 / 77e9) / 2.0;

int bitrev_2048[2048]; int bitrev_1024[1024]; int bitrev_512[512]; 
int bitrev_256[256];   int bitrev_128[128];   int bitrev_64[64];

float twiddle_real_2048[1024]; float twiddle_imag_2048[1024];
float twiddle_real_1024[512];  float twiddle_imag_1024[512];
float twiddle_real_512[256];   float twiddle_imag_512[256];
float twiddle_real_256[128];   float twiddle_imag_256[128];
float twiddle_real_128[64];    float twiddle_imag_128[64];
float twiddle_real_64[32];     float twiddle_imag_64[32];

float win_2048[2048]; float win_1024[1024]; float win_512[512]; 
float win_256[256];   float win_128[128];