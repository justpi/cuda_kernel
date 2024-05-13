#include "utils.hpp"


#define BLOCK_SIZE 1024
#define EPSK 1e-5f



void layernorm_kernel_launcher(float *h_a, float *h_o, float gamma, float beta, int length, int stride);