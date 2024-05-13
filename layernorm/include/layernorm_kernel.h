#include "utils.hpp"


#define BLOCK_SIZE 1024
#define EPS 1e-5



void layernorm_kernel_launcher(float *h_a, float *h_o, float gamma, float beta, int length, int stride);