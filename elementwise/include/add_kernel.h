#include "utils.hpp"
#define BLOCK_SIZE 256

__global__ void add(float* __restrict__ d_a, float* __restrict__ d_b, float* __restrict__ d_c, int N);


void add_kernel_launcher(float* h_a, float* h_b, float* h_c, int N);