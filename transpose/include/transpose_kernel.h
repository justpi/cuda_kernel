#include "utils.hpp"

#define BLOCK_SIZE 32


__global__ void transpose_baseline(float* d_a, float * d_b, int M, int N);

void transpose_kernel_launcher(float* h_a, float* h_b, int M, int N);