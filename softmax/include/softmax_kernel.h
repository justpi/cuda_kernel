#include "utils.hpp"

#define BLOCK_SIZE 1024

__global__ void softmax_online_base(float *d_a, float *d_o, int length, int stride);


void softmax_kernel_launcher(float* a, float* h_o, int length, int stride);