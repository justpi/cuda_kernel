#include "utils.hpp"


#define THREADS_PER_BLOCK 1024

#define WARP_SIZE 32

__global__ void reduce_kernel_baseline(float* d_in, float* d_out);

__global__ void reduce_kernel_div(float* d_in, float* d_out);

__global__ void reduce_kernel_bankconflic(float* d_in, float* d_out);

__global__ void reduce_kernel_idle(float* d_in, float* d_out);

__global__ void reduce_kernel_unrollLast32(float* d_in, float* d_out);

__global__ void reduce_kernel_unroll(float* d_in, float* d_out);

void reduce_kernel_launcher(float* h_in, float* h_out, int N);