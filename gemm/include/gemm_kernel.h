#include "utils.hpp"

#define BLOCK_SIZE 32

#define TILE 64

#define SHARED_BLOCK_SIZE 32

__global__ void gemm_baseline();


void gemm_kernel_launcher(float* a, float* b, float* c, int N, int M, int K);