#include "utils.hpp"

#define BLOCK_SIZE 32

#define TILE 64

#define SHARED_BLOCK_SIZE 32

/*WMMA参数*/
#define WMMA_M 16
#define WMMA_K 16
#define WMMA_N 16
#define WARP_SIZE 32


void gemm_kernel_launcher(float* a, float* b, float* c, int N, int M, int K);