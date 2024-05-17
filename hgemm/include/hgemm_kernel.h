#include "utils.hpp"
#include <cuda_fp16.h>

#define BLOCK_SIZE 32

#define TILE 64

#define SHARED_BLOCK_SIZE 32

/*WMMA参数*/
#define WMMA_M 16
#define WMMA_K 16
#define WMMA_N 16
#define WARP_SIZE 32

void hgemm_cublas_launcher(half* a, half* b, half* c, int N, int K, int M);

void hgemm_kernel_launcher(half* a, half* b, half* c, int N, int M, int K);