#include "utils.hpp"
#include <cuda_fp16.h>

#define BLOCK_SIZE 32

#define TILE 64

#define SHARED_BLOCK_SIZE 32

/* wmma shared tile size*/
#define BM 128
#define BN 128
#define BK 32

/*WMMA参数*/
#define WMMA_M 16
#define WMMA_K 16
#define WMMA_N 16
#define WARP_SIZE 32

void hgemm_cublas_launcher(half* a, half* b, half* c, int M, int K, int N);

void hgemm_cutlass_launcher(half* a, half* b, half* c, int M, int K, int N);

void hgemm_kernel_launcher(half* a, half* b, half* c, int M, int K, int N);