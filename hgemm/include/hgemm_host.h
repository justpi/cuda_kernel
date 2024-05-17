#include <cuda_fp16.h>

void gemm_cpu_base(half *a, half *b, half *c, int N, int K, int M);