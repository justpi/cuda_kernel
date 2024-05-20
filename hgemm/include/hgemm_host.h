#include <cuda_fp16.h>

void hgemm_cpu_base(half *a, half *b, half *c, int M, int K, int N);