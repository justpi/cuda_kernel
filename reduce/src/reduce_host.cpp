#include "reduce_host.h"
#include "reduce_kernel.h"
#include <iostream>

void reduce_cpu_base(float* h_in, float* h_out, int N) {
    /*base 版本的reduce算子*/
    int num_block = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    for (int i=0; i < num_block; ++i) {
        float cur = 0.0;
        for (int j=0; j < THREADS_PER_BLOCK; ++j) {
            cur += h_in[i * THREADS_PER_BLOCK + j];
        }
        h_out[i] = cur;
    }
}


