#include "transpose_host.h"
#include <iostream>

void transpose_cpu_base(float* a, float* b, int M, int N) {
    /* base版本的transpose */
    for (int i=0; i < M; ++i) {
        for (int j=0; j < N; ++j) {
            b[j * M + i] = a[i * N + j];
        }
    }
}
