#include <iostream>
#include "gemm.h"
/* baseline:
        d_a: M * K
        d_b: K * N
        d_c: M * N
*/
void gemm_cpu_base(float *a, float *b, float *c, int N, int K, int M) {
    for (int i=0; i < N; i++) {
        for (int j=0; j < M; j++) {
            float tmpValue = 0.0f;
            for (int k=0; k < K; k++) {
                tmpValue += a[i * K + k] * b[k * N + j];
            }
            c[i * N + j] = tmpValue;
        }
    }
}

