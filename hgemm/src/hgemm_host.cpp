#include <iostream>
#include "hgemm_host.h"
/* baseline:
        d_a: M * K
        d_b: K * N
        d_c: M * N
*/
void hgemm_cpu_base(half *a, half *b, half *c, int M, int K, int N) {
    for (int i=0; i < M; i++) {
        for (int j=0; j < N; j++) {
            float tmpValue = 0.0f;
            for (int k=0; k < K; k++) {
                tmpValue += __half2float(a[i * K + k]) * __half2float(b[k * N + j]);
            }
            c[i * N + j] = __float2half(tmpValue);
        }
    }
}

