#include <iostream>
#include "elementwise_host.h"

void add_cpu_base(float* a, float* b, float* c, int N) {
    for (int i=0; i < N; ++i) {
        c[i] = a[i] + b[i];
    }
}