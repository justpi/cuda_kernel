#include <iostream>
#include "add_host.h"
#include "add_kernel.h"
#include "timer.hpp"
#include "utils.hpp"

int seed;
int main(int argc, char **argv) {
    Timer timer;
    /* 设置add计算参数 */
    int width = atoi(argv[1]);
    int min = 0;
    int max = 1;
    int size = width * width;

    /* 分配host内存 */
    float* a = (float*)malloc(size * sizeof(float));
    float* b = (float*)malloc(size * sizeof(float));
    float* h_c = (float*)malloc(size * sizeof(float));
    float* d_c = (float*)malloc(size * sizeof(float));

    /* 初始化矩阵 */
    seed = 42;
    initMatrix(a, size, min, max, seed);
    initMatrix(b, size, min, max, seed+1);

    /* CPU */
    timer.start();
    add_cpu_base(a, b, h_c, size);
    timer.stop();
    timer.duration<Timer::ms>("gemm in cpu");

    /* GPU warmup */
    timer.start();
    add_kernel_launcher(a, b, d_c, size);
    timer.stop();
    timer.duration<Timer::ms>("gemm in gpu(warmup)");

    /* GPU */
    timer.start();
    add_kernel_launcher(a, b, d_c, size);
    timer.stop();
    timer.duration<Timer::ms>("gemm in gpu");

    /* 验证结果 */
    compareMat(h_c, d_c, size);

    return 0;
}