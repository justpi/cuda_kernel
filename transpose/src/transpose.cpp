#include <iostream>
#include "transpose_host.h"
#include "transpose_kernel.h"
#include "timer.hpp"
#include "utils.hpp"

int seed;
int main(int argc, char **argv) {
    Timer timer;
    /* 设置tranpose计算参数 */
    int width = atoi(argv[1]);
    int height = atoi(argv[2]);
    int min = 0;
    int max = 1;
    int size = width * height;

    /* 分配host内存 */
    float* a = (float*)malloc(size * sizeof(float));
    float* h_b = (float*)malloc(size * sizeof(float));
    float* d_b = (float*)malloc(size * sizeof(float));

    /* 初始化矩阵 */
    seed = 42;
    initMatrix(a, size, min, max, seed);

    /* CPU */
    timer.start();
    transpose_cpu_base(a, h_b, height, width);
    timer.stop();
    timer.duration<Timer::ms>("gemm in cpu");

    /* GPU warmup */
    timer.start();
    transpose_kernel_launcher(a, d_b, height, width);
    timer.stop();
    timer.duration<Timer::ms>("gemm in gpu(warmup)");

    /* GPU */
    timer.start();
    transpose_kernel_launcher(a, h_b, height, width);
    timer.stop();
    timer.duration<Timer::ms>("gemm in gpu");

    /* 验证结果 */
    compareMat(h_b, d_b, size);

    return 0;
}