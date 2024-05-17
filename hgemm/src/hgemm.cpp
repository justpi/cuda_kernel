#include <iostream>
#include "hgemm_host.h"
#include "hgemm_kernel.h"
#include "timer.hpp"
#include "utils.hpp"

int seed;
int main(int argc, char **argv) {
    Timer timer;
    /* 设置gemm计算参数 */
    int width = atoi(argv[1]);
    int min = 0;
    int max = 1;
    int size = width * width;

    /* 分配host内存 */
    half* a = (half*)malloc(size * sizeof(half));
    half* b = (half*)malloc(size * sizeof(half));
    half* h_c = (half*)malloc(size * sizeof(half));
    half* d_c = (half*)malloc(size * sizeof(half));

    /* 初始化矩阵 */
    seed = 42;
    initMatrix(a, size, min, max, seed);
    initMatrix(b, size, min, max, seed+1);

    /* CPU */
    // timer.start();
    // hgemm_cpu_base(a, b, h_c, width, width, width);
    // timer.stop();
    // timer.duration<Timer::ms>("gemm in cpu");

    /* GPU warmup */
    timer.start();
    hgemm_cublas_launcher(a, b, h_c, width, width, width);
    timer.stop();
    timer.duration<Timer::ms>("gemm in gpu(warmup)");

    timer.start();
    hgemm_kernel_launcher(a, b, d_c, width, width, width);
    timer.stop();
    timer.duration<Timer::ms>("gemm in gpu(warmup)");


    /* GPU */
    timer.start();
    hgemm_cublas_launcher(a, b, h_c, width, width, width);
    timer.stop();
    timer.duration<Timer::ms>("gemm in gpu");

    timer.start();
    hgemm_kernel_launcher(a, b, d_c, width, width, width);
    timer.stop();
    timer.duration<Timer::ms>("gemm in gpu");

    /* 验证结果 */
    compareMat(h_c, d_c, size);

    return 0;
}