#include <iostream>
#include "reduce_host.h"
#include "reduce_kernel.h"
#include "timer.hpp"
#include "utils.hpp"

int seed;

int main() {
    Timer timer;
    /* 设置reduce计算元素 */
    int size = 1 << 25;
    int min = 0;
    int max = 1;
    
    /* 分配host内存 */
    float* h_in = (float*)malloc(sizeof(float) * size);
    float* h_out = (float*)malloc(sizeof(float) * size / THREADS_PER_BLOCK);
    float* d_out = (float*)malloc(sizeof(float) * size / THREADS_PER_BLOCK);

    /* 初始化矩阵 */
    seed = 42;
    initMatrix(h_in, size, min, max, seed);

    /* CPU */
    timer.start();
    reduce_cpu_base(h_in, h_out, size);
    timer.stop();
    timer.duration<Timer::ms>("reduce in cpu");

    /* GPU warmup*/
    timer.start();
    reduce_kernel_launcher(h_in, d_out, size);
    timer.stop();
    timer.duration<Timer::ms>("reduce in gpu(warmup)");

    /* GPU warmup*/
    timer.start();
    reduce_kernel_launcher(h_in, d_out, size);
    timer.stop();
    timer.duration<Timer::ms>("reduce in gpu");

    /*测试结果*/
    compareMat(h_out, d_out, size / THREADS_PER_BLOCK);

    return 0;
}