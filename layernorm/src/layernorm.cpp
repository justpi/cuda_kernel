#include <iostream>
#include "layernorm_host.h"
#include "layernorm_kernel.h"
#include "timer.hpp"
#include "utils.hpp"

int seed;
int main(int argc, char **argv) {
    Timer timer;
    /* 设置layernorm计算参数 */
    int width = atoi(argv[1]);
    int min = 0;
    int max = 1;
    int size = width * width;
    float gamma = 0.5;
    float beta = 0.5;

    /* 分配host内存 */
    float* a = (float*)malloc(size * sizeof(float));
    float* h_o = (float*)malloc(size * sizeof(float));
    float* d_o = (float*)malloc(size * sizeof(float));

    /* 初始化矩阵 */
    seed = 42;
    initMatrix(a, size, min, max, seed);

    /* CPU */
    timer.start();
    layernorm_host_base(a, h_o, gamma, beta, size, width);
    timer.stop();
    timer.duration<Timer::ms>("layernorm in cpu");

    /* CPU */
    timer.start();
    layernorm_host_naive(a, d_o, gamma, beta, size, width);
    timer.stop();
    timer.duration<Timer::ms>("layernorm in cpu");

        /* CPU */
    timer.start();
    layernorm_host_welford(a, d_o, gamma, beta, size, width);
    timer.stop();
    timer.duration<Timer::ms>("layernorm in cpu");

    // /* GPU warmup */
    // timer.start();
    // layernorm_kernel_launcher(a, d_o, gamma, beta, size, width);
    // timer.stop();
    // timer.duration<Timer::ms>("layernorm in gpu(warmup)");

    // /* GPU */
    // timer.start();
    // layernorm_kernel_launcher(a, d_o, gamma, beta, size, width);
    // timer.stop();
    // timer.duration<Timer::ms>("layernorm in gpu");

    /* 验证结果 */
    compareMat(h_o, d_o, size);

    return 0;
}