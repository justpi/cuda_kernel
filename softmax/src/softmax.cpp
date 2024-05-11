#include <iostream>
#include "softmax_host.h"
#include "softmax_kernel.h"
#include "timer.hpp"
#include "utils.hpp"

int seed;
int main(int argc, char **argv) {
    Timer timer;
    /* 设置softmax计算参数 */
    int width = atoi(argv[1]);
    int min = 0;
    int max = 1;
    int size = width * width;

    /* 分配host内存 */
    float* a = (float*)malloc(size * sizeof(float));
    float* h_c = (float*)malloc(size * sizeof(float));
    float* d_c = (float*)malloc(size * sizeof(float));

    /* 初始化矩阵 */
    seed = 42;
    initMatrix(a, size, min, max, seed);

    // /* CPU */
    // timer.start();
    // softmax_host_base(a, h_c, size, width);
    // timer.stop();
    // timer.duration<Timer::ms>("softmax in cpu");

    /* CPU */
    timer.start();
    softmax_host_online(a, h_c, size, width);
    timer.stop();
    timer.duration<Timer::ms>("softmax in cpu");

    /* GPU warmup */
    timer.start();
    softmax_kernel_launcher(a, d_c, size, width);
    timer.stop();
    timer.duration<Timer::ms>("softmax in gpu(warmup)");

    /* GPU */
    timer.start();
    softmax_kernel_launcher(a, d_c, size, width);
    timer.stop();
    timer.duration<Timer::ms>("softmax in gpu");

    /* 验证结果 */
    compareMat(h_c, d_c, size);

    return 0;
}