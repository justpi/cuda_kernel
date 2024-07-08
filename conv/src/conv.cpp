#include <iostream>
#include "conv_host.h"
#include "conv_kernel.h"
#include "timer.hpp"
#include "utils.hpp"

int seed;
int main(int argc, char **argv) {
    Timer timer;
    /* 设置conv计算参数 */
    int batch = atoi(argv[1]);
    int channel = atoi(argv[2]);
    int height = atoi(argv[3]);
    int width = atoi(argv[4]);

    int knum = 64;
    int kheight = 3;
    int kwidth = 3;
    int padding_h = 1;
    int padding_w = 1;
    int stride_h = 1;
    int stride_w = 1;
    int dilation_h = 1;
    int dilation_w = 1;

    int height_out = (height + 2 * padding_h - dilation_h * (kheight - 1) - 1) / stride_h + 1;
    int width_out = (width + 2 * padding_w - dilation_w * (kwidth - 1) - 1) / stride_w + 1;

    int min = 0;
    int max = 1;
    
    int size_input = batch * channel * height * width;
    int size_output = batch * knum * height_out * width_out;
    int size_weight = knum * channel * kheight * kwidth;

    /* 分配host内存 */
    float* input = (float*)malloc(size_input * sizeof(float));
    float* weight = (float*)malloc(size_weight * sizeof(float));
    float* output_h = (float*)malloc(size_output * sizeof(float));
    float* output_d = (float*)malloc(size_output * sizeof(float));

    /* 初始化矩阵 */
    seed = 42;
    initMatrix(input, size_input, min, max, seed);
    initMatrix(weight, size_weight, min, max, seed+1);

    /* CPU */
//     timer.start();
//     conv_host_naive(input, output_h, weight, batch, channel, knum, height, width, 
//             kheight, kwidth, padding_h, padding_w, stride_h, stride_w);
//     timer.stop();
//     timer.duration<Timer::ms>("conv in cpu");

    /* GPU warmup */
    timer.start();
    conv_cudnn_launcher(input, output_h, weight, batch, channel, knum, height, width, 
            kheight, kwidth, padding_h, padding_w, stride_h, stride_w);
    timer.stop();
    timer.duration<Timer::ms>("conv cudnn in gpu(warmup)");

    timer.start();
    conv_kernel_launcher(input, output_d, weight, batch, channel, knum, height, width, 
            kheight, kwidth, padding_h, padding_w, stride_h, stride_w);
    timer.stop();
    timer.duration<Timer::ms>("conv kernel in gpu(warmup)");

    /* GPU */
    timer.start();
    conv_cudnn_launcher(input, output_h, weight, batch, channel, knum, height, width, 
            kheight, kwidth, padding_h, padding_w, stride_h, stride_w);
    timer.stop();
    timer.duration<Timer::ms>("conv cudnn in gpu(warmup)");

    timer.start();
    conv_kernel_launcher(input, output_d, weight, batch, channel, knum, height, width, 
            kheight, kwidth, padding_h, padding_w, stride_h, stride_w);
    timer.stop();
    timer.duration<Timer::ms>("conv kernel in gpu(warmup)");

    /* 验证结果 */
    compareMat(output_h, output_d, size_output);

    return 0;
}