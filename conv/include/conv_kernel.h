#include "utils.hpp"

#define TILE 32

void conv_cudnn_launcher(float* input, float* output, float* weight, 
                        int batch, int in_channel, int out_channel, int height, int width, 
                        int kheight, int kwidth, int pad_height, int pad_width, int stride_height, int stride_width);


void conv_kernel_launcher(float* input, float* output, float* weight, 
    int batch, int in_channel, int out_channel, int height, int width, 
    int kheight, int kwidth, int pad_height, int pad_width, int stride_height, int stride_width);


