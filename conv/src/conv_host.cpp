#include "conv_host.h"

void conv_host_naive(float* input, float* output, float* weight, 
                        int batch, int in_channel, int out_channel, int height, int width, 
                        int kh, int kw, int pad_h, int pad_w, int stride_h, int stride_w) {

    /*计算输出维度*/
    int height_o = (height + 2 * pad_h - (kh - 1) - 1) / stride_h + 1;
    int width_o = (width + 2 * pad_w - (kw - 1) - 1) / stride_w + 1;
    
    for (int b=0; b < batch; ++b) {
        for (int c_out=0; c_out < out_channel; ++c_out) {
            for(int h_o=0; h_o < height_o; ++h_o) {
                for(int w_o=0; w_o < width_o; ++w_o) {
                    long out_idx = b * out_channel * height_o * width_o + c_out * height_o * width_o + h_o * width_o + w_o;
                    output[out_idx] = 0.0f;
                    /*计算一个输出值*/
                    for(int h_k=0; h_k < kh; ++h_k) {
                        for (int w_k=0; w_k < kw; ++w_k) {
                            int in_h = h_o * stride_h + h_k - pad_h;
                            int in_w = w_o * stride_w + w_k - pad_w;
                            if (in_h >= 0 && in_h < height && in_w >= 0 && in_w < width) {  // 确保在有效范围内
                                for (int c_in=0; c_in < in_channel; ++c_in) {
                                    /*计算一个输入特征点和对应卷积核的点积*/
                                    long weight_idx = c_out * in_channel * kh * kw + c_in * kh * kw + h_k * kw + w_k;
                                    long input_idx = b * in_channel * height * width + c_in * height * width + in_h * width + in_w;
                                    output[out_idx] += weight[weight_idx] * input[input_idx];
                                }
                            }
                            
                        }
                    }
                }
            }
        }
    }
}