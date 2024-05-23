#include "conv_kernel.h"
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>

void conv_cudnn_launcher(float* input, float* output, float* weight, 
                        int batch, int in_channel, int out_channel, int height, int width, 
                        int kheight, int kwidth, int pad_height, int pad_width, int stride_height, int stride_width) {
    

    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);
    /*设置输入张量参数*/
    cudnnTensorDescriptor_t input_desc;
    cudnnCreateTensorDescriptor(&input_desc);
    cudnnSetTensor4dDescriptor(input_desc,
                            /*format=*/CUDNN_TENSOR_NCHW,
                            /*dataType=*/CUDNN_DATA_FLOAT,
                            /*batch_size=*/batch,
                            /*channel=*/in_channel,
                            /*height=*/height,
                            /*width=*/width
                        );
     
    /*设置过滤器参数*/
    cudnnFilterDescriptor_t filter_desc;
    cudnnCreateFilterDescriptor(&filter_desc);
    cudnnSetFilter4dDescriptor(filter_desc,
                            CUDNN_DATA_FLOAT,
                            CUDNN_TENSOR_NCHW,
                            out_channel,
                            in_channel,
                            kheight,
                            kwidth
    );

    /*设置卷积算子参数*/
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnSetConvolution2dDescriptor(conv_desc,
                                /*pad_height=*/pad_height,
                                /*pad_width=*/pad_width,
                                /*stride_height=*/stride_height,
                                /*stride_width=*/stride_width,
                                /*dilation_height=*/1,
                                /*dilation_width=*/1,
                                /*mode=*/CUDNN_CROSS_CORRELATION,
                                /*computeType=*/CUDNN_DATA_FLOAT

    );

    /*输出张量设置*/
    int b_out, c_out, h_out, w_out;
    cudnnGetConvolution2dForwardOutputDim(conv_desc,
                                        input_desc,
                                        filter_desc,
                                        &b_out,
                                        &c_out,
                                        &h_out,
                                        &w_out

    );
    cudnnTensorDescriptor_t output_desc;
    cudnnCreateTensorDescriptor(&output_desc);
    cudnnSetTensor4dDescriptor(output_desc,
                            CUDNN_TENSOR_NCHW,
                            CUDNN_DATA_FLOAT,
                            b_out,
                            c_out,
                            h_out,
                            w_out

    );

    /*分配显存*/
    int input_size = batch * in_channel * height * width;
    int weight_size = out_channel * kheight * kwidth;
    int output_size = b_out * c_out * h_out * w_out;

    float *input_d, *weight_d, *output_d;
    CUDA_CHECK(cudaMalloc(&input_d, input_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&weight_d, weight_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&output_d, output_size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(input_d, input, input_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(weight_d, weight, weight_size * sizeof(float), cudaMemcpyHostToDevice));

    /*执行计算*/
    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(cudnn,
                        &alpha,
                        input_desc,
                        input_d,
                        filter_desc,
                        weight_d,
                        conv_desc,
                        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                        NULL,
                        0,
                        &beta,
                        output_desc,
                        output_d

    );

    /*数据拷回*/
    CUDA_CHECK(cudaMemcpy(output, output_d,output_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    LAST_KERNEL_CHECK();

    /*清除空间*/
    cudaFree(output_d);
    cudaFree(weight_d);
    cudaFree(input_d);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn);
    
}