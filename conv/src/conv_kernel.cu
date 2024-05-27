#include "conv_kernel.h"
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>

#define div_ceil(a, b) ((a + b - 1) / b)

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
    int weight_size = out_channel * in_channel * kheight * kwidth;
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

__global__ void conv_implicit_gemm_base(float* input, float* output, float* weight, 
    int batch, int in_channel, int out_channel, int height, int width, int height_out, int width_out,
    int kheight, int kwidth, int pad_height, int pad_width, int stride_h, int stride_w) {
    /*首先计算当前线程的输出索引，然后计算weight的im2col索引，从而计算出weight的各项参数，最后计算input的各项参数*/
    __shared__ float sdata_i[TILE][TILE];
    __shared__ float sdata_w[TILE][TILE];
    const int out_ch = blockIdx.y * TILE + threadIdx.y;
    const int out_hw = blockIdx.x * TILE + threadIdx.x;
    const int b = blockIdx.z;
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    const int out_w = out_hw % height_out;
    const int out_h = out_hw / height_out;
    /*输出纬度为K*PQ，矩阵A为K*CRS，，im2col_weight的形状是K*CRS，表示矩阵A的M和K，矩阵B为CRS*PQ，计算CRS和PQ，即为im2col_input的K和N*/
    // const int weight_im2col_M = out_channel;
    const int weight_im2col_K = in_channel * kheight * kwidth;
    // const int input_im2col_N = height_out * width_out;
    const int k_tile = div_ceil(weight_im2col_K, TILE);
    int in_ch, kh, kw, in_h, in_w;
    float output_value = 0.0f;
    for (int i=0; i < k_tile; ++i) {
        /*计算weight的索引*/
        int idx_k = i * TILE + ty;
        in_ch = idx_k / (kheight * kwidth);
        kh = (idx_k % (kheight * kwidth)) / kwidth;
        kw = idx_k % kwidth;
        /*计算input索引*/
        in_h = out_h * stride_h + kh - pad_height;
        in_w = out_w * stride_w + kw - pad_width;
        /*将weight矩阵读取到shared memory中*/
        if (i * TILE + tx >= weight_im2col_K) sdata_w[ty][tx] = 0.0f;
        else {
            sdata_w[ty][tx] = weight[out_ch * weight_im2col_K + i * TILE + tx];
        }
        /*将input矩阵读取到shared memory中*/
        if (in_h >=0 && in_h < height && in_w >= 0 && in_w < width) {
            sdata_i[ty][tx] = input[b * in_channel * height * width + in_ch * height * width + in_h * width + in_w];
        }
        else sdata_i[ty][tx] = 0.0f;
        __syncthreads();
        /*计算当前tile的输出值*/
        for (int k=0; k < TILE; ++k) {
            output_value += sdata_w[ty][k] * sdata_i[k][tx];
        }
        __syncthreads();

    }
    if (out_ch < out_channel && out_hw < height_out * width_out)
    output[b * out_channel * height_out * width_out + out_ch * height_out * width_out + out_h * width_out + out_w] = output_value;

}


void conv_kernel_launcher(float* input, float* output, float* weight, 
    int batch, int in_channel, int out_channel, int height, int width, 
    int kheight, int kwidth, int pad_height, int pad_width, int stride_height, int stride_width) {
    /*分配显存*/
    int height_out = (height + 2 * pad_height - (kheight - 1) - 1) / stride_height + 1;
    int width_out = (width + 2 * pad_width - (kwidth - 1) - 1) / stride_width + 1;

    int size_input = batch * in_channel * height * width;
    int size_output = batch * out_channel * height_out * width_out;
    int size_weight = out_channel * in_channel * kheight * kwidth;

    float *input_d, *weight_d, *output_d;
    CUDA_CHECK(cudaMalloc(&input_d, size_input * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&weight_d, size_weight * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&output_d, size_output * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(input_d, input, size_input * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(weight_d, weight, size_weight * sizeof(float), cudaMemcpyHostToDevice));

    /*baseline*/

    dim3 block_base(TILE, TILE);
    dim3 grid_base(div_ceil(height_out * width_out, TILE), div_ceil(out_channel, TILE), batch);
    conv_implicit_gemm_base<<<grid_base, block_base>>>(input_d, output_d, weight_d, batch, in_channel, out_channel, height, width, height_out, width_out, kheight, kwidth, pad_height, pad_width, stride_height, stride_width);

    CUDA_CHECK(cudaMemcpy(output, output_d, size_output * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(output_d));
    CUDA_CHECK(cudaFree(weight_d));
    CUDA_CHECK(cudaFree(input_d));

}