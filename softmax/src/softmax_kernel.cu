#include "softmax_kernel.h"
#include <cuda_runtime.h>
#include <climits>
#include <iostream>

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

__global__ void softmax_online_base(float *d_a, float *d_o, int length, int stride) {
    /*
    线程组织：block设置为一维，每个线程处理stride / blockDim.x个值；
    算法：将cpu版本的online softmax翻译为cuda代码
    */
    int iters = stride / blockDim.x;
    int idx;
    float m = -INFINITY;
    float m_ = -INFINITY;
    float sum = 0.0;
    float value;
    /*求最值&求指数和*/
    #pragma unroll
    for (int i=0; i < iters; ++i) {
        idx = blockIdx.x * stride + i * blockDim.x + threadIdx.x;
        value = d_a[idx];
        m_ = m;
        m = fmaxf(value, m);
        sum = sum * expf(m - m_) + expf(value - m);
    }

    __syncthreads();
    /*求softmax*/
    d_o[idx] = expf(value - m) * 1./sum;

}

__global__ void softmax_online_vec(float *d_a, float *d_o, int length, int stride) {
    /*在baseline基础上添加向量化访存，一次访问4个元素用于计算*/
    int iters = stride / (blockDim.x * 4);
    int idx;
    float m[4]={-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    float m_[4]={-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    float sums[4]={0.0, 0.0, 0.0, 0.0};
    float values[4];
    for (int i=0; i < iters; ++i) {
        idx = blockIdx.x * stride + i * blockDim.x * 4 + threadIdx.x * 4;
        FETCH_FLOAT4(values[0]) = FETCH_FLOAT4(d_a[idx]);
        #pragma unroll 
        for (int j=0; j < 4; ++j) {
            m_[j] = m[j];
            m[j] = fmaxf(values[j], m[j]);
            sums[j] = sums[j] * expf(m[j] - m_[j]) + expf(values[j] - m[j]);
        }
        
    }

    __syncthreads();

    for(int i=0; i < 4; ++i) {
        sums[i] = expf(values[i]- m[i]) * 1./sums[i];
    }
    

    d_o[idx] = sums[0];
    d_o[idx+1] = sums[1];
    d_o[idx+2] = sums[2];
    d_o[idx+3] = sums[3];
}

void softmax_kernel_launcher(float* a, float* h_o, int length, int stride) {
    /* 分配GPU资源 */
    float *d_a;
    float *d_o;
    CUDA_CHECK(cudaMalloc(&d_a, length * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_o, length * sizeof(float)));
    /* 拷贝数据 */
    CUDA_CHECK(cudaMemcpy(d_a, a, length * sizeof(float), cudaMemcpyHostToDevice));

    /* 组织线程并launch kernel */

    /* base */
    const int rows = (length + stride - 1) / stride;
    // dim3 block(BLOCK_SIZE);
    // dim3 grid(rows);
    // softmax_online_base<<<grid, block>>>(d_a, d_o, length, stride);

    dim3 block_vec(BLOCK_SIZE/4);
    dim3 grid_vec(rows);
    softmax_online_vec<<<grid_vec, block_vec>>>(d_a, d_o, length, stride);

    /* 拷贝数据 */
    CUDA_CHECK(cudaMemcpy(h_o, d_o, length * sizeof(float), cudaMemcpyDeviceToHost));

    /* free */
    CUDA_CHECK(cudaFree(d_o));
    CUDA_CHECK(cudaFree(d_a));
}