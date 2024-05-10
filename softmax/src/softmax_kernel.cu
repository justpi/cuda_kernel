#include "softmax_kernel.h"
#include <cuda_runtime.h>
#include <climits>
#include <iostream>


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
        m = m > value ? m:value;
        sum = sum * expf(m - m_) + expf(value - m);
    }

    __syncthreads();
    /*求softmax*/
    d_o[idx] = expf(value - m) * 1/sum;

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
    dim3 block(BLOCK_SIZE);
    dim3 grid(rows);
    softmax_online_base<<<grid, block>>>(d_a, d_o, length, stride);

    /* 拷贝数据 */
    CUDA_CHECK(cudaMemcpy(h_o, d_o, length * sizeof(float), cudaMemcpyDeviceToHost));

    /* free */
    CUDA_CHECK(cudaFree(d_o));
    CUDA_CHECK(cudaFree(d_a));
}