#include "layernorm_kernel.h"

__global__ void layernorm_kernel_base(float *d_a, float *d_o, float gamma, float beta, int length, int stride) {
    /*一个block计算一个row，一个thread计算stride/blocksize个数*/
    int iters = stride / blockDim.x;
    float m = 0.0, m_= 0.0;
    float M = 0.0;
    for (int i=0; i < iters; ++i) {
        
    }

}



void layernorm_kernel_launcher(float *h_a, float *h_o, float gamma, float beta, int length, int stride) {
    /*分配内存*/
    float *d_a;
    float *d_o;
    CUDA_CHECK(cudaMalloc(&d_a, length * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_o, length * sizeof(float)));
    /*拷贝数据*/
    CUDA_CHECK(cudaMemcpy(d_a, h_a, length * sizeof(float), cudaMemcpyHostToDevice));

    /*组织线程以及执行核函数*/
    dim3 block_base();
    dim3 grid_base();
    layernorm_kernel_base<<<grid_base, block_base>>>(d_a, d_o, gamma, beta, length, stride);

    /*数据拷回*/
    CUDA_CHECK(cudaMemcpy(h_o, d_o, length * sizeof(float), cudaMemcpyDeviceToHost));

    /*free*/
    CUDA_CHECK(cudaFree(d_o));
    CUDA_CHECK(cudaFree(d_a));
}