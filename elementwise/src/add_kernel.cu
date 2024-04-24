#include "add_kernel.h"
#include "utils.hpp"
#include <cuda_runtime.h>
#include <iostream>

/* 向量化访存：取两个float */
#define FETCH_FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer))[0])
/* 向量化访存：取4个float */
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

__global__ void add(float* __restrict__ d_a, float* __restrict__ d_b, float* __restrict__ d_c, int N) {
    /* 矩阵加法实现 */
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) d_c[idx] = d_a[idx] + d_b[idx];
}


__global__ void add_float2(float* __restrict__ d_a, float* __restrict__ d_b, float* __restrict__ d_c, int N) {
    /* 向量化访存，一个thread一次访存读取两个float，执行两次计算.此时若block不变，那么grid需对应减半 */
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 2;
    if (idx < N) {
        float2 reg_a = FETCH_FLOAT2(d_a[idx]);
        float2 reg_b = FETCH_FLOAT2(d_b[idx]);
        float2 reg_c;
        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.y + reg_b.y;
        FETCH_FLOAT2(d_c[idx]) = reg_c;
    }
    
}

__global__ void add_float4(float* __restrict__ d_a, float* __restrict__ d_b, float* __restrict__ d_c, int N) {
    /* 向量化访存，一个thread一次访存读取4个float，执行4次计算.此时若block不变，那么grid需对应除以4 */
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    if (idx < N) {
        float4 reg_a = FETCH_FLOAT4(d_a[idx]);
        float4 reg_b = FETCH_FLOAT4(d_b[idx]);
        float4 reg_c;
        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.y + reg_b.y;
        reg_c.z = reg_a.z + reg_b.z;
        reg_c.w = reg_a.w + reg_b.w;
        FETCH_FLOAT4(d_c[idx]) = reg_c;
    }
    
    
}


void add_kernel_launcher(float* h_a, float* h_b, float* h_c, int N) {
    /* 逐元素加法 */
    /* 分配GPU空间 */
    float *d_a;
    float *d_b;
    float *d_c;

    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));

    /* 拷贝数据到global mem */
    CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));
    
    /* 运行核函数 */
    /* baseline */
    // dim3 block(BLOCK_SIZE);
    // dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    // add<<<grid, block>>>(d_a, d_b, d_c, N);

    /* 向量化访存-2个float */
    // dim3 block_f2(BLOCK_SIZE);
    // dim3 grid_f2((N + BLOCK_SIZE - 1) / (BLOCK_SIZE * 2));
    // add_float2<<<grid_f2, block_f2>>>(d_a, d_b, d_c, N);

    /* 向量化访存-4个float */
    dim3 block_f4(BLOCK_SIZE);
    dim3 grid_f4((N + BLOCK_SIZE - 1) / (BLOCK_SIZE * 4));
    add_float4<<<grid_f4, block_f4>>>(d_a, d_b, d_c, N);


    /* 数据拷回 */
    CUDA_CHECK(cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    /* free */
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_a));
}