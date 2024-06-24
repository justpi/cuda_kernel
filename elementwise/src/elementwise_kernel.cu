#include "elementwise_kernel.h"
#include "utils.hpp"
#include <cuda_runtime.h>
#include <iostream>

/* 向量化访存：取两个float */
#define FETCH_FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer))[0])
/* 向量化访存：取4个float */
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

#define FETECH_INT4(pointer) (reinterpret_cast<int4*>(&(pointer))[0])

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


template<const int WarpSize=32>
__device__ __forceinline__ float warp_reduce_sum(int val) {
    #pragma unroll
    for(int mask=WarpSize>>1; mask>=1; mask>>=1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}


// Dot Product-baseline
// a: Nx1, b:Nx1, y=sum(elementwise_mul(a, b))
template <const int THREADS_PER_BLOCK>
__global__ void dot_product(float* d_a, float* d_b, float* d_c, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int NUM_WARPS = THREADS_PER_BLOCK / WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    __shared__ float sdata[NUM_WARPS];

    int sum = (idx < N) ? d_a[idx] * d_b[idx] : 0.0f;
    sum = warp_reduce_sum<WARP_SIZE>(sum);
    if (lane == 0) sdata[warp] = sum;
    __syncthreads();

    sum = (lane < NUM_WARPS) ? sdata[lane] : 0.0f;
    if (warp == 0) sum = warp_reduce_sum<WARP_SIZE>(sum);
    // 写回结果
    if(tid == 0) atomicAdd(d_c, sum);
}

// Dot Product-vec4
// a: Nx1, b:Nx1, y=sum(elementwise_mul(a, b))
template<const int THREADS_PER_BLOCK>
__global__ void dot_product_vec4(float* d_a, float* d_b, float* d_c, int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    constexpr int NUM_WARPS = THREADS_PER_BLOCK / WARP_SIZE;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    __shared__ float sdata[NUM_WARPS];

    float4 reg_a = FETCH_FLOAT4(d_a[idx]);
    float4 reg_b = FETCH_FLOAT4(d_b[idx]);
    float sum = (idx < N) ? (reg_a.x * reg_b.x + reg_a.y * reg_b.y + reg_a.z * reg_b.z + reg_a.w * reg_b.w):0.0f;
    sum = warp_reduce_sum<WARP_SIZE>(sum);
    if (lane == 0) sdata[warp] = sum;
    __syncthreads();
    // 处理shared mem中求出的sum值
    sum = (lane < NUM_WARPS) ? sdata[lane]:0.0f;
    if(warp == 0) sum = warp_reduce_sum<WARP_SIZE>(sum);
    if(tid == 0) atomicAdd(d_c, sum);
}


void dot_product_kernel_launcher(float* h_a, float* h_b, float* h_c, int N) {
    /* 点乘 */
    /* 分配GPU空间 */
    float *d_a;
    float *d_b;
    float *d_c;

    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, 1 * sizeof(float)));

    /* 拷贝数据到global mem */
    CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));
    
    /* 运行核函数 */
    /* baseline */
    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dot_product<BLOCK_SIZE><<<grid, block>>>(d_a, d_b, d_c, N);

    /* 向量化访存-4个float */
    dim3 block_f4(BLOCK_SIZE/4);
    dim3 grid_f4((N + BLOCK_SIZE - 1) / (BLOCK_SIZE));
    dot_product_vec4<BLOCK_SIZE/4><<<grid_f4, block_f4>>>(d_a, d_b, d_c, N);


    /* 数据拷回 */
    CUDA_CHECK(cudaMemcpy(h_c, d_c, 1 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    /* free */
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_a));
}


// histgram-baseline 
// block(BLOCK_SIZE), grid(N/BLOCK_SIZE)
// a: Nx1. o: counted gistgram

__global__ void histgram(int* d_a, int* d_o, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) atomicAdd(d_o[d_a[dix]], 1);
}

// histgram-vec4
// block(BLOCK_SIZE/4), grid(N/BLOCK_SIZE)
// a: Nx1. o: counted gistgram
__global__ void histgram_vec4(int* d_a, int* d_o, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx < N) {
        int4 reg_val = FETECH_INT4(d_a[idx]);
        atomicAdd(d_o[reg_val.x], 1);
        atomicAdd(d_o[reg_val.y], 1);
        atomicAdd(d_o[reg_val.z], 1);
        atomicAdd(d_o[reg_val.w], 1);
    }
}


__device__ __forceinline__ float _sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// sigmoid-baseline
// block(BLOCK_SIZE), grid(N/BLOCK_SIZE)
// a:Nx1, o:Nx1
__global__ void sigmoid(float* d_a, float* d_o, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float output = _sigmoid(d_a[idx]);
        d_o[idx] = output;
    }
}


// sigmoid-vec4
// block(BLOCK_SIZE), grid(N/BLOCK_SIZE)
// a:Nx1, o:Nx1
__global__ void sigmoid_vec4(float* d_a, float* d_o, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx < N) {
        float4 reg_val = FETCH_FLOAT4(d_a[idx]);
        float4 output;
        output.x = _sigmoid(reg_val.x);
        output.y = _sigmoid(reg_val.y);
        output.z = _sigmoid(reg_val.z);
        output.w = _sigmoid(reg_val.w);
        FETCH_FLOAT4(d_o[idx]) = output;
    }
}


