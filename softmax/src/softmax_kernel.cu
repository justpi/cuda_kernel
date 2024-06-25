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
    int iters = (stride + blockDim.x - 1) / blockDim.x;
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
    int iters = (stride + (blockDim.x * 4) - 1) / (blockDim.x * 4);
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


__global__ void softmax_online_vec_share(float *d_a, float *d_o, int length, int stride) {
    /*在向量化访存的基础上，先将数据存入共享内存*/
    __shared__ float sdata[BLOCK_SIZE];
    int iters = (stride + (blockDim.x * 4) - 1) / (blockDim.x * 4);
    int idx;
    float m[4]={-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    float m_[4]={-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    float sums[4]={0.0, 0.0, 0.0, 0.0};
    float values[4];

    /*加载输入到共享内存*/
    for (int i=0; i < iters; ++i) {
        idx = blockIdx.x * stride + i * blockDim.x * 4 + threadIdx.x * 4;
        FETCH_FLOAT4(sdata[threadIdx.x * 4]) = FETCH_FLOAT4(d_a[idx]);
    }
    __syncthreads();

    for (int i=0; i < iters; ++i) {
        idx = threadIdx.x * 4;
        FETCH_FLOAT4(values[0]) = FETCH_FLOAT4(sdata[idx]);
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

// 用于求最值
template<const int WarpSize=32>
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int mask=WarpSize>>1; mask >=1; mask>=1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}
// 用于求和
template<const int WarpSize=32>
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for(int mask=WarpSize>>1; mask>=1; mask>>=1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// block_reduce_sum
template<const int NUM_THREAD=BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum(int val) {
    constexpr int NUM_WARPS = NUM_THREAD / WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    __shared__ float sdata[NUM_WARPS];
    
    val = warp_reduce_sum<WARP_SIZE>(val);
    if (lane == 0) sdata[warp] = val;
    __syncthreads;

    val = (lane < NUM_WARPS) ? sdata[lane]:0.0f;
    val = warp_reduce_sum<WARP_SIZE>(val);
    return  val;
}

// 1-dim softmax-baseline
// block(BLOCK_SIZE), grid(N/BLOCK_SIZE)
// a:N, o: N
template<const int NUM_THREAD=BLOCK_SIZE>
__global__ void softmax_1d(float* d_a, float* d_o, float* total, int N) {
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int NUM_WARPS = NUM_THREAD / WARP_SIZE;
    int warp = tid /  WARP_SIZE;
    int lane = tid % WARP_SIZE;
    __shared__ float  sdata[NUM_WARPS];

    float exp_value = (idx < N) ? expf(d_a[idx]): 0.0f;
    float sum = warp_reduce_sum<WARP_SIZE>(exp_value);
    if (lane == 0) sdata[warp] = sum;
    __syncthreads();
    sum = (lane < NUM_WARPS) ? sdata[lane]:0.0f;
    sum = warp_reduce_sum<WARP_SIZE>(sum);
    if (tid == 0) atomicAdd(total, sum);
    __threadfence();
    if (idx < N) d_o[idx] = exp_value / (*total);
}

// 1-dim softmax-block reduce
// block(BLOCK_SIZE), grid(N/BLOCK_SIZE)
// a:N, o:N
template<const int NUM_THREAD=BLOCK_SIZE>
__global__ void softmax_1d_block(float* d_a, float* d_o, float* total, int N) {
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float exp_val = (idx < N) ? expf(d_a[idx]):0.0f;
    float sum = block_reduce_sum<BLOCK_SIZE>(exp_val);
    if (tid == 0) atomicAdd(total, sum);
    __threadfence();
    if (idx < N) d_o[idx] = exp_val / (*total);
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

    /* online-baseline */
    const int rows = (length + stride - 1) / stride;
    dim3 block(BLOCK_SIZE);
    dim3 grid(rows);
    softmax_online_base<<<grid, block>>>(d_a, d_o, length, stride);
    /* online-向量化访存 */
    if (stride >= BLOCK_SIZE) {
        dim3 block_vec(BLOCK_SIZE/4);
        dim3 grid_vec(rows);
        softmax_online_vec<<<grid_vec, block_vec>>>(d_a, d_o, length, stride);
    }
    else {
        dim3 block_vec(stride/4);
        dim3 grid_vec(rows);
        softmax_online_vec<<<grid_vec, block_vec>>>(d_a, d_o, length, stride);
    }
    

    /* online-共享内存，TODO: 实现尚有问题 */
    if (stride >= BLOCK_SIZE) {
        dim3 block_vec(BLOCK_SIZE/4);
        dim3 grid_vec(rows);
        softmax_online_vec_share<<<grid_vec, block_vec>>>(d_a, d_o, length, stride);
    }
    else {
        dim3 block_vec(stride/4);
        dim3 grid_vec(rows);
        softmax_online_vec_share<<<grid_vec, block_vec>>>(d_a, d_o, length, stride);
    }

    /* 一维softmax实现 */

    /* 拷贝数据 */
    CUDA_CHECK(cudaMemcpy(h_o, d_o, length * sizeof(float), cudaMemcpyDeviceToHost));

    /* free */
    CUDA_CHECK(cudaFree(d_o));
    CUDA_CHECK(cudaFree(d_a));
}