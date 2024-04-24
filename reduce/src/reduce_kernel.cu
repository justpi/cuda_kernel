#include "reduce_kernel.h"
#include <cuda_runtime.h>
#include <iostream>


__global__ void reduce_kernel_baseline(float* d_in, float* d_out) {
    /*base版本的reduce算子
    d_in: reduce算子的输入，表示为一维数组；
    d_out: reduce算子的输出，表示为一维数组；
    使用共享内存进行访存, block和grid均为一维
    */
   __shared__ float sdata[THREADS_PER_BLOCK];

   /* 获取输入的index和当前进程的index */
   unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
   unsigned int tidx = threadIdx.x;

   /* 读取输入到共享内存 */
    sdata[tidx] = d_in[idx];
   __syncthreads();

    /*  进行reduce操作:
        这个过程需要进行多轮迭代，
        在第一轮迭代中，如果tid%2 ==0, 则第tid号线程将shared memory中第tid号位置的值和第tid+1号的值进行相加，而后放在第tid号位置。
        在第二轮迭代中，如果tid%4==0,则第tid号线程将shared memory中第tid号位置的值和第tid+2号的值进行相加，而后放在第tid号位置。
        不断迭代，则所有元素都将被累加到第0号位置。
     */
    for (int s=1; s < blockDim.x; s*=2) {
        if (tidx % (2*s) == 0)
            sdata[tidx] += sdata[s + tidx];
        __syncthreads();
    }
    /* 写回到主存中 */
    if (tidx == 0) d_out[blockIdx.x] = sdata[0];
}

__global__ void reduce_kernel_div(float* d_in, float* d_out) {
    /*在base版本的reduce算子上改善了warp的divergence
    d_in: reduce算子的输入，表示为一维数组；
    d_out: reduce算子的输出，表示为一维数组；
    使用共享内存进行访存, block和grid均为一维
    */
   __shared__ float sdata[THREADS_PER_BLOCK];

   /* 获取输入的index和当前进程的index */
   unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
   unsigned int tidx = threadIdx.x;

   /* 读取输入到共享内存 */
    sdata[tidx] = d_in[idx];
   __syncthreads();

    /*  进行reduce操作:
        虽然代码依旧存在着if语句，但是却与reduce0代码有所不同。
        我们继续假定block中存在256个thread，即拥有256/32=8个warp。当进行第1次迭代时，0-3号warp的index<blockDim.x， 4-7号warp的index>=blockDim.x。
        对于每个warp而言，都只是进入到一个分支内，所以并不会存在warp divergence的情况。
        当进行第2次迭代时，0、1号两个warp进入计算分支。
        当进行第3次迭代时，只有0号warp进入计算分支。
        当进行第4次迭代时，只有0号warp的前16个线程进入分支。
        此时开始产生warp divergence。通过这种方式，我们消除了前3次迭代的warp divergence。
     */
    for (int s=1; s < blockDim.x; s*=2) {
        int index = 2 * s * tidx;
        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }
    /* 写回到主存中 */
    if (tidx == 0) d_out[blockIdx.x] = sdata[0];
}


__global__ void reduce_kernel_bankconflic(float* d_in, float* d_out) {
    /*在div版本的reduce算子上改善了bank conflict
    d_in: reduce算子的输入，表示为一维数组；
    d_out: reduce算子的输出，表示为一维数组；
    使用共享内存进行访存, block和grid均为一维
    */
   __shared__ float sdata[THREADS_PER_BLOCK];

   /* 获取输入的index和当前进程的index */
   unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
   unsigned int tidx = threadIdx.x;

   /* 读取输入到共享内存 */
    sdata[tidx] = d_in[idx];
   __syncthreads();

    /*  进行reduce操作:
        把目光继续看到这个for循环中，并且只分析0号warp。
        0号线程需要load shared memory的0号元素以及128号元素。
        1号线程需要load shared memory中的1号元素和129号元素。
        这一轮迭代中，在读取第一个数时，warp中的32个线程刚好load 一行shared memory数据。
        再分析第2轮迭代，0号线程load 0号元素和64号元素，1号线程load 1号元素和65号元素。每次load shared memory的一行。
        再来分析第3轮迭代，0号线程load 0号元素和32号元素，
        接下来不写了，总之，一个warp load shared memory的一行。
        没有bank冲突。
        到了4轮迭代，0号线程load 0号元素和16号元素。
        那16号线程呢，16号线程啥也不干，因为s=16，16-31号线程啥也不干，跳过去了。
     */
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tidx < s) {
            sdata[tidx] += sdata[tidx + s];
        }
        __syncthreads();
    }
    /* 写回到主存中 */
    if (tidx == 0) d_out[blockIdx.x] = sdata[0];
}


__global__ void reduce_kernel_idle(float* d_in, float* d_out) {
    /*在bank conflict版本的reduce算子上改善了线程利用率
    d_in: reduce算子的输入，表示为一维数组；
    d_out: reduce算子的输出，表示为一维数组；
    使用共享内存进行访存, block和grid均为一维
    */
   __shared__ float sdata[THREADS_PER_BLOCK];

   /* 获取输入的index和当前进程的index，在原来基础上一个线程计算两个 */
   unsigned int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
   unsigned int tidx = threadIdx.x;

   /* 读取输入到共享内存，添加一个访存和求和操作 */
    sdata[tidx] = d_in[idx] + d_in[idx+blockDim.x];
   __syncthreads();

    /*  进行reduce操作:
        把目光继续看到这个for循环中，并且只分析0号warp。
        0号线程需要load shared memory的0号元素以及128号元素。
        1号线程需要load shared memory中的1号元素和129号元素。
        这一轮迭代中，在读取第一个数时，warp中的32个线程刚好load 一行shared memory数据。
        再分析第2轮迭代，0号线程load 0号元素和64号元素，1号线程load 1号元素和65号元素。每次load shared memory的一行。
        再来分析第3轮迭代，0号线程load 0号元素和32号元素，
        接下来不写了，总之，一个warp load shared memory的一行。
        没有bank冲突。
        到了4轮迭代，0号线程load 0号元素和16号元素。
        那16号线程呢，16号线程啥也不干，因为s=16，16-31号线程啥也不干，跳过去了。
     */
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tidx < s) {
            sdata[tidx] += sdata[tidx + s];
        }
        __syncthreads();
    }
    /* 写回到主存中 */
    if (tidx == 0) d_out[blockIdx.x] = sdata[0];
}


__device__ void warpReduce32(float* cache, int tidx) {
    cache[tidx] += cache[tidx + 32];
    __syncthreads();
    cache[tidx] += cache[tidx + 16];
    __syncthreads();
    cache[tidx] += cache[tidx + 8];
    __syncthreads();
    cache[tidx] += cache[tidx + 4];
    __syncthreads();
    cache[tidx] += cache[tidx + 2];
    __syncthreads();
    cache[tidx] += cache[tidx + 1];
    __syncthreads();
}

__global__ void reduce_kernel_unrollLast32(float* d_in, float* d_out) {
    /*在idel版本的reduce算子上去掉最后几次循环，将最后5个循环写成一个
    d_in: reduce算子的输入，表示为一维数组；
    d_out: reduce算子的输出，表示为一维数组；
    使用共享内存进行访存, block和grid均为一维
    */
   __shared__ float sdata[THREADS_PER_BLOCK];

   /* 获取输入的index和当前进程的index，在原来基础上一个线程计算两个 */
   unsigned int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
   unsigned int tidx = threadIdx.x;

   /* 读取输入到共享内存，添加一个访存和求和操作 */
    sdata[tidx] = d_in[idx] + d_in[idx + blockDim.x];
   __syncthreads();

    /*  进行reduce操作:
        把目光继续看到这个for循环中，并且只分析0号warp。
        0号线程需要load shared memory的0号元素以及128号元素。
        1号线程需要load shared memory中的1号元素和129号元素。
        这一轮迭代中，在读取第一个数时，warp中的32个线程刚好load 一行shared memory数据。
        再分析第2轮迭代，0号线程load 0号元素和64号元素，1号线程load 1号元素和65号元素。每次load shared memory的一行。
        再来分析第3轮迭代，0号线程load 0号元素和32号元素，
        接下来不写了，总之，一个warp load shared memory的一行。
        没有bank冲突。
        到了4轮迭代，0号线程load 0号元素和16号元素。
        那16号线程呢，16号线程啥也不干，因为s=16，16-31号线程啥也不干，跳过去了。
     */
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tidx < s) {
            sdata[tidx] += sdata[tidx + s];
        }
        __syncthreads();
    }
    if (tidx <= 32) warpReduce32(sdata, tidx);
    /* 写回到主存中 */
    if (tidx == 0) {
        d_out[blockIdx.x] = sdata[0];
    }
        
}


__global__ void reduce_kernel_unroll(float* d_in, float* d_out) {
    /*在idel版本的reduce算子上去掉最后几次循环，将最后5个循环写成一个
    d_in: reduce算子的输入，表示为一维数组；
    d_out: reduce算子的输出，表示为一维数组；
    使用共享内存进行访存, block和grid均为一维
    */
   __shared__ float sdata[THREADS_PER_BLOCK];

   /* 获取输入的index和当前进程的index，在原来基础上一个线程计算两个 */
   unsigned int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
   unsigned int tidx = threadIdx.x;

   /* 读取输入到共享内存，添加一个访存和求和操作 */
    sdata[tidx] = d_in[idx] + d_in[idx + blockDim.x];
   __syncthreads();

    /*  进行reduce操作: 添加循环展开
        把目光继续看到这个for循环中，并且只分析0号warp。
        0号线程需要load shared memory的0号元素以及128号元素。
        1号线程需要load shared memory中的1号元素和129号元素。
        这一轮迭代中，在读取第一个数时，warp中的32个线程刚好load 一行shared memory数据。
        再分析第2轮迭代，0号线程load 0号元素和64号元素，1号线程load 1号元素和65号元素。每次load shared memory的一行。
        再来分析第3轮迭代，0号线程load 0号元素和32号元素，
        接下来不写了，总之，一个warp load shared memory的一行。
        没有bank冲突。
        到了4轮迭代，0号线程load 0号元素和16号元素。
        那16号线程呢，16号线程啥也不干，因为s=16，16-31号线程啥也不干，跳过去了。
     */
    #pragma unroll
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tidx < s) {
            sdata[tidx] += sdata[tidx + s];
        }
        __syncthreads();
    }
    if (tidx <= 32) warpReduce32(sdata, tidx);
    /* 写回到主存中 */
    if (tidx == 0) {
        d_out[blockIdx.x] = sdata[0];
    }
        
}


void reduce_kernel_launcher(float* h_in, float* h_out, int N) {
    /* reduce算子的执行函数 */
    int M = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;


    /* 分配GPU内存 */
    float* d_in;
    float* d_out;

    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, M * sizeof(float)));

    /* 拷贝数据到global mem */
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));
    
    /* 设置block和grid:每个block计算一个结果d_out值 */
    /* baseline/1/2使用blocks和grids组织运行时线程 */
    dim3 blocks(THREADS_PER_BLOCK);
    dim3 grids(M);
    /* 3/4/5/6使用blocks_idle, grid_idle组织运行时线程 */
    dim3 blocks_idle(THREADS_PER_BLOCK / 2);
    dim3 grids_idle(M);

    /* 执行核函数 */

    /* baseline*/
    // reduce_kernel_baseline<<<grids, blocks>>>(d_in, d_out);

    /* 优化1：改善线程的divergence */
    // reduce_kernel_div<<<grids, blocks>>>(d_in, d_out);
    
    /* 优化2： 在上面的基础上改善了bank conflict */
    // reduce_kernel_bankconflic<<<grids, blocks>>>(d_in, d_out);
    
    /* 优化3：在上面的基础上改善了进程使用率（idle） */
    // reduce_kernel_idle<<<grids_idle, blocks_idle>>>(d_in, d_out);
    
    /* 优化4：在优化3的基础上unroll了最后的几个循环计算 */
    // reduce_kernel_unrollLast32<<<grids_idle, blocks_idle>>>(d_in, d_out);
    
    /* 优化5：在优化4的基础上直接循环展开 */
    reduce_kernel_unroll<<<grids_idle, blocks_idle>>>(d_in, d_out);
    
    /* 优化6：TODO:在使用shuffle指令 */


    /* 数据拷回host */
    CUDA_CHECK(cudaMemcpy(h_out, d_out, M * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_in));

}


