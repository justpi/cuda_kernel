#include "transpose_kernel.h"
#include <cuda_runtime.h>


__global__ void transpose_baseline(float* d_a, float * d_b, int M, int N) {
    /* base版本的transpose函数 */
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_y = blockDim.x * blockIdx.x + threadIdx.x;
    d_b[idx_x * M + idx_y] = d_a[idx_y * N + idx_x];
}



void transpose_kernel_launcher(float* h_a, float* h_b, int M, int N) {
    /* transpose核函数launcher:
        h_a: host端数组
        h_b: host端输出
        N: 数组a的col
        M：数组a的row
     */
    int size = N * M;
    /* 分配global mem空间 */
    float* d_a;
    float* d_b;

    CUDA_CHECK(cudaMalloc(&d_a, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, size * sizeof(float)));

    /* 数据拷贝 */
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size * sizeof(float), cudaMemcpyHostToDevice));

    /* 核函数执行 */
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    transpose_baseline<<<grid, block>>>(d_a, d_b, M, N);

    /* 数据拷回 */
    CUDA_CHECK(cudaMemcpy(h_b, d_b, size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Free */
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_a));
}