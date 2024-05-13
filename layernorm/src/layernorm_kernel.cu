#include "layernorm_kernel.h"



__global__ void layernorm_kernel_base(float *d_a, float *d_o, float gamma, float beta, int length, int stride) {
    /*按照计算公式直接计算*/
    __shared__ float sdata[3][BLOCK_SIZE];  // m, m_, M, count四个数据
    int iters = stride / blockDim.x;
    float m = 0.0, m_= 0.0;
    float M = 0.0;
    float count = 0;

    for (int i=0; i < iters; ++i) {
        int idx = blockIdx.x * stride + blockDim.x * i + threadIdx.x;
        float value = d_a[idx];
        m_ = m;
        m = m * count / (count + 1.0) + value / (count + 1.0);
        count++;
        M = M + (value - m) * (value - m_);
    }
    sdata[0][threadIdx.x] = m;
    sdata[1][threadIdx.x] = M;
    sdata[2][threadIdx.x] = count;
    __syncthreads();

    /*归约计算*/
    for (int s=blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            float old_m = sdata[0][threadIdx.x + s];
            float old_M = sdata[1][threadIdx.x + s];
            float old_count = sdata[2][threadIdx.x + s];

            float new_count = sdata[2][threadIdx.x] + old_count;
            float new_m = (sdata[0][threadIdx.x] * sdata[2][threadIdx.x] + old_m * old_count) / new_count;
            float delta_m = old_m - sdata[0][threadIdx.x];
            float new_M = sdata[1][threadIdx.x] + old_M + delta_m * delta_m * sdata[2][threadIdx.x] * old_count / new_count;

            sdata[0][threadIdx.x] = new_m;
            sdata[1][threadIdx.x] = new_M;
            sdata[2][threadIdx.x] = new_count;
        }
        __syncthreads();
    }

    /*求解layernorm*/
    float mean = sdata[0][0];
    float variance = sdata[1][0] / sdata[2][0];
    float stddev = sqrt(variance + EPSK);
    for (int i=0; i < iters; ++i) {
        int idx = blockIdx.x * stride + blockDim.x * i + threadIdx.x;
        float value = d_a[idx];
        float output = (value - mean) / stddev * gamma + beta;
        d_o[idx] = output;
    }


}

__global__ void layernorm_kernel_unroll(float *d_a, float *d_o, float gamma, float beta, int length, int stride) {
    /*在base基础上添加unroll以及修改计算为fp32*/
    __shared__ float sdata[3][BLOCK_SIZE];  // m, m_, M, count四个数据
    int iters = stride / blockDim.x;
    float m = 0.0f, m_= 0.0f;
    float M = 0.0f;
    float count = 0.0f;
    #pragma unroll
    for (int i=0; i < iters; ++i) {
        int idx = blockIdx.x * stride + blockDim.x * i + threadIdx.x;
        float value = d_a[idx];
        m_ = m;
        m = m * count / (count + 1.0f) + value / (count + 1.0f);
        count++;
        M = M + (value - m) * (value - m_);
    }
    sdata[0][threadIdx.x] = m;
    sdata[1][threadIdx.x] = M;
    sdata[2][threadIdx.x] = count;
    __syncthreads();

    /*归约计算*/
    #pragma unroll 
    for (int s=blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            float old_m = sdata[0][threadIdx.x + s];
            float old_M = sdata[1][threadIdx.x + s];
            float old_count = sdata[2][threadIdx.x + s];

            float new_count = sdata[2][threadIdx.x] + old_count;
            float new_m = (sdata[0][threadIdx.x] * sdata[2][threadIdx.x] + old_m * old_count) / new_count;
            float delta_m = old_m - sdata[0][threadIdx.x];
            float new_M = sdata[1][threadIdx.x] + old_M + delta_m * delta_m * sdata[2][threadIdx.x] * old_count / new_count;

            sdata[0][threadIdx.x] = new_m;
            sdata[1][threadIdx.x] = new_M;
            sdata[2][threadIdx.x] = new_count;
        }
        __syncthreads();
    }

    /*求解layernorm*/
    float mean = sdata[0][0];
    float variance = sdata[1][0] / sdata[2][0];
    float stddev = sqrt(variance + EPSK);
    #pragma unroll
    for (int i=0; i < iters; ++i) {
        int idx = blockIdx.x * stride + blockDim.x * i + threadIdx.x;
        float value = d_a[idx];
        float output = (value - mean) / stddev * gamma + beta;
        d_o[idx] = output;
    }

}

__global__ void layernorm_kernel_bankconflict(float *d_a, float *d_o, float gamma, float beta, int length, int stride) {
    /*在unroll的基础上将访存变为向量化访存*/
    __shared__ float sdata[3][BLOCK_SIZE + 1];  // m, m_, M, count四个数据, pad一个数，第0个位置不存值
    int iters = stride / blockDim.x;
    float m = 0.0f, m_= 0.0f;
    float M = 0.0f;
    float count = 0.0f;
    #pragma unroll
    for (int i=0; i < iters; ++i) {
        int idx = blockIdx.x * stride + blockDim.x * i + threadIdx.x;
        float value = d_a[idx];
        m_ = m;
        m = m * count / (count + 1.0f) + value / (count + 1.0f);
        count++;
        M = M + (value - m) * (value - m_);
    }
    sdata[0][threadIdx.x+1] = m;
    sdata[1][threadIdx.x+1] = M;
    sdata[2][threadIdx.x+1] = count;
    __syncthreads();

    /*归约计算*/
    #pragma unroll 
    for (int s=blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            float old_m = sdata[0][threadIdx.x+1 + s];
            float old_M = sdata[1][threadIdx.x+1 + s];
            float old_count = sdata[2][threadIdx.x+1 + s];

            float new_count = sdata[2][threadIdx.x+1] + old_count;
            float new_m = (sdata[0][threadIdx.x+1] * sdata[2][threadIdx.x+1] + old_m * old_count) / new_count;
            float delta_m = old_m - sdata[0][threadIdx.x+1];
            float new_M = sdata[1][threadIdx.x+1] + old_M + delta_m * delta_m * sdata[2][threadIdx.x+1] * old_count / new_count;

            sdata[0][threadIdx.x+1] = new_m;
            sdata[1][threadIdx.x+1] = new_M;
            sdata[2][threadIdx.x+1] = new_count;
        }
        __syncthreads();
    }

    /*求解layernorm*/
    float mean = sdata[0][1];
    float variance = sdata[1][1] / sdata[2][1];
    float stddev = sqrt(variance + EPSK);
    #pragma unroll
    for (int i=0; i < iters; ++i) {
        int idx = blockIdx.x * stride + blockDim.x * i + threadIdx.x;
        float value = d_a[idx];
        float output = (value - mean) / stddev * gamma + beta;
        d_o[idx] = output;
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
    // if (stride >= BLOCK_SIZE) {
    //     dim3 block_base(BLOCK_SIZE);    // 每个block处理stride个元素
    //     dim3 grid_base(length / stride);
    //     layernorm_kernel_base<<<grid_base, block_base>>>(d_a, d_o, gamma, beta, length, stride);
    // }
    // else {
    //     dim3 block_base(stride);    // 每个block处理stride个元素
    //     dim3 grid_base(length / stride);
    //     layernorm_kernel_base<<<grid_base, block_base>>>(d_a, d_o, gamma, beta, length, stride);
    // }

    // /*unroll 循环&去掉FP64计算*/
    // if (stride >= BLOCK_SIZE) {
    //     dim3 block_base(BLOCK_SIZE);    // 每个block处理stride个元素
    //     dim3 grid_base(length / stride);
    //     layernorm_kernel_unroll<<<grid_base, block_base>>>(d_a, d_o, gamma, beta, length, stride);
    // }
    // else {
    //     dim3 block_base(stride);    // 每个block处理stride个元素
    //     dim3 grid_base(length / stride);
    //     layernorm_kernel_unroll<<<grid_base, block_base>>>(d_a, d_o, gamma, beta, length, stride);
    // }
    /*使用padding解决bank conflict（并没有bank conflict）*/
    if (stride >= BLOCK_SIZE) {
        dim3 block_base(BLOCK_SIZE);    // 每个block处理stride个元素
        dim3 grid_base(length / stride);
        layernorm_kernel_bankconflict<<<grid_base, block_base>>>(d_a, d_o, gamma, beta, length, stride);
    }
    else {
        dim3 block_base(stride);    // 每个block处理stride个元素
        dim3 grid_base(length / stride);
        layernorm_kernel_bankconflict<<<grid_base, block_base>>>(d_a, d_o, gamma, beta, length, stride);
    }

    // /*使用padding解决bank conflict（并没有bank conflict）*/
    // if (stride >= BLOCK_SIZE) {
    //     dim3 block_base(BLOCK_SIZE);    // 每个block处理stride个元素
    //     dim3 grid_base(length / stride);
    //     layernorm_kernel_vec4<<<grid_base, block_base>>>(d_a, d_o, gamma, beta, length, stride);
    // }
    // else {
    //     dim3 block_base(stride);    // 每个block处理stride个元素
    //     dim3 grid_base(length / stride);
    //     layernorm_kernel_vec4<<<grid_base, block_base>>>(d_a, d_o, gamma, beta, length, stride);
    // }

    

    /*数据拷回*/
    CUDA_CHECK(cudaMemcpy(h_o, d_o, length * sizeof(float), cudaMemcpyDeviceToHost));

    /*free*/
    CUDA_CHECK(cudaFree(d_o));
    CUDA_CHECK(cudaFree(d_a));
}