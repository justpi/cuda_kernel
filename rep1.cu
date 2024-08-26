#include <cuda_runtime.h>

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0]) 
/*1. element-wise op*/

/*1.1 add */

// block(256)：每个线程计算一个element
// grid(round(N/256))
// a: N, b: N, out:N
__global__ void add_base(float* a, float* b, float* out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = a[idx] * b[idx];
}

// block(256)   每个线程计算4个elements
// grid(round(N/(256*4)))
// a: N, b: N, out:N
__global__ void add_vec4(float* a, float* b, float* out) {
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    float4 reg_a = FETCH_FLOAT4(a[idx]);
    float4 reg_b = FETCH_FLOAT4(b[idx]);
    float4 reg_out;
    reg_out.x = reg_a.x + reg_b.x;
    reg_out.y = reg_a.y + reg_b.y;
    reg_out.z = reg_a.z + reg_b.z;
    reg_out.w = reg_a.w + reg_b.w;
    FETCH_FLOAT4(out[idx]) = reg_out;
}


/*1.2 dot product */

// block(256): 每个线程计算一个element
// grid([N/256])
// a: N, b: N, out: N
__global__ void dot_base(float* a, float* b, float* out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = a[idx] * b[idx];
}

// block(256): 每个线程计算4个人element
// grid([N/(256 * 4)])
// a:N, b:N, out:N
__global__ void dot_vec4(float* a, float* b, float* out) {
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    float4 reg_a = FETCH_FLOAT4(a[idx]);
    float4 reg_b = FETCH_FLOAT4(b[idx]);
    float4 reg_out;
    reg_out.x = reg_a.x * reg_b.x;
    reg_out.y = reg_a.y * reg_b.y;
    reg_out.z = reg_a.z * reg_b.z;
    reg_out.w = reg_a.w * reg_b.w;
    FETCH_FLOAT4(out[idx]) = reg_out;
}


