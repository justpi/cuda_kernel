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

/*1.3 dot product pro: 先求点积再求和，相较1.2多了求和的过程*/
template <const int WarpSize=32>
__device__ int warp_reduce_add(float val) {
    #pragma unroll
    for (int mask=WarpSize>>1; mask >= 1; mask>>=1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}
template <const int NUM_THREADS,
        const int WarpSize
        >
__device__ block_reduce_add(float val) {

    
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    const int warp_idx = tid / WarpSize;
    const int lane_idx = tid % WarpSize;

    const int NUM_WARPS = (NUM_THREADS + WarpSize - 1) / WarpSize;
    __shared__ float sdata[NUM_WAPRS];
    val = warp_reduce_add(val);
    sdata[warp_idx] = val;

    val = (lane < NUM_WAPRS) ? sdata[lane]:0.;
    if (warp_idx == 0) val = warp_reduce_add(val);
    return val;
}

// block(256)
// grid([N/256])
// a: Nx1, b:Nx1, out:1
template <const int N>
__global__ void dot_product_base(float* a, float* b, float* out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = a[idx] * b[idx];
    float output = block_reduce_add(val);
    if (tid == 0) out[0] = output;
}

// block(256)
// grid([N/ 256*4])
// a: Nx1, b: Nx1, out:1
template <const int N> 
__global__ void dot_product_vec4(float* a, float* b, float *out) {
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    float4 reg_out;
    float4 reg_a = FETCH_FLOAT4(a[idx]);
    float4 reg_b = FETCH_FLOAT4(b[idx]);
    reg_out.x = reg_a.x * reg_b.x;
    reg_out.y = reg_a.y * reg_b.y;
    reg_out.z = reg_a.z * reg_b.z;
    reg_out.w = reg_a.w * reg_b.w;
    float val = reg.x + reg_out.y + reg_out.z + reg_out.w;
    float output = block_reduce_add(val);
    if (tid == 0) out[0] = output;
}

/*1.4 histgram*/

// block(256)
// grid([N/256])
// a: Nx1, out: counted hisgram
__global__ void histgram_base(int* a, int* out, int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) atomicAdd(&out[a[idx]], 1);
}


// block(256)
// grid([N / (256*4)])
// a: Nx1, out: counted histgram
__global__ void histgram_vec4(int* a, int* out, int N) {
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx < N) {
        float4 reg_a = FETCH_FLOAT4(a[idx]);
        atomicAdd(&out[reg_a.x], 1);
        atomicAdd(&out[reg_a.y], 1);
        atomicAdd(&out[reg_a.z], 1);
        atomicAdd(&out[reg_a.w], 1)
    }
}

