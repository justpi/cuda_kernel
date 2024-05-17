#include "hgemm_kernel.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

/* half4读取数据的宏 */
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
// 此宏将传递给它的指针 pointer 解释为指向 half4 类型数据的指针，并提取该位置的 half4 值。
#define FETCH_HALF2(pointer) (reinterpret_cast<half2*>(&(pointer))[0])

#define div_ceil(a, b) ((a + b - 1) / b)


__global__ void gemm_baseline(half* d_a, half* d_b, half* d_c, int N, int K, int M) {
    /* baseline:
        d_a: M * K
        d_b: K * N
        d_c: M * N
    */
    /* 计算d_c索引 (i, j) */
    unsigned int c_i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int c_j = blockDim.y * blockIdx.y + threadIdx.y;
    /* 执行计算 */
    half output = 0.0;
    for (int k=0; k < K; ++k) {
        output += d_a[c_i * K + k] * d_b[k * N + c_j];
    }
    /* 结果写回 */
    d_c[c_i * N + c_j] = output;

}

template <const int tile>
__global__ void sgemm_tile(half* d_a, half* d_b, half* d_c, int M, int N, int K) {
    /* 当K过大时，一个block无法存入所有值，需要对K进行分块
       在grid上添加一维z来存储K/TILE个block，每个thread计算TILE个乘加
    */
    unsigned int c_i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int c_j = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int tile_idx = blockIdx.z;

    half output = 0.0;
    size_t a_idx, b_idx;

    if (c_i < M && c_j < N) {
        #pragma unroll
        for (int k=0; k < tile; ++k) {
            a_idx = c_i * K + k + tile_idx * tile;
            b_idx = (k + tile_idx * tile) * N + c_j;
            if (a_idx < M*K && b_idx < K*N) {
                output += d_a[a_idx] * d_b[b_idx];
            }
        }
        atomicAdd(&d_c[c_i * N + c_j], output);
    }
}


__global__ void sgemm_tile_share(half* d_a, half* d_b, half* d_c, int M, int N, int K) {
    /* 在tile的基础上，增加共享内存 */
    __shared__ half sdata_a[TILE][TILE];
    __shared__ half sdata_b[TILE][TILE];

    unsigned int c_i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int c_j = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int tile_idx = blockIdx.z;
    
    unsigned int tidx_y = threadIdx.y;
    unsigned int tidx_x = threadIdx.x;

    half output = 0.0;
    size_t a_idx, b_idx;
    /* 读取数据到共享内存中 */
    for(int i=0; i < TILE; ++i) {
        a_idx = c_i * K + i + TILE * tile_idx;
        b_idx = (i + TILE * tile_idx) * N + c_j;
        sdata_a[tidx_y][tidx_x] = d_a[a_idx];
        sdata_b[tidx_x][tidx_y] = d_b[b_idx];
        __syncthreads();
    } 

    /* 从共享内存读取数据完成计算 */
    for (int k=0; k < TILE; ++k) {
        output += sdata_a[tidx_y][k] * sdata_b[tidx_x][k];
    }
    atomicAdd(&d_c[c_i * N + c_j], output);
}


__global__ void gemm_share(half* d_a, half* d_b, half* d_c, int N, int K, int M) {
    /* 在baseline的基础上添加共享内存来存放输入值
        TODO:待完成，目前的线程组织分配还有问题
    */
    __shared__ half sdata_a[SHARED_BLOCK_SIZE][SHARED_BLOCK_SIZE];
    __shared__ half sdata_b[SHARED_BLOCK_SIZE][SHARED_BLOCK_SIZE];

    unsigned int c_i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int c_j = blockDim.y * blockIdx.y + threadIdx.y;

    /* 拷贝数据到共享内存,将数据拆分成K/SHARED_BLOCK_SIZE(此处为方便写代码假设K=M=N) */
    unsigned int sblocks = (K + SHARED_BLOCK_SIZE -1 ) / SHARED_BLOCK_SIZE;
    for (int m=0; m < sblocks; ++m) {
        sdata_a[threadIdx.y][threadIdx.x] = d_a[(c_i + m * SHARED_BLOCK_SIZE + threadIdx.y) * K + m * SHARED_BLOCK_SIZE + threadIdx.x];
        sdata_b[threadIdx.y][threadIdx.x] = d_b[(m * SHARED_BLOCK_SIZE + threadIdx.x) * N + c_j + m * SHARED_BLOCK_SIZE + threadIdx.y];
        __syncthreads();
    }
    /* 计算矩阵乘法 */
    half output = 0.0;
    for (int i=0; i < SHARED_BLOCK_SIZE; ++i) {
        output += sdata_a[threadIdx.y][threadIdx.x] * sdata_b[threadIdx.y][threadIdx.x];
    }
    /* 写回global mem */
    d_c[c_i * N + c_j] += output;
    
}

template<
    const int BLOCK_SIZE_M,     // 每个线程块计算的矩阵C的连续行的数量
    const int BLOCK_SIZE_K,     // 每个线程块加载到共享内存中的矩阵A的连续列的数量
    const int BLOCK_SIZE_N,     // 每个线程块计算的矩阵C的连续列的数量
    const int THREAD_SIZE_Y,    // 每个线程计算矩阵C的block的行数
    const int THREAD_SIZE_X,    // 每个线程计算矩阵C的block的列数
    const bool ENABLE_DOUBLE_BUFFER //是否启用数据预取
    >
__global__ void gemm_without_prefectch(half* __restrict__ d_a, half* __restrict__ d_b, half* __restrict__ d_c, int M, int K, int N) {
    /*没有采用数据预取的矩阵乘法函数
    线程组织：
        设置grid：M / BLOCK_SIZE_M, block: BLOCK_SZIE_M / THREAD_SIZE_Y
        假如输入M=2048，N=2048，那么grid= 16 * 16,block=16 * 16
    每个block的计算逻辑：
        一共需要256次迭代，每次迭代：
            1.将矩阵A里面的128x8个元素和矩阵B里面的8x128个元素存入共享内存中。
            2.然后这个block中的256个线程把结果计算出来。每个线程的计算逻辑：
                2.1 每个线程需要进行8次迭代（BLOCK_SIZE_K）
                2.2 每次迭代中，每个线程从共享内存拿到A矩阵的一小列（8个数）和B矩阵的一小行（8个数）
                2.3 线程将这8+8个元素存入寄存器中。
                2.4 每个线程负责8x8=64个元素计算，一共会有64个FFMA指令
            3.计算完成后进入下一次迭代
        

    */
}

template<
    const int BLOCK_SIZE_M,     // 每个线程块计算的矩阵C的连续行的数量
    const int BLOCK_SIZE_K,     // 每个线程块加载到共享内存中的矩阵A的连续列的数量
    const int BLOCK_SIZE_N,     // 每个线程块计算的矩阵C的连续列的数量
    const int THREAD_SIZE_Y,    // 每个线程计算矩阵C的block的行数
    const int THREAD_SIZE_X,    // 每个线程计算矩阵C的block的列数
    const bool ENABLE_DOUBLE_BUFFER //是否启用数据预取
    >
__global__ void gemm_prefetch(half* __restrict__ d_a, half* __restrict__ d_b, half* __restrict__ d_c, int M, int K, int N) {
    /*采用数据预取的矩阵乘法函数，与上面函数的区别主要是两个方面：
        1. 开启的共享内存和寄存器数量：需要开启两倍的共享内存和寄存器数量。共享内存：bmxbk->bmxbkx2, bkxbn->bkxbnx2，寄存器：rm+rn->rmx2+rnx2
        2. 提前将一些数据放置在共享内存中
    每个block的计算逻辑：
        1. 先将第0轮迭代的数据存入共享内存，将线程第0次迭代的数据存入寄存器
        2. 执行第i次迭代，读取第i-1次存入的共享内存数据，线程第i此迭代读取第i-1次迭代的数据；同时存入第i+1此迭代的共享内存数据和线程第i+1次迭代的寄存器数据
        3. 通过多个FFMA计算来掩盖从global mem读取数据产生的延迟（需要有足够多的FFMA才能掩盖，所以这里设置了64个计算）
    */
    
    /* block idx */
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;

    /* thread idx */
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;

    /* 每个block有多少个x方向和y方向的线程 */
    const int THREAD_X_PER_BLOCK = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int THREAD_Y_PER_BLOCK = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int THREAD_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;

    /* 当前线程在此block中的id号 */
    const int tid = ty * THREAD_X_PER_BLOCK + tx;

    /* 设置共享内存和寄存器 */
    __shared__ half As[2][BLOCK_SIZE_K][BLOCK_SIZE_M];     // 为加快后续的访存，进行了一次转置，为了预取，开了两倍的buffer，一半用来read数据，一半用来write数据
    __shared__ half Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N];     //预取，开了两倍的buffer，一半用来read数据，一半用来write数据

    /* 用来临时存储矩阵C的计算结果 */
    half accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};
    /* 矩阵A的寄存器存储，直接申请一个小数组即可，编译时编译器会将其放置在寄存器中，如果数组较大，会溢出到本地内存中，这时访问数组的时钟周期会达到几百 */
    half frag_a[2][THREAD_SIZE_Y];
    /* 矩阵B的寄存器存储 */
    half frag_b[2][THREAD_SIZE_X];

    /* (global->shared)加载数据次数：加载矩阵A的一个共享内存块，使用half4从主存读取数据，需要加载的次数 */
    const int ldg_num_a = BLOCK_SIZE_M * BLOCK_SIZE_K / (THREAD_PER_BLOCK * 2);
    const int ldg_num_b = BLOCK_SIZE_K * BLOCK_SIZE_N / (THREAD_PER_BLOCK * 2);
    /* (global->shared)将数据存入共享内存需要调用寄存器数量 */
    half ldg_a_reg[2 * ldg_num_a];
    half ldg_b_reg[2 * ldg_num_b];

    /* 每四个数进行一次访存操作，计算每行和每列需要进行多少次访存 */
    const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 2;
    const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 2;

    const int A_TILE_THREAD_PER_COL = BLOCK_SIZE_M / 2;
    const int B_TILE_THREAD_PER_COL = BLOCK_SIZE_K / 2;

    /* 计算当前进程负责的数据访存（x, y）->(此block mem中起始点的坐标) */
    const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;

    const int A_TILE_COL_START = tid % A_TILE_THREAD_PER_ROW * 2;
    const int B_TILE_COL_START = tid % B_TILE_THREAD_PER_ROW * 2;

    /* 计算每个线程需要加载多少次数据 */
    const int A_TILE_ROW_STRIDE = THREAD_PER_BLOCK / A_TILE_THREAD_PER_ROW;
    const int B_TILE_ROW_STRIDE = THREAD_PER_BLOCK / B_TILE_THREAD_PER_ROW;

    d_a = &d_a[(BLOCK_SIZE_M * by) * K];    // 当前block负责的A的首地址；
    d_b = &d_b[BLOCK_SIZE_N * bx];          // 当前block负责的B的首地址；

    /* 加载矩阵A到共享内存：数据流动方向global mem -> register -> shared mem */
    #pragma unroll
    for (int i=0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) {
        int ldg_index = i / A_TILE_ROW_STRIDE * 2;
        FETCH_HALF2(ldg_a_reg[ldg_index]) = FETCH_HALF2(d_a[OFFSET(A_TILE_ROW_START + i,    // row
                                                                 A_TILE_COL_START,                              //col
                                                                 K)]);
        As[0][A_TILE_COL_START][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index];
        As[0][A_TILE_COL_START+1][A_TILE_ROW_START + i] = ldg_a_reg[ldg_index+1];
    }
    /* 加载矩阵B到共享内存 */
    #pragma unroll
    for (int i=0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
        FETCH_HALF2(Bs[0][B_TILE_ROW_START + i][B_TILE_COL_START]) = FETCH_HALF2(d_b[OFFSET(
                                                                                            B_TILE_ROW_START + i,                   // row
                                                                                            B_TILE_COL_START,   // col
                                                                                            N
                                                                                    )]);
    }
    __syncthreads();

    /* 将共享内存数据预取到寄出器中 */
    #pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 2) {
        FETCH_HALF2(frag_a[0][thread_y]) = FETCH_HALF2(As[0][0][THREAD_SIZE_Y * ty + thread_y]);
    }

    #pragma unroll
    for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 2) {
        FETCH_HALF2(frag_b[0][thread_x]) = FETCH_HALF2(Bs[0][0][THREAD_SIZE_X * tx + thread_x]);
    }

    /* 执行迭代：包括外层的256个迭代和每个线程的迭代 */
    int write_stage_idx = 1;
    int tile_idx = 0;
    do{
        tile_idx += BLOCK_SIZE_K;
        /* 如果还有下一个迭代，加载下一个block的数据到寄存器 */
        if (tile_idx < K) {
            #pragma unroll
            for (int i=0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) {
                int ldg_index = i / A_TILE_ROW_STRIDE * 2;
                FETCH_HALF2(ldg_a_reg[ldg_index]) = FETCH_HALF2(d_a[OFFSET(
                                                                            A_TILE_ROW_START + i,
                                                                            A_TILE_COL_START + tile_idx,
                                                                            K
                                                                        )]);
            }
            #pragma unroll
            for (int i=0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
                int ldg_index = i / B_TILE_ROW_STRIDE * 2;
                FETCH_HALF2(ldg_b_reg[ldg_index]) = FETCH_HALF2(d_b[OFFSET(
                                                                            tile_idx + B_TILE_ROW_START + i,
                                                                            B_TILE_COL_START,
                                                                            N                                                        
                                                                        )]);
            }
        }
        /* 该变量表示需要从As的哪个空间进行读数 */
        int load_stage_idx = write_stage_idx ^ 1;
        /* 需要完成7次小迭代，由于在小迭代中也采用了双缓冲的方式，需要将下一轮的小迭代数据提前写入寄存器中 */
        #pragma unroll
        for (int j = 0; j < BLOCK_SIZE_K-1; ++j) {
            /* 从共享内存As读取数据到寄存器frag_a中 */
            #pragma unroll
            for (int thread_y=0; thread_y < THREAD_SIZE_Y; thread_y += 2) {
                FETCH_HALF2(frag_a[(j+1)%2][thread_y]) = FETCH_HALF2(As[load_stage_idx][j+1][THREAD_SIZE_Y * ty + thread_y]);
            }
            /* 从共享内存Bs读取数据到寄存器frag_b中 */
            #pragma unroll
            for (int thread_x=0; thread_x < THREAD_SIZE_X; thread_x +=2) {
                FETCH_HALF2(frag_b[(j+1)%2][thread_x]) = FETCH_HALF2(Bs[load_stage_idx][j+1][THREAD_SIZE_X * tx + thread_x]);
            }

            /* 计算矩阵C的值，并将其存入accum寄存器中 */
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
                #pragma unroll
                for (int thread_x=0; thread_x < THREAD_SIZE_X; ++thread_x) {
                    accum[thread_y][thread_x] += frag_a[j%2][thread_y] * frag_b[j%2][thread_x];
                }
            }
        }

        /* 将存储在临时寄存器的数据搬运到共享内存中 */
        if (tile_idx < K) {
            /* 将A的数据写回共享内存 */
            #pragma unroll
            for (int i=0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) {
                int ldg_index = i / A_TILE_ROW_STRIDE * 2;
                As[write_stage_idx][A_TILE_COL_START][A_TILE_ROW_START+i] = ldg_a_reg[ldg_index];
                As[write_stage_idx][A_TILE_COL_START+1][A_TILE_ROW_START+i] = ldg_a_reg[ldg_index+1];
            }
            /*将B的数据写回共享内存*/
            #pragma unroll
            for (int i=0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
                int ldg_index = i / B_TILE_ROW_STRIDE * 2;
                FETCH_HALF2(Bs[write_stage_idx][B_TILE_ROW_START + i][B_TILE_COL_START]) = FETCH_HALF2(ldg_b_reg[ldg_index]);
            }
            /* 使用double buffer，只需要一个sync */
            __syncthreads();

            /* 切换状态 */
            write_stage_idx ^= 1;
        }

        /* 完成最后一个小迭代以及寄存器的预取 */
        #pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 2) {
            FETCH_HALF2(frag_a[0][thread_y]) = FETCH_HALF2(As[load_stage_idx^1][0][THREAD_SIZE_Y * ty + thread_y]);
        }
        /* 从共享内存加载矩阵B的数据 */
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 2) {
            FETCH_HALF2(frag_b[0][thread_x]) = FETCH_HALF2(Bs[load_stage_idx^1][0][THREAD_SIZE_X * tx + thread_x]);
        }

        /* 计算最后一个tile的值 */
        #pragma unroll
        for (int thread_y=0; thread_y < THREAD_SIZE_Y; ++thread_y) {
            #pragma unroll
            for (int thread_x=0; thread_x <THREAD_SIZE_X; ++thread_x) {
                accum[thread_y][thread_x] += frag_a[1][thread_y] * frag_b[1][thread_x];
            }
        }
    }while(tile_idx < K);

    /* 将计算结果写回主存 */
    #pragma unroll
    for (int thread_y=0; thread_y < THREAD_SIZE_Y; ++thread_y) {
        #pragma unroll
        for (int thread_x=0; thread_x < THREAD_SIZE_X; thread_x += 2) {
            FETCH_HALF2(d_c[OFFSET(
                BLOCK_SIZE_M * by + ty * THREAD_SIZE_Y + thread_y,
                BLOCK_SIZE_N * bx + tx * THREAD_SIZE_X + thread_x,
                N
            )]) = FETCH_HALF2(accum[thread_y][thread_x]);
        }
    }
}


__global__ void hgemm_tensorcore(half *d_a, half *d_b, half *d_c, int N, int K, int M) {
    /*混合精度计算，使用tensorcore完成矩阵计算*/
    int k_tile = div_ceil(K, WMMA_K);

    int idx_row = blockIdx.y * WMMA_M;
    int idx_col = blockIdx.x * WMMA_N;

    if (idx_row < M && idx_col < N) {
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> frag_c;
        /*frag_c赋值*/
        wmma::fill_fragment(frag_c, 0.0f);
        /*frag_c需要从主存读数赋值*/
        wmma::load_matrix_sync(frag_c, d_c + idx_row * N + idx_col, N, wmma::mem_row_major);
        for (int i=0; i < k_tile; ++i) {
            /**/
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_a;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_b;

            /*从global mem拷贝数据*/
            wmma::load_matrix_sync(frag_a, d_a + idx_row * K + i * WMMA_K, K);
            wmma::load_matrix_sync(frag_b, d_b + (i * WMMA_K) * N + idx_col, N);

            /*计算*/
            wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        }

        wmma::store_matrix_sync(d_c + idx_row * N + idx_col, frag_c, N, wmma::mem_row_major);
    }

}



void cublas_gemm(half* d_a, half* d_b, half* d_c, int N, int K, int M) {
    /* cublas版本单精度矩阵乘法实现 */
    
    cublasHandle_t blas_handle;
    cublasCreate(&blas_handle);
    half alpha = 1.0f;
    half beta = 0.0f;
    cublasHgemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                M, N, K, &alpha,
                d_a, M, d_b, K, &beta, d_c, M 
                );
}

void hgemm_cublas_launcher(half* a, half* b, half* c, int N, int K, int M) {
    /* 分配GPU资源 */
    int sizeA = M*K;
    int sizeB = K*N;
    int sizeC = M*N;
    half* d_a;
    half* d_b;
    half* d_c;
    CUDA_CHECK(cudaMalloc(&d_a, sizeA * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_b, sizeB * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_c, sizeC * sizeof(half)));

    /* 拷贝数据到global mem */
    CUDA_CHECK(cudaMemcpy(d_a, a, sizeA * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, sizeB * sizeof(half), cudaMemcpyHostToDevice));

    /* cublas实现 */
    cublas_gemm(d_a, d_b, d_c, N, K, M);

    /* 结果拷回host内存 */
    CUDA_CHECK(cudaMemcpy(c, d_c, sizeC * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    LAST_KERNEL_CHECK();

    /* 释放显存 */
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_a));

}

void hgemm_kernel_launcher(half* a, half* b, half* c, int N, int K, int M) {
    /* 分配GPU资源 */
    int sizeA = M*K;
    int sizeB = K*N;
    int sizeC = M*N;
    half* d_a;
    half* d_b;
    half* d_c;
    CUDA_CHECK(cudaMalloc(&d_a, sizeA * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_b, sizeB * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_c, sizeC * sizeof(half)));

    /* 拷贝数据到global mem */
    CUDA_CHECK(cudaMemcpy(d_a, a, sizeA * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, sizeB * sizeof(half), cudaMemcpyHostToDevice));


    /* baseline：naive版本的gemm，每个线程处理一个output元素 */
    // dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    // dim3 grid((M + BLOCK_SIZE - 1)/ BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    // gemm_baseline<<<grid, block>>>(d_a, d_b, d_c, N, K, M);

    /*优化一：对K使用tile*/
    // dim3 block_1(BLOCK_SIZE, BLOCK_SIZE);
    // dim3 grid_1((M + BLOCK_SIZE - 1)/ BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE, (K + TILE - 1) / TILE);
    // sgemm_tile<TILE><<<grid_1, block_1>>>(d_a, d_b, d_c, M, N, K);

    // /* 优化二：tileK + shared mem */
    // dim3 block_2(BLOCK_SIZE, BLOCK_SIZE);
    // dim3 grid_2((M + BLOCK_SIZE - 1)/ BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE, (K + TILE - 1) / TILE);
    // sgemm_tile_share<<<grid_2, block_2>>>(d_a, d_b, d_c, M, N, K);
    /* 优化二：使用共享内存+tile */
    // gemm_share<<<grid, block>>>(d_a, d_b, d_c, N, K, M);

    // /* 优化三：使用tile+prefetch策略 */
    // const int BLOCK_SIZE_M = 128;     // 每个线程块计算的矩阵C的连续行的数量
    // const int BLOCK_SIZE_K = 8;     // 每个线程块加载到共享内存中的矩阵A的连续列的数量
    // const int BLOCK_SIZE_N = 128;     // 每个线程块计算的矩阵C的连续列的数量
    // const int THREAD_SIZE_Y = 8;    // 每个线程计算矩阵C的block的行数
    // const int THREAD_SIZE_X = 8;    // 每个线程计算矩阵C的block的列数
    // const bool ENABLE_DOUBLE_BUFFER = false;

    // dim3 block_size(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
    // dim3 grid_size(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
    
    // gemm_prefetch<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X, ENABLE_DOUBLE_BUFFER>
    // <<<grid_size, block_size>>>(d_a, d_b, d_c, M, K, N);


    /*tensor core实现*/

    dim3 block_h(WARP_SIZE);
    dim3 grid_h(div_ceil(M, WMMA_M), div_ceil(N, WMMA_N));
    hgemm_tensorcore<<<grid_h, block_h>>>(d_a, d_b, d_c, M, K, N);


    /* 结果拷回host内存 */
    CUDA_CHECK(cudaMemcpy(c, d_c, sizeC * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    LAST_KERNEL_CHECK();

    /* 释放显存 */
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_a));
}