# CUDA基础知识

## 1. 编程模型


## 2. 运行模型


## 3. 优化方式


# 算子开发

## 1. 矩阵乘法-sgemm

使用下列指令编译matmul的算子
```shell
cd matmul
mkdir build & cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
# cmake -DCMAKE_BUILD_TYPE=Release ..
```
### baseline
基础实现参照Host端的基础矩阵乘法代码，线程组织形式为一个thread计算输出的一个值，block组织成二维方式，核心代码如下：
```c++
__global__ void gemm_baseline(float* d_a, float* d_b, float* d_c, int N, int K, int M) {
    /* baseline:
        d_a: M * K
        d_b: K * N
        d_c: M * N
    */
    /* 计算d_c索引 (i, j) */
    unsigned int c_i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int c_j = blockDim.y * blockIdx.y + threadIdx.y;
    /* 执行计算 */
    float output = 0.0;
    for (int k=0; k < K; ++k) {
        output += d_a[c_i * K + k] * d_b[k * N + c_j];
    }
    /* 结果写回 */
    d_c[c_i * N + c_j] = output;

}
```

*分析*：在baseline的最小循环内，一共有两次从global mem的访存，一次FFMA操作，该核函数的计算密度0.5。
核函数计算效率很低，核函数一直在等访存结果，这样整体的SM占用率极低，通过ncu分析可以发现，核函数带宽接近94%，而计算利用率约7%，所以下一步的优化策略是提升一个线程的FFMA次数。
*下一步策略*：根据从global mem的带宽（204.8GB/s）与CUDA Core计算频率（1.3 GHz）对比，使用tile策略，每个thread计算约64-128个FFMA。
在此之前，需要先将数据存入共享内存防止频繁从global mem中做访存。

### tile_share




### tileMN


## 2. reduce

## 3. element-wise

## 4. softmax

softmax是一个基本的激活函数，它可以将一个数值向量归一化为一个概率分布向量。

softmax的计算公式如下：
$$
m = max(x)
$$


$$
softmax(x_i) = \frac {e^{x_i-m}} {\sum_j {e^{x_j-m}}}
$$

## 5. transpose

## 6. conv

## 7. layernorm

## 8. attention


