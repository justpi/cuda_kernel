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

safe softmax的计算公式如下：

$$
m = max(x)
$$


$$
softmax(x_i) = \frac {e^{x_i-m}} {\sum_j {e^{x_j-m}}}
$$

使用safe softmax的原因是exp函数计算结果具有不稳定性，部分值在经过exp后超出数值表示范围，会引起softmax计算不准确。根据exp函数的特性，均减去最大值，这样可以保证softmax的计算结果准确。

为了后续便于与其他算子进行融合，对safe softmax进行拆分，有online softmax函数，具体公式：

$$
\begin{align*}
m_{i-1} = m_i, m_i = max(x, m_i)
\\
sum_i = sum_{i-1} * e ^ {m_{i-1} - m_i} + e^{x_i - m_i}
\end{align*}
$$

$$
softmax(x_i) = \frac {e^{x_i-m}} {sum}
$$

online softmax在每次循环中只需访存2次，写入1次，而safe softmax在每次循环中需要访存3次，写入1次。这种写法会有速度提升。



## 5. transpose

## 6. conv

## 7. layernorm

layernorm算子在attention中非常常用，一般用来减少网络层和层之间的Covariate Shift，提高网络的收敛速度。layernorm的计算公式如下：

$$
y = \frac {x-\mu} {\sqrt{\delta^2 + \varepsilon}} \cdot \gamma + \beta
$$

layernorm和softmax很相似，都是在一行元素内计算做两次归约然后再做点积。

计算均值和方差的方法：

1. 直接计算：

$$
\mu_i = \sum_{j=0}^{stride} {a[i*stride+j]}
$$

$$
\delta_{i}^2 = \sum_{j=0}^{stride}{(a[i*stride] - \mu)^2}
$$

2. 根据下面公式直接计算均值和方差，仅需在一个循环内计算$x$的均值和$x^2$的均值即可，这个方法的缺点是当数据量过大时，容易出现数值溢出，从而影响计算结果；维基百科的介绍是：缺点是其结果依赖于数据的排序，存在累加的舍入误差，对于大数据集效果较差：

$$
\delta^2 = E(X^2) - E(x)^2
$$

3. 在线算法Welford：具体计算公式如下所示，

$$
M_{2,n} =M_{2,n-1}+(x_n-\bar{x}_{n-1})(x_n-\bar{x}_n)
$$

$$
\delta_n^2=\frac {M_{2,n}} {n}
$$



## 8. attention


