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

reduce是一类算子，一般常用的求和、求最值等操作实际执行时均采用reduce操作，也叫归约操作。

在GPU中，reduce有两种写法，第一种是顺序分块归约，第二种是递归归约。递归归约的时间复杂度较顺序分块归约的时间复杂度低。

reduce的计算公式可以写成下面的方式：

$$
reduce(x) = x_0 \otimes x_1 \otimes x_2 ...\otimes x_n
$$

我们在这里实现的是reduceSum操作，首先根据reduce的计算方式实现一个baseline，首先将数据存入共享内存中：

```c++
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

```

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

transpose算子很简单，两次访存，没有计算。

## 6. conv

### 1. naive版本

卷积的基本实现过程是：卷积核在输入的一个特征图上滑动，计算卷积核于特征图上对应区域的点积，这个卷积核在C个输入特征图上均需进行上述操作，一个卷积核那么可以生成C个特征图，然后将这C个特征图对应位置做加法操作，这就计算出一个卷积核的输出特征图，一共有K个特征图，那么就会计算出KHW的输出特征向量（不包含B）。

卷积的naive版本计算公式如下：

$$
O_{b, c_o, h_o, w_o} = \sum_{ci=0}^{input\ channel} {\sum_{h_k, w_k}^{H_k, W_k} {weight_{b,ci,h_k,w_k} \cdot I_{b,ci,h_i+h_k,w_i+w_k}}}
$$

其中$h_i$和$h_o$之间的转换关系为：

$$
h_i = h_o * stride_r + h_k - padding_r
$$

$$
w_i = w_o * stride_c + h_w - padding_c
$$

这种方式的大致cpu实现如下：
```
for(b:B) {
    for(c_o:C) {
        for (h_o:H) {
            for (w_o:W) {
                out_idx = b * CHW + c_o * HW + h_o * W + w_o;
                O[out_idx] = bias[c_o];
                for (h_k:H_k) {
                    for (w_k:W_k) {
                        input_h = h_o * stride_r + h_k - padding_r;
                        input_w = h_o * stride_r + h_k - padding_r;
                        for (c_i:C_I) {
                            O[out_idx] += weight[c,h_k,w_k] * I[b, c_i,input_h,input_w];
                        }
                    }
                }
            }
        }
    }
}
```


### 2. im2col+gemm

根据前面的naive实现，我们将输入的一个特征图进行展开，将每次滑动对应的特征值拉平成一个列向量，一共需要进行HW次拉平，所以会产生一个$H_kW_k * HK$的二维矩阵，而卷积核则拉平成一个行向量；K个卷积核组合成一个$K * H_kW_k$的矩阵；最终将$weight \otimes Input$则可以一次计算得到输入的一个特征图的输出结果。一共有C个特征图，计算C次此矩阵，最终对应位置相加得到最终结果。



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


