#include <iostream>
#include <math.h>
#include <climits>


void softmax_host_base(float *a, float *h_o, int length, int stride) {
    /* softmax的cpu实现
    输入：a[width, width]
    输出：h_o[width, width]
    length: 向量a大小
    stride: 步长
    */
    float m;
    float sum;
    for (int i=0; i < length; i+=stride) {
        /* 求最大值 */
        m = INT_MIN;
        for(int j=0; j < stride; ++j) {
            m = m > a[i+j]? m:a[i+j];
        }

        /* 求指数和 */
        sum = 0.0;
        for(int j=0; j < stride; ++j) {
            sum += exp(a[i+j] - m);
        }
        /* 求softmax */
        for(int j=0; j < stride; ++j) {
            h_o[i+j] = exp(a[i+j] - m) / sum;
        }
    }

}

void softmax_host_online(float *a, float *h_o, int length, int stride) {
    float m;
    float sum;
    for (int i=0; i < length; i+=stride) {
        /* max & exp sum */
        m = INT_MIN;
        sum = 0.0;
        for (int j=0; j < stride; ++j) {
            m = m > a[i+j] ? m:a[i+j];
            sum = sum * exp(a[i+j] - m) + exp(a[i+j] - m);
        }

        /* softmax */
        for(int j=0; j < stride; ++j) {
            h_o[i+j] = exp(a[i+j] - m) / sum;
        }
    }
}