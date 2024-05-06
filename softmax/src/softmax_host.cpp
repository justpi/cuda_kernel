#include <iostream>
#include <math.h>


void softmax_host_base(float *a, float *o_h, int length, int stride) {
    /* softmax的cpu实现
    输入：a[width, width]
    输出：o_h[width, width]
    length: 向量a大小
    stride: 步长
    */
    float m;
    float sum;
    for (int i=0; i < length; i+=stride) {
        /* 求最大值 */
        m = 0.0;
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
            o_h[i+j] = exp(a[i+j] - m) / sum;
        }
    }

}