#include "layernorm_host.h"

void layernorm_host_base(float* h_a, float *h_o, float gamma, float beta, int length, int stride) {
    /*基础版本的layernorm*/
    
    for (int i=0; i < length; i+=stride) {
        /* 1. 计算一个stirde的均值 */
        float mu = 0.0;
        for (int j=0; j < stride; ++j) {
            mu += h_a[i+j];
        }
        mu /= stride;
        /* 2. 计算一个stride内的方差 */
        float delta = 0.0;
        for (int j=0; j < stride; ++j) {
            delta += (h_a[i+j] - mu) * (h_a[i+j] - mu);
        }
        delta /= stride;
        /* 3. 计算点积 */
        float output;
        for (int j=0; j < stride; ++j) {
            output = (h_a[i+j] - mu) / (delta + EPS) * gamma + beta;
            h_o[i+j] = output;
        }
    }
    
}

void layernorm_host_naive(float *h_a, float *h_o, float gamma, float beta, int length, int stride) {
    /* 将方差和均值一起计算 */
    for (int i=0; i < length;i+=stride) {
        /*计算E(X^2),E(x)*/
        float mu = 0.0;
        float mu_2 = 0.0;
        for (int j=0; j < stride; ++j) {
            mu += h_a[i+j];
            mu_2 += h_a[i+j] * h_a[i+j];
        }
        mu /= stride;
        mu_2 /= stride;

        float delta = mu_2 - mu * mu;
        /* layernorm计算 */
        for(int j=0; j < stride; ++j) {
            float output = (h_a[i+j] - mu) / (delta + EPS) * gamma + beta;
            h_o[i+j] = output;
        }
    }

    
}