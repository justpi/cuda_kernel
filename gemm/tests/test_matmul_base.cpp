#include <iostream>
#include <gtest/gtest.h>

extern void matmul_cpu_base(float *a, float *b, float *c, int N, int K, int M);

bool matrixAreEqual(float *a, float *b, int rows, int cols) {
    for (int i=0; i < rows; ++i) {
        for (int j=0; j < cols; ++j) {
            if (a[i * cols + j] != b[i * cols + j]) return false;
        }
    }
    return true;
}


TEST(MatMulCpuBaseTest, MultiPlyTwoMatrices) {
    const int N = 2;
    const int K = 3;
    const int M = 2;

    /*初始化矩阵*/
    float a[N * K] = {1, 2, 3, 4, 5, 6};
    float b[K * M] = {7, 8, 9, 10, 11, 12};
    float c[N * M] = {0};
    float expected[N * M] = {58, 64, 139, 154};

    /*测试函数*/
    matmul_cpu_base(a, b, c, N, K, M);

    /*测试结果*/
    EXPECT_TRUE(matrixAreEqual(c, expected, N, M));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}