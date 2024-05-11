
#define EPS 1e-5

void layernorm_host_base(float* h_a, float *h_o, float gamma, float beta, int length, int stride);

void layernorm_host_naive(float *h_a, float *h_o, float gamma, float beta, int length, int stride);