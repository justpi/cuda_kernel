

void conv_host_naive(float* input, float* output, float* weight, 
                        int batch, int in_channel, int out_channel, int height, int width, 
                        int kh, int kw, int pad_h, int pad_w, int stride_h, int stride_w);