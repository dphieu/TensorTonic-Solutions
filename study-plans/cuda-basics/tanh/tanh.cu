#include <cuda_runtime.h>
#include <math.h>

__global__ void tanh_kernel(const float* input, float* output, int N) {
    // Write code here
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // eplus = expf(input[idx]);
        // esub = expf(-input[idx]);
        // output[idx] = (eplus - esub) / (eplus + esub);
        output[idx] = tanhf(input[idx]);
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    tanh_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}