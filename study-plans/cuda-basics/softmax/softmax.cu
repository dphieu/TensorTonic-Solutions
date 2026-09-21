#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {
    // Write code here
    if (blockIdx.x != 0) return;

    __shared__ float shared[256];

    int tid = threadIdx.x;

    float local_max = -INFINITY;

    for (int i = tid; i < N; i += blockDim.x) {
        local_max = fmaxf(local_max, input[i]);
    }

    shared[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }

    float max_val = shared[0];
    __syncthreads();

    float local_sum = 0.0f;

    for (int i = tid; i < N; i += blockDim.x) {
        local_sum += __expf(input[i] - max_val);
    }

    shared[tid] = local_sum;
    __syncthreads();

    // Block reduction: sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    float sum_exp = shared[0];
    __syncthreads();

    for (int i = tid; i < N; i += blockDim.x) {
        output[i] = __expf(input[i] - max_val) / sum_exp;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    softmax_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}