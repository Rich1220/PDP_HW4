#include "../math_utils.h"
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void StudentKernel(int M, int N, int K, float alpha,
                              const float *__restrict__ A,
                              const float *__restrict__ B,
                              float beta, float *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

void runStudent(int M, int N, int K, float alpha,
                float *A, float *B, float beta, float *C) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);

    StudentKernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "StudentKernel launch error: %s\n", cudaGetErrorString(err));
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "StudentKernel sync error: %s\n", cudaGetErrorString(err));
    }
}