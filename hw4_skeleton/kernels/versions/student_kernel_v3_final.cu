#include "../math_utils.h"
#include <stdio.h>
#include <cuda_runtime.h>

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;
constexpr int TM = 8;
constexpr int TN = 8;



__global__ void StudentKernel(int M, int N, int K, float alpha,
                              const float * __restrict__ A, 
                              const float * __restrict__ B, 
                              float beta, float *C) {
    __shared__ float sA[2][BK][BM]; 
    __shared__ float sB[2][BK][BN]; 

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x; 
    int ty = threadIdx.y; 
    int tid = ty * blockDim.x + tx; 

    float acc[TM][TN] = {0.0f};

    const float4* A_f4 = reinterpret_cast<const float4*>(A);
    const float4* B_f4 = reinterpret_cast<const float4*>(B);

    int a_g_row = by * BM + (tid / 2);      
    int a_g_col_f4 = (tid % 2);             

    int b_g_row = tid / 32;                 
    int b_g_col_f4 = (bx * BN / 4) + (tid % 32); 

    int write_idx = 0;
    
    if (a_g_row < M && (a_g_col_f4 * 4) < K) {
        float4 tmpA = A_f4[a_g_row * (K / 4) + a_g_col_f4];
        sA[write_idx][a_g_col_f4 * 4 + 0][tid / 2] = tmpA.x;
        sA[write_idx][a_g_col_f4 * 4 + 1][tid / 2] = tmpA.y;
        sA[write_idx][a_g_col_f4 * 4 + 2][tid / 2] = tmpA.z;
        sA[write_idx][a_g_col_f4 * 4 + 3][tid / 2] = tmpA.w;
    } else {
        sA[write_idx][a_g_col_f4 * 4 + 0][tid / 2] = 0.0f;
        sA[write_idx][a_g_col_f4 * 4 + 1][tid / 2] = 0.0f;
        sA[write_idx][a_g_col_f4 * 4 + 2][tid / 2] = 0.0f;
        sA[write_idx][a_g_col_f4 * 4 + 3][tid / 2] = 0.0f;
    }

    if (b_g_row < K && (b_g_col_f4 * 4) < N) {
        float4 tmpB = B_f4[b_g_row * (N / 4) + b_g_col_f4];
        reinterpret_cast<float4*>(sB[write_idx][b_g_row])[tid % 32] = tmpB;
    } else {
        reinterpret_cast<float4*>(sB[write_idx][b_g_row])[tid % 32] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    __syncthreads(); 

    int read_idx = 0;
    int numTiles = (K + BK - 1) / BK;

    for (int t = 0; t < numTiles; ++t) {
        int next_bk = (t + 1) * BK;
        write_idx = 1 - read_idx;

        float4 next_A_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 next_B_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        if (t + 1 < numTiles) {
            int next_a_col = next_bk + (a_g_col_f4 * 4);
            if (a_g_row < M && next_a_col < K) {
                next_A_val = A_f4[a_g_row * (K / 4) + (next_bk / 4) + a_g_col_f4];
            }
            int next_b_row = next_bk + b_g_row;
            if (next_b_row < K && (b_g_col_f4 * 4) < N) {
                next_B_val = B_f4[next_b_row * (N / 4) + b_g_col_f4];
            }
        }

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            float regA[TM];
            float regB[TN];

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                regA[i] = sA[read_idx][k][ty * TM + i];
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                regB[j] = sB[read_idx][k][tx * TN + j];
            }

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += regA[i] * regB[j];
                }
            }
        }

        if (t + 1 < numTiles) {
            sA[write_idx][a_g_col_f4 * 4 + 0][tid / 2] = next_A_val.x;
            sA[write_idx][a_g_col_f4 * 4 + 1][tid / 2] = next_A_val.y;
            sA[write_idx][a_g_col_f4 * 4 + 2][tid / 2] = next_A_val.z;
            sA[write_idx][a_g_col_f4 * 4 + 3][tid / 2] = next_A_val.w;

            reinterpret_cast<float4*>(sB[write_idx][b_g_row])[tid % 32] = next_B_val;
        }

        __syncthreads();
        read_idx = write_idx; 
    }

    // 向量化寫回 C 矩陣
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int g_row = by * BM + ty * TM + i;
        if (g_row < M) {
            int g_col_base = bx * BN + tx * TN;
            if (g_col_base < N) {
                int idx_f4_0 = g_row * (N / 4) + (g_col_base / 4);
                float4 oldC_0 = reinterpret_cast<float4*>(C)[idx_f4_0];
                float4 newC_0;
                newC_0.x = alpha * acc[i][0] + beta * oldC_0.x;
                newC_0.y = alpha * acc[i][1] + beta * oldC_0.y;
                newC_0.z = alpha * acc[i][2] + beta * oldC_0.z;
                newC_0.w = alpha * acc[i][3] + beta * oldC_0.w;
                reinterpret_cast<float4*>(C)[idx_f4_0] = newC_0;

                int idx_f4_1 = idx_f4_0 + 1;
                float4 oldC_1 = reinterpret_cast<float4*>(C)[idx_f4_1];
                float4 newC_1;
                newC_1.x = alpha * acc[i][4] + beta * oldC_1.x;
                newC_1.y = alpha * acc[i][5] + beta * oldC_1.y;
                newC_1.z = alpha * acc[i][6] + beta * oldC_1.z;
                newC_1.w = alpha * acc[i][7] + beta * oldC_1.w;
                reinterpret_cast<float4*>(C)[idx_f4_1] = newC_1;
            }
        }
    }
}

void runStudent(int M, int N, int K, float alpha,
                float *A, float *B, float beta, float *C) {
    // 16x16 = 256 threads per block
    dim3 block(BN / TN, BM / TM); 
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

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
