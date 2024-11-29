#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

using namespace std;

#define cudaCheckError() { \
    cudaError_t e = cudaGetLastError(); \
    if (e != cudaSuccess) { \
        cerr << "CUDA Error: " << cudaGetErrorString(e) << endl; \
        exit(1); \
    } \
}

void matrixMultiplyAndSumCPU(const vector<float>& A, const vector<float>& B, vector<float>& result, int N, int K, int M) {
    vector<float> C(N * M, 0.0f);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            for (int k = 0; k < K; ++k) {
                C[i * M + j] += A[i * K + k] * B[k * M + j];
            }
        }
        float sumSquares = 0.0f;
        for (int j = 0; j < M; ++j) {
            sumSquares += C[i * M + j] * C[i * M + j];
        }
        result[i] = sumSquares;
    }
}

__global__ void matrixMultiplyKernel(const float* A, const float* B, float* C, int N, int K, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < M) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * M + col];
        }
        C[row * M + col] = sum;
    }
}

__global__ void sumOfSquaresKernel(const float* C, float* result, int N, int M) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N) {
        float sumSquares = 0.0f;
        for (int j = 0; j < M; ++j) {
            float val = C[row * M + j];
            sumSquares += val * val;
        }
        result[row] = sumSquares;
    }
}

int main() {
    int N = 1024, K = 1024, M = 1024;

    vector<float> A(N * K), B(K * M);
    for (int i = 0; i < N * K; ++i) A[i] = rand() % 10;
    for (int i = 0; i < K * M; ++i) B[i] = rand() % 10;

    vector<float> resultCPU(N), resultGPU(N);

    auto startCPU = chrono::high_resolution_clock::now();
    matrixMultiplyAndSumCPU(A, B, resultCPU, N, K, M);
    auto endCPU = chrono::high_resolution_clock::now();
    cout << "CPU Execution Time: " << chrono::duration_cast<chrono::milliseconds>(endCPU - startCPU).count() << " ms\n";

    float sumCPU = 0.0f;
    for (float value : resultCPU) sumCPU += value;
    cout << "CPU Result Sum: " << sumCPU << endl;

    float* d_A, * d_B, * d_C, * d_result;
    cudaMalloc(&d_A, N * K * sizeof(float));
    cudaMalloc(&d_B, K * M * sizeof(float));
    cudaMalloc(&d_C, N * M * sizeof(float));
    cudaMalloc(&d_result, N * sizeof(float));
    cudaMemcpy(d_A, A.data(), N * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), K * M * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((M + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    auto startGPU = chrono::high_resolution_clock::now();
    matrixMultiplyKernel << <gridSize, blockSize >> > (d_A, d_B, d_C, N, K, M);
    cudaCheckError();

    dim3 blockSum(256);
    dim3 gridSum((N + blockSum.x - 1) / blockSum.x);
    sumOfSquaresKernel << <gridSum, blockSum >> > (d_C, d_result, N, M);
    cudaDeviceSynchronize();
    auto endGPU = chrono::high_resolution_clock::now();

    cout << "GPU Execution Time: " << chrono::duration_cast<chrono::milliseconds>(endGPU - startGPU).count() << " ms\n";

    cudaMemcpy(resultGPU.data(), d_result, N * sizeof(float), cudaMemcpyDeviceToHost);

    float sumGPU = 0.0f;
    for (float value : resultGPU) sumGPU += value;
    cout << "GPU Result Sum: " << sumGPU << endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_result);

    return 0;
}
