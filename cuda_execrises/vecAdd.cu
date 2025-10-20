#include <cuda_runtime.h>
#include <cstdio>

#define CUDA_CHECK(expr) do {                                   \
    cudaError_t _err = (expr);                                   \
    if (_err != cudaSuccess) {                                   \
        fprintf(stderr, "CUDA error %s at %s:%d\n",              \
                cudaGetErrorString(_err), __FILE__, __LINE__);   \
        return;                                                  \
    }                                                            \
} while (0)

__global__
void vecAddKernel(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

void vecAdd(const float* A, const float* B, float* C, int n) {
    if (n <= 0) return;

    float *A_d = nullptr, *B_d = nullptr, *C_d = nullptr;
    size_t size = static_cast<size_t>(n) * sizeof(float);

    CUDA_CHECK(cudaMalloc((void**)&A_d, size));
    CUDA_CHECK(cudaMalloc((void**)&B_d, size));
    CUDA_CHECK(cudaMalloc((void**)&C_d, size));

    CUDA_CHECK(cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;

    vecAddKernel<<<blocks, threads>>>(A_d, B_d, C_d, n);
    CUDA_CHECK(cudaGetLastError());       // 檢查 kernel launch
    CUDA_CHECK(cudaDeviceSynchronize());  // 等待 kernel 完成

    CUDA_CHECK(cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost));

    cudaFree(C_d);
    cudaFree(B_d);
    cudaFree(A_d);
}

int main() {
    const int n = 1000;
    float *A = new float[n];
    float *B = new float[n];
    float *C = new float[n];

    // 初始化向量 A 和 B
    for (int i = 0; i < n; i++) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(i * 2);
    }

    // 執行向量加法
    vecAdd(A, B, C, n);

    // 驗證結果
    bool correct = true;
    for (int i = 0; i < n; i++) {
        float expected = A[i] + B[i];
        if (fabs(C[i] - expected) > 1e-5) {
            printf("錯誤：C[%d] = %f, 期望值 = %f\n", i, C[i], expected);
            correct = false;
            break;
        }
    }

    if (correct) {
        printf("向量加法測試通過！\n");
        printf("前 10 個結果：\n");
        for (int i = 0; i < 10; i++) {
            printf("C[%d] = %f (A[%d] + B[%d] = %f + %f)\n", 
                   i, C[i], i, i, A[i], B[i]);
        }
    }

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}