#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

#define BLOCKSIZE 32

__global__ void sgemm_naive(float* A, float* B, float* C, int M, int N, int K, 
    float alpha, float beta) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < M && y < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[x * K + k] * B[k * N + y];
        }
        C[x * N + y] = alpha * sum + beta * C[x * N + y];
    }
}

__global__ void sgemm_naive_coalescing(float* A, float* B, float* C, int M, int N, int K, 
    float alpha, float beta) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (y < M && x < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[y * K + k] * B[k * N + x];
        }
        C[y * N + x] = alpha * sum + beta * C[y * N + x];
    }
}

__global__ void sgemm_shared_memory(float* A, float* B, float* C, int M, int N, int K, 
    float alpha, float beta) {
    __shared__ float shared_A[BLOCKSIZE][BLOCKSIZE];
    __shared__ float shared_B[BLOCKSIZE][BLOCKSIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (K + BLOCKSIZE - 1) / BLOCKSIZE; tile++) {
        // Load A tile into shared memory
        int a_row = row;
        int a_col = tile * BLOCKSIZE + threadIdx.x;
        if (a_row < M && a_col < K) {
            shared_A[threadIdx.y][threadIdx.x] = A[a_row * K + a_col];
        } else {
            shared_A[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load B tile into shared memory
        int b_row = tile * BLOCKSIZE + threadIdx.y;
        int b_col = col;
        if (b_row < K && b_col < N) {
            shared_B[threadIdx.y][threadIdx.x] = B[b_row * N + b_col];
        } else {
            shared_B[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        if (row < M && col < N) {
            for (int k = 0; k < BLOCKSIZE; k++) {
                sum += shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];
            }
        }
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

void sgemm_launcher_naive(float* A, float* B, float* C, int M, int N, int K, 
    float alpha, float beta) {
    dim3 blockDim(32, 32);
    // gridDim.x corresponds to M (rows), gridDim.y corresponds to N (columns)
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
    sgemm_naive<<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in sgemm_naive: %s\n", cudaGetErrorString(err));
    }
}

void sgemm_launcher_coalescing(float* A, float* B, float* C, int M, int N, int K, 
    float alpha, float beta) {
    dim3 blockDim(32 , 32);
    // gridDim.x corresponds to M (rows), gridDim.y corresponds to N (columns)
    dim3 gridDim((M + 32 - 1) / 32, (N + 32 - 1) / 32);
    sgemm_naive_coalescing<<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in sgemm_naive_coalescing: %s\n", cudaGetErrorString(err));
    }
}

void sgemm_launcher_shared_memory(float* A, float* B, float* C, int M, int N, int K, 
    float alpha, float beta) {
    dim3 blockDim(32, 32);
    // gridDim.x corresponds to N (columns), gridDim.y corresponds to M (rows)
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    sgemm_shared_memory<<<gridDim, blockDim>>>(A, B, C, M, N, K, alpha, beta);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in sgemm_shared_memory: %s\n", cudaGetErrorString(err));
    }
}

void sgemm_cpu(float* A, float* B, float* C, int M, int N, int K, 
    float alpha, float beta) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}

void print_usage(const char* prog_name) {
    printf("Usage: %s [options] [kernel_type] [M] [N] [K]\n", prog_name);
    printf("Options:\n");
    printf("  -v, --validate    Enable CPU validation (default: disabled)\n");
    printf("  -h, --help       Show this help message\n");
    printf("Arguments:\n");
    printf("  kernel_type: 0=naive, 1=coalescing, 2=shared_memory (default: 0)\n");
    printf("  M, N, K: matrix dimensions (default: 4096)\n");
    printf("\nExamples:\n");
    printf("  %s 0 1024 1024 1024\n", prog_name);
    printf("  %s -v 0 1024 1024 1024\n", prog_name);
    printf("  %s --validate 1 2048 2048 2048\n", prog_name);
    printf("  %s 2 1024 1024 1024\n", prog_name);
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    bool enable_validation = false;
    int kernel_type = 0;  // 0=naive, 1=coalescing, 2=shared_memory
    int M = 4096;
    int N = 4096;
    int K = 4096;
    
    int arg_idx = 1;
    
    // Parse options
    while (arg_idx < argc) {
        if (strcmp(argv[arg_idx], "-h") == 0 || strcmp(argv[arg_idx], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[arg_idx], "-v") == 0 || strcmp(argv[arg_idx], "--validate") == 0) {
            enable_validation = true;
            arg_idx++;
        } else {
            break;
        }
    }
    
    // Parse positional arguments
    if (arg_idx < argc) {
        kernel_type = atoi(argv[arg_idx]);
        if (kernel_type < 0 || kernel_type > 2) {
            printf("Error: kernel_type must be 0 (naive), 1 (coalescing), or 2 (shared_memory)\n");
            print_usage(argv[0]);
            return 1;
        }
        arg_idx++;
    }
    if (arg_idx < argc) M = atoi(argv[arg_idx++]);
    if (arg_idx < argc) N = atoi(argv[arg_idx++]);
    if (arg_idx < argc) K = atoi(argv[arg_idx++]);
    
    // Select kernel launcher based on kernel_type
    void (*sgemm_launcher)(float*, float*, float*, int, int, int, float, float);
    const char* kernel_name;
    
    if (kernel_type == 0) {
        sgemm_launcher = sgemm_launcher_naive;
        kernel_name = "naive";
    } else if (kernel_type == 1) {
        sgemm_launcher = sgemm_launcher_coalescing;
        kernel_name = "coalescing";
    } else {
        sgemm_launcher = sgemm_launcher_shared_memory;
        kernel_name = "shared_memory";
    }
    float* A = new float[M * K];
    float* B = new float[K * N];
    float* C = new float[M * N];
    float* C_cpu = enable_validation ? new float[M * N] : nullptr;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            A[i * K + j] = rand() % 100;
        }
    }
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            B[i * N + j] = rand() % 100;
        }
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = 0.0f;
        }
    }
    // GPU mamory allocation
    float* A_d = nullptr;
    float* B_d = nullptr;
    float* C_d = nullptr;
    cudaMalloc((void**)&A_d, M * K * sizeof(float));
    cudaMalloc((void**)&B_d, K * N * sizeof(float));
    cudaMalloc((void**)&C_d, M * N * sizeof(float));
    cudaMemcpy(A_d, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C, M * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Check for CUDA errors after memory operations
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after memory copy: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // Warmup
    for (int i = 0; i < 3; i++) {
        sgemm_launcher(A_d, B_d, C_d, M, N, K, 1.0f, 0.0f);
    }
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after warmup: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // Measure kernel execution time using CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const int num_iterations = 10;
    float total_time_ms = 0.0f;
    
    for (int i = 0; i < num_iterations; i++) {
        cudaEventRecord(start);
        sgemm_launcher(A_d, B_d, C_d, M, N, K, 1.0f, 0.0f);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        total_time_ms += elapsed_ms;
    }
    
    float avg_time_ms = total_time_ms / num_iterations;
    float latency_ms = avg_time_ms;
    
    // Calculate throughput (GFLOPS)
    // For matrix multiplication C = A * B (M×K * K×N = M×N)
    // Each output element requires K multiplications and K additions = 2*K FLOPS
    // Total FLOPS = 2 * M * N * K
    long long total_flops = 2LL * M * N * K;
    float throughput_gflops = (total_flops / 1e9) / (avg_time_ms / 1000.0f);
    
    // Calculate effective memory bandwidth
    // Memory access pattern for C = A * B:
    // - Read A matrix: M * K elements
    // - Read B matrix: K * N elements
    // - Read C matrix: M * N elements (if beta != 0, but here beta = 0, so no read)
    // - Write C matrix: M * N elements
    // Total bytes transferred = (M*K + K*N + M*N) * sizeof(float)
    long long total_bytes = (long long)(M * K + K * N + M * N) * sizeof(float);
    float effective_bandwidth_gbps = (total_bytes / 1e9) / (avg_time_ms / 1000.0f);
    
    printf("\n=== GPU Kernel Performance (sgemm) ===\n");
    printf("Kernel type: %s\n", kernel_name);
    printf("Matrix size: %d x %d x %d\n", M, N, K);
    printf("Average latency: %.4f ms\n", latency_ms);
    printf("Throughput: %.2f GFLOPS\n", throughput_gflops);
    printf("Effective memory bandwidth: %.2f GB/s\n", effective_bandwidth_gbps);
    printf("========================================\n\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cudaMemcpy(C, C_d, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // CPU validation if enabled
    if (enable_validation) {
        printf("Running CPU validation...\n");
        sgemm_cpu(A, B, C_cpu, M, N, K, 1.0f, 0.0f);
        
        int total_elements = M * N;
        int error_count = 0;
        int shown_errors = 0;
        const float tolerance = 1e-5f;
        const int max_show_errors = 10;
        
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float diff = fabs(C[i * N + j] - C_cpu[i * N + j]);
                if (diff > tolerance) {
                    error_count++;
                    if (shown_errors < max_show_errors) {
                        if (shown_errors == 0) {
                            printf("\n=== Validation Results ===\n");
                            printf("Total elements: %d\n", total_elements);
                        }
                        printf("Error at (%d, %d): GPU=%.6f, CPU=%.6f, diff=%.6f\n", 
                               i, j, C[i * N + j], C_cpu[i * N + j], diff);
                        shown_errors++;
                    }
                }
            }
        }
        
        if (error_count == 0) {
            printf("\n=== Validation Results ===\n");
            printf("Total elements: %d\n", total_elements);
            printf("Validation PASSED: GPU and CPU results match (tolerance: %.1e)\n", tolerance);
        } else {
            if (shown_errors == 0) {
                printf("\n=== Validation Results ===\n");
                printf("Total elements: %d\n", total_elements);
            }
            printf("Validation FAILED: Found %d errors out of %d elements", error_count, total_elements);
            if (error_count > max_show_errors) {
                printf(" (showing first %d)", max_show_errors);
            }
            printf("\n");
        }
        printf("===========================\n\n");
        
        delete[] C_cpu;
    }
    
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}