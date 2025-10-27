#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " -> " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
}

// ======================
// GPU Kernel
// ======================
__global__ void spmm_block_kernel(
    const float* __restrict__ A,     // dense row-major [M × d]
    const float* __restrict__ B,     // dense column-major [d × N]
    const int* __restrict__ row_ptr, // CSR-like block-row pointer
    const int* __restrict__ col_ind, // block-column index
    const int* __restrict__ val_idx, // block-to-val index mapping
    float* __restrict__ val,         // output nonzero block values
    int d, int k)
{
    __shared__ int y_idx, x_idx, out_block_idx;

    // --- one thread per block does binary search ---
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        int blk = blockIdx.x;
        int lo = 0, hi = gridDim.x;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (row_ptr[mid + 1] <= blk) lo = mid + 1;
            else hi = mid;
        }
        int expert_id   = lo;
        int expert_beg  = row_ptr[expert_id];
        int offset      = blk - expert_beg;

        y_idx = offset;              // block row index within expert
        x_idx = col_ind[offset];     // block col index
        out_block_idx = val_idx[blk];
    }
    __syncthreads();

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int a_row = y_idx * k + ty;
    int b_col = x_idx * k + tx;

    float acc = 0.f;
    for (int i = 0; i < d; ++i) {
        float a_val = A[a_row * d + i];   // A: row-major
        float b_val = B[i + b_col * d];   // B: column-major
        acc += a_val * b_val;
    }

    int local_offset = ty * k + tx;
    val[out_block_idx * (k * k) + local_offset] = acc;
}

// ======================
// Host helper
// ======================
void launch_spmm_block(
    const std::vector<float>& A_h,
    const std::vector<float>& B_h,
    const std::vector<int>& row_ptr_h,
    const std::vector<int>& col_ind_h,
    int k, int d,
    std::vector<float>& val_h)
{
    int num_blocks = row_ptr_h.back();
    int num_vals = num_blocks * k * k;

    // build val_idx = [0, 1, 2, ...]
    std::vector<int> val_idx_h(num_blocks);
    for (int i = 0; i < num_blocks; ++i) val_idx_h[i] = i;

    // allocate device memory
    float *A_d, *B_d, *val_d;
    int *row_ptr_d, *col_ind_d, *val_idx_d;
    CHECK_CUDA(cudaMalloc(&A_d, A_h.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&B_d, B_h.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&row_ptr_d, row_ptr_h.size() * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&col_ind_d, col_ind_h.size() * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&val_idx_d, val_idx_h.size() * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&val_d, num_vals * sizeof(float)));

    // copy to device
    CHECK_CUDA(cudaMemcpy(A_d, A_h.data(), A_h.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(B_d, B_h.data(), B_h.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(row_ptr_d, row_ptr_h.data(), row_ptr_h.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(col_ind_d, col_ind_h.data(), col_ind_h.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(val_idx_d, val_idx_h.data(), val_idx_h.size() * sizeof(int), cudaMemcpyHostToDevice));

    // launch kernel
    dim3 blockDim(k, k);
    dim3 gridDim(num_blocks);
    spmm_block_kernel<<<gridDim, blockDim>>>(A_d, B_d, row_ptr_d, col_ind_d, val_idx_d, val_d, d, k);
    CHECK_CUDA(cudaDeviceSynchronize());

    // copy result back
    val_h.resize(num_vals);
    CHECK_CUDA(cudaMemcpy(val_h.data(), val_d, num_vals * sizeof(float), cudaMemcpyDeviceToHost));

    // free
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(row_ptr_d);
    cudaFree(col_ind_d);
    cudaFree(val_idx_d);
    cudaFree(val_d);
}

// ======================
// Example main
// ======================
int main() {
    int k = 2;     // block size
    int d = 4;     // inner dimension

    // Example dense A (row-major): 4×4
    std::vector<float> A = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9,10,11,12,
        13,14,15,16
    };

    // Example dense B (column-major): 4×4
    std::vector<float> B = {
        1, 5, 9, 13,
        2, 6,10, 14,
        3, 7,11, 15,
        4, 8,12, 16
    };

    // Suppose we have 2 nonzero blocks
    // row_ptr: [0, 1, 2] → two block-rows, each one block
    std::vector<int> row_ptr = {0, 1, 2};
    std::vector<int> col_ind = {0, 1};  // each block-row has one block

    std::vector<float> val;

    launch_spmm_block(A, B, row_ptr, col_ind, k, d, val);

    std::cout << "Non-zero block values:\n";
    for (int i = 0; i < val.size(); ++i) {
        std::cout << val[i] << " ";
    }
    std::cout << std::endl;
}