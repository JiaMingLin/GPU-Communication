// spmm_bsr_example.cu
// Compile: nvcc -O3 spmm_bsr_example.cu -lcusparse -o spmm_bsr_example
// Run:     ./spmm_bsr_example

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <cinttypes>

#define CUDA_CHECK(expr)                                                     \
  do {                                                                       \
    cudaError_t _err = (expr);                                               \
    if (_err != cudaSuccess) {                                               \
      fprintf(stderr, "[CUDA] %s:%d: %s\n", __FILE__, __LINE__,              \
              cudaGetErrorString(_err));                                     \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

#define CUSPARSE_CHECK(expr)                                                 \
  do {                                                                       \
    cusparseStatus_t _st = (expr);                                           \
    if (_st != CUSPARSE_STATUS_SUCCESS) {                                    \
      fprintf(stderr, "[cuSPARSE] %s:%d: status=%d\n", __FILE__, __LINE__,   \
              (int)_st);                                                     \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

// A: (M x K) in BSR (blockDim x blockDim blocks), values are ROW-major inside each block
// B: (K x N) dense ROW-major
// C: (M x N) dense ROW-major
void spmm_bsr_rowmajor_float(
    int M, int K, int N,
    int blockDim,
    int nnzb,                         // number of nonzero blocks (== length of bsrColInd)
    const int*   d_bsrRowPtr,         // length = Mb+1
    const int*   d_bsrColInd,         // length = nnzb
    const float* d_bsrVal,            // length = nnzb * blockDim * blockDim
    const float* d_B,                 // length = K*N (row-major)
    float*       d_C                  // length = M*N (row-major)
) {
  if ((M % blockDim) || (K % blockDim)) {
    fprintf(stderr, "[Error] M (%d) and K (%d) must be divisible by blockDim (%d).\n",
            M, K, blockDim);
    std::exit(EXIT_FAILURE);
  }

  const int Mb = M / blockDim;
  const int Nb = K / blockDim;

  cusparseHandle_t handle;
  CUSPARSE_CHECK(cusparseCreate(&handle));

  // ---- Create A as BSR (Generic API, CUDA 12+/cuSPARSE 13+) ----
  // New signature requires brows, bcols, bnnz, rowBlockSize, colBlockSize, ... , order
  cusparseSpMatDescr_t matA;
  {
    const int64_t brows = (int64_t)Mb;
    const int64_t bcols = (int64_t)Nb;
    const int64_t bnnz  = (int64_t)nnzb;

    CUSPARSE_CHECK(cusparseCreateBsr(
        &matA,
        /*brows*/ brows,
        /*bcols*/ bcols,
        /*bnnz */ bnnz,
        /*rowBlockSize*/ (int64_t)blockDim,
        /*colBlockSize*/ (int64_t)blockDim,
        /*bsrRowOffsets*/ (void*)d_bsrRowPtr,     // device pointers
        /*bsrColInd    */ (void*)d_bsrColInd,
        /*bsrValues    */ (void*)d_bsrVal,
        /*bsrRowOffsetsType*/ CUSPARSE_INDEX_32I,
        /*bsrColIndType    */ CUSPARSE_INDEX_32I,
        /*idxBase          */ CUSPARSE_INDEX_BASE_ZERO,
        /*valueType        */ CUDA_R_32F,
        /*order (inside block)*/ CUSPARSE_ORDER_ROW)); // your val is row-major inside each block
  }

  // ---- Create B, C as DnMat (row-major) ----
  cusparseDnMatDescr_t matB, matC;
  {
    // For row-major matrices in cuSPARSE DnMat, leading dimension = number of columns
    const int ldb = N;
    const int ldc = N;
    CUSPARSE_CHECK(cusparseCreateDnMat(&matB, K, N, ldb, (void*)d_B, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    CUSPARSE_CHECK(cusparseCreateDnMat(&matC, M, N, ldc, (void*)d_C, CUDA_R_32F, CUSPARSE_ORDER_ROW));
  }

  float alpha = 1.0f, beta = 0.0f;

  // ---- Query buffer size ----
  size_t bufferSize = 0;
  CUSPARSE_CHECK(cusparseSpMM_bufferSize(
      handle,
      CUSPARSE_OPERATION_NON_TRANSPOSE,  // opA
      CUSPARSE_OPERATION_NON_TRANSPOSE,  // opB
      &alpha, matA, matB, &beta, matC,
      CUDA_R_32F,
      CUSPARSE_SPMM_ALG_DEFAULT,
      &bufferSize));

  void* dBuffer = nullptr;
  if (bufferSize > 0) CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));

  // ---- Compute ----
  CUSPARSE_CHECK(cusparseSpMM(
      handle,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      &alpha, matA, matB, &beta, matC,
      CUDA_R_32F,
      CUSPARSE_SPMM_ALG_DEFAULT,
      dBuffer));

  CUDA_CHECK(cudaDeviceSynchronize());

  // ---- Destroy descriptors & buffer ----
  if (dBuffer) CUDA_CHECK(cudaFree(dBuffer));
  CUSPARSE_CHECK(cusparseDestroyDnMat(matB));
  CUSPARSE_CHECK(cusparseDestroyDnMat(matC));
  CUSPARSE_CHECK(cusparseDestroySpMat(matA));
  CUSPARSE_CHECK(cusparseDestroy(handle));
}

int main() {
  // Example: M=4, K=4, N=3, blockDim=2
  // A in BSR has two 2x2 blocks:
  //   block(0,0) = [1 2; 3 4]
  //   block(1,1) = [5 6; 7 8]
  // So A =
  // [1 2 0 0
  //  3 4 0 0
  //  0 0 5 6
  //  0 0 7 8]

  const int M = 4, K = 4, N = 3, blockDim = 2;
  const int Mb = M / blockDim;
  const int Nb = K / blockDim;

  std::vector<int>   h_bsrRowPtr = {0, 1, 2};  // length Mb+1=3
  std::vector<int>   h_bsrColInd = {0, 1};     // nnzb=2
  std::vector<float> h_bsrVal = {
      // block(0,0), row-major inside block
      1, 2,
      3, 4,
      // block(1,1), row-major inside block
      5, 6,
      7, 8
  };
  const int nnzb = static_cast<int>(h_bsrColInd.size());

  // B: KxN = 4x3, row-major
  // [ 1  2  3
  //   4  5  6
  //   7  8  9
  //  10 11 12 ]
  std::vector<float> h_B = {
      1,2,3,
      4,5,6,
      7,8,9,
      10,11,12
  };

  std::vector<float> h_C(M*N, 0.0f);

  // ---- Device buffers ----
  int   *d_bsrRowPtr=nullptr, *d_bsrColInd=nullptr;
  float *d_bsrVal=nullptr, *d_B=nullptr, *d_C=nullptr;

  CUDA_CHECK(cudaMalloc((void**)&d_bsrRowPtr, (Mb+1)*sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)&d_bsrColInd, nnzb*sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)&d_bsrVal,    nnzb*blockDim*blockDim*sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&d_B,         K*N*sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&d_C,         M*N*sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_bsrRowPtr, h_bsrRowPtr.data(), (Mb+1)*sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_bsrColInd, h_bsrColInd.data(), nnzb*sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_bsrVal,    h_bsrVal.data(),    nnzb*blockDim*blockDim*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B,         h_B.data(),         K*N*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_C, 0, M*N*sizeof(float)));

  // ---- Run SpMM ----
  spmm_bsr_rowmajor_float(M, K, N, blockDim, nnzb,
                          d_bsrRowPtr, d_bsrColInd, d_bsrVal,
                          d_B, d_C);

  // ---- Copy back & print ----
  CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost));

  printf("C = A * B (row-major, %dx%d):\n", M, N);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      printf("%6.1f ", h_C[i*N + j]);
    }
    printf("\n");
  }

  // (Optional) quick expected check:
  // Top 2x3 = [1 2;3 4] * [1 2 3;4 5 6] = [[9 12 15],[19 26 33]]
  // Bottom 2x3 = [5 6;7 8] * [7 8 9;10 11 12] = [[95 106 117],[129 144 159]]

  // ---- Free ----
  CUDA_CHECK(cudaFree(d_bsrRowPtr));
  CUDA_CHECK(cudaFree(d_bsrColInd));
  CUDA_CHECK(cudaFree(d_bsrVal));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  return 0;
}