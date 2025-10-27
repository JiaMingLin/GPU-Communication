// file: dxd_to_bsr_values.cpp
#include <vector>
#include <cstddef>
#include <cstdint>
#include <cassert>
#include <iostream>

template <class T>
std::vector<T> denseXdense_to_blockedSparse_values(
    const T* A,              // [M, K] row-major
    std::size_t M,
    std::size_t K,
    const T* B,              // [K, N] column-major
    std::size_t N,
    const int32_t* blk_row_ptr, // length = num_block_rows + 1
    const int32_t* blk_col_ind, // length = blk_row_ptr[num_block_rows]
    std::size_t ky, std::size_t kx)
{
    // 基本檢查（此版本假設整除）
    assert(M % ky == 0 && "M must be a multiple of ky");
    assert(N % kx == 0 && "N must be a multiple of kx");

    const std::size_t num_block_rows = M / ky;
    const std::size_t num_blocks = static_cast<std::size_t>(blk_row_ptr[num_block_rows]);
    std::vector<T> blk_val;
    blk_val.reserve(num_blocks * ky * kx);

    auto A_at = [&](std::size_t m, std::size_t k) -> T {
        // A: row-major
        return A[m * K + k];
    };
    auto B_at = [&](std::size_t k, std::size_t n) -> T {
        // B: column-major  (K x N)
        return B[k + K * n];
    };

    for (std::size_t i = 0; i < num_block_rows; ++i) {
        int32_t start = blk_row_ptr[i];
        int32_t end   = blk_row_ptr[i + 1];
        if (start == end) continue; // 這個 block row 沒有任何非零 block

        // 取出 A 的這個 block-row 的 ky 列視窗: rows [ky*i, ky*(i+1))
        const std::size_t a_row0 = i * ky;

        for (int32_t p = start; p < end; ++p) {
            // 這個非零 block 的 block-column index
            const std::size_t bc = static_cast<std::size_t>(blk_col_ind[p]);

            // 取出 B 的這個 block-column 的 kx 欄視窗: cols [kx*bc, kx*(bc+1))
            const std::size_t b_col0 = bc * kx;

            // 計算 ky x kx 的小矩陣：C_block = A_sub(ky x K) * B_sub(K x kx)
            // 並以 row-major 順序 append 到 blk_val
            for (std::size_t r = 0; r < ky; ++r) {
                const std::size_t a_row = a_row0 + r;
                for (std::size_t c = 0; c < kx; ++c) {
                    const std::size_t b_col = b_col0 + c;
                    T acc = T(0);
                    for (std::size_t k = 0; k < K; ++k) {
                        acc += A_at(a_row, k) * B_at(k, b_col);
                    }
                    blk_val.push_back(acc); // row-major: 先 r 再 c
                }
            }
        }
    }
    return blk_val;
}

// ---- 簡單示範 ----
#ifdef DEMO_MAIN
int main() {
    using T = float;
    // 例：M=4, K=3, N=4, ky=2, kx=2
    const std::size_t M=4, K=3, N=4, ky=2, kx=2;

    // A: row-major [4x3]
    std::vector<T> A = {
        1,2,3,
        4,5,6,
        7,8,9,
        1,0,1
    };

    // B: column-major [3x4]
    // 先建一個 row-major，然後轉成 column-major 方便示範
    std::vector<T> B_row = {
        1,2,3,4,
        0,1,0,1,
        2,0,2,0
    };
    std::vector<T> B(N*K);
    for (std::size_t k=0;k<K;++k)
        for (std::size_t n=0;n<N;++n)
            B[k + K*n] = B_row[k*N + n]; // 轉成 column-major

    // 輸出稀疏的 block 結構（BSR 的 row_ptr/col_ind）
    // num_block_rows = M/ky = 2, num_block_cols = N/kx = 2
    // 假設非零 block 分佈：
    // block-row 0 -> col {0,1}
    // block-row 1 -> col {1}
    int32_t blk_row_ptr[] = {0, 2, 3};
    int32_t blk_col_ind[] = {0, 1, 1};

    auto blk_val = denseXdense_to_blockedSparse_values(
        A.data(), M, K,
        B.data(), N,
        blk_row_ptr, blk_col_ind,
        ky, kx);

    std::cout << "blk_val size = " << blk_val.size() << "\n";
    for (std::size_t i=0;i<blk_val.size();++i) {
        std::cout << blk_val[i] << ( (i+1)%kx? ' ' : '\n');
    }
    return 0;
}
#endif