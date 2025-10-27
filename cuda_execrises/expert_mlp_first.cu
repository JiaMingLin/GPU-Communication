#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <unordered_set>
#include <chrono>
#include <cassert>
#include <iomanip>
#include <cmath>
using namespace std;

// --------------------- CUDA Errcheck ---------------------
#define CHECK_CUDA(call) do {                               \
  cudaError_t err = (call);                                  \
  if (err != cudaSuccess) {                                  \
    cerr << "CUDA error " << cudaGetErrorString(err)         \
         << " at " << __FILE__ << ":" << __LINE__ << endl;   \
    exit(1);                                                 \
  }                                                          \
} while(0)

// --------------------- 計時工具 --------------------------
struct TimeMs { double ms = 0.0; };
template <class F>
auto with_timer(TimeMs& t, F&& fn) {
  auto beg = chrono::steady_clock::now();
  auto ret = fn();
  auto end = chrono::steady_clock::now();
  t.ms = chrono::duration<double, milli>(end - beg).count();
  return ret;
}

// ========================================================
// 1) 產生高度不均勻 Routing（保證指定數量的 experts 沒有 tokens；token 全數分配給其餘人）
// ========================================================
static vector<vector<int>>
make_imbalanced_routing(int num_experts, int num_tokens,
                        int zero_expert_cnt, unsigned seed,
                        TimeMs& t_route)
{
  return with_timer(t_route, [&](){
    if (num_experts <= 1) throw runtime_error("num_experts must be >= 2");
    zero_expert_cnt = max(0, min(zero_expert_cnt, num_experts - 1));

    mt19937 gen(seed);

    // 隨機選出沒有 token 的 experts：Z
    vector<int> ids(num_experts); iota(ids.begin(), ids.end(), 0);
    shuffle(ids.begin(), ids.end(), gen);
    unordered_set<int> Z(ids.begin(), ids.begin() + zero_expert_cnt);

    // 非 Z 的權重為 power-law（越小 id 權重越大），Z 的權重=0
    vector<double> w(num_experts, 0.0);
    for (int i = 0; i < num_experts; ++i)
      if (!Z.count(i)) w[i] = 1.0 / (i + 1.0);

    double sumw = accumulate(w.begin(), w.end(), 0.0);
    if (sumw <= 0.0) throw runtime_error("No available experts to assign tokens.");

    discrete_distribution<int> dist(w.begin(), w.end());

    vector<vector<int>> T(num_experts);
    for (int t = 0; t < num_tokens; ++t) {
      int e = dist(gen); // 只會落在非 Z
      T[e].push_back(t);
    }

    // 確認：Z 內為空、token 總數不變
    size_t total = 0;
    for (int i = 0; i < num_experts; ++i) total += T[i].size();
    if ((int)total != num_tokens) throw runtime_error("Token conservation violated.");
    for (int e : Z) assert(T[e].empty());

    return T;
  });
}

// ========================================================
// 2) Routing -> Block-CSR（依投影片）：blk_col_ind / blk_row_ptr
//    - 允許 |G_i|=0 → row 仍要在 row_ptr 中出現（空列）
// ========================================================
static void compute_block_csr_from_routing(
    const vector<vector<int>>& T,
    const vector<int>& F,
    int ky, int kx,
    vector<int>& blk_col_ind,
    vector<int>& blk_row_ptr,
    TimeMs& t_csr)
{
  with_timer(t_csr, [&](){
    int num_experts = (int)T.size();
    if ((int)F.size() != num_experts) throw runtime_error("F.size() must equal T.size()");

    blk_col_ind.clear();
    blk_row_ptr.clear();
    blk_row_ptr.push_back(0); // CSR 起點

    int base_col = 0; // b = sum_{j<i} |G_j|

    for (int i = 0; i < num_experts; ++i) {
      int tokens_i = (int)T[i].size();
      int Fi       = F[i];
      int Gi = (tokens_i == 0) ? 0 : ((tokens_i + kx - 1) / kx); // # column blocks for expert i
      int Ri = (Fi       == 0) ? 0 : ((Fi       + ky - 1) / ky); // # row blocks for expert i

      for (int r = 0; r < Ri; ++r) {
        int start = blk_row_ptr.back();
        if (Gi > 0) {
          for (int n = 0; n < Gi; ++n) blk_col_ind.push_back(base_col + n);
        }
        blk_row_ptr.push_back(start + Gi); // Gi==0 也推進（空列）
      }
      base_col += Gi;
    }
    return 0;
  });
}

// ========================================================
// 3) 依 CSR 建立 padding 後的 dense A(row-major) 與 B(col-major)
//    - A：每個 expert 補到 ceil(F_i/ky)*ky 列，補零
//    - B：每個 expert 的最後一個 token-group 若寬度 < kx，補零欄
// ========================================================
struct DenseAB {
  vector<float> A;  // row-major, shape [M_pad x d]
  vector<float> B;  // column-major, shape [d x N_pad]
  int M_pad = 0;    // 總列數（含 padding）
  int N_pad = 0;    // 總欄數（含 padding）
};

static DenseAB make_padded_AB(const vector<vector<int>>& T,
                              const vector<int>& F,
                              int ky, int kx, int d,
                              unsigned seed)
{
  int num_experts = (int)T.size();
  // A 的總列數（補到 ky 的倍數）
  int M_pad = 0;
  for (int i = 0; i < num_experts; ++i) {
    int Ri = (F[i] == 0) ? 0 : ((F[i] + ky - 1) / ky);
    M_pad += Ri * ky;
  }

  // B 的總欄數（每個 expert 的 Gi*kx）
  int total_col_blocks = 0;
  for (int i = 0; i < num_experts; ++i) {
    int tokens_i = (int)T[i].size();
    int Gi = (tokens_i == 0) ? 0 : ((tokens_i + kx - 1) / kx);
    total_col_blocks += Gi;
  }
  int N_pad = total_col_blocks * kx;

  DenseAB out; out.M_pad = M_pad; out.N_pad = N_pad;
  out.A.assign((size_t)M_pad * d, 0.0f);
  out.B.assign((size_t)N_pad * d, 0.0f); // column-major：每欄 d 個元素連續

  mt19937 gen(seed);
  uniform_real_distribution<float> ur(-1.f, 1.f);

  // ---- 填 A（每個 expert 的實際 Fi 列為隨機，padding 列為 0）----
  int a_row_cursor = 0;
  for (int i = 0; i < num_experts; ++i) {
    int Fi = F[i];
    int Ri = (Fi == 0) ? 0 : ((Fi + ky - 1) / ky);
    int rows_pad = Ri * ky;

    // 寫入 Fi 列隨機
    for (int r = 0; r < Fi; ++r)
      for (int c = 0; c < d; ++c)
        out.A[(a_row_cursor + r) * d + c] = ur(gen);

    // 後面 (rows_pad - Fi) 列保持 0
    a_row_cursor += rows_pad;
  }

  // ---- 填 B（每個 expert 的每個 group kx 欄；若最後一組不足 kx，超出的欄全 0）----
  int b_col_block_base = 0;
  for (int i = 0; i < num_experts; ++i) {
    int tokens_i = (int)T[i].size();
    int Gi = (tokens_i == 0) ? 0 : ((tokens_i + kx - 1) / kx);
    for (int g = 0; g < Gi; ++g) {
      int valid_w = min(kx, max(0, tokens_i - g * kx)); // 該 group 真正的欄數（<=kx）
      int col_block_idx = b_col_block_base + g;
      for (int tx = 0; tx < kx; ++tx) {
        int col = col_block_idx * kx + tx;
        if (tx < valid_w) {
          // 填隨機
          for (int r = 0; r < d; ++r)
            out.B[r + col * d] = ur(gen); // column-major
        } else {
          // 補零（保留 0）
        }
      }
    }
    b_col_block_base += Gi;
  }

  return out;
}

// ========================================================
// 4) CUDA Kernel：每個 block 計算一個 (ky x kx) tile，輸出到 val[]
//    - B 是 column-major；A 是 row-major；內積長度為 d（B 的列數）
//    - blockIdx.x 直接代表「第幾個非零 block」
//    - 由一個 thread 做 binary search：在 blk_row_ptr 中找出 y，使
//         row_ptr[y] <= block < row_ptr[y+1]
//      offset = block - row_ptr[y]
//      x = blk_col_ind[row_ptr[y] + offset]
// ========================================================
// 未完成版本的 kernel（保留以供參考，已註解避免警告）
/*
__global__ void spmm_block_kernel_rect(
    const float* __restrict__ A,     // row-major [M_pad x d]
    const float* __restrict__ B,     // col-major [d x N_pad]
    const int*   __restrict__ blk_row_ptr, // size: num_block_rows + 1
    const int*   __restrict__ blk_col_ind, // size: num_blocks
    float*       __restrict__ val,   // size: num_blocks * (ky*kx)
    int d, int ky, int kx)
{
  __shared__ int y_block, x_block, out_block_idx;
  int blockIdxLinear = blockIdx.x;

  // 一個 thread 做 binary-search，找到 y（block row id）與 x（block col id）
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    int lo = 0, hi = gridDim.y;
  }
}
*/

__global__ void spmm_block_kernel_rect_final(
    const float* __restrict__ A,     // row-major [M_pad x d]
    const float* __restrict__ B,     // col-major [d x N_pad]
    const int*   __restrict__ blk_row_ptr, // size: R+1
    const int*   __restrict__ blk_col_ind, // size: NNZb
    float*       __restrict__ val,   // size: NNZb * (ky*kx)
    int d, int ky, int kx, int num_block_rows)
{
  __shared__ int y_block, x_block, out_block_idx;

  int blk = blockIdx.x; // 第 blk 個非零 block
  // 使用 block 的第一個 thread (threadIdx.x=0, threadIdx.y=0) 來做 binary search
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    // binary search 找到 y，使 row_ptr[y] <= blk < row_ptr[y+1]
    int lo = 0, hi = num_block_rows; // row_ptr 有 R+1 個元素
    while (lo < hi) {
      int mid = (lo + hi) >> 1;
      if (blk_row_ptr[mid+1] <= blk) lo = mid + 1;
      else hi = mid;
    }
    y_block = lo;
    int start = blk_row_ptr[y_block];
    int offset = blk - start;
    x_block = blk_col_ind[start + offset];
    out_block_idx = blk;
  }
  __syncthreads();

  // 每個 thread 負責 (ty, tx) in ky x kx
  // threadIdx.x 是 row index (範圍 [0, ky))
  // threadIdx.y 是 column index (範圍 [0, kx))
  int ty = threadIdx.x; // [0, ky) - row index
  int tx = threadIdx.y; // [0, kx) - column index

  int a_row = y_block * ky + ty; // row-major
  int b_col = x_block * kx + tx; // col-major

  float acc = 0.f;
  for (int i = 0; i < d; ++i) {
    float a_val = A[a_row * d + i];
    float b_val = B[i + b_col * d];
    acc += a_val * b_val;
  }

  int local = ty * kx + tx;
  val[out_block_idx * (ky * kx) + local] = acc;
}

// ========================================================
// CPU 參考實作：denseXdense_to_blockedSparse_values
// ========================================================
template <class T>
vector<T> denseXdense_to_blockedSparse_values(
    const T* A,              // [M, K] row-major
    size_t M,
    size_t K,
    const T* B,              // [K, N] column-major
    size_t N,
    const int32_t* blk_row_ptr, // length = num_block_rows + 1
    const int32_t* blk_col_ind, // length = blk_row_ptr[num_block_rows]
    size_t ky, size_t kx)
{
    // 基本檢查（此版本假設整除）
    assert(M % ky == 0 && "M must be a multiple of ky");
    assert(N % kx == 0 && "N must be a multiple of kx");

    const size_t num_block_rows = M / ky;
    const size_t num_blocks = static_cast<size_t>(blk_row_ptr[num_block_rows]);
    vector<T> blk_val;
    blk_val.reserve(num_blocks * ky * kx);

    auto A_at = [&](size_t m, size_t k) -> T {
        // A: row-major
        return A[m * K + k];
    };
    auto B_at = [&](size_t k, size_t n) -> T {
        // B: column-major  (K x N)
        return B[k + K * n];
    };

    for (size_t i = 0; i < num_block_rows; ++i) {
        int32_t start = blk_row_ptr[i];
        int32_t end   = blk_row_ptr[i + 1];
        if (start == end) continue; // 這個 block row 沒有任何非零 block

        // 取出 A 的這個 block-row 的 ky 列視窗: rows [ky*i, ky*(i+1))
        const size_t a_row0 = i * ky;

        for (int32_t p = start; p < end; ++p) {
            // 這個非零 block 的 block-column index
            const size_t bc = static_cast<size_t>(blk_col_ind[p]);

            // 取出 B 的這個 block-column 的 kx 欄視窗: cols [kx*bc, kx*(bc+1))
            const size_t b_col0 = bc * kx;

            // 計算 ky x kx 的小矩陣：C_block = A_sub(ky x K) * B_sub(K x kx)
            // 並以 row-major 順序 append 到 blk_val
            for (size_t r = 0; r < ky; ++r) {
                const size_t a_row = a_row0 + r;
                for (size_t c = 0; c < kx; ++c) {
                    const size_t b_col = b_col0 + c;
                    T acc = T(0);
                    for (size_t k = 0; k < K; ++k) {
                        acc += A_at(a_row, k) * B_at(k, b_col);
                    }
                    blk_val.push_back(acc); // row-major: 先 r 再 c
                }
            }
        }
    }
    return blk_val;
}

// ------------------------- 小工具 -------------------------
static void print_vec(const char* name, const vector<int>& v) {
  cout << name << " (size=" << v.size() << "): ";
  for (int x : v) cout << x << ' ';
  cout << '\n';
}

// ========================================================
// 5) 主程式：Routing→CSR→產生 A/B→CUDA 計算 val[]
// ========================================================
int main(int argc, char** argv) {
  // ---- 可調參數（可通過命令行參數調整） ----
  int num_experts = 6;
  int num_tokens  = 40;
  int zero_expert_cnt = 2;       // 指定多少 experts 完全沒 token
  int ky = 4;                     // tile rows (block height)
  int kx = 3;                     // tile cols (block width)
  int d  = 16;                    // token embedding 維度
  
  // 解析命令行參數
  if (argc > 1) num_experts = atoi(argv[1]);
  if (argc > 2) num_tokens = atoi(argv[2]);
  if (argc > 3) zero_expert_cnt = atoi(argv[3]);
  if (argc > 4) ky = atoi(argv[4]);
  if (argc > 5) kx = atoi(argv[5]);
  if (argc > 6) d = atoi(argv[6]);
  
  if (argc == 1) {
    cout << "用法: " << argv[0] << " [num_experts] [num_tokens] [zero_expert_cnt] [ky] [kx] [d]\n";
    cout << "預設值: num_experts=6, num_tokens=40, zero_expert_cnt=2, ky=4, kx=3, d=16\n\n";
  }
  
  unsigned seed = 20251025;
  
  cout << "=== Blocked SpMM 驗證程式 ===\n";
  cout << "參數：\n";
  cout << "  num_experts = " << num_experts << "\n";
  cout << "  num_tokens = " << num_tokens << "\n";
  cout << "  zero_expert_cnt = " << zero_expert_cnt << " (完全沒 token 的 experts 數量)\n";
  cout << "  ky (block rows) = " << ky << "\n";
  cout << "  kx (block cols) = " << kx << "\n";
  cout << "  d (token embedding 維度) = " << d << "\n";
  cout << "  各 Expert 的 Hidden 維度 = 4 × " << d << " = " << (4 * d) << "\n\n";

  // 1) routing
  TimeMs t_route;
  auto T = make_imbalanced_routing(num_experts, num_tokens, zero_expert_cnt, seed, t_route);

  // 2) 設定各 expert 的 hidden 維度
  // 在 MoE 架構中，expert MLP 的 hidden layer 通常是輸入維度的 4 倍
  // 這裡的 d 是 token embedding 維度，F[i] = 4 * d 代表每個 expert 的 hidden 維度
  vector<int> F(num_experts, 4 * d);
  
  cout << "[Routing] tokens per expert:\n";
  for (int i = 0; i < num_experts; ++i) {
    cout << "  E" << i << " (|T_i|=" << T[i].size() << "): ";
    for (int t : T[i]) cout << t << ' ';
    cout << '\n';
  }
  std::cout << std::fixed << std::setprecision(3)
          << "Routing latency: " << t_route.ms << " ms\n";
          
  cout << "\n各 Expert 的 Hidden 維度 (F):\n";
  for (int i = 0; i < num_experts; ++i) {
    cout << "  Expert " << i << ": " << F[i] << " (4 × " << d << ")\n";
  }

  // 3) CSR
  vector<int> blk_col_ind, blk_row_ptr;
  TimeMs t_csr;
  compute_block_csr_from_routing(T, F, ky, kx, blk_col_ind, blk_row_ptr, t_csr);

  // print_vec("blk_col_ind", blk_col_ind);
  // print_vec("blk_row_ptr", blk_row_ptr);
  cout << "Total non-zero blocks = " << blk_col_ind.size()
       << " (should equal " << blk_row_ptr.back() << ")\n";
  cout << "CSR latency: " << t_csr.ms << " ms\n\n";

  // 4) 產生 padding 後的 dense A(row-major) / B(col-major)
  TimeMs t_gen_data;
  auto AB = with_timer(t_gen_data, [&]() {
    return make_padded_AB(T, F, ky, kx, d, seed+7);
  });
  
  // 統一矩陣維度表示：A [M_pad, d] row-major, B [d, N_pad] column-major
  const int M = AB.M_pad;  // A 的列數
  const int N = AB.N_pad;  // B 的欄數
  const int num_blocks = (int)blk_col_ind.size();
  // d 已在前面定義：A 的欄數 = B 的列數
  
  cout << "\n矩陣維度：\n";
  cout << "  A: [" << M << ", " << d << "] row-major\n";
  cout << "  B: [" << d << ", " << N << "] column-major\n";
  cout << "  output blocks: " << num_blocks << " blocks of size [" << ky << ", " << kx << "]\n";
  cout << "  資料生成耗時: " << fixed << setprecision(3) << t_gen_data.ms << " ms\n\n";

  // 5) GPU 完整流程（包含記憶體管理與拷貝）的時間計測
  TimeMs t_gpu_total;
  vector<float> val = with_timer(t_gpu_total, [&]() {
    const int num_block_rows = (int)blk_row_ptr.size() - 1;
    const size_t A_bytes = (size_t)M * d * sizeof(float);
    const size_t B_bytes = (size_t)N * d * sizeof(float);
    const size_t rowptr_bytes = (size_t)(num_block_rows + 1) * sizeof(int);
    const size_t colind_bytes = (size_t)num_blocks * sizeof(int);
    const size_t val_bytes = (size_t)num_blocks * ky * kx * sizeof(float);

    float *A_d=nullptr, *B_d=nullptr, *val_d=nullptr;
    int *rowptr_d=nullptr, *colind_d=nullptr;
    CHECK_CUDA(cudaMalloc(&A_d, A_bytes));
    CHECK_CUDA(cudaMalloc(&B_d, B_bytes));
    CHECK_CUDA(cudaMalloc(&rowptr_d, rowptr_bytes));
    CHECK_CUDA(cudaMalloc(&colind_d, colind_bytes));
    CHECK_CUDA(cudaMalloc(&val_d, val_bytes));

    CHECK_CUDA(cudaMemcpy(A_d, AB.A.data(), A_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(B_d, AB.B.data(), B_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(rowptr_d, blk_row_ptr.data(), rowptr_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(colind_d, blk_col_ind.data(), colind_bytes, cudaMemcpyHostToDevice));

    // 啟動 Kernel（注意：blockDim = (ky, kx) 
    //   → threadIdx.x 對應 row [0, ky)，threadIdx.y 對應 column [0, kx)）
    dim3 blockDim(ky, kx);
    dim3 gridDim(num_blocks);
    spmm_block_kernel_rect_final<<<gridDim, blockDim>>>(
        A_d, B_d, rowptr_d, colind_d, val_d, d, ky, kx, num_block_rows);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 取回 val
    vector<float> val_result(num_blocks * ky * kx);
    CHECK_CUDA(cudaMemcpy(val_result.data(), val_d, val_bytes, cudaMemcpyDeviceToHost));
    
    // 釋放記憶體
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(rowptr_d);
    cudaFree(colind_d);
    cudaFree(val_d);
    
    return val_result;
  });

  // 6) GPU 時間報告
  cout << "\n=== GPU 計算完成 ===\n";
  cout << "總耗時（含資料搬移）: " << fixed << setprecision(3) << t_gpu_total.ms << " ms\n";
  cout << "  val.size() = " << val.size() << " (= " << num_blocks << " blocks × " 
       << ky << " × " << kx << ")\n";

  // 7) CPU 參考計算與驗證
  cout << "\n=== CPU 參考計算 ===\n";
  TimeMs t_cpu;
  auto val_cpu = with_timer(t_cpu, [&]() {
    // 轉換 blk_row_ptr 和 blk_col_ind 為 int32_t 陣列
    vector<int32_t> blk_row_ptr_32(blk_row_ptr.begin(), blk_row_ptr.end());
    vector<int32_t> blk_col_ind_32(blk_col_ind.begin(), blk_col_ind.end());
    
    return denseXdense_to_blockedSparse_values<float>(
        AB.A.data(), M, d,
        AB.B.data(), N,
        blk_row_ptr_32.data(), blk_col_ind_32.data(),
        ky, kx);
  });
  
  cout << "總耗時: " << fixed << setprecision(3) << t_cpu.ms << " ms\n";
  cout << "  val_cpu.size() = " << val_cpu.size() 
       << " (預期 " << num_blocks * ky * kx << ")\n";
  
  // 9) 驗證結果
  cout << "\n--- 驗證 GPU vs CPU 結果 ---\n";
  if (val.size() != val_cpu.size()) {
    cerr << "錯誤：size 不一致！val.size()=" << val.size() 
         << ", val_cpu.size()=" << val_cpu.size() << endl;
    return 1;
  }
  
  int mismatch_count = 0;
  double max_abs_diff = 0.0;
  double max_rel_diff = 0.0;
  double sum_abs_diff = 0.0;
  
  const float tolerance = 1e-5f; // float 精度的容忍度
  
  for (size_t i = 0; i < val.size(); ++i) {
    float diff = fabsf(val[i] - val_cpu[i]);
    sum_abs_diff += diff;
    max_abs_diff = max(max_abs_diff, (double)diff);
    
    float ref_abs = max(fabsf(val_cpu[i]), 1e-10f);
    float rel_diff = diff / ref_abs;
    max_rel_diff = max(max_rel_diff, (double)rel_diff);
    
    if (diff > tolerance) {
      if (mismatch_count < 10) {
        cout << "Mismatch[" << i << "]: GPU=" << val[i] 
             << ", CPU=" << val_cpu[i] << ", diff=" << diff << endl;
      }
      mismatch_count++;
    }
  }
  
  cout << "\n驗證結果：\n";
  cout << "  總元素數: " << val.size() << "\n";
  cout << "  不匹配數: " << mismatch_count << "\n";
  cout << "  匹配率: " << fixed << setprecision(2) 
       << 100.0 * (1.0 - (double)mismatch_count / val.size()) << "%\n";
  cout << "  最大絕對誤差: " << scientific << setprecision(6) << max_abs_diff << "\n";
  cout << "  最大相對誤差: " << scientific << setprecision(6) << max_rel_diff << "\n";
  cout << "  平均絕對誤差: " << scientific << setprecision(6) 
       << sum_abs_diff / val.size() << "\n";
  
  if (mismatch_count == 0) {
    cout << "\n✓ GPU 和 CPU 結果完全匹配！\n";
  } else {
    cout << "\n⚠ GPU 和 CPU 結果有差異（請檢查是否為浮點精度問題）\n";
  }

  // 8) 時間比較總結
  cout << "\n=== 效能比較 ===\n";
  cout << "  GPU 總時間: " << fixed << setprecision(3) << t_gpu_total.ms << " ms\n";
  cout << "  CPU 總時間: " << fixed << setprecision(3) << t_cpu.ms << " ms\n";
  if (t_cpu.ms > 0) {
    double speedup = t_cpu.ms / t_gpu_total.ms;
    cout << "  加速比: " << fixed << setprecision(2) << speedup << "x\n";
  }

  // 9) 展示前幾個 block
  if (false) { //num_blocks > 0
    cout << "\n=== 前 " << min(num_blocks, 3) << " 個 Blocks 預覽 ===\n";
    int num_block_rows = (int)blk_row_ptr.size() - 1;
    for (int b = 0; b < min(num_blocks, 3); ++b) {
      cout << "\nBlock " << b << " (rows y=";
      // 找到其 y 以印漂亮一點
      int lo = 0, hi = num_block_rows;
      while (lo < hi) { int mid = (lo + hi) >> 1; if (blk_row_ptr[mid+1] <= b) lo = mid + 1; else hi = mid; }
      int y = lo;
      int start = blk_row_ptr[y];
      int x = blk_col_ind[start + (b - start)];
      cout << y << ", cols x=" << x << ") :\n";
      
      cout << "GPU 結果:\n";
      for (int r = 0; r < ky; ++r) {
        for (int c = 0; c < kx; ++c) {
          cout << fixed << setprecision(4) << setw(10) << val[b*ky*kx + r*kx + c] << ' ';
        }
        cout << '\n';
      }
      cout << "CPU 結果:\n";
      for (int r = 0; r < ky; ++r) {
        for (int c = 0; c < kx; ++c) {
          cout << fixed << setprecision(4) << setw(10) << val_cpu[b*ky*kx + r*kx + c] << ' ';
        }
        cout << '\n';
      }
    }
  }

  return 0;
}