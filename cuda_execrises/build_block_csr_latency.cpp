#include <bits/stdc++.h>
using namespace std;

// ------------------------- 工具：量測時間 -------------------------
struct TimeMs {
    double ms = 0.0;
};
template <class F>
auto with_timer(TimeMs& t, F&& fn) {
    auto beg = chrono::steady_clock::now();
    auto ret = fn();
    auto end = chrono::steady_clock::now();
    t.ms = chrono::duration<double, milli>(end - beg).count();
    return ret;
}

// ------------------------- 1) 產生不均勻 routing -------------------------
// 需求：讓一部份 experts 完全沒有 tokens；token 全數分配給其餘 experts（總數不變）。
// 參數：
//   num_experts, num_tokens
//   zero_expert_cnt : 指定要「沒 token」的 expert 數（<= num_experts-1）。
//   seed            : 隨機種子
//   t               : 量測 latency（毫秒）
// 輸出：T[i] = 第 i 個 expert 的 token indices
static vector<vector<int>>
make_imbalanced_routing(int num_experts, int num_tokens,
                        int zero_expert_cnt, unsigned seed,
                        TimeMs& t)
{
    return with_timer(t, [&]() {
        if (num_experts <= 1) throw runtime_error("num_experts must be >= 2");
        zero_expert_cnt = max(0, min(zero_expert_cnt, num_experts - 1));

        mt19937 gen(seed);

        // 隨機選出 zero set Z：這些 expert 將「完全沒有 token」
        vector<int> ids(num_experts); iota(ids.begin(), ids.end(), 0);
        shuffle(ids.begin(), ids.end(), gen);
        unordered_set<int> Z(ids.begin(), ids.begin() + zero_expert_cnt);

        // 為其餘 experts 構造不均勻權重（power-law 風格），Z 內權重=0
        vector<double> w(num_experts, 0.0);
        for (int i = 0; i < num_experts; ++i) {
            if (Z.count(i) == 0) w[i] = 1.0 / (i + 1.0);
        }
        // 保證至少有一個可分配對象
        double sumw = accumulate(w.begin(), w.end(), 0.0);
        if (sumw <= 0.0) throw runtime_error("No available experts to assign tokens.");

        std::discrete_distribution<int> dist(w.begin(), w.end());

        // 分配 token：只會落在非 Z 的 experts 上
        vector<vector<int>> T(num_experts);
        for (int t = 0; t < num_tokens; ++t) {
            int e = dist(gen);
            T[e].push_back(t);
        }

        // 檢查：Z 內必為空；總 token 不變
        size_t total = 0;
        for (int i = 0; i < num_experts; ++i) {
            if (Z.count(i)) { /* 應為 0 */ }
            total += T[i].size();
        }
        if ((int)total != num_tokens) throw runtime_error("Token conservation violated.");

        return T;
    });
}

// ------------------------- 2) Routing → Block-CSR -------------------------
// 參數：
//   T[i]  : 第 i 個 expert 的 token 索引集合（只用其大小）
//   F[i]  : 第 i 個 expert 的 hidden 維度
//   ky,kx : block 形狀（row, col）
//   t     : 量測 latency（毫秒）
// 輸出：
//   blk_col_ind : 非零 block 的 column indices（全域）
//   blk_row_ptr : CSR row pointers；每個 block-row 皆有一筆，允許 0 nnz
static void compute_block_csr_from_routing(
    const vector<vector<int>>& T,
    const vector<int>& F,
    int ky, int kx,
    vector<int>& blk_col_ind,
    vector<int>& blk_row_ptr,
    TimeMs& t)
{
    with_timer(t, [&]() {
        const int num_experts = (int)T.size();
        if ((int)F.size() != num_experts) {
            throw runtime_error("F.size() must equal T.size()");
        }
        blk_col_ind.clear();
        blk_row_ptr.clear();
        blk_row_ptr.push_back(0); // CSR 起點

        int base_col = 0; // b = sum_{j < i} |G_j|

        for (int i = 0; i < num_experts; ++i) {
            const int tokens_i = (int)T[i].size();
            const int Fi       = F[i];

            // |G_i| = ceil(|T_i| / kx)，允許 0（該 expert 無 column blocks）
            const int Gi = (tokens_i == 0) ? 0 : ((tokens_i + kx - 1) / kx);
            // Ri = ceil(Fi / ky)
            const int Ri = (Fi == 0) ? 0 : ((Fi + ky - 1) / ky);

            for (int r = 0; r < Ri; ++r) {
                int start = blk_row_ptr.back();   // 本 row 的起始 offset
                if (Gi > 0) {
                    for (int n = 0; n < Gi; ++n) {
                        blk_col_ind.push_back(base_col + n);
                    }
                }
                blk_row_ptr.push_back(start + Gi); // 即使 Gi==0 也要推進（空列）
            }
            base_col += Gi;
        }
        return 0;
    });
}

// ------------------------- Demo -------------------------
static void print_routing(const vector<vector<int>>& T) {
    cout << "[Routing] tokens per expert:\n";
    for (int i = 0; i < (int)T.size(); ++i) {
        cout << "  E" << i << " (|T_i|=" << T[i].size() << "): ";
        for (int t : T[i]) cout << t << " ";
        cout << "\n";
    }
}

int main() {
    // 超參數
    const int num_experts = 8;
    const int num_tokens  = 40;
    const int ky = 4;   // block rows
    const int kx = 3;   // block cols
    const int zero_expert_cnt = 2; // 讓 2 個 expert 完全沒有 tokens
    const unsigned seed  = 20251025;

    // 1) 產生 routing（高度不均勻，且確保有部分 experts 無 tokens）
    TimeMs t_route;
    auto T = make_imbalanced_routing(num_experts, num_tokens,
                                     zero_expert_cnt, seed, t_route);

    // 假設每個 expert 的 hidden 維度（可自行調整；允許 0）
    vector<int> F = {16, 16, 16, 16, 16, 16, 16, 16};

    // 印出 routing 與延遲
    print_routing(T);
    cout << fixed << setprecision(3)
         << "\nRouting latency: " << t_route.ms << " ms\n\n";

    // 2) Routing → Block-CSR
    vector<int> blk_col_ind, blk_row_ptr;
    TimeMs t_csr;
    compute_block_csr_from_routing(T, F, ky, kx, blk_col_ind, blk_row_ptr, t_csr);

    // 輸出結果與延遲
    cout << "Block shape (ky,kx) = (" << ky << "," << kx << ")\n";
    cout << "blk_col_ind (size=" << blk_col_ind.size() << "): ";
    for (int v : blk_col_ind) cout << v << " ";
    cout << "\n";

    cout << "blk_row_ptr (size=" << blk_row_ptr.size() << "): ";
    for (int v : blk_row_ptr) cout << v << " ";
    cout << "\n";

    cout << "Total non-zero blocks = " << blk_col_ind.size()
         << " (should equal " << blk_row_ptr.back() << ")\n";

    cout << "CSR build latency: " << t_csr.ms << " ms\n\n";

    // 逐列檢視（可看到空列）
    cout << "Per block-row columns:\n";
    for (size_t r = 0; r + 1 < blk_row_ptr.size(); ++r) {
        int beg = blk_row_ptr[r], end = blk_row_ptr[r+1];
        cout << "  row " << r << " : ";
        for (int p = beg; p < end; ++p) cout << blk_col_ind[p] << " ";
        if (beg == end) cout << "(empty)";
        cout << "\n";
    }
    return 0;
}