# 矩陣維度說明

## 為什麼 A 是 [65536, 1024]，B 是 [1024, 296]？

### 參數設定
- `d = 1024` (token embedding 維度)
- `num_experts = 16`
- `F[i] = 4 × d = 4096` (每個 expert 的 hidden 維度)
- `ky = 4` (block rows)
- `kx = 3` (block cols)

### A 矩陣 [M=65536, d=1024]
**含義**：所有 experts 的 hidden layer weights，經過 padding

計算過程：
1. 每個 expert 的 hidden 維度：`F[i] = 4 × 1024 = 4096`
2. 需要 ceil(4096/4) = 1024 個 block rows（因為 ky=4）
3. 每個 expert 在 A 中佔用：1024 × 4 = 4096 行
4. 總共 num_experts = 16 個 experts
5. **M = 16 × 4096 = 65536 行**

所以 A 是將 16 個 experts 的 weights 垂直堆疊起來。

### B 矩陣 [d=1024, N=296] (column-major)
**含義**：Token embeddings，分配到各個 experts

計算過程：
- B 的欄數 N = 296 取決於 routing 結果
- 這表示總共處理了約 296 個 tokens（經過 group 和 padding）
- 每個 token embedding 是 d=1024 維

### 為什麼 N 這麼小？

B 的欄數（296）相比 A 的行數（65536）小很多，因為：
1. B 存的是實際的 token embeddings
2. tokens 經 routing 分配給各 experts
3. 經過 block 分組（kx=3）和 padding
4. 最終只有約 296 個完整的 columns

這是合理的，因為：
- Expert weights (A) 需要準備所有可能用到的參數
- 而實際輸入 (B) 只包含當前 batch 的 tokens

