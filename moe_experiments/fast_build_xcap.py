import torch

@torch.inference_mode()
def pad_grouped_to_cap_fast(
    X_grouped: torch.Tensor,      # [N, d]，已經是依 expert 排好且連續
    counts: torch.Tensor,         # [E]，每個 expert 的 token 數
    off: torch.Tensor,            # [E+1]，prefix-sum，X_grouped[off[e]:off[e+1]] 是第 e 個 expert
    cap: int,                     # capacity
    xcap_workspace: torch.Tensor = None,  # 可選，預先配置好的 [E, cap, d] 工作區
) -> torch.Tensor:
    """
    回傳 Xcap: [E, cap, d]；只清 pad 尾端；全程 GPU。
    需求：X_grouped, counts, off 同在 CUDA，且 X_grouped.contiguous()
    """
    assert X_grouped.is_cuda and counts.is_cuda and off.is_cuda
    assert X_grouped.is_contiguous(), "X_grouped 請先 .contiguous()"
    device = X_grouped.device
    E = counts.numel()
    d = X_grouped.shape[1]
    kept = torch.clamp(counts, max=cap)                     # [E] int
    N = int(counts.sum().item())                            # 只在這裡取一次 N（不會影響正確性）
                                                            # 若要 100% 避免同步，傳入 N 也可。

    # 1) 工作區（不清零）
    if xcap_workspace is None or xcap_workspace.numel() == 0 \
       or xcap_workspace.shape != (E, cap, d) \
       or xcap_workspace.dtype != X_grouped.dtype \
       or xcap_workspace.device != device:
        Xcap = torch.empty((E, cap, d), device=device, dtype=X_grouped.dtype)
    else:
        Xcap = xcap_workspace

    # 2) 建立目的地/來源索引（GPU）
    # 需要拷入的 rows：對每個 expert，row in [0, kept[e])
    rows = torch.arange(cap, device=device, dtype=kept.dtype)                # [cap]
    mask = rows.unsqueeze(0) < kept.unsqueeze(1)                             # [E, cap] bool
    e_idx, r_idx = torch.nonzero(mask, as_tuple=True)                        # [M], [M] where M = kept.sum()

    # src = off[e] + r
    base = off[:-1]                                                          # [E]
    src_idx = base.gather(0, e_idx) + r_idx                                  # [M] in X_grouped
    # dst = e*cap + r  (在扁平化後的 [E*cap, d] 中的 row)
    dst_idx = e_idx * cap + r_idx                                            # [M]

    # 3) 扁平視圖，一次性搬運（read N rows, write N rows）
    Xcap_flat = Xcap.view(E * cap, d)
    # 等價於：Xcap_flat[dst_idx] = X_grouped.index_select(0, src_idx)
    Xcap_flat.index_copy_(0, dst_idx, X_grouped.index_select(0, src_idx))

    # 4) 僅清 pad 尾端（寫 (E*cap - N) rows；避免整塊 new_zeros）
    if mask.numel() != E * cap:   # 防守式；實際上一定成立
        pass
    # pad rows 的位置：~mask
    pad_e, pad_r = torch.nonzero(~mask, as_tuple=True)
    if pad_e.numel() > 0:
        pad_dst = pad_e * cap + pad_r
        # 針對 row 維度做 index_fill_，整 row 置零；單一 kernel，寫入量 = pad_rows * d
        Xcap_flat.index_fill_(0, pad_dst, 0)

    return Xcap

# ---------- Zipf 分佈生成 ----------
def make_zipf_counts(E: int, tokens_total: int, alpha: float, seed: int = 0) -> torch.Tensor:
    """
    以 Zipf 機率 p_e ∝ 1/rank^alpha 將 tokens_total 分配到 E 個 experts。
    回傳 [E] tensor，總和 == tokens_total。
    alpha 越大，分佈越不均（第一個 expert 會得到更多 tokens）。
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    ranks = torch.arange(1, E + 1, dtype=torch.float64)
    probs = (1.0 / (ranks ** alpha))
    probs = probs / probs.sum()
    # 使用 multinomial 分配所有 tokens
    samples = torch.multinomial(probs, num_samples=tokens_total, replacement=True, generator=g)
    counts = torch.bincount(samples, minlength=E).to(dtype=torch.int32)
    return counts

# ---------- 範例用法 ----------
def example(E: int = 4, N: int = 2048, d: int = 1024, zipf_alpha: float = 1.2, seed: int = 0):
    """
    Args:
        E: expert 數量
        N: 總 token 數
        d: token 維度
        zipf_alpha: Zipf 分佈參數（越大越不均，1.2 為常用值）
        seed: 隨機種子
    """
    torch.cuda.synchronize()
    cap = N // E
    X = torch.randn(N, d, device='cuda', dtype=torch.float16).contiguous()

    # 使用 Zipf 分佈生成每個 expert 的 token 數
    counts_cpu = make_zipf_counts(E, N, alpha=zipf_alpha, seed=seed)
    # 確保每個 expert 的 token 數不超過 cap
    counts_cpu = torch.clamp(counts_cpu, max=cap)
    
    # 如果總和減少（因為 clamp），重新分配剩餘的 tokens
    # 按照 Zipf 權重分配到尚未達 cap 的 experts
    total = int(counts_cpu.sum().item())
    if total < N:
        remaining = N - total
        # 計算 Zipf 權重（只考慮未達 cap 的 experts）
        ranks = torch.arange(1, E + 1, dtype=torch.float64)
        weights = (1.0 / (ranks ** zipf_alpha))
        available = (counts_cpu < cap).float()
        
        max_iter = 100  # 防止無限循環
        iter_count = 0
        while remaining > 0 and available.sum() > 0 and iter_count < max_iter:
            iter_count += 1
            # 只考慮還有空間的 experts
            available_weights = weights * available
            if available_weights.sum() == 0:
                break
            available_weights = available_weights / available_weights.sum()
            
            # 計算每個 expert 還可以接受多少
            capacity_left = cap - counts_cpu
            
            # 按權重分配剩餘 tokens，但不超過每個 expert 的剩餘容量
            allocation_float = available_weights * remaining
            allocation = allocation_float.long()
            allocation = torch.clamp(allocation, max=capacity_left)
            
            allocated = int(allocation.sum().item())
            if allocated == 0:
                # 如果無法分配（可能是因為容量太小），嘗試逐個分配
                for e in range(E):
                    if remaining == 0:
                        break
                    if counts_cpu[e] < cap:
                        counts_cpu[e] += 1
                        remaining -= 1
                break
            
            counts_cpu += allocation
            remaining -= allocated
            
            # 更新可用空間
            available = (counts_cpu < cap).float()
    
    counts = counts_cpu.to(device='cuda', dtype=torch.int32)
    off = torch.zeros(E + 1, device='cuda', dtype=torch.int32)
    off[1:] = torch.cumsum(counts, dim=0)
    X_grouped = X  # 已依 off 切段

    # 預先配置可重用的工作區（建議在外層長期保存）
    xcap_ws = torch.empty((E, cap, d), device=X.device, dtype=X.dtype)

    # 暖機
    for _ in range(10):
        pad_grouped_to_cap_fast(X_grouped, counts, off, cap, xcap_ws)
    torch.cuda.synchronize()

    # 量測
    t0 = torch.cuda.Event(True); t1 = torch.cuda.Event(True)
    t0.record()
    for _ in range(50):
        pad_grouped_to_cap_fast(X_grouped, counts, off, cap, xcap_ws)
    t1.record(); torch.cuda.synchronize()
    print('avg ms:', t0.elapsed_time(t1)/50)

# example()
if __name__ == "__main__":
    example(E=16, N=2048, d=1024, zipf_alpha=1.2, seed=0)