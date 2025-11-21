#!/usr/bin/env python3
# verify_loop_vs_vectorized_copy.py
from __future__ import annotations
import argparse, time
import torch

# --------------------------
# Grouping（呼叫一次，兩法共用）
# --------------------------
def make_grouping(expert_idx: torch.Tensor, E: int):
    """
    回傳：
      order: [N]    — 將同 expert 排在一起的索引（argsort）
      counts: [E]   — 各 expert token 數（int32, CUDA）
      off: [E+1]    — exclusive prefix-sum of counts（int32, CUDA）
    """
    counts = torch.bincount(expert_idx, minlength=E).to(torch.int32)   # [E]
    order  = torch.argsort(expert_idx)                                 # [N]
    off    = torch.zeros(E + 1, device=expert_idx.device, dtype=torch.int32)
    off[1:] = torch.cumsum(counts, dim=0)
    return order, counts, off

# --------------------------
# 方法 A1：逐 expert 迴圈 copy（使用 .item()，會造成 CPU-GPU 同步）
# --------------------------
@torch.inference_mode()
def build_xcap_loop_with_item(
    X: torch.Tensor, E: int, cap: int,
    order: torch.Tensor, counts: torch.Tensor, off: torch.Tensor
) -> torch.Tensor:
    """
    逐 expert 迴圈：Xcap[e, :n_keep_e] = X_grouped[...]
    使用 .item() 直接從 GPU 取值（會造成 CPU-GPU 同步，效能較差）
    輸出：Xcap ∈ [E, cap, d]
    """
    N, d = X.shape
    n_keep = torch.clamp(counts, max=cap)                # [E] int32
    Xg = X[order]                                        # [N, d]
    Xcap = X.new_zeros((E, cap, d))

    # 直接使用 .item()，每次都會造成 CPU-GPU 同步
    for e in range(E):
        nke = int(n_keep[e].item())
        if nke > 0:
            s = int(off[e].item())
            Xcap[e, :nke] = Xg[s:s+nke]
    return Xcap

# --------------------------
# 方法 A2：逐 expert 迴圈 copy（避免 .item() 同步）
# --------------------------
@torch.inference_mode()
def build_xcap_loop(
    X: torch.Tensor, E: int, cap: int,
    order: torch.Tensor, counts: torch.Tensor, off: torch.Tensor
) -> torch.Tensor:
    """
    逐 expert 迴圈：Xcap[e, :n_keep_e] = X_grouped[...]
    為避免 .item() 造成同步，改在 CPU side 取 list（效能較好）
    輸出：Xcap ∈ [E, cap, d]
    """
    N, d = X.shape
    n_keep = torch.clamp(counts, max=cap)                # [E] int32
    Xg = X[order]                                        # [N, d]
    Xcap = X.new_zeros((E, cap, d))

    # 為避免 .item() 造成同步，改在 CPU side 取 list
    n_keep_cpu = n_keep.cpu().tolist()
    off_cpu    = off[:-1].cpu().tolist()

    for e in range(E):
        nke = n_keep_cpu[e]
        if nke > 0:
            s = off_cpu[e]
            Xcap[e, :nke] = Xg[s:s+nke]
    return Xcap

# --------------------------
# 方法 B：一次性向量化 copy
# --------------------------
@torch.inference_mode()
def build_xcap_vectorized(
    X: torch.Tensor, E: int, cap: int,
    order: torch.Tensor, counts: torch.Tensor, off: torch.Tensor
) -> torch.Tensor:
    """
    一次性向量化：
      1) 產生 (expert, row) 有效配對的遮罩
      2) 算出來源/目的的扁平索引
      3) 用 index_copy_ 單次搬運
    輸出：Xcap ∈ [E, cap, d]
    """
    N, d = X.shape
    kept = torch.clamp(counts, max=cap)                  # [E] int32
    Xg = X[order]                                        # [N, d]
    Xcap = X.new_zeros((E, cap, d))
    if int(kept.sum().item()) == 0:
        return Xcap

    Xcap_flat = Xcap.view(E * cap, X.shape[1])

    row = torch.arange(cap, device=X.device, dtype=torch.int32)            # [cap]
    mask = row.unsqueeze(0) < kept.unsqueeze(1)                            # [E, cap] bool
    e_idx, r_idx = mask.nonzero(as_tuple=True)                             # 1-D long

    src_idx = off[:-1].to(torch.long)[e_idx] + r_idx.to(torch.long)        # in Xg
    dst_idx = (e_idx * cap + r_idx).to(torch.long)                          # in Xcap_flat

    Xcap_flat.index_copy_(0, dst_idx, Xg.index_select(0, src_idx))
    return Xcap

# --------------------------
# 方法 C：pad_grouped_to_cap_fast（從 fast_build_xcap.py 移植）
# --------------------------
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

@torch.inference_mode()
def build_xcap_fast(
    X: torch.Tensor, E: int, cap: int,
    order: torch.Tensor, counts: torch.Tensor, off: torch.Tensor,
    xcap_workspace: torch.Tensor = None
) -> torch.Tensor:
    """
    包裝 pad_grouped_to_cap_fast，使其與其他 build_xcap_* 函數具有相同的接口。
    輸出：Xcap ∈ [E, cap, d]
    """
    Xg = X[order].contiguous()  # 確保連續
    return pad_grouped_to_cap_fast(Xg, counts, off, cap, xcap_workspace)

# --------------------------
# 量測工具
# --------------------------
def time_cuda_ms(fn, warmup=5, repeats=50):
    # CUDA events
    torch.cuda.synchronize()
    # warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))  # ms
    return sum(times) / len(times), times

def time_cpu_ms(fn, warmup=2, repeats=20):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return sum(times) / len(times), times

# --------------------------
# 主程式
# --------------------------
def main():
    ap = argparse.ArgumentParser(description="Loop vs Vectorized copy (one-time grouping, warmup+repeats).")
    ap.add_argument("--N", type=int, default=2048)
    ap.add_argument("--E", type=int, default=4)
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu"])
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16","bf16","fp32"])
    ap.add_argument("--alpha", type=float, default=1.2, help="Zipf alpha; 0=uniform")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=50)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    dev = torch.device("cuda" if use_cuda else "cpu")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    # 建資料
    N, E, d, cap = args.N, args.E, args.d, args.cap
    cap = N // E
    X = torch.randn(N, d, device=dev, dtype=dtype)

    # Zipf routing（同一份）
    ranks = torch.arange(1, E+1, dtype=torch.float64)
    p = (ranks.pow(-float(args.alpha)) / ranks.pow(-float(args.alpha)).sum()).cpu()
    r_samp = torch.multinomial(p, N, replacement=True)  # CPU
    perm = torch.randperm(E)
    expert_idx = perm[r_samp].to(device=dev, dtype=torch.long)

    # 只呼叫一次 grouping，兩方法共用
    order, counts, off = make_grouping(expert_idx, E)

    # 計算統計信息：padding 和 dropped rows
    kept = torch.clamp(counts, max=cap)  # [E] 每個 expert 保留的 rows
    dropped = counts - kept  # [E] 每個 expert 被 dropped 的 rows（可能為負，取 max(0)）
    total_dropped = int((dropped.clamp(min=0)).sum().item())  # 總共被 dropped 的 rows
    total_kept = int(kept.sum().item())  # 總共保留的 rows
    total_padded = E * cap - total_kept  # padding 後的總 rows 減去保留的 = padding 的 rows
    total_after_padding = E * cap  # padding 後的總 rows
    
    # 顯示統計信息
    print(f"\n[Statistics]")
    print(f"  Original rows (N): {N}")
    print(f"  Rows kept: {total_kept} (after clamping to cap={cap})")
    print(f"  Rows dropped: {total_dropped} (超過 capacity 的部分)")
    print(f"  Rows padded: {total_padded} (填充的零 rows)")
    print(f"  Total rows after padding: {total_after_padding} (= E * cap = {E} * {cap})")
    if total_dropped > 0:
        print(f"  Drop rate: {100.0 * total_dropped / N:.2f}%")
    print(f"  Padding rate: {100.0 * total_padded / total_after_padding:.2f}%")

    # 先做一次輸出並驗證一致
    Xcap_loop_item = build_xcap_loop_with_item(X, E, cap, order, counts, off)
    Xcap_loop = build_xcap_loop(X, E, cap, order, counts, off)
    Xcap_vec  = build_xcap_vectorized(X, E, cap, order, counts, off)
    
    # 對於 fast 版本，需要 CUDA 且預先準備 workspace
    if use_cuda:
        xcap_workspace = torch.empty((E, cap, d), device=dev, dtype=dtype)
        Xcap_fast = build_xcap_fast(X, E, cap, order, counts, off, xcap_workspace)
    else:
        Xcap_fast = None
    
    # 驗證所有方法輸出相同
    same_1 = torch.equal(Xcap_loop_item, Xcap_loop)
    same_2 = torch.equal(Xcap_loop, Xcap_vec)
    max_abs_1 = (Xcap_loop_item - Xcap_loop).abs().max().item()
    max_abs_2 = (Xcap_loop - Xcap_vec).abs().max().item()
    print(f"[Equal] loop_with_item vs loop: {same_1} | max_abs_diff = {max_abs_1:.3e}")
    print(f"[Equal] loop vs vectorized: {same_2} | max_abs_diff = {max_abs_2:.3e}")
    
    if use_cuda and Xcap_fast is not None:
        same_3 = torch.equal(Xcap_vec, Xcap_fast)
        max_abs_3 = (Xcap_vec - Xcap_fast).abs().max().item()
        print(f"[Equal] vectorized vs fast: {same_3} | max_abs_diff = {max_abs_3:.3e}")

    # 量測
    if use_cuda:
        avg_loop_item_ms, _ = time_cuda_ms(lambda: build_xcap_loop_with_item(X, E, cap, order, counts, off),
                                           warmup=args.warmup, repeats=args.repeats)
        avg_loop_ms, _ = time_cuda_ms(lambda: build_xcap_loop(X, E, cap, order, counts, off),
                                      warmup=args.warmup, repeats=args.repeats)
        avg_vec_ms,  _ = time_cuda_ms(lambda: build_xcap_vectorized(X, E, cap, order, counts, off),
                                      warmup=args.warmup, repeats=args.repeats)
        # fast 版本需要 workspace（可重用）
        xcap_workspace = torch.empty((E, cap, d), device=dev, dtype=dtype)
        avg_fast_ms, _ = time_cuda_ms(lambda: build_xcap_fast(X, E, cap, order, counts, off, xcap_workspace),
                                      warmup=args.warmup, repeats=args.repeats)
    else:
        avg_loop_item_ms, _ = time_cpu_ms(lambda: build_xcap_loop_with_item(X, E, cap, order, counts, off),
                                           warmup=args.warmup, repeats=args.repeats)
        avg_loop_ms, _ = time_cpu_ms(lambda: build_xcap_loop(X, E, cap, order, counts, off),
                                     warmup=args.warmup, repeats=args.repeats)
        avg_vec_ms,  _ = time_cpu_ms(lambda: build_xcap_vectorized(X, E, cap, order, counts, off),
                                     warmup=args.warmup, repeats=args.repeats)
        avg_fast_ms = None

    print(f"[Setting] N={N}, E={E}, d={d}, cap={cap}, device={dev}, dtype={dtype}")
    print(f"[Timing] (warmup={args.warmup}, repeats={args.repeats})")
    print(f"  loop_with_item:  {avg_loop_item_ms:.3f} ms  (使用 .item()，會造成 CPU-GPU 同步)")
    print(f"  loop:            {avg_loop_ms:.3f} ms  (避免 .item()，先轉 CPU)")
    print(f"  vectorized:      {avg_vec_ms:.3f} ms  (向量化版本)")
    if use_cuda and avg_fast_ms is not None:
        print(f"  fast:            {avg_fast_ms:.3f} ms  (pad_grouped_to_cap_fast 版本)")
    
    # 計算加速比
    if avg_loop_item_ms > 0:
        speedup_vs_item = avg_loop_item_ms / avg_loop_ms
        print(f"\n[Speedup] loop (避免同步) 相對 loop_with_item: {speedup_vs_item:.2f}x")
    if avg_loop_ms > 0:
        speedup_vec_vs_loop = avg_loop_ms / avg_vec_ms
        print(f"[Speedup] vectorized 相對 loop: {speedup_vec_vs_loop:.2f}x")
    if use_cuda and avg_fast_ms is not None and avg_vec_ms > 0:
        speedup_fast_vs_vec = avg_vec_ms / avg_fast_ms
        print(f"[Speedup] fast 相對 vectorized: {speedup_fast_vs_vec:.2f}x")
    if use_cuda and avg_fast_ms is not None and avg_loop_ms > 0:
        speedup_fast_vs_loop = avg_loop_ms / avg_fast_ms
        print(f"[Speedup] fast 相對 loop: {speedup_fast_vs_loop:.2f}x")

if __name__ == "__main__":
    main()