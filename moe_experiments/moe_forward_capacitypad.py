#!/usr/bin/env python3
# moe_forward_capacitypad.py
from __future__ import annotations
import math
import torch

__all__ = ["moe_forward_capacitypad"]

@torch.inference_mode()
def moe_forward_capacitypad(
    X: torch.Tensor,             # [N, d]
    expert_idx: torch.Tensor,    # [N] long, 每個 token 指派到的 expert（此實作為 k=1）
    W1: torch.Tensor,            # [E, d, d_ff]
    W2: torch.Tensor,            # [E, d_ff, d]
    activation: str = "gelu",
    group_same_n: bool = True,   # 僅為與 packed/tile 介面一致；在 Cap-Pad 中不需使用
    **kwargs,                    # 參數以 kwargs 傳入：cap 或 capacity_factor、overflow_policy 等
) -> torch.Tensor:
    """
    Capacity Padding (Cap-Pad) 的 MoE-MLP 前向實作。
    - 給定容量上限 cap（或以 capacity_factor 推得 cap），每個 expert 的 M 維補到 cap；
      對 n_e > cap 的 token 採 overflow_policy（預設 'drop'：丟棄多出的 token）。
    - 計算兩段 GEMM 後，將各 expert 有效的前 n_keep_e 行回填到輸出 Y 的原 token 位置。

    kwargs 支援：
      - cap: int，容量上限；若提供則優先使用。
      - capacity_factor: float（預設 1.0），cap = ceil(capacity_factor * ceil(N/E))。
      - overflow_policy: str in {"drop","error"}（預設 "drop"）。
      - pad_value: float（預設 0.0），padding 行的填充值（通常為 0）。
    輸出：
      - Y: [N, d]，與輸入 token 原順序對齊；若 token 被 drop，對應輸出為 0（或 pad_value 的線性映射結果，對線性層即 0）。
    備註：
      - 本函式聚焦在 MLP 計算與容量補齊；不含 gating、通訊與 k>1 聚合。
      - 若上游已確保 n_e <= cap，則本函式不會 drop。
    """
    # ---- 基本檢查 ----
    assert X.dim() == 2 and expert_idx.dim() == 1, "X:[N,d], expert_idx:[N]"
    N, d = X.shape
    E, d_in, d_ff = W1.shape
    assert d == d_in and W2.shape == (E, d_ff, d), "權重形狀不符"
    assert expert_idx.shape[0] == N and expert_idx.dtype in (torch.long, torch.int64)

    device = X.device
    dtype  = X.dtype

    # ---- 讀取 kwargs ----
    cap = kwargs.get("cap", None)
    capacity_factor = float(kwargs.get("capacity_factor", 1.0))
    overflow_policy = kwargs.get("overflow_policy", "drop")  # "drop" | "error"
    pad_value = float(kwargs.get("pad_value", 0.0))

    if cap is None:
        # 平均每 expert 的負載估計（k=1 假設）
        avg = math.ceil(N / max(1, E))
        cap = int(math.ceil(capacity_factor * avg))
    if cap <= 0:
        raise ValueError(f"cap 必須為正整數，取得 cap={cap}")

    # ---- 分組（將同 expert 的 token 連續化）----
    # 使用排序避免逐 expert nonzero 的高開銷
    counts = torch.bincount(expert_idx, minlength=E)     # [E]
    order  = torch.argsort(expert_idx)                   # [N]
    off    = torch.zeros(E + 1, device=device, dtype=torch.int32)
    off[1:] = torch.cumsum(counts, dim=0)

    # 每 expert 的保留數 n_keep_e 與 overflow
    n = counts.to(torch.int32)                           # [E]
    n_keep = torch.clamp(n, max=cap)                     # [E]
    n_over = (n - n_keep).clamp_min(0)                   # [E]
    total_keep = int(n_keep.sum().item())
    total_over = int(n_over.sum().item())

    if total_over > 0 and overflow_policy == "error":
        raise RuntimeError(f"有 {total_over} 個指派超過 cap={cap}，且 overflow_policy='error'。")

    # ---- 建立 cap 對齊的輸入批（[E, cap, d]）並填入有效行，其餘補 pad_value ----
    # 使用向量化版本取代 for 循環
    Xgrp = X[order]
    Xcap = build_xcap_vectorized(Xgrp, order, n_keep, off, cap, pad_value)

    # ---- 兩段 GEMM：使用 batched bmm ----
    # H = Xcap @ W1  -> [E, cap, d_ff]
    # Ycap = act(H) @ W2 -> [E, cap, d]
    act = torch.nn.functional.gelu if activation == "gelu" else torch.nn.functional.relu
    # bmm 需 [E, cap, d] x [E, d, d_ff]
    H = torch.bmm(Xcap, W1)                  # [E, cap, d_ff]
    H = act(H)
    Ycap = torch.bmm(H, W2)                  # [E, cap, d]

    # ---- 回填到輸出 Y（與原 token 順序對齊）----
    Y = X.new_zeros((N, d))                  # 預設 0（被 drop 的 token 仍為 0）
    # 我們需要把每個 expert 的前 n_keep_e 行依原順序放回
    # 被 drop 的 token（每 expert 超過 cap 的尾端）不寫入（保持 0）
    # 為避免 .item() 造成同步，改在 CPU side 取 list
    n_keep_cpu = n_keep.cpu().tolist()
    off_cpu = off[:-1].cpu().tolist()
    
    for e in range(E):
        nke = n_keep_cpu[e]
        if nke == 0:
            continue
        # 原 grouped 區段起點（在 order 上）
        s = off_cpu[e]
        # 該 expert 保留的 token 的「原始索引」
        kept_indices = order[s:s+nke]        # [nke]
        # 取 Ycap 的有效前 nke 行
        Ye = Ycap[e, :nke]                   # [nke, d]
        Y[kept_indices] = Ye

    return Y

def make_grouping(expert_idx: torch.Tensor, E: int):
    """
    回傳：
      order: [N]    — 將同 expert 排在一起的索引（argsort）
      counts: [E]   — 各 expert token 數
      off: [E+1]    — exclusive prefix-sum of counts
    """
    counts = torch.bincount(expert_idx, minlength=E)
    order  = torch.argsort(expert_idx)
    off    = torch.zeros(E + 1, device=expert_idx.device, dtype=torch.int32)
    off[1:] = torch.cumsum(counts, dim=0)
    return order, counts.to(torch.int32), off

@torch.inference_mode()
def build_xcap_vectorized(
    Xgrouped: torch.Tensor,
    order: torch.Tensor,
    n_keep: torch.Tensor,
    off: torch.Tensor,
    cap: int,
    pad_value: float = 0.0
) -> torch.Tensor:
    """
    一次性向量化建立 Xcap：
      1) 產生 (expert, row) 有效配對的布林遮罩
      2) 算出來源/目的的扁平索引
      3) 用 index_copy_ 單次搬運
    參數：
      Xgrouped: [N, d] 分組後的輸入
      order: [N] 已排序的索引（同 expert 連續）
      n_keep: [E] 每個 expert 保留的 token 數（已 clamp 到 cap）
      off: [E+1] 前綴和偏移
      cap: 容量上限
      pad_value: padding 填充值（預設 0.0）
    輸出：Xcap ∈ [E, cap, d]
    """
    E = n_keep.shape[0]
    d = Xgrouped.shape[1]
    
    # Xg = X[order]                               # [N, d]
    Xcap = Xgrouped.new_full((E, cap, d), fill_value=pad_value)
    Xcap_flat = Xcap.view(E * cap, d)

    if int(n_keep.sum().item()) == 0:
        return Xcap  # 所有 expert 都沒有 token

    row = torch.arange(cap, device=Xgrouped.device, dtype=torch.int32)         # [cap]
    mask = row.unsqueeze(0) < n_keep.unsqueeze(1)                        # [E, cap] bool
    e_idx, r_idx = mask.nonzero(as_tuple=True)                           # 1-D long vectors

    src_idx = off[:-1][e_idx] + r_idx                                    # in Xg (len = sum kept)
    dst_idx = (e_idx * cap + r_idx).to(torch.long)                        # in Xcap_flat

    # 單次 kernel 完成所有有效行搬運
    Xcap_flat.index_copy_(0, dst_idx, Xgrouped.index_select(0, src_idx.to(torch.long)))
    return Xcap

# ---------- 小測試（可直接執行） ----------
if __name__ == "__main__":
    # 從 benchmark_moe_padding.py 導入 Zipf 路由函數
    import sys
    import os
    # 導入 Zipf 路由函數和 profiling 工具
    sys.path.insert(0, os.path.dirname(__file__))
    from benchmark_moe_padding import sample_zipf_routing
    from metrics_profiling import Metrics, Profiling
    
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="測試 moe_forward_capacitypad 函數")
    parser.add_argument("--N", type=int, default=1024, help="Token 數量")
    parser.add_argument("--E", type=int, default=8, help="Expert 數量")
    parser.add_argument("--d", type=int, default=512, help="特徵維度")
    parser.add_argument("--dff", type=int, default=2048, help="FFN 隱藏層維度")
    parser.add_argument("--alpha", type=float, default=1.2, help="Zipf alpha (0=uniform, bigger=more skew)")
    parser.add_argument("--capacity_factor", type=float, default=1.0, help="容量係數")
    parser.add_argument("--seed", type=int, default=0, help="隨機種子")
    parser.add_argument("--repeats", type=int, default=10, help="重複測試次數")
    parser.add_argument("--profile", action="store_true", help="啟用詳細 profiling")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    N, E, d, dff = args.N, args.E, args.d, args.dff
    
    # 選擇設備和數據類型
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA 不可用，自動改用 CPU")
        args.device = "cpu"
    dev = torch.device(args.device)
    
    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    X  = torch.randn(N, d, device=dev, dtype=dtype)
    W1 = torch.randn(E, d, dff, device=dev, dtype=dtype)
    W2 = torch.randn(E, dff, d, device=dev, dtype=dtype)
    
    # 使用 Zipf 分佈生成 expert 索引
    expert_idx = sample_zipf_routing(N, E, args.alpha, device=dev)
    
    # 顯示 Expert 分配
    counts = torch.bincount(expert_idx, minlength=E)
    print(f"== Workload ==")
    print(f"N={N}, E={E}, d={d}, dff={dff}, alpha={args.alpha}")
    print(f"device={dev}, dtype={dtype}")
    print(f"Expert 分配:")
    for e in range(E):
        print(f"  Expert {e}: {int(counts[e])} tokens")
    
    if args.profile:
        # --- Padding/Drop 統計（固定於本路由與參數下不變）---
        cap_once = int(math.ceil(args.capacity_factor * math.ceil(N / max(1, E))))
        n_keep_once = torch.clamp(counts.to(torch.int32), max=cap_once)
        total_keep_once = int(n_keep_once.sum().item())
        total_drop_once = int((counts.to(torch.int32) - n_keep_once).clamp_min(0).sum().item())
        padded_rows_once = int(E * cap_once)
        added_rows_once = int(max(0, padded_rows_once - total_keep_once))

        print("\n== Padding/Drop 統計（單次）==")
        print(f"  原始 tokens: {N}")
        print(f"  保留 tokens: {total_keep_once}")
        print(f"  Drop tokens: {total_drop_once}")
        print(f"  Padded rows: {padded_rows_once}")
        print(f"  Padding 增加行數: {added_rows_once}")

        # 建立 Metrics 和 Profiling 對象
        metrics_grouping = Metrics()
        metrics_padding = Metrics()
        metrics_matmul = Metrics()
        metrics_scatter = Metrics()
        metrics_total = Metrics()
        
        use_cuda_events = args.device == "cuda" and torch.cuda.is_available()
        prof_grouping = Profiling(metrics_grouping, use_cuda_events=use_cuda_events)
        prof_padding = Profiling(metrics_padding, use_cuda_events=use_cuda_events)
        prof_matmul = Profiling(metrics_matmul, use_cuda_events=use_cuda_events)
        prof_scatter = Profiling(metrics_scatter, use_cuda_events=use_cuda_events)
        prof_total = Profiling(metrics_total, use_cuda_events=use_cuda_events)
        
        # 暖機
        for _ in range(3):
            _ = moe_forward_capacitypad(X, expert_idx, W1, W2, capacity_factor=args.capacity_factor)
        
        if args.device == "cuda":
            torch.cuda.synchronize()
        
        print(f"\n== 開始 Profiling (重複 {args.repeats} 次) ==\n")
        
        # 模擬各步驟的 profiling
        for i in range(args.repeats):
            # --- 步驟 1: Token Grouping ---
            prof_grouping.start()
            counts_group = torch.bincount(expert_idx, minlength=E)
            order_group = torch.argsort(expert_idx)
            off_group = torch.zeros(E + 1, device=dev, dtype=torch.int32)
            off_group[1:] = torch.cumsum(counts_group, dim=0)
            Xgrouped = X[order_group]
            if args.device == "cuda":
                torch.cuda.synchronize()
            prof_grouping.end()
            
            # --- 步驟 2: Padding/Dropping ---
            prof_padding.start()
            n_group = counts_group.to(torch.int32)
            cap_group = int(math.ceil(args.capacity_factor * math.ceil(N / max(1, E))))
            n_keep_group = torch.clamp(n_group, max=cap_group)
            # 使用向量化版本取代 for 循環
            Xcap_group = build_xcap_vectorized(Xgrouped, order_group, n_keep_group, off_group, cap_group, pad_value=0.0)
            if args.device == "cuda":
                torch.cuda.synchronize()
            # 在 padding 步驟的 profiling 中回報行數與 drop
            real_rows_group = int(n_keep_group.sum().item())
            padded_rows_group = int(E * cap_group)
            drops_group = int((n_group - n_keep_group).clamp_min(0).sum().item())
            prof_padding.end(padding_real_rows=real_rows_group,
                             padding_padded_rows=padded_rows_group,
                             drops=drops_group)
            
            # --- 步驟 3: Batch MatMul ---
            prof_matmul.start()
            H_group = torch.bmm(Xcap_group, W1)
            H_group = torch.nn.functional.gelu(H_group)
            Ycap_group = torch.bmm(H_group, W2)
            if args.device == "cuda":
                torch.cuda.synchronize()
            prof_matmul.end()
            
            # --- 步驟 4: 取回有效行並還原原順序 ---
            
            prof_scatter.start()
            Y_group = Xgrouped.new_zeros((N, d))
            # 為避免 .item() 造成同步，改在 CPU side 取 list
            n_keep_cpu = n_keep_group.cpu().tolist()
            off_cpu = off_group[:-1].cpu().tolist()
            
            for e in range(E):
                nke = n_keep_cpu[e]
                if nke > 0:
                    s = off_cpu[e]
                    kept_indices = order_group[s:s+nke]
                    Ye_group = Ycap_group[e, :nke]
                    Y_group[kept_indices] = Ye_group
            
            if args.device == "cuda":
                torch.cuda.synchronize()
            prof_scatter.end()
            
            # 總體量測
            prof_total.start()
            Y_total = moe_forward_capacitypad(X, expert_idx, W1, W2, capacity_factor=args.capacity_factor)
            prof_total.end()
            
            if (i + 1) % max(1, args.repeats // 10) == 0:
                print(f"完成 {i+1}/{args.repeats} 次...")
        
        print("\n== Profiling 結果 ==\n")
        
        # 步驟 1: Token Grouping
        grouping_summary = metrics_grouping.summary()
        print("(1) Token Grouping:")
        if "latency" in grouping_summary:
            lat = grouping_summary["latency"]
            print(f"  平均: {lat.get('mean_ms', 0):.4f} ms")
            print(f"  P50:  {lat.get('p50_ms', 0):.4f} ms")
            print(f"  P90:  {lat.get('p90_ms', 0):.4f} ms")
            print(f"  P99:  {lat.get('p99_ms', 0):.4f} ms")
        
        # 步驟 2: Padding/Dropping
        padding_summary = metrics_padding.summary()
        print("\n(2) Padding/Dropping:")
        if "latency" in padding_summary:
            lat = padding_summary["latency"]
            print(f"  平均: {lat.get('mean_ms', 0):.4f} ms")
            print(f"  P50:  {lat.get('p50_ms', 0):.4f} ms")
            print(f"  P90:  {lat.get('p90_ms', 0):.4f} ms")
            print(f"  P99:  {lat.get('p99_ms', 0):.4f} ms")
        
        # 步驟 3: Batch MatMul
        matmul_summary = metrics_matmul.summary()
        print("\n(3) Batch MatMul:")
        if "latency" in matmul_summary:
            lat = matmul_summary["latency"]
            print(f"  平均: {lat.get('mean_ms', 0):.4f} ms")
            print(f"  P50:  {lat.get('p50_ms', 0):.4f} ms")
            print(f"  P90:  {lat.get('p90_ms', 0):.4f} ms")
            print(f"  P99:  {lat.get('p99_ms', 0):.4f} ms")
        
        # 步驟 4: 取回有效行並還原原順序
        scatter_summary = metrics_scatter.summary()
        print("\n(4) 取回有效行並還原原順序:")
        if "latency" in scatter_summary:
            lat = scatter_summary["latency"]
            print(f"  平均: {lat.get('mean_ms', 0):.4f} ms")
            print(f"  P50:  {lat.get('p50_ms', 0):.4f} ms")
            print(f"  P90:  {lat.get('p90_ms', 0):.4f} ms")
            print(f"  P99:  {lat.get('p99_ms', 0):.4f} ms")
        
        # 總體
        total_summary = metrics_total.summary()
        print("\n總體執行:")
        if "latency" in total_summary:
            lat = total_summary["latency"]
            print(f"  平均: {lat.get('mean_ms', 0):.4f} ms")
            print(f"  P50:  {lat.get('p50_ms', 0):.4f} ms")
            print(f"  P90:  {lat.get('p90_ms', 0):.4f} ms")
            print(f"  P99:  {lat.get('p99_ms', 0):.4f} ms")
        
        # 計算時間比例
        if "latency" in total_summary:
            total_mean = total_summary["latency"].get('mean_ms', 0)
            grouping_mean = grouping_summary.get("latency", {}).get('mean_ms', 0)
            padding_mean = padding_summary.get("latency", {}).get('mean_ms', 0)
            matmul_mean = matmul_summary.get("latency", {}).get('mean_ms', 0)
            scatter_mean = scatter_summary.get("latency", {}).get('mean_ms', 0)
            
            print("\n時間比例:")
            if total_mean > 0:
                print(f"  Token Grouping:         {grouping_mean/total_mean*100:.1f}%")
                print(f"  Padding/Dropping:       {padding_mean/total_mean*100:.1f}%")
                print(f"  Batch MatMul:           {matmul_mean/total_mean*100:.1f}%")
                print(f"  取回有效行並還原順序:  {scatter_mean/total_mean*100:.1f}%")
                print(f"  其他開銷:               {(total_mean-grouping_mean-padding_mean-matmul_mean-scatter_mean)/total_mean*100:.1f}%")
        
        # 輸出完整 JSON
        print("\n== 完整 JSON 結果 ==")
        print(json.dumps({
            "grouping": grouping_summary,
            "padding": padding_summary,
            "matmul": matmul_summary,
            "scatter": scatter_summary,
            "total": total_summary
        }, indent=2, ensure_ascii=False))
    else:
        # 簡單執行（無 profiling）
        print("\n執行 moe_forward_capacitypad...")
        
        # 例1：cap 充足（不 drop）
        Y1 = moe_forward_capacitypad(X, expert_idx, W1, W2, capacity_factor=args.capacity_factor)
        print(f"OK: 完成前向。輸出形狀: {Y1.shape}")
        print(f"輸出統計: mean={Y1.mean():.6f}, std={Y1.std():.6f}")
    
    print("\n測試完成！")