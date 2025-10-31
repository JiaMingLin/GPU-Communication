# moe_forward_packed.py
from __future__ import annotations
from collections import defaultdict
import torch

__all__ = ["moe_forward_packed"]

@torch.inference_mode()
def moe_forward_packed(
    X: torch.Tensor,             # [N, d]
    expert_idx: torch.Tensor,    # [N] long
    W1: torch.Tensor,            # [E, d, d_ff]
    W2: torch.Tensor,            # [E, d_ff, d]
    activation: str = "gelu",
    group_same_n: bool = True,
    tile: int | None = None,     # 保持與 tilepad 一致；此函式不使用
) -> torch.Tensor:
    """
    Packed（不補零）MoE-MLP 前向：
      - 將相同 expert 的 token 連續化
      - 對每個 expert 以實際行數 n_e 執行兩段 GEMM
      - （選用）相同 n 的 experts 以 batched bmm 合併，降低 kernel 啟動成本
    參數：
      X: [N, d]、expert_idx: [N]、W1: [E, d, d_ff]、W2: [E, d_ff, d]
      activation: "gelu" 或 "relu"
      group_same_n: 將相同 n 的 experts 用 batched bmm
    回傳：
      Y: [N, d]（與輸入 token 原順序對齊）
    """
    assert X.dim() == 2 and expert_idx.dim() == 1
    N, d = X.shape
    E, d_in, d_ff = W1.shape
    assert d == d_in and W2.shape == (E, d_ff, d)
    assert expert_idx.shape[0] == N and expert_idx.dtype in (torch.int64, torch.long)

    # 分組與前綴和
    counts = torch.bincount(expert_idx, minlength=E)            # [E]
    order  = torch.argsort(expert_idx)                          # [N]：同 expert 連續
    off    = torch.zeros(E + 1, device=X.device, dtype=torch.int32)
    off[1:] = torch.cumsum(counts, dim=0)

    Xg = X[order]                                               # [N, d]
    Yg = Xg.new_empty((N, d))
    act = torch.nn.functional.gelu if activation == "gelu" else torch.nn.functional.relu

    # 以 "相同 n" 分組
    groups: dict[int, list[int]] = defaultdict(list)
    for e in range(E):
        ne = int(counts[e].item())
        if ne > 0:
            groups[ne].append(e)

    for ne, es in groups.items():
        if (not group_same_n) or len(es) == 1:
            for e in es:
                s = int(off[e].item())
                Xe = Xg[s:s+ne]                      # [ne, d]
                He = Xe @ W1[e]                      # [ne, d_ff]
                He = act(He)
                Ye = He @ W2[e]                      # [ne, d]
                Yg[s:s+ne] = Ye
        else:
            G  = len(es)
            Xb = torch.stack([Xg[int(off[e].item()):int(off[e].item())+ne] for e in es], dim=0)  # [G, ne, d]
            W1b = torch.stack([W1[e] for e in es], dim=0)                                        # [G, d, d_ff]
            W2b = torch.stack([W2[e] for e in es], dim=0)                                        # [G, d_ff, d]
            Hb = torch.bmm(Xb, W1b)            # [G, ne, d_ff]
            Hb = act(Hb)
            Yb = torch.bmm(Hb, W2b)            # [G, ne, d]
            for i, e in enumerate(es):
                s = int(off[e].item())
                Yg[s:s+ne] = Yb[i]

    # 還原原順序
    Y = X.new_empty((N, d))
    Y[order] = Yg
    return Y


if __name__ == "__main__":
    import argparse
    
    # 從 benchmark_moe_padding.py 導入 Zipf 路由函數
    from benchmark_moe_padding import zipf_probs, sample_zipf_routing
    # 從 metrics_profiling.py 導入量測工具
    from metrics_profiling import Metrics, Profiling
    
    parser = argparse.ArgumentParser(description="測試 moe_forward_packed 函數")
    parser.add_argument("--N", type=int, default=2048, help="Token 數量")
    parser.add_argument("--d", type=int, default=1024, help="特徵維度")
    parser.add_argument("--dff", type=int, default=4096, help="FFN 隱藏層維度")
    parser.add_argument("--E", type=int, default=4, help="Expert 數量")
    parser.add_argument("--alpha", type=float, default=1.2, help="Zipf alpha (0=uniform, bigger=more skew)")
    parser.add_argument("--activation", type=str, default="gelu", choices=["gelu", "relu"], help="激活函數")
    parser.add_argument("--group_same_n", action="store_true", help="啟用相同 n 的 batched BMM")
    parser.add_argument("--seed", type=int, default=42, help="隨機種子")
    parser.add_argument("--repeats", type=int, default=100, help="重複測試次數")
    parser.add_argument("--profile", action="store_true", help="啟用詳細時間量測")
    
    args = parser.parse_args()
    
    # 設定隨機種子
    torch.manual_seed(args.seed)
    
    # 生成測試數據
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"生成測試數據: N={args.N}, d={args.d}, dff={args.dff}, E={args.E}, alpha={args.alpha}, device={device}")
    X = torch.randn(args.N, args.d, device=device, dtype=torch.float32)
    
    # 使用 Zipf 分佈生成 expert 索引
    expert_idx = sample_zipf_routing(args.N, args.E, args.alpha, device=device)
    
    # 生成權重矩陣
    W1 = torch.randn(args.E, args.d, args.dff, device=device, dtype=torch.float32)
    W2 = torch.randn(args.E, args.dff, args.d, device=device, dtype=torch.float32)
    
    # 顯示 Expert 分配
    counts = torch.bincount(expert_idx, minlength=args.E)
    print(f"Expert 分配:")
    for e in range(args.E):
        print(f"  Expert {e}: {int(counts[e])} tokens")
    
    if args.profile:
        # 建立量測工具
        device_str = "cuda" if device.type == "cuda" else "cpu"
        use_cuda_events = device.type == "cuda"
        
        # 總體量測
        metrics_total = Metrics()
        prof_total = Profiling(metrics_total, use_cuda_events=use_cuda_events)
        
        # Token 分組量測
        metrics_grouping = Metrics()
        prof_grouping = Profiling(metrics_grouping, use_cuda_events=use_cuda_events)
        
        # 矩陣乘法量測
        metrics_matmul = Metrics()
        prof_matmul = Profiling(metrics_matmul, use_cuda_events=use_cuda_events)
        
        print(f"\n開始量測 ({args.repeats} 次重複)...")
        
        # 暖機
        for _ in range(3):
            _ = moe_forward_packed(X, expert_idx, W1, W2, activation=args.activation, group_same_n=args.group_same_n)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        # 執行量測
        for i in range(args.repeats):
            
            # 總體量測
            prof_total.start()
            Y = moe_forward_packed(X, expert_idx, W1, W2, activation=args.activation, group_same_n=args.group_same_n)
            prof_total.end()
            
            # Token 分組量測（模擬分組操作）
            prof_grouping.start()
            # 模擬分組操作：排序和計算偏移
            order = torch.argsort(expert_idx)
            counts_temp = torch.bincount(expert_idx, minlength=args.E)
            off_temp = torch.zeros(args.E + 1, device=device, dtype=torch.int32)
            off_temp[1:] = torch.cumsum(counts_temp, dim=0)
            X_temp = X[order]  # 包含數據重排
            if device.type == "cuda":
                torch.cuda.synchronize()
            prof_grouping.end()
            
            # 矩陣乘法量測（模擬核心計算）
            prof_matmul.start()
            # 模擬矩陣乘法：執行一些 GEMM 操作
            for e in range(args.E):
                ne = int(counts[e].item())
                if ne > 0:
                    s = int(off_temp[e].item())
                    Xe = X_temp[s:s+ne]
                    He = Xe @ W1[e]
                    He = torch.nn.functional.gelu(He) if args.activation == "gelu" else torch.nn.functional.relu(He)
                    Ye = He @ W2[e]
            if device.type == "cuda":
                torch.cuda.synchronize()
            prof_matmul.end()
            if (i + 1) % max(1, args.repeats // 10) == 0:
                print(f"完成 {i+1}/{args.repeats} 次...")
        
        # 輸出量測結果
        print("\n=== 量測結果 ===")
        print("總體執行時間:")
        total_summary = metrics_total.summary()
        if "latency" in total_summary:
            lat = total_summary["latency"]
            print(f"  平均: {lat.get('mean_ms', 0):.3f} ms")
            print(f"  P50:  {lat.get('p50_ms', 0):.3f} ms")
            print(f"  P90:  {lat.get('p90_ms', 0):.3f} ms")
            print(f"  P99:  {lat.get('p99_ms', 0):.3f} ms")
        
        print("\nToken 分組時間:")
        grouping_summary = metrics_grouping.summary()
        if "latency" in grouping_summary:
            lat = grouping_summary["latency"]
            print(f"  平均: {lat.get('mean_ms', 0):.3f} ms")
            print(f"  P50:  {lat.get('p50_ms', 0):.3f} ms")
        
        print("\n矩陣乘法時間:")
        matmul_summary = metrics_matmul.summary()
        if "latency" in matmul_summary:
            lat = matmul_summary["latency"]
            print(f"  平均: {lat.get('mean_ms', 0):.3f} ms")
            print(f"  P50:  {lat.get('p50_ms', 0):.3f} ms")
        
        # 計算比例
        if "latency" in total_summary and "latency" in grouping_summary and "latency" in matmul_summary:
            total_mean = total_summary["latency"].get('mean_ms', 0)
            grouping_mean = grouping_summary["latency"].get('mean_ms', 0)
            matmul_mean = matmul_summary["latency"].get('mean_ms', 0)
            
            if total_mean > 0:
                print(f"\n時間比例:")
                print(f"  Token 分組: {grouping_mean/total_mean*100:.1f}%")
                print(f"  矩陣乘法:   {matmul_mean/total_mean*100:.1f}%")
                print(f"  其他開銷:   {(total_mean-grouping_mean-matmul_mean)/total_mean*100:.1f}%")
    
    else:
        # 簡單執行（無量測）
        print(f"\n執行 moe_forward_packed (activation={args.activation}, group_same_n={args.group_same_n})...")
        Y = moe_forward_packed(
            X, expert_idx, W1, W2,
            activation=args.activation,
            group_same_n=args.group_same_n
        )
        
        # 顯示結果
        print(f"輸出形狀: {Y.shape}")
        print(f"輸出統計: mean={Y.mean():.6f}, std={Y.std():.6f}")
    
    print("\n測試完成！")