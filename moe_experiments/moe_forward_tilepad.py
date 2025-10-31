# moe_forward_tilepad.py
from __future__ import annotations
from collections import defaultdict
import torch

__all__ = ["moe_forward_tilepad"]

def _round_up(x: int, tile: int) -> int:
    return ((x + tile - 1) // tile) * tile

def _tile_from_dtype(dtype: torch.dtype) -> int:
    # 依常見 Tensor Core 對齊；實務可基於 GPU/庫微調
    if dtype in (torch.float16, torch.bfloat16):
        return 8
    if dtype == torch.int8:
        return 16
    return 8  # FP32/TF32 先用 8 作為保守值

@torch.inference_mode()
def moe_forward_tilepad(
    X: torch.Tensor,             # [N, d]
    expert_idx: torch.Tensor,    # [N] long
    W1: torch.Tensor,            # [E, d, d_ff]
    W2: torch.Tensor,            # [E, d_ff, d]
    activation: str = "gelu",
    group_same_n: bool = True,   # 為與 packed 保持一致之介面；此參數對 tilepad 無實質作用
    tile: int | None = None,
) -> torch.Tensor:
    """
    Tile-Padding MoE-MLP 前向：
      - 同 expert token 連續化
      - 每個 expert 的行數 m_e = round_up(n_e, tile)（n_e=0 直接跳過）
      - 以 "相同 m" 的 experts 做 batched bmm；最後只取前 n_e 行回填
    參數：
      X: [N, d]、expert_idx: [N]、W1: [E, d, d_ff]、W2: [E, d_ff, d]
      activation: "gelu" 或 "relu"
      tile: 若為 None 則依 dtype 自動選擇（FP16/BF16→8，INT8→16）
    回傳：
      Y: [N, d]（與輸入 token 原順序對齊）
    """
    assert X.dim() == 2 and expert_idx.dim() == 1
    N, d = X.shape
    E, d_in, d_ff = W1.shape
    assert d == d_in and W2.shape == (E, d_ff, d)
    assert expert_idx.shape[0] == N and expert_idx.dtype in (torch.int64, torch.long)

    tile = _tile_from_dtype(X.dtype) if tile is None else int(tile)

    # 分組與前綴和（n_e）
    n = torch.bincount(expert_idx, minlength=E)            # [E]
    order = torch.argsort(expert_idx)                      # [N]
    off_n = torch.zeros(E + 1, device=X.device, dtype=torch.int32)
    off_n[1:] = torch.cumsum(n, dim=0)

    Xg = X[order]                                          # [N, d]
    act = torch.nn.functional.gelu if activation == "gelu" else torch.nn.functional.relu

    # 決定 m_e（n_e=0 直接跳過，避免做無效 GEMM）
    m = torch.empty_like(n)
    for e in range(E):
        ne = int(n[e].item())
        m[e] = 0 if ne == 0 else _round_up(ne, tile)

    off_m = torch.zeros(E + 1, device=X.device, dtype=torch.int32)
    off_m[1:] = torch.cumsum(m, dim=0)
    M_pad = int(off_m[-1].item())

    # 為避免 .item() 造成同步，將 n, m, off_n, off_m 轉移到 CPU
    n_cpu = n.cpu().tolist()
    m_cpu = m.cpu().tolist()
    off_n_cpu = off_n.cpu().tolist()
    off_m_cpu = off_m.cpu().tolist()

    # 建立 padded 連續輸入
    Xpad = X.new_zeros((M_pad, d))
    for e in range(E):
        ne, me = n_cpu[e], m_cpu[e]
        if ne == 0:
            continue
        s_n, s_m = off_n_cpu[e], off_m_cpu[e]
        Xpad[s_m:s_m+ne] = Xg[s_n:s_n+ne]     # 後段維持 0 作為 padding

    # 依 "相同 m" 分組，做 batched bmm
    Ypad = X.new_zeros((M_pad, d))
    groups: dict[int, list[int]] = defaultdict(list)
    for e in range(E):
        me = m_cpu[e]
        if me > 0:
            groups[me].append(e)

    for mval, es in groups.items():
        G  = len(es)
        Xm  = torch.stack([Xpad[off_m_cpu[e]:off_m_cpu[e]+mval] for e in es], dim=0)  # [G, m, d]
        W1m = torch.stack([W1[e] for e in es], dim=0)                                 # [G, d, d_ff]
        W2m = torch.stack([W2[e] for e in es], dim=0)                                 # [G, d_ff, d]
        H   = torch.bmm(Xm, W1m)                # [G, m, d_ff]
        H   = act(H)
        Y   = torch.bmm(H, W2m)                 # [G, m, d]
        for i, e in enumerate(es):
            base = off_m_cpu[e]
            Ypad[base:base+mval] = Y[i]

    # 取回有效行並還原原順序
    Yg = X.new_empty((N, d))
    for e in range(E):
        ne = n_cpu[e]
        if ne == 0:
            continue
        s_n, s_m = off_n_cpu[e], off_m_cpu[e]
        Yg[s_n:s_n+ne] = Ypad[s_m:s_m+ne]

    Y = X.new_empty((N, d))
    Y[order] = Yg
    return Y


# ---------- 小測試（可直接執行） ----------
if __name__ == "__main__":
    # 從 benchmark_moe_padding.py 導入 Zipf 路由函數
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from benchmark_moe_padding import sample_zipf_routing
    from metrics_profiling import Metrics, Profiling
    
    import argparse
    import json
    import statistics
    # python moe_forward_tilepad.py --profile --repeats 100
    parser = argparse.ArgumentParser(description="測試 moe_forward_tilepad 函數")
    parser.add_argument("--N", type=int, default=1024, help="Token 數量")
    parser.add_argument("--E", type=int, default=16, help="Expert 數量")
    parser.add_argument("--d", type=int, default=1024, help="特徵維度")
    parser.add_argument("--dff", type=int, default=4096, help="FFN 隱藏層維度")
    parser.add_argument("--alpha", type=float, default=1.2, help="Zipf alpha (0=uniform, bigger=more skew)")
    parser.add_argument("--tile", type=int, default=None, help="Tile 大小（None 則依 dtype 自動選擇）")
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

    # 決定 tile
    tile = args.tile if args.tile is not None else _tile_from_dtype(dtype)
    print(f"Tile size: {tile}")
    
    if args.profile:
        # --- Tile Padding 統計（固定於本路由與參數下不變）---
        n_stat = counts.to(torch.int32)
        m_stat = torch.empty_like(n_stat)
        for e in range(E):
            ne = int(n_stat[e].item())
            m_stat[e] = 0 if ne == 0 else _round_up(ne, tile)
        
        total_real = int(n_stat.sum().item())
        total_padded = int(m_stat.sum().item())
        added_rows = total_padded - total_real

        print("\n== Tile Padding 統計（單次）==")
        print(f"  原始 tokens: {N}")
        print(f"  Real rows: {total_real}")
        print(f"  Padded rows: {total_padded}")
        print(f"  Padding 增加行數: {added_rows}")

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
            _ = moe_forward_tilepad(X, expert_idx, W1, W2, activation="gelu", tile=tile)
        
        if args.device == "cuda":
            torch.cuda.synchronize()
        
        print(f"\n== 開始 Profiling (重複 {args.repeats} 次) ==\n")
        
        # 記錄 bmm kernel 呼叫次數（groups 長度）
        bmm_calls_list = []
        
        # 模擬各步驟的 profiling
        for i in range(args.repeats):
            # --- 步驟 1: 分組與前綴和（n_e）---
            prof_grouping.start()
            n_group = torch.bincount(expert_idx, minlength=E)
            order_group = torch.argsort(expert_idx)
            off_n_group = torch.zeros(E + 1, device=dev, dtype=torch.int32)
            off_n_group[1:] = torch.cumsum(n_group, dim=0)
            Xg_group = X[order_group]
            act = torch.nn.functional.gelu
            
            # 決定 m_e
            m_group = torch.empty_like(n_group)
            for e in range(E):
                ne = int(n_group[e].item())
                m_group[e] = 0 if ne == 0 else _round_up(ne, tile)
            
            off_m_group = torch.zeros(E + 1, device=dev, dtype=torch.int32)
            off_m_group[1:] = torch.cumsum(m_group, dim=0)
            M_pad_group = int(off_m_group[-1].item())
            
            # 轉移到 CPU
            n_cpu_group = n_group.cpu().tolist()
            m_cpu_group = m_group.cpu().tolist()
            off_n_cpu_group = off_n_group.cpu().tolist()
            off_m_cpu_group = off_m_group.cpu().tolist()
            
            if args.device == "cuda":
                torch.cuda.synchronize()
            prof_grouping.end()
            
            # --- 步驟 2: 建立 padded 連續輸入 ---
            prof_padding.start()
            Xpad_group = X.new_zeros((M_pad_group, d))
            for e in range(E):
                ne, me = n_cpu_group[e], m_cpu_group[e]
                if ne == 0:
                    continue
                s_n, s_m = off_n_cpu_group[e], off_m_cpu_group[e]
                Xpad_group[s_m:s_m+ne] = Xg_group[s_n:s_n+ne]
            
            Ypad_group = X.new_zeros((M_pad_group, d))
            groups: dict[int, list[int]] = defaultdict(list)
            for e in range(E):
                me = m_cpu_group[e]
                if me > 0:
                    groups[me].append(e)
            if args.device == "cuda":
                torch.cuda.synchronize()
            prof_padding.end(
                padding_real_rows=total_real,
                padding_padded_rows=total_padded
            )
            # --- 步驟 3: 依 "相同 m" 分組，做 batched bmm ---
            prof_matmul.start()
            num_bmm_calls = len(groups)  # groups 的長度就是 bmm kernel 的呼叫次數
            for mval, es in groups.items():
                G  = len(es)
                Xm  = torch.stack([Xpad_group[off_m_cpu_group[e]:off_m_cpu_group[e]+mval] for e in es], dim=0)
                W1m = torch.stack([W1[e] for e in es], dim=0)
                W2m = torch.stack([W2[e] for e in es], dim=0)
                H   = torch.bmm(Xm, W1m)
                H   = act(H)
                Y   = torch.bmm(H, W2m)
                for i_idx, e in enumerate(es):
                    base = off_m_cpu_group[e]
                    Ypad_group[base:base+mval] = Y[i_idx]
            
            if args.device == "cuda":
                torch.cuda.synchronize()
            prof_matmul.end()
            
            # 記錄這次的 bmm kernel 呼叫次數
            bmm_calls_list.append(num_bmm_calls)
            
            # --- 步驟 4: 取回有效行並還原原順序 ---
            prof_scatter.start()
            Yg_group = X.new_empty((N, d))
            for e in range(E):
                ne = n_cpu_group[e]
                if ne == 0:
                    continue
                s_n, s_m = off_n_cpu_group[e], off_m_cpu_group[e]
                Yg_group[s_n:s_n+ne] = Ypad_group[s_m:s_m+ne]

            Y_final = X.new_empty((N, d))
            Y_final[order_group] = Yg_group
            
            if args.device == "cuda":
                torch.cuda.synchronize()
            prof_scatter.end()
            
            # 總體量測
            prof_total.start()
            Y_total = moe_forward_tilepad(X, expert_idx, W1, W2, activation="gelu", tile=tile)
            prof_total.end()
            
            if (i + 1) % max(1, args.repeats // 10) == 0:
                print(f"完成 {i+1}/{args.repeats} 次...")
        
        print("\n== Profiling 結果 ==\n")
        
        # 步驟 1: 分組與前綴和（n_e）
        grouping_summary = metrics_grouping.summary()
        print("(1) 分組與前綴和（n_e）:")
        if "latency" in grouping_summary:
            lat = grouping_summary["latency"]
            print(f"  平均: {lat.get('mean_ms', 0):.4f} ms")
            print(f"  P50:  {lat.get('p50_ms', 0):.4f} ms")
            print(f"  P90:  {lat.get('p90_ms', 0):.4f} ms")
            print(f"  P99:  {lat.get('p99_ms', 0):.4f} ms")
        
        # 步驟 2: 建立 padded 連續輸入
        padding_summary = metrics_padding.summary()
        print("\n(2) 建立 padded 連續輸入:")
        if "latency" in padding_summary:
            lat = padding_summary["latency"]
            print(f"  平均: {lat.get('mean_ms', 0):.4f} ms")
            print(f"  P50:  {lat.get('p50_ms', 0):.4f} ms")
            print(f"  P90:  {lat.get('p90_ms', 0):.4f} ms")
            print(f"  P99:  {lat.get('p99_ms', 0):.4f} ms")
        
        # 步驟 3: 依 "相同 m" 分組，做 batched bmm
        matmul_summary = metrics_matmul.summary()
        print("\n(3) 依 \"相同 m\" 分組，做 batched bmm:")
        if "latency" in matmul_summary:
            lat = matmul_summary["latency"]
            print(f"  平均: {lat.get('mean_ms', 0):.4f} ms")
            print(f"  P50:  {lat.get('p50_ms', 0):.4f} ms")
            print(f"  P90:  {lat.get('p90_ms', 0):.4f} ms")
            print(f"  P99:  {lat.get('p99_ms', 0):.4f} ms")
        
        # 顯示 bmm kernel 呼叫次數統計
        if bmm_calls_list:
            print(f"  BMM kernel 呼叫次數 (groups 長度):")
            print(f"    平均: {statistics.fmean(bmm_calls_list):.2f}")
            print(f"    最小: {min(bmm_calls_list)}")
            print(f"    最大: {max(bmm_calls_list)}")
            if len(set(bmm_calls_list)) == 1:
                print(f"    固定: {bmm_calls_list[0]} 次")
            else:
                print(f"    變動: {sorted(set(bmm_calls_list))}")
        
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
                print(f"  分組與前綴和（n_e）:          {grouping_mean/total_mean*100:.1f}%")
                print(f"  建立 padded 連續輸入:         {padding_mean/total_mean*100:.1f}%")
                print(f"  依 \"相同 m\" 分組，做 batched bmm: {matmul_mean/total_mean*100:.1f}%")
                print(f"  取回有效行並還原原順序:      {scatter_mean/total_mean*100:.1f}%")
                print(f"  其他開銷:                     {(total_mean-grouping_mean-padding_mean-matmul_mean-scatter_mean)/total_mean*100:.1f}%")
        
        # 輸出完整 JSON
        print("\n== 完整 JSON 結果 ==")
        json_output = {
            "grouping": grouping_summary,
            "padding": padding_summary,
            "matmul": matmul_summary,
            "scatter": scatter_summary,
            "total": total_summary
        }
        # 添加 bmm kernel 呼叫次數統計
        if bmm_calls_list:
            json_output["bmm_calls"] = {
                "mean": statistics.fmean(bmm_calls_list),
                "min": min(bmm_calls_list),
                "max": max(bmm_calls_list),
                "values": bmm_calls_list
            }
        print(json.dumps(json_output, indent=2, ensure_ascii=False))
    else:
        # 簡單執行（無 profiling）
        print("\n執行 moe_forward_tilepad...")
    Y = moe_forward_tilepad(X, expert_idx, W1, W2, activation="gelu", tile=tile)
    print(f"OK: 完成前向。輸出形狀: {Y.shape}")
    print(f"輸出統計: mean={Y.mean():.6f}, std={Y.std():.6f}")
    
    print("\n測試完成！")