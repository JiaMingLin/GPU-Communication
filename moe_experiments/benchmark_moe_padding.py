#!/usr/bin/env python3
# benchmark_moe_padding.py
from __future__ import annotations
import argparse, math, json
from typing import Tuple
import torch

from metrics_profiling import Metrics, Profiling
from moe_forward_packed import moe_forward_packed
from moe_forward_tilepad import moe_forward_tilepad
from moe_forward_capacitypad import moe_forward_capacitypad

# ---------- helpers ----------
def set_matmul_flags():
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

def tile_from_dtype(dtype: torch.dtype) -> int:
    if dtype in (torch.float16, torch.bfloat16):
        return 8
    if dtype == torch.int8:
        return 16
    return 8

def zipf_probs(E: int, alpha: float, device="cpu") -> torch.Tensor:
    # p_r ∝ r^{-alpha}; alpha=0 → 均勻
    ranks = torch.arange(1, E + 1, dtype=torch.float64, device=device)
    w = ranks.pow(-float(alpha))
    p = w / w.sum()
    return p

def sample_zipf_routing(N: int, E: int, alpha: float, device) -> torch.Tensor:
    # 先以 Zipf 機率在 rank 空間取樣，再亂數打散 rank→expert 映射，避免固定頭部 expert
    p = zipf_probs(E, alpha, device="cpu")  # 用 CPU 產生機率，之後放回 device
    idx_ranks = torch.multinomial(p, N, replacement=True)   # [N] on CPU
    perm = torch.randperm(E)                                # rank→expert 對應
    expert_idx = perm[idx_ranks].to(device=device, dtype=torch.long)
    return expert_idx

def round_up(x: int, tile: int) -> int:
    return ((x + tile - 1) // tile) * tile

# 統計 padding/drops（以「實際參與計算的有效行數」為 real_rows）
def stats_packed(n: torch.Tensor) -> Tuple[int, int, int]:
    # n: [E] 每 expert token 數
    real_rows = int(n.sum().item())
    padded_rows = real_rows
    drops = 0
    return real_rows, padded_rows, drops

def stats_tilepad(n: torch.Tensor, tile: int) -> Tuple[int, int, int]:
    real_rows = int(n.sum().item())
    m = [(round_up(int(ne.item()), tile) if int(ne.item()) > 0 else 0) for ne in n]
    padded_rows = int(sum(m))
    drops = 0
    return real_rows, padded_rows, drops

def stats_capacity(n: torch.Tensor, cap: int) -> Tuple[int, int, int]:
    n_keep = torch.clamp(n, max=cap)
    real_rows = int(n_keep.sum().item())     # 真正參與計算的行數（超 cap 的算 drop）
    padded_rows = int(n.shape[0] * cap)      # [E, cap, d] 的固定計算量
    drops = int((n - n_keep).clamp_min(0).sum().item())
    return real_rows, padded_rows, drops

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Benchmark MoE padding strategies with Zipf routing.")
    ap.add_argument("--N", type=int, default=1024, help="number of tokens")
    ap.add_argument("--E", type=int, default=16, help="number of experts")
    ap.add_argument("--d", type=int, default=1024, help="model dim")
    ap.add_argument("--dff", type=int, default=4096, help="FFN hidden dim")
    ap.add_argument("--alpha", type=float, default=1.2, help="Zipf alpha (0=uniform, bigger=more skew)")
    ap.add_argument("--cap", type=int, default=0, help="expert capacity (rows per expert). 0→auto from capacity_factor")
    ap.add_argument("--capacity_factor", type=float, default=1.0, help="used when --cap=0: cap=ceil(cf*ceil(N/E))")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda","cpu"])
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16","bf16","fp32"])
    ap.add_argument("--tile", type=int, default=0, help="tile size for tile-padding; 0→auto by dtype")
    ap.add_argument("--repeats", type=int, default=50, help="number of timed repeats (after warmup)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
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

    set_matmul_flags()

    # 構建資料與權重
    N, E, d, dff = args.N, args.E, args.d, args.dff
    X  = torch.randn(N, d, device=dev, dtype=dtype)
    W1 = torch.randn(E, d, dff, device=dev, dtype=dtype)
    W2 = torch.randn(E, dff, d, device=dev, dtype=dtype)

    # 生成 Zipf 路由（同一份供三種方法）
    expert_idx = sample_zipf_routing(N, E, args.alpha, device=dev)
    n = torch.bincount(expert_idx, minlength=E)  # [E]

    # 參數準備
    tile = args.tile if args.tile > 0 else tile_from_dtype(dtype)
    cap  = args.cap if args.cap > 0 else int(math.ceil(args.capacity_factor * math.ceil(N / max(E,1))))

    print("== Workload ==")
    print(json.dumps({
        "N": N, "E": E, "d": d, "dff": dff,
        "alpha": args.alpha, "cap": cap, "tile": tile,
        "device": str(dev), "dtype": str(dtype).replace("torch.",""),
        "routing": {
            "max_n": int(n.max().item()),
            "min_n": int(n.min().item()),
            "mean_n": float(n.float().mean().item()),
        }
    }, indent=2, ensure_ascii=False))

    # 建立 Metrics / Profilers
    m_packed = Metrics();     p_packed = Profiling(metrics=m_packed, use_cuda_events=(args.device=="cuda"))
    m_tile   = Metrics();     p_tile   = Profiling(metrics=m_tile,   use_cuda_events=(args.device=="cuda"))
    m_cap    = Metrics();     p_cap    = Profiling(metrics=m_cap,    use_cuda_events=(args.device=="cuda"))

    # 暖機（不記錄）
    for _ in range(5):
        _ = moe_forward_packed(X, expert_idx, W1, W2)
        _ = moe_forward_tilepad(X, expert_idx, W1, W2, tile=tile)
        _ = moe_forward_capacitypad(X, expert_idx, W1, W2, cap=cap)

    torch.cuda.synchronize() if args.device=="cuda" else None

    # --------- 測試 Packed ---------
    real_rows, padded_rows, drops = stats_packed(n)
    for _ in range(args.repeats):
        p_packed.start()
        _ = moe_forward_packed(X, expert_idx, W1, W2)
        p_packed.end(padding_real_rows=real_rows, padding_padded_rows=padded_rows, drops=drops)

    # --------- 測試 Tile-Padding ---------
    real_rows_t, padded_rows_t, drops_t = stats_tilepad(n, tile)
    for _ in range(args.repeats):
        p_tile.start()
        _ = moe_forward_tilepad(X, expert_idx, W1, W2, tile=tile)
        p_tile.end(padding_real_rows=real_rows_t, padding_padded_rows=padded_rows_t, drops=drops_t)

    # --------- 測試 Capacity-Padding ---------
    real_rows_c, padded_rows_c, drops_c = stats_capacity(n, cap)
    for _ in range(args.repeats):
        p_cap.start()
        _ = moe_forward_capacitypad(X, expert_idx, W1, W2, cap=cap)
        p_cap.end(padding_real_rows=real_rows_c, padding_padded_rows=padded_rows_c, drops=drops_c)

    # 輸出摘要
    print("\n== Summary: Packed ==")
    print(json.dumps(m_packed.summary(), indent=2, ensure_ascii=False))

    print("\n== Summary: Tile-Padding ==")
    print(json.dumps(m_tile.summary(), indent=2, ensure_ascii=False))

    print("\n== Summary: Capacity-Padding ==")
    print(json.dumps(m_cap.summary(), indent=2, ensure_ascii=False))

    # 方便複製到表格
    def flat(d):
        lat = d.get("latency", {})
        pad = d.get("padding", {})
        return {
            "count": lat.get("count", 0),
            "p50_ms": lat.get("p50_ms", None),
            "p90_ms": lat.get("p90_ms", None),
            "p99_ms": lat.get("p99_ms", None),
            "mean_ms": lat.get("mean_ms", None),
            "pad_added_rows": pad.get("added_rows", 0),
            "pad_overhead_ratio": pad.get("overhead_ratio", None),
            "drops_total": pad.get("drops_total", 0),
        }

    row = {
        "packed": flat(m_packed.summary()),
        "tilepad": flat(m_tile.summary()),
        "capacity": flat(m_cap.summary()),
    }
    print("\n== Flat Table ==")
    print(json.dumps(row, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()