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
    ap.add_argument("--N", type=int, default=8192)
    ap.add_argument("--E", type=int, default=16)
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
    X = torch.randn(N, d, device=dev, dtype=dtype)

    # Zipf routing（同一份）
    ranks = torch.arange(1, E+1, dtype=torch.float64)
    p = (ranks.pow(-float(args.alpha)) / ranks.pow(-float(args.alpha)).sum()).cpu()
    r_samp = torch.multinomial(p, N, replacement=True)  # CPU
    perm = torch.randperm(E)
    expert_idx = perm[r_samp].to(device=dev, dtype=torch.long)

    # 只呼叫一次 grouping，兩方法共用
    order, counts, off = make_grouping(expert_idx, E)

    # 先做一次輸出並驗證一致
    Xcap_loop_item = build_xcap_loop_with_item(X, E, cap, order, counts, off)
    Xcap_loop = build_xcap_loop(X, E, cap, order, counts, off)
    Xcap_vec  = build_xcap_vectorized(X, E, cap, order, counts, off)
    
    # 驗證三種方法輸出相同
    same_1 = torch.equal(Xcap_loop_item, Xcap_loop)
    same_2 = torch.equal(Xcap_loop, Xcap_vec)
    max_abs_1 = (Xcap_loop_item - Xcap_loop).abs().max().item()
    max_abs_2 = (Xcap_loop - Xcap_vec).abs().max().item()
    print(f"[Equal] loop_with_item vs loop: {same_1} | max_abs_diff = {max_abs_1:.3e}")
    print(f"[Equal] loop vs vectorized: {same_2} | max_abs_diff = {max_abs_2:.3e}")

    # 量測
    if use_cuda:
        avg_loop_item_ms, _ = time_cuda_ms(lambda: build_xcap_loop_with_item(X, E, cap, order, counts, off),
                                           warmup=args.warmup, repeats=args.repeats)
        avg_loop_ms, _ = time_cuda_ms(lambda: build_xcap_loop(X, E, cap, order, counts, off),
                                      warmup=args.warmup, repeats=args.repeats)
        avg_vec_ms,  _ = time_cuda_ms(lambda: build_xcap_vectorized(X, E, cap, order, counts, off),
                                      warmup=args.warmup, repeats=args.repeats)
    else:
        avg_loop_item_ms, _ = time_cpu_ms(lambda: build_xcap_loop_with_item(X, E, cap, order, counts, off),
                                           warmup=args.warmup, repeats=args.repeats)
        avg_loop_ms, _ = time_cpu_ms(lambda: build_xcap_loop(X, E, cap, order, counts, off),
                                     warmup=args.warmup, repeats=args.repeats)
        avg_vec_ms,  _ = time_cpu_ms(lambda: build_xcap_vectorized(X, E, cap, order, counts, off),
                                     warmup=args.warmup, repeats=args.repeats)

    print(f"[Setting] N={N}, E={E}, d={d}, cap={cap}, device={dev}, dtype={dtype}")
    print(f"[Timing] (warmup={args.warmup}, repeats={args.repeats})")
    print(f"  loop_with_item:  {avg_loop_item_ms:.3f} ms  (使用 .item()，會造成 CPU-GPU 同步)")
    print(f"  loop:            {avg_loop_ms:.3f} ms  (避免 .item()，先轉 CPU)")
    print(f"  vectorized:      {avg_vec_ms:.3f} ms  (向量化版本)")
    
    # 計算加速比
    if avg_loop_item_ms > 0:
        speedup_vs_item = avg_loop_item_ms / avg_loop_ms
        print(f"\n[Speedup] loop (避免同步) 相對 loop_with_item: {speedup_vs_item:.2f}x")
    if avg_loop_ms > 0:
        speedup_vec_vs_loop = avg_loop_ms / avg_vec_ms
        print(f"[Speedup] vectorized 相對 loop: {speedup_vec_vs_loop:.2f}x")

if __name__ == "__main__":
    main()