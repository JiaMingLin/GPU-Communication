#!/usr/bin/env python3
"""
Minimal, self-contained experiment: sequential experts vs. batched (grouped) experts
with padding/dropping controlled by a capacity factor.

- Framework: PyTorch only
- Device: works on CPU/GPU (use CUDA on Jetson Orin for real measurements)

Steps mapped to the user's plan:
(1) Generate an imbalanced routing (expert assignment per token)
(2) Group token embeddings by expert
(3) Sequential: compute per expert one-by-one GEMM
(4) Grouped: apply capacity factor -> drop/pad -> batched GEMM via torch.bmm
(5) Compare latency & memory consumption (and record wasted MACs / metadata-free stats)

Usage example (GPU):
  python moe_batch_vs_seq.py --device cuda --B 4 --S 256 --H 1024 --F 4096 \
      --E 32 --cf 1.25 --imbalance zipf --zipf-s 1.6 --runs 50

CSV output is written to ./results/batch_vs_seq.csv
"""

import argparse
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import matplotlib.pyplot as plt
import numpy as np

torch.set_float32_matmul_precision("high")  # TF32 on Ampere if float32 is used
# Prefer Tensor Cores when possible
import torch.backends.cuda as cuda_backends
cuda_backends.matmul.allow_tf32 = True
cuda_backends.matmul.allow_fp16_reduced_precision_reduction = True


# ----------------------------- utils -----------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def pretty_size(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.1f}{unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f}PB"


def plot_expert_loading(counts: torch.Tensor, title: str = "Expert Loading Distribution", 
                       save_path: str = None):
    """繪製每個 expert 的 loading 長條圖"""
    counts_cpu = counts.detach().cpu().numpy()
    expert_ids = np.arange(len(counts_cpu))
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(expert_ids, counts_cpu, alpha=0.7, color='skyblue', edgecolor='navy')
    
    # 添加數值標籤在長條上方
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(counts_cpu[i])}', ha='center', va='bottom', fontsize=8)
    
    plt.xlabel('Expert ID')
    plt.ylabel('Number of Tokens')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # 添加統計資訊
    mean_count = np.mean(counts_cpu)
    max_count = np.max(counts_cpu)
    min_count = np.min(counts_cpu)
    plt.axhline(y=mean_count, color='red', linestyle='--', alpha=0.7, 
                label=f'Mean: {mean_count:.1f}')
    
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Expert loading plot saved to: {save_path}")
    
    plt.show()


@dataclass
class Routing:
    expert_ids: torch.Tensor  # [N] int64
    counts: torch.Tensor      # [E]
    probs: torch.Tensor       # [E] probability used to sample


# ------------------------ routing generation ---------------------

def make_imbalanced_routing(B: int, S: int, E: int, *, kind: str = "zipf", zipf_s: float = 1.6,
                             dirichlet_alpha: float = 0.2, seed: int = 0, device="cpu") -> Routing:
    """Return top-1 routing assignments with an imbalanced expert distribution.

    kind="zipf": p_e ∝ 1 / rank^s  (then randomly permuted across experts)
    kind="dirichlet": p ~ Dirichlet(alpha=alpha * ones)
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    if kind == "zipf":
        ranks = torch.arange(1, E + 1, dtype=torch.float64)
        p = 1.0 / torch.pow(ranks, zipf_s)
        p = p / p.sum()
        # random permutation of experts so expert id isn't tied to rank
        perm = torch.randperm(E, generator=g)
        p = p[perm].to(torch.float64)
    elif kind == "dirichlet":
        alpha = torch.full((E,), dirichlet_alpha, dtype=torch.float64)
        p = torch.distributions.Dirichlet(alpha).sample(generator=g)
    else:
        raise ValueError(f"Unknown imbalance kind: {kind}")

    N = B * S
    expert_ids = torch.multinomial(p, num_samples=N, replacement=True, generator=g)
    counts = torch.bincount(expert_ids, minlength=E)
    return Routing(expert_ids=expert_ids.to(torch.long).to(device),
                   counts=counts.to(device), probs=p.to(device))


# ---------------------- model / data creation --------------------

def make_inputs(B: int, S: int, H: int, dtype, device):
    N = B * S
    x = torch.randn(N, H, device=device, dtype=dtype)
    return x


def make_expert_weights(E: int, H: int, F: int, dtype, device):
    # Each expert has its own [H, F] matrix
    W = torch.randn(E, H, F, device=device, dtype=dtype)
    return W


# ---------------------- grouping utilities ----------------------

def group_indices_by_expert(expert_ids: torch.Tensor, E: int):
    idx_list = []
    for e in range(E):
        idx = (expert_ids == e).nonzero(as_tuple=False).flatten()
        idx_list.append(idx)
    return idx_list


def make_capacity(B: int, S: int, E: int, capacity_factor: float) -> int:
    avg = math.ceil((B * S) / E)
    return int(math.ceil(capacity_factor * avg))


# ---------------------- compute paths ---------------------------

def sequential_path(x: torch.Tensor, W: torch.Tensor, expert_ids: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
    """One-by-one GEMM per expert (no padding/dropping)."""
    E = W.shape[0]
    H, F = W.shape[1], W.shape[2]
    out = []
    device = x.device
    
    # 測量 grouping 時間
    if device.type == "cuda":
        torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
    else:
        start_time = time.perf_counter()
    
    # 先將 token 按 expert 分組，就像 batch GEMM 那邊的處理方式
    idx_list = group_indices_by_expert(expert_ids, E)
    
    # 結束測量 grouping 時間
    if device.type == "cuda":
        end_time.record()
        torch.cuda.synchronize()
        grouping_time = start_time.elapsed_time(end_time)  # 毫秒
    else:
        end_time = time.perf_counter()
        grouping_time = (end_time - start_time) * 1000  # 轉換為毫秒
    
    for e in range(E):
        idx = idx_list[e]
        if idx.numel() == 0:
            continue
        xe = x.index_select(0, idx)  # [n_e, H]
        ye = xe @ W[e]               # [n_e, F]
        out.append(ye)
    if len(out) == 0:
        return torch.empty(0, F, device=x.device, dtype=x.dtype), {"grouping_time_ms": float(grouping_time)}
    return torch.cat(out, dim=0), {"grouping_time_ms": float(grouping_time)}


def grouped_path(x: torch.Tensor, W: torch.Tensor, expert_ids: torch.Tensor, *,
                 capacity_factor: float, drop_policy: str = "tail", pad_value: float = 0.0):
    """Group-by expert, apply capacity (drop/pad) so all token matrices share shape [cap, H],
    then run batched GEMM via torch.bmm.

    drop_policy: 'tail' (stable: keep first cap tokens seen per expert), 'random'.
    Returns output tensor [E*cap, F] (order: expert-major then within-cap order),
    and accounting dict with padding_ratio, dropped_tokens, wasted_macs etc.
    """
    device, dtype = x.device, x.dtype
    E, H, F = W.shape
    cap = make_capacity(B=1, S=x.shape[0], E=E, capacity_factor=capacity_factor)  # uses total tokens

    # build [E, cap, H]
    tokens = torch.full((E, cap, H), pad_value, device=device, dtype=dtype)
    dropped = 0
    padded = 0
    kept = 0

    # 測量 grouping 時間
    if device.type == "cuda":
        torch.cuda.synchronize()
        grouping_start = torch.cuda.Event(enable_timing=True)
        grouping_end = torch.cuda.Event(enable_timing=True)
        grouping_start.record()
    else:
        grouping_start = time.perf_counter()

    # indices grouped once
    idx_list = group_indices_by_expert(expert_ids, E)

    # 結束測量 grouping 時間
    if device.type == "cuda":
        grouping_end.record()
        torch.cuda.synchronize()
        grouping_time = grouping_start.elapsed_time(grouping_end)  # 毫秒
    else:
        grouping_end = time.perf_counter()
        grouping_time = (grouping_end - grouping_start) * 1000  # 轉換為毫秒

    # 測量 padding/dropping 時間
    if device.type == "cuda":
        torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
    else:
        start_time = time.perf_counter()

    for e in range(E):
        idx = idx_list[e]
        n = idx.numel()
        if n == 0:
            padded += cap
            continue
        if n > cap:
            if drop_policy == "random":
                sel = idx[torch.randperm(n, device=device)[:cap]]
            else:  # tail -> take first seen
                sel = idx[:cap]
            xe = x.index_select(0, sel)
            dropped += (n - cap)
            kept += cap
            tokens[e, :, :] = xe
        else:
            # pad to cap
            xe = x.index_select(0, idx)
            tokens[e, :n, :] = xe
            padded += (cap - n)
            kept += n

    # 結束測量 padding/dropping 時間
    if device.type == "cuda":
        end_time.record()
        torch.cuda.synchronize()
        padding_time = start_time.elapsed_time(end_time)  # 毫秒
    else:
        end_time = time.perf_counter()
        padding_time = (end_time - start_time) * 1000  # 轉換為毫秒

    # Batched GEMM: [E, cap, H] x [E, H, F] -> [E, cap, F]
    y = torch.bmm(tokens, W)

    # metrics
    total = kept + padded  # == E*cap
    padding_ratio = padded / float(total) if total else 0.0
    wasted_macs = padded * H * F  # MACs spent on padded rows
    # Flatten to [E*cap, F]
    return y.reshape(E * cap, F), {
        "cap": cap,
        "kept_tokens": int(kept),
        "padded_tokens": int(padded),
        "dropped_tokens": int(dropped),
        "padding_ratio": float(padding_ratio),
        "wasted_macs": int(wasted_macs),
        "padding_time_ms": float(padding_time),
        "grouping_time_ms": float(grouping_time),
    }


# ----------------------- benchmarking ---------------------------

def benchmark(fn, runs: int = 50, synchronize: bool = True, device="cuda") -> Tuple[float, float]:
    """Return (avg_ms, p95_ms)."""
    # warmup
    for _ in range(min(10, max(2, runs // 5))):
        _ = fn()
    if synchronize and device.startswith("cuda"):
        torch.cuda.synchronize()

    times = []
    if device.startswith("cuda"):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(runs):
            start.record()
            _ = fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))  # ms
    else:
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = fn()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1e3)

    times.sort()
    avg = sum(times) / len(times)
    p95 = times[int(0.95 * (len(times) - 1))]
    return avg, p95


def benchmark_with_padding(fn, runs: int = 50, synchronize: bool = True, device="cuda") -> Tuple[float, float, float, float]:
    """Return (avg_ms, p95_ms, padding_avg_ms, padding_p95_ms)."""
    # warmup
    for _ in range(min(10, max(2, runs // 5))):
        _ = fn()
    if synchronize and device.startswith("cuda"):
        torch.cuda.synchronize()

    times = []
    padding_times = []
    
    if device.startswith("cuda"):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(runs):
            start.record()
            result = fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))  # ms
            
            # 從結果中提取 padding 時間
            if isinstance(result, tuple) and len(result) == 2:
                _, stats = result
                if "padding_time_ms" in stats:
                    padding_times.append(stats["padding_time_ms"])
                else:
                    padding_times.append(0.0)
            else:
                padding_times.append(0.0)
    else:
        for _ in range(runs):
            t0 = time.perf_counter()
            result = fn()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1e3)
            
            # 從結果中提取 padding 時間
            if isinstance(result, tuple) and len(result) == 2:
                _, stats = result
                if "padding_time_ms" in stats:
                    padding_times.append(stats["padding_time_ms"])
                else:
                    padding_times.append(0.0)
            else:
                padding_times.append(0.0)

    times.sort()
    padding_times.sort()
    
    avg = sum(times) / len(times)
    p95 = times[int(0.95 * (len(times) - 1))]
    padding_avg = sum(padding_times) / len(padding_times)
    padding_p95 = padding_times[int(0.95 * (len(padding_times) - 1))]
    
    return avg, p95, padding_avg, padding_p95


def benchmark_with_timing(fn, runs: int = 50, synchronize: bool = True, device="cuda") -> Tuple[float, float, float, float, float, float]:
    """Return (avg_ms, p95_ms, padding_avg_ms, padding_p95_ms, grouping_avg_ms, grouping_p95_ms)."""
    # warmup
    for _ in range(min(10, max(2, runs // 5))):
        _ = fn()
    if synchronize and device.startswith("cuda"):
        torch.cuda.synchronize()

    times = []
    padding_times = []
    grouping_times = []
    
    if device.startswith("cuda"):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(runs):
            start.record()
            result = fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))  # ms
            
            # 從結果中提取時間資訊
            if isinstance(result, tuple) and len(result) == 2:
                _, stats = result
                padding_times.append(stats.get("padding_time_ms", 0.0))
                grouping_times.append(stats.get("grouping_time_ms", 0.0))
            else:
                padding_times.append(0.0)
                grouping_times.append(0.0)
    else:
        for _ in range(runs):
            t0 = time.perf_counter()
            result = fn()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1e3)
            
            # 從結果中提取時間資訊
            if isinstance(result, tuple) and len(result) == 2:
                _, stats = result
                padding_times.append(stats.get("padding_time_ms", 0.0))
                grouping_times.append(stats.get("grouping_time_ms", 0.0))
            else:
                padding_times.append(0.0)
                grouping_times.append(0.0)

    times.sort()
    padding_times.sort()
    grouping_times.sort()
    
    avg = sum(times) / len(times)
    p95 = times[int(0.95 * (len(times) - 1))]
    padding_avg = sum(padding_times) / len(padding_times)
    padding_p95 = padding_times[int(0.95 * (len(padding_times) - 1))]
    grouping_avg = sum(grouping_times) / len(grouping_times)
    grouping_p95 = grouping_times[int(0.95 * (len(grouping_times) - 1))]
    
    return avg, p95, padding_avg, padding_p95, grouping_avg, grouping_p95


def measure_peak_mem(callable_fn, device: str) -> int:
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        _ = callable_fn()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        return int(peak - before)
    else:
        # Not available for CPU; return 0
        _ = callable_fn()
        return 0


# ----------------------------- main -----------------------------

def main():
    p = argparse.ArgumentParser(description="Sequential vs Batched Experts (Padding)")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])  
    p.add_argument("--B", type=int, default=4)
    p.add_argument("--S", type=int, default=256)
    p.add_argument("--H", type=int, default=1024)
    p.add_argument("--F", type=int, default=4096)
    p.add_argument("--E", type=int, default=32)
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])  
    p.add_argument("--imbalance", type=str, default="zipf", choices=["zipf", "dirichlet"])  
    p.add_argument("--zipf-s", type=float, default=1.6)
    p.add_argument("--dirichlet-alpha", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cf", type=float, default=1.25, help="capacity factor for grouped path")
    p.add_argument("--drop-policy", type=str, default="tail", choices=["tail", "random"])  
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--out", type=str, default="results/batch_vs_seq.csv")
    args = p.parse_args()

    # ---- enforce GPU if requested ----
    use_cuda = (args.device == "cuda")
    if use_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
        torch.cuda.init()
        print(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    device = "cuda" if use_cuda else "cpu"
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    ensure_dir(os.path.dirname(args.out))

    # Create data & model
    x = make_inputs(args.B, args.S, args.H, dtype=dtype, device=device)
    W = make_expert_weights(args.E, args.H, args.F, dtype=dtype, device=device)

    # Routing
    routing = make_imbalanced_routing(args.B, args.S, args.E, kind=args.imbalance,
                                      zipf_s=args.zipf_s, dirichlet_alpha=args.dirichlet_alpha,
                                      seed=args.seed, device=device)

    # Print quick stats
    print(f"x.device={x.device}, W.device={W.device}, dtype={dtype}")
    total_tokens = args.B * args.S
    probs_cpu = routing.probs.detach().to("cpu")
    counts_cpu = routing.counts.detach().to("cpu")
    gini = 1.0 - (probs_cpu.sort()[0].cumsum(0) * 2 - probs_cpu).sum().item() / args.E
    print("== Routing summary ==")
    print(f"tokens={total_tokens}, experts={args.E}, imbalance kind={args.imbalance}")
    print(f"counts: min={int(counts_cpu.min())}, max={int(counts_cpu.max())}, mean={float(counts_cpu.float().mean()):.2f}")
    print(f"capacity_factor(cf)={args.cf:.2f}; zipf_s={args.zipf_s:.2f} dirichlet_alpha={args.dirichlet_alpha:.2f}")
    print(f"approx-gini(prob)={gini:.3f}")

    # 繪製 expert loading 長條圖
    plot_title = f"Expert Loading Distribution\nB={args.B}, S={args.S}, E={args.E}, imbalance={args.imbalance}"
    plot_save_path = f"results/expert_loading_B{args.B}_S{args.S}_E{args.E}_{args.imbalance}.png"
    plot_expert_loading(routing.counts, title=plot_title, save_path=plot_save_path)

    # -------- sequential path --------
    def seq_call():
        return sequential_path(x, W, routing.expert_ids)

    seq_mem = measure_peak_mem(seq_call, args.device)
    seq_avg, seq_p95, seq_padding_avg, seq_padding_p95, seq_grouping_avg, seq_grouping_p95 = benchmark_with_timing(seq_call, runs=args.runs, device=args.device)

    # -------- grouped (batched) path --------
    def grouped_call():
        y, stats = grouped_path(x, W, routing.expert_ids, capacity_factor=args.cf, drop_policy=args.drop_policy)
        return y, stats

    # run once to get accounting numbers
    y_sample, group_stats = grouped_path(x, W, routing.expert_ids, capacity_factor=args.cf, drop_policy=args.drop_policy)
    group_mem = measure_peak_mem(lambda: grouped_call()[0], args.device)
    group_avg, group_p95, group_padding_avg, group_padding_p95, group_grouping_avg, group_grouping_p95 = benchmark_with_timing(grouped_call, runs=args.runs, device=args.device)

    # Derived metrics
    macs_per_token = args.H * args.F
    true_macs = int((total_tokens - group_stats["padded_tokens"]) * macs_per_token)
    wasted_macs = int(group_stats["wasted_macs"])
    padding_ratio = group_stats["padding_ratio"]

    # Print summary
    from csv import writer as csv_writer
    print("")
    print(f"Sequential:   avg={seq_avg:.3f}, p95={seq_p95:.3f}, peak_mem={pretty_size(seq_mem)}")
    print(f"  Padding:    avg={seq_padding_avg:.3f}, p95={seq_padding_p95:.3f} ms")
    print(f"  Grouping:   avg={seq_grouping_avg:.3f}, p95={seq_grouping_p95:.3f} ms")
    print(f"Grouped/PAD:  avg={group_avg:.3f}, p95={group_p95:.3f}, peak_mem={pretty_size(group_mem)}")
    print(f"  Padding:    avg={group_padding_avg:.3f}, p95={group_padding_p95:.3f} ms")
    print(f"  Grouping:   avg={group_grouping_avg:.3f}, p95={group_grouping_p95:.3f} ms")
    print(f"cap={group_stats['cap']}, kept={group_stats['kept_tokens']}, padded={group_stats['padded_tokens']}, dropped={group_stats['dropped_tokens']}")
    print(f"padding_ratio={padding_ratio:.3f}, wasted_MACs={wasted_macs/1e9:.3f} G, true_MACs={true_macs/1e9:.3f} G")

    # write CSV
    header = [
        "device", "dtype", "B", "S", "H", "F", "E", "cf", "imbalance", "zipf_s", "dirichlet_alpha",
        "tokens", "cap", "kept", "padded", "dropped", "padding_ratio",
        "seq_avg_ms", "seq_p95_ms", "seq_peak_mem_bytes", "seq_padding_avg_ms", "seq_padding_p95_ms", "seq_grouping_avg_ms", "seq_grouping_p95_ms",
        "group_avg_ms", "group_p95_ms", "group_peak_mem_bytes", "group_padding_avg_ms", "group_padding_p95_ms", "group_grouping_avg_ms", "group_grouping_p95_ms",
        "wasted_MACs", "true_MACs"
    ]
    row = [
        args.device, args.dtype, args.B, args.S, args.H, args.F, args.E, args.cf, args.imbalance, args.zipf_s, args.dirichlet_alpha,
        total_tokens, group_stats['cap'], group_stats['kept_tokens'], group_stats['padded_tokens'], group_stats['dropped_tokens'], padding_ratio,
        seq_avg, seq_p95, seq_mem, seq_padding_avg, seq_padding_p95, seq_grouping_avg, seq_grouping_p95,
        group_avg, group_p95, group_mem, group_padding_avg, group_padding_p95, group_grouping_avg, group_grouping_p95,
        wasted_macs, true_macs
    ]
    with open(args.out, "a", newline="") as f:
        w = csv_writer(f)
        if f.tell() == 0:
            w.writerow(header)
        w.writerow(row)


if __name__ == "__main__":
    main()
