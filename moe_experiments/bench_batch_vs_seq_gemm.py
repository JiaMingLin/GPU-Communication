#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch GEMM vs Sequential GEMM latency/throughput benchmark (PyTorch)
可在 RTX 3060 與 Jetson Xavier NX 執行

功能：
- 逐筆 Sequential GEMM:  loop over B 次 torch.mm
- 一次 Batched GEMM:    torch.bmm
- 計時模式：wall-clock（含框架/同步）或 CUDA event（kernel time）
- 支援 dtype: fp32 / fp16 / bf16（若裝置不支援會自動降級）
- 控制 TF32（Ampere 以上有效，Jetson NX 無效）
- 匯出 CSV

用法範例：
# RTX 3060（Ampere, 可開 TF32）
python bench_batch_vs_seq_gemm.py --device cuda --M 1024 --K 1024 --N 1024 \
  --batches 1 2 4 8 16 32 --dtype fp32 --allow-tf32 1 --matmul-prec high \
  --repeats 5 --warmup 2 --timing gpu --out result_rtx3060.csv

# Jetson Xavier NX（建議 fp16, 無 TF32）
python bench_batch_vs_seq_gemm.py --device cuda --M 512 --K 512 --N 512 \
  --batches 1 2 4 8 16 --dtype fp16 --repeats 5 --warmup 3 \
  --timing gpu --out result_jetson_nx.csv
"""
import argparse
import time
from typing import Dict, List

import torch
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Batch vs Sequential GEMM benchmark (PyTorch)")
    p.add_argument("--M", type=int, default=512, help="矩陣 M（A: BxMxK, C: BxMxN）")
    p.add_argument("--K", type=int, default=512, help="矩陣 K（A: BxMxK, B: BxKxN）")
    p.add_argument("--N", type=int, default=512, help="矩陣 N（B: BxKxN, C: BxMxN）")
    p.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16],
                   help="批次大小列表（B）")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                   help="裝置（預設 cuda，若無 GPU 會自動退回 cpu）")
    p.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"],
                   help="資料型別（依裝置支援自動降級）")
    p.add_argument("--seed", type=int, default=2025, help="隨機種子")
    p.add_argument("--warmup", type=int, default=2, help="每種方法預熱次數")
    p.add_argument("--repeats", type=int, default=5, help="正式量測重複次數（取最小值）")
    p.add_argument("--timing", type=str, default="gpu", choices=["gpu", "wall"],
                   help="gpu: CUDA events（kernel time）；wall: perf_counter + sync")
    p.add_argument("--out", type=str, default="batch_vs_seq_gemm.csv", help="輸出 CSV 路徑")
    p.add_argument("--allow-tf32", type=int, default=1, help="是否允許 TF32（Ampere+ 有效）")
    p.add_argument("--matmul-prec", type=str, default="high",
                   choices=["high", "medium", "highest"],
                   help="torch.set_float32_matmul_precision（Ampere+ 生效）")
    return p.parse_args()


def get_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def pick_dtype(device: torch.device, want: str) -> torch.dtype:
    """依裝置支援選 dtype，不支援就降級到 fp32。"""
    if want == "fp16":
        if device.type == "cuda":
            return torch.float16
        # CPU 上 fp16 mm 很慢/不穩定，降級
        print("[WARN] CPU 不建議 fp16，改用 fp32")
        return torch.float32
    if want == "bf16":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if device.type == "cpu" and torch.cpu.is_bf16_supported():
            return torch.bfloat16
        print("[WARN] 此裝置不支援 bfloat16，改用 fp32")
        return torch.float32
    return torch.float32


def config_tf32(allow_tf32: bool, matmul_prec: str):
    """控制 TF32（Ampere 有效；Jetson NX 無效）。"""
    try:
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision(matmul_prec)
    except Exception:
        pass


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def measure_gpu_kernel_ms(fn) -> float:
    """CUDA events 計時（毫秒）；需裝置為 cuda。"""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)  # ms


def measure_wall_ms(fn, device: torch.device) -> float:
    """wall-clock 計時（毫秒），含 synchronize。"""
    sync(device)
    t0 = time.perf_counter()
    fn()
    sync(device)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


@torch.inference_mode()
def run_one_case(M: int, K: int, N: int, B: int, device: torch.device,
                 dtype: torch.dtype, warmup: int, repeats: int,
                 timing: str) -> Dict:
    # 建立輸入（固定一組資料，兩種方法共用）
    A = torch.randn(B, M, K, device=device, dtype=dtype)
    Bm = torch.randn(B, K, N, device=device, dtype=dtype)
    # 逐筆 mm 預先配置 C，避免迴圈內 alloc 影響計時
    C_seq = torch.empty(B, M, N, device=device, dtype=dtype)

    def fn_seq():
        for i in range(B):
            C_seq[i] = torch.mm(A[i], Bm[i])

    def fn_bmm():
        _ = torch.bmm(A, Bm)

    # 預熱
    for _ in range(max(0, warmup)):
        fn_seq()
        fn_bmm()

    # 選擇計時器
    meas = (lambda f: measure_gpu_kernel_ms(f)) if (timing == "gpu" and device.type == "cuda") \
        else (lambda f: measure_wall_ms(f, device))

    # 多次取最小值，降低抖動
    seq_ms = min(meas(fn_seq) for _ in range(max(1, repeats)))
    bmm_ms = min(meas(fn_bmm) for _ in range(max(1, repeats)))

    # FLOPs：2*M*K*N per GEMM × B
    flops = 2.0 * M * K * N * B
    gflops_seq = flops / (seq_ms / 1e3) / 1e9
    gflops_bmm = flops / (bmm_ms / 1e3) / 1e9

    return dict(
        device=str(device),
        capability=(torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"),
        dtype=str(dtype).replace("torch.", ""),
        timing_mode=("cuda_events" if timing == "gpu" and device.type == "cuda" else "wall_clock"),
        M=M, K=K, N=N, B=B,
        sequential_time_ms=seq_ms,
        batched_time_ms=bmm_ms,
        avg_ms_per_gemm_seq=seq_ms / B,
        avg_ms_per_gemm_bmm=bmm_ms / B,
        sequential_gflops=gflops_seq,
        batched_gflops=gflops_bmm,
        speedup_seq_over_batched=(seq_ms / bmm_ms)
    )


def pretty_mem(M, K, N, B, dtype_bytes: int) -> float:
    """簡估 A+B+C 佔用（GiB），僅供規劃。"""
    total_elems = (B * M * K) + (B * K * N) + (B * M * N)
    return total_elems * dtype_bytes / (1024 ** 3)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = get_device(args.device)
    dtype = pick_dtype(device, args.dtype)
    if device.type == "cuda":
        config_tf32(args.allow_tf32, args.matmul_prec)

    # 顯示設定與記憶體估算
    bytes_per = {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2}[dtype]
    est_gib = pretty_mem(args.M, args.K, args.N, max(args.batches), bytes_per)
    print(f"[INFO] device={device}, dtype={dtype}, timing={args.timing}, "
          f"TF32={'on' if args.allow_tf32 else 'off'}")
    print(f"[INFO] Size: M={args.M}, K={args.K}, N={args.N}, B in {args.batches}")
    print(f"[INFO] Rough peak memory (A+B+C at B=max): ~{est_gib:.3f} GiB")

    results: List[Dict] = []
    for B in args.batches:
        stats = run_one_case(args.M, args.K, args.N, B, device, dtype,
                             warmup=args.warmup, repeats=args.repeats,
                             timing=args.timing)
        results.append(stats)
        sp = stats["speedup_seq_over_batched"]
        print(f"[DONE] B={B:>4d} | seq={stats['sequential_time_ms']:.3f} ms  "
              f"bmm={stats['batched_time_ms']:.3f} ms  speedup={sp:.3f}x")

    df = pd.DataFrame(results).sort_values(["B"]).reset_index(drop=True)
    df.to_csv(args.out, index=False)
    print(f"[SAVE] CSV -> {args.out}")
    try:
        # 友善顯示前幾行
        from tabulate import tabulate  # 若沒裝也沒關係
        print(tabulate(df.head(len(args.batches)), headers="keys", tablefmt="github", floatfmt=".4f"))
    except Exception:
        print(df)


if __name__ == "__main__":
    main()