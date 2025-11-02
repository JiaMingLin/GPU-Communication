#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch GEMM vs Sequential GEMM benchmark (latency + energy, with window extension)
- 支援 RTX 3060（NVML）與 Jetson Orin NX（tegrastats）
- 量測：Sequential (loop mm) vs Batched (bmm)
- 指標：時間(ms)、GFLOP/s、能耗(J)、平均/峰值功率、J/GFLOP
- 新增：自動把每次量測的區間拉長到 >= --min-energy-window-ms，避免短 kernel 被功率取樣漏掉

RTX 3060
$ pip install -U nvidia-ml-py3
$ nvidia-smi -pm 1
$ python bench_batch_vs_seq_gemm.py --device cuda --M 1024 --K 1024 --N 1024 \
    --batches 1 2 4 8 16 32 --dtype fp32 --allow-tf32 1 --matmul-prec high \
    --repeats 5 --warmup 2 --timing gpu --energy auto \
    --energy-interval-ms 10 --min-energy-window-ms 1000 \
    --out result_rtx3060.csv

Jetson Orin NX
$ sudo nvpmodel -m 0
$ sudo jetson_clocks
$ python3 bench_batch_vs_seq_gemm.py --device cuda --M 4096 --K 4096 --N 4096 \
    --batches 1 2 4 8 16 --dtype fp16 --repeats 100 --warmup 2 \
    --timing gpu --energy tegrastats --tegrastats-keys VDD_IN \
    --energy-interval-ms 10 --min-energy-window-ms 1000 \
    --out result_orin_vddin.csv

"""
import argparse
import os
import re
import time
import math
import threading
import subprocess
from typing import Dict, List, Tuple, Optional

import torch
import pandas as pd

# ---------------------------
# Energy meters
# ---------------------------

class EnergyMeterBase:
    def start(self): ...
    def stop(self) -> None: ...
    def stats(self) -> Dict[str, float]:
        return {"energy_j": 0.0, "duration_s": 0.0, "mean_w": 0.0, "peak_w": 0.0}

class NoopEnergyMeter(EnergyMeterBase):
    def __init__(self): self._t0 = None
    def start(self): self._t0 = time.perf_counter()
    def stop(self): pass
    def stats(self):
        dur = max(0.0, time.perf_counter() - (self._t0 or time.perf_counter()))
        return {"energy_j": 0.0, "duration_s": dur, "mean_w": 0.0, "peak_w": 0.0}

class NvmlEnergyMeter(EnergyMeterBase):
    """板卡功率（W）取樣 -> 能耗（J）。需要 pip install nvidia-ml-py3"""
    def __init__(self, index=0, interval_ms=10):
        self.index = index
        self.dt = max(5, int(interval_ms)) / 1000.0
        self.samples: List[Tuple[float, float]] = []
        self._stop = threading.Event()
        self._thr = None
        self._t0 = 0.0
        self._nvml = None
        self._handle = None

    def _init_nvml(self):
        from pynvml import (
            nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage
        )
        self._nvml = {
            "init": nvmlInit,
            "shutdown": nvmlShutdown,
            "get_handle": nvmlDeviceGetHandleByIndex,
            "get_power": nvmlDeviceGetPowerUsage
        }

    def start(self):
        try:
            self._init_nvml()
        except Exception as e:
            raise RuntimeError(
                f"[NVML] 請先安裝: pip install nvidia-ml-py3；錯誤：{e}"
            )
        self._nvml["init"]()
        self._handle = self._nvml["get_handle"](self.index)
        self.samples.clear()
        self._stop.clear()
        self._t0 = time.perf_counter()
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    def _loop(self):
        while not self._stop.is_set():
            try:
                mw = self._nvml["get_power"](self._handle)  # 毫瓦
                pw = mw / 1000.0
                t = time.perf_counter() - self._t0
                self.samples.append((t, pw))
            except Exception:
                pass
            time.sleep(self.dt)

    def stop(self):
        self._stop.set()
        if self._thr:
            self._thr.join()
        if self._nvml:
            try:
                self._nvml["shutdown"]()
            except Exception:
                pass

    def stats(self):
        if len(self.samples) < 2:
            return {"energy_j": 0.0, "duration_s": 0.0, "mean_w": 0.0, "peak_w": 0.0}
        e = 0.0
        for (t0, p0), (t1, p1) in zip(self.samples, self.samples[1:]):
            e += 0.5 * (p0 + p1) * (t1 - t0)
        duration = self.samples[-1][0] - self.samples[0][0]
        mean_w = sum(p for _, p in self.samples) / len(self.samples)
        peak_w = max(p for _, p in self.samples)
        return {"energy_j": e, "duration_s": duration, "mean_w": mean_w, "peak_w": peak_w}

def shutil_which(cmd: str) -> Optional[str]:
    from shutil import which
    return which(cmd)

class TegraStatsEnergyMeter(EnergyMeterBase):
    """
    以 tegrastats 抓 Jetson 功率，支援多欄位相加（預設 VDD_IN）。
    例：
      keys = ["VDD_IN"]                    -> 整機輸入功率
      keys = ["VDD_CPU_GPU_CV"]           -> 計算叢集近似功率（CPU+GPU+CV）
      keys = ["VDD_CPU_GPU_CV","VDD_SOC"] -> 計算 + SoC/記憶體 近似總功率
    """
    def __init__(self, interval_ms=50, keys=("VDD_IN",)):
        self.interval_ms = max(10, int(interval_ms))
        self.keys = list(keys)
        self.samples: List[Tuple[float, float]] = []
        self._stop = threading.Event()
        self._thr = None
        self._proc: Optional[subprocess.Popen] = None
        self._t0 = 0.0
        self._patterns: List[re.Pattern] = []

    def start(self):
        if not shutil_which("tegrastats"):
            raise RuntimeError("找不到 tegrastats，請確認已安裝在 Jetson 上（通常預裝）。")
        self.samples.clear()
        self._stop.clear()
        self._patterns = [re.compile(rf"{re.escape(k)}\s+(\d+)mW") for k in self.keys]
        self._proc = subprocess.Popen(
            ["tegrastats", "--interval", str(self.interval_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1
        )
        self._t0 = time.perf_counter()
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    def _loop(self):
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            total_w = 0.0
            matched = False
            for pat in self._patterns:
                m = pat.search(line)
                if m:
                    matched = True
                    total_w += int(m.group(1)) / 1000.0  # mW -> W
            if matched:
                t = time.perf_counter() - self._t0
                self.samples.append((t, total_w))
            if self._stop.is_set():
                break

    def stop(self):
        self._stop.set()
        try:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
        except Exception:
            pass
        if self._thr:
            self._thr.join(timeout=1.0)

    def stats(self):
        if len(self.samples) < 2:
            return {"energy_j": 0.0, "duration_s": 0.0, "mean_w": 0.0, "peak_w": 0.0}
        e = 0.0
        for (t0, p0), (t1, p1) in zip(self.samples, self.samples[1:]):
            e += 0.5 * (p0 + p1) * (t1 - t0)
        duration = self.samples[-1][0] - self.samples[0][0]
        mean_w = sum(p for _, p in self.samples) / len(self.samples)
        peak_w = max(p for _, p in self.samples)
        return {"energy_j": e, "duration_s": duration, "mean_w": mean_w, "peak_w": peak_w}

# ---------------------------
# Benchmark core
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Batch vs Sequential GEMM benchmark (latency + energy)")
    p.add_argument("--M", type=int, default=512)
    p.add_argument("--K", type=int, default=512)
    p.add_argument("--N", type=int, default=512)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--timing", type=str, default="gpu", choices=["gpu", "wall"])
    p.add_argument("--out", type=str, default="batch_vs_seq_gemm_energy.csv")
    p.add_argument("--allow-tf32", type=int, default=1)
    p.add_argument("--matmul-prec", type=str, default="high",
                   choices=["high", "medium", "highest"])
    # Energy options
    p.add_argument("--energy", type=str, default="auto",
                   choices=["auto", "none", "nvml", "tegrastats"])
    p.add_argument("--energy-interval-ms", type=int, default=10,
                   help="功率取樣間隔（建議 10 ms）")
    p.add_argument("--tegrastats-keys", type=str, default="VDD_IN",
                   help="Jetson 欲量測的功率欄位，逗號分隔，例如：VDD_IN 或 VDD_CPU_GPU_CV,VDD_SOC")
    # Copies inclusion
    p.add_argument("--include-copies", type=int, default=0,
                   help="1=把 H2D+D2H 也包進計時/能耗區間（需 pinned memory）")
    # Window extension
    p.add_argument("--min-energy-window-ms", type=float, default=1000.0,
                   help="每次量測至少延長到此時間，再均分回單次（避免功率取樣漏峰）")
    p.add_argument("--repeat-iters", type=int, default=0,
                   help="手動覆蓋自動估計的重複次數（>0 有效）")
    return p.parse_args()

def get_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def pick_dtype(device: torch.device, want: str) -> torch.dtype:
    if want == "fp16":
        if device.type == "cuda":
            return torch.float16
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

def measure_gpu_ms(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)

def measure_wall_ms(fn, device: torch.device) -> float:
    sync(device)
    t0 = time.perf_counter()
    fn()
    sync(device)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0

def auto_energy_meter(kind: str, interval_ms: int, tegra_keys: List[str]) -> EnergyMeterBase:
    if kind == "none":
        return NoopEnergyMeter()
    if kind == "nvml":
        return NvmlEnergyMeter(interval_ms=interval_ms)
    if kind == "tegrastats":
        return TegraStatsEnergyMeter(interval_ms=interval_ms, keys=tegra_keys)
    # auto
    if shutil_which("tegrastats"):
        return TegraStatsEnergyMeter(interval_ms=interval_ms, keys=tegra_keys)
    try:
        import pynvml  # noqa
        return NvmlEnergyMeter(interval_ms=interval_ms)
    except Exception:
        pass
    print("[WARN] 找不到 tegrastats / NVML，改用 --energy none")
    return NoopEnergyMeter()

def pretty_mem(M, K, N, B, dtype_bytes: int) -> float:
    total_elems = (B * M * K) + (B * K * N) + (B * M * N)
    return total_elems * dtype_bytes / (1024 ** 3)

@torch.inference_mode()
def run_one_case(M: int, K: int, N: int, B: int, device: torch.device,
                 dtype: torch.dtype, warmup: int, repeats: int,
                 timing: str, energy_kind: str, energy_interval_ms: int,
                 tegra_keys: List[str], include_copies: bool,
                 min_window_ms: float, repeat_override: int) -> Dict:

    # -------- 準備 compute 內容 --------
    if include_copies and device.type == "cuda":
        cpu_A  = torch.randn(B, M, K, pin_memory=True, dtype=dtype)
        cpu_Bm = torch.randn(B, K, N, pin_memory=True, dtype=dtype)
        cpu_C  = torch.empty(B, M, N, pin_memory=True, dtype=dtype)
        def make_seq_once():
            def fn():
                dA  = cpu_A.to(device, non_blocking=True)
                dBm = cpu_Bm.to(device, non_blocking=True)
                dCseq = torch.empty(B, M, N, device=device, dtype=dtype)
                for i in range(B):
                    dCseq[i] = torch.mm(dA[i], dBm[i])
                _ = dCseq.to("cpu", non_blocking=True, out=cpu_C)
            return fn
        def make_bmm_once():
            def fn():
                dA  = cpu_A.to(device, non_blocking=True)
                dBm = cpu_Bm.to(device, non_blocking=True)
                dC  = torch.bmm(dA, dBm)
                _ = dC.to("cpu", non_blocking=True, out=cpu_C)
            return fn
    else:
        A = torch.randn(B, M, K, device=device, dtype=dtype)
        Bm = torch.randn(B, K, N, device=device, dtype=dtype)
        C_seq = torch.empty(B, M, N, device=device, dtype=dtype)
        def make_seq_once():
            def fn():
                for i in range(B):
                    C_seq[i] = torch.mm(A[i], Bm[i])
            return fn
        def make_bmm_once():
            def fn():
                _ = torch.bmm(A, Bm)
            return fn

    # -------- 預熱 --------
    fn_seq_once = make_seq_once()
    fn_bmm_once = make_bmm_once()
    for _ in range(max(0, warmup)):
        fn_seq_once(); fn_bmm_once()

    # -------- 先做一次「試跑」以估計單次時間，決定需要重複幾次 --------
    meas = (lambda f: measure_gpu_ms(f)) if (timing == "gpu" and device.type == "cuda") \
        else (lambda f: measure_wall_ms(f, device))

    seq_one_ms = meas(fn_seq_once)
    bmm_one_ms = meas(fn_bmm_once)

    def choose_repeats(one_ms: float) -> int:
        if repeat_override > 0:
            return repeat_override
        if one_ms <= 0:  # 防呆
            return 1
        need = math.ceil(max(min_window_ms, one_ms) / max(one_ms, 1e-6))
        return int(max(1, need))

    seq_repeats = choose_repeats(seq_one_ms)
    bmm_repeats = choose_repeats(bmm_one_ms)

    def make_seq_repeated():
        def fn():
            for _ in range(seq_repeats):
                fn_seq_once()
        return fn
    def make_bmm_repeated():
        def fn():
            for _ in range(bmm_repeats):
                fn_bmm_once()
        return fn

    fn_seq = make_seq_repeated()
    fn_bmm = make_bmm_repeated()

    # -------- 正式量測（時間 + 能耗），再平均回「單次」 --------
    def run_with_energy(repeats_block, fn_kernel):
        meter = auto_energy_meter(energy_kind, energy_interval_ms, tegra_keys)
        times_ms, energies, meanw, peakw, durs = [], [], [], [], []
        for _ in range(max(1, repeats)):
            meter.start()
            t_ms = meas(fn_kernel)
            meter.stop()
            es = meter.stats()
            times_ms.append(t_ms)
            energies.append(es["energy_j"])
            meanw.append(es["mean_w"])
            peakw.append(es["peak_w"])
            durs.append(es["duration_s"])
        # 取最短時間那次的能耗（或中位數），並平均回單次
        t_min = min(times_ms); idx = times_ms.index(t_min)
        E = energies[idx] if energies[idx] > 0 else (sorted(energies)[len(energies)//2] if energies else 0.0)
        mW = meanw[idx] if meanw[idx] > 0 else (sum(meanw)/len(meanw) if meanw else 0.0)
        pW = peakw[idx] if peakw[idx] > 0 else (max(peakw) if peakw else 0.0)
        dur = durs[idx] if durs[idx] > 0 else (sum(durs)/len(durs) if durs else 0.0)
        # 平均回單次
        return (t_min / repeats_block, E / repeats_block, mW, pW, dur)

    seq_time_ms, seq_energy_j, seq_mean_w, seq_peak_w, seq_dur_s = run_with_energy(seq_repeats, fn_seq)
    bmm_time_ms, bmm_energy_j, bmm_mean_w, bmm_peak_w, bmm_dur_s = run_with_energy(bmm_repeats, fn_bmm)

    # FLOPs & throughput（以「單次」為基準）
    flops = 2.0 * M * K * N * B
    gflops_seq = flops / (seq_time_ms / 1e3) / 1e9
    gflops_bmm = flops / (bmm_time_ms / 1e3) / 1e9
    seq_j_per_gflop = (seq_energy_j / (flops / 1e9)) if flops > 0 else 0.0
    bmm_j_per_gflop = (bmm_energy_j / (flops / 1e9)) if flops > 0 else 0.0

    return dict(
        device=str(device),
        capability=(torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"),
        dtype=str(dtype).replace("torch.", ""),
        timing_mode=("cuda_events" if timing == "gpu" and device.type == "cuda" else "wall_clock"),
        energy_meter=("none" if energy_kind == "none" else ("tegrastats" if shutil_which("tegrastats") else "nvml")),
        include_copies=bool(include_copies),
        M=M, K=K, N=N, B=B,
        seq_repeats=seq_repeats,
        bmm_repeats=bmm_repeats,

        sequential_time_ms=seq_time_ms,
        sequential_gflops=gflops_seq,
        sequential_energy_j=seq_energy_j,
        sequential_duration_s=seq_dur_s,
        sequential_mean_w=seq_mean_w,
        sequential_peak_w=seq_peak_w,
        sequential_j_per_gflop=seq_j_per_gflop,

        batched_time_ms=bmm_time_ms,
        batched_gflops=gflops_bmm,
        batched_energy_j=bmm_energy_j,
        batched_duration_s=bmm_dur_s,
        batched_mean_w=bmm_mean_w,
        batched_peak_w=bmm_peak_w,
        batched_j_per_gflop=bmm_j_per_gflop,

        speedup_seq_over_batched=(seq_time_ms / bmm_time_ms) if bmm_time_ms > 0 else 0.0,
        energy_ratio_seq_over_batched=(seq_energy_j / bmm_energy_j) if bmm_energy_j > 0 else 0.0,
    )

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = get_device(args.device)
    dtype = pick_dtype(device, args.dtype)
    if device.type == "cuda":
        config_tf32(args.allow_tf32, args.matmul_prec)

    bytes_per = {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2}[dtype]
    est_gib = pretty_mem(args.M, args.K, args.N, max(args.batches), bytes_per)
    print(f"[INFO] device={device}, dtype={dtype}, timing={args.timing}, "
          f"TF32={'on' if args.allow_tf32 else 'off'}, energy={args.energy}")
    print(f"[INFO] Size: M={args.M}, K={args.K}, N={args.N}, B in {args.batches}, "
          f"include_copies={bool(args.include_copies)}, min_window_ms={args.min_energy_window_ms}")
    print(f"[INFO] Rough peak memory (A+B+C at B=max): ~{est_gib:.3f} GiB")

    tegra_keys = [s.strip() for s in args.tegrastats_keys.split(",") if s.strip()]
    results: List[Dict] = []
    for B in args.batches:
        stats = run_one_case(args.M, args.K, args.N, B, device, dtype,
                             warmup=args.warmup, repeats=args.repeats,
                             timing=args.timing, energy_kind=args.energy,
                             energy_interval_ms=args.energy_interval_ms,
                             tegra_keys=tegra_keys,
                             include_copies=args.include_copies,
                             min_window_ms=args.min_energy_window_ms,
                             repeat_override=args.repeat_iters)
        results.append(stats)
        print(f"[DONE] B={B:>4d} | seq {stats['sequential_time_ms']:.3f} ms, "
              f"bmm {stats['batched_time_ms']:.3f} ms | "
              f"seqE {stats['sequential_energy_j']:.5f} J, "
              f"bmmE {stats['batched_energy_j']:.5f} J | "
              f"seq_rep={stats['seq_repeats']}, bmm_rep={stats['bmm_repeats']}")

    df = pd.DataFrame(results).sort_values(["B"]).reset_index(drop=True)
    out = args.out
    df.to_csv(out, index=False)
    print(f"[SAVE] CSV -> {out}")
    try:
        from tabulate import tabulate
        print(tabulate(df, headers="keys", tablefmt="github", floatfmt=".4f"))
    except Exception:
        print(df)

if __name__ == "__main__":
    main()