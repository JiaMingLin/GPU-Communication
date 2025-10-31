#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
breakdown_zipf_padded_stack_vs_bmm.py

- 以 Zipf 分佈產生各 expert 的 token 數 m[e]
- 將每個 m[e] 以「進位到指定倍數」pad 成 m_pad[e]（例：--pad_multiple 8，150->152）
- 依 m_pad 分組後執行：stack(Xm/W1m/W2m) -> bmm -> act -> bmm -> writeback
- 計時拆分：stack_total（含填零 padding + 三個 stack）、bmm_total（兩次 bmm）、act、writeback
- CUDA 用 cudaEvent；CPU 退回 perf_counter（CPU 上強制使用 float32）

用法：
  GPU：
    python breakdown_zipf_padded_stack_vs_bmm.py --device cuda --E 64 --d 512 --d_ff 2048 \
      --tokens_total 4096 --zipf_s 1.2 --pad_multiple 8 --warmup 10 --repeat 30

  若未給 tokens_total，就用 avg_m*E 當總 tokens：
    python breakdown_zipf_padded_stack_vs_bmm.py --device cuda --E 64 --avg_m 32 \
      --zipf_s 1.2 --pad_multiple 8
"""

import argparse
import time
from collections import defaultdict
import torch

# -------------------- Timer --------------------

def _now_ms():
    return time.perf_counter() * 1000.0

class SegmentTimer:
    """CUDA 用 cudaEvent；CPU 用 perf_counter。回傳毫秒。"""
    def __init__(self, device: torch.device):
        self.is_cuda = (device.type == "cuda")
        if self.is_cuda:
            self.start_evt = torch.cuda.Event(enable_timing=True)
            self.end_evt = torch.cuda.Event(enable_timing=True)
        else:
            self._t0 = 0.0
            self._t1 = 0.0

    def start(self):
        if self.is_cuda:
            self.start_evt.record()
        else:
            self._t0 = _now_ms()

    def stop(self) -> float:
        if self.is_cuda:
            self.end_evt.record()
            torch.cuda.synchronize()
            return self.start_evt.elapsed_time(self.end_evt)  # ms
        else:
            self._t1 = _now_ms()
            return self._t1 - self._t0

# -------------------- Zipf & padding --------------------

def make_zipf_counts(E: int, tokens_total: int, s: float, min_m: int, seed: int = 0) -> torch.Tensor:
    """
    以 Zipf 機率 p_e ∝ 1/rank^s 將 tokens_total 分配到 E 個 experts。
    每個 expert 至少 min_m（常用 0 或 1）。
    回傳 [E] long tensor（總和 == tokens_total）。
    """
    if tokens_total < E * min_m:
        raise ValueError(f"tokens_total={tokens_total} < E*min_m={E*min_m}")
    g = torch.Generator(device="cpu").manual_seed(seed)
    ranks = torch.arange(1, E + 1, dtype=torch.float64)
    probs = (1.0 / (ranks ** s))
    probs = probs / probs.sum()
    remain = tokens_total - E * min_m
    if remain == 0:
        return torch.full((E,), min_m, dtype=torch.long)
    # 使用 torch.multinomial 替代 Multinomial.sample(generator=...)
    # 進行 remain 次抽樣（with replacement），然後用 bincount 計數
    samples = torch.multinomial(probs, num_samples=remain, replacement=True, generator=g)
    m_extra = torch.bincount(samples, minlength=E).to(dtype=torch.long)
    m = m_extra + min_m
    return m

def round_up_to_multiple(x: int, k: int) -> int:
    """把 x 進位到 k 的倍數（k>0）。"""
    if k <= 0:
        raise ValueError("pad_multiple 必須 > 0")
    return ((x + k - 1) // k) * k if x > 0 else 0  # m=0 保持 0

# -------------------- Case builder --------------------

def build_case_zipf_padded(
    E=64, d=512, d_ff=2048, zipf_s=1.2,
    avg_m=32, tokens_total=None, min_m=0,
    pad_multiple=8, seed=0, device_str="cuda", dtype_str="float16", act_name="gelu"
):
    """
    產生：
      - m_orig[e]：Zipf 分配的原始 tokens 數
      - m_pad[e] ：進位到 pad_multiple 的 tokens 數（m=0 → 0）
      - groups: 依 m_pad 分組（僅 m_pad>0）
      - off_m_cpu：以「原始 m」累加的 offsets（Python int）
      - Xpad, W1, W2：都在指定 device
    """
    device = torch.device(device_str)
    if device.type == "cpu":
        dtype = torch.float32
    else:
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_str]

    torch.manual_seed(seed)

    if tokens_total is None:
        tokens_total = int(E * avg_m)

    m_orig = make_zipf_counts(E, tokens_total, s=zipf_s, min_m=min_m, seed=seed)
    m_pad = torch.tensor([round_up_to_multiple(int(m_orig[e].item()), pad_multiple) for e in range(E)], dtype=torch.long)

    # offsets（依「原始 m」建）：確保只回寫真實 tokens
    off_m_cpu = [0] * E
    cursor = 0
    for e in range(E):
        off_m_cpu[e] = int(cursor)
        cursor += int(m_orig[e].item())
    total_m_orig = int(m_orig.sum().item())
    assert cursor == total_m_orig

    # 依 m_pad 分組（只保留 m_pad>0 的 experts）
    groups = defaultdict(list)
    for e in range(E):
        mp = int(m_pad[e].item())
        if mp > 0:
            groups[mp].append(e)
    groups = dict(groups)

    # 準備資料
    Xpad = torch.randn((total_m_orig, d), device=device, dtype=dtype)
    W1   = torch.randn((E, d, d_ff), device=device, dtype=dtype)
    W2   = torch.randn((E, d_ff, d), device=device, dtype=dtype)

    # 活化
    act_map = {
        "gelu": torch.nn.functional.gelu,
        "relu": torch.relu,
        "silu": torch.nn.functional.silu,
        "identity": (lambda x: x),
    }
    act = act_map[act_name]

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True  # 讓 FP32 走 TF32

    return {
        "groups": groups,
        "Xpad": Xpad,
        "W1": W1, "W2": W2,
        "m_orig": m_orig,     # [E]
        "m_pad": m_pad,       # [E]
        "off_m_cpu": off_m_cpu,
        "device": device,
        "dtype": dtype,
        "d": d, "d_ff": d_ff,
        "act": act,
        "tokens_total": tokens_total,
        "pad_multiple": pad_multiple,
    }

# -------------------- Core pass --------------------

@torch.inference_mode()
def run_one_pass_padded(groups, Xpad, off_m_cpu, W1, W2, act, m_orig, m_pad, device):
    """
    依 m_pad 分組做計算；stack 階段會把每個 expert 的真實 tokens 複製到 [G, m_pad, d] 的前 m_orig 列，
    其餘補 0。回寫只寫回前 m_orig 列。
    回傳：各段毫秒數。
    """
    d = Xpad.shape[1]
    total_m_orig = Xpad.shape[0]
    Ypad = torch.empty((total_m_orig, d), dtype=Xpad.dtype, device=Xpad.device)

    t_stack = t_bmm = t_act = t_write = 0.0
    timer = SegmentTimer(device)

    for mval_pad, es in groups.items():
        G = len(es)

        # ---- 1) stack Xm（含 padding）----
        timer.start()
        Xm = torch.zeros((G, mval_pad, d), dtype=Xpad.dtype, device=Xpad.device)
        # 將每個 expert 的真實 tokens 複製到前 m_orig 列
        for i, e in enumerate(es):
            mo = int(m_orig[e].item())
            if mo > 0:
                base = off_m_cpu[e]
                Xm[i, :mo] = Xpad[base:base + mo]
        t_stack += timer.stop()

        # ---- 2) stack W1m ----
        timer.start()
        W1m = torch.stack([W1[e] for e in es], dim=0)  # [G, d, d_ff]
        t_stack += timer.stop()

        # ---- 3) stack W2m ----
        timer.start()
        W2m = torch.stack([W2[e] for e in es], dim=0)  # [G, d_ff, d]
        t_stack += timer.stop()

        # ---- 4) bmm 1 ----
        timer.start()
        H = torch.bmm(Xm, W1m)  # [G, m_pad, d_ff]
        t_bmm += timer.stop()

        # ---- 5) act ----
        timer.start()
        H = act(H)
        t_act += timer.stop()

        # ---- 6) bmm 2 ----
        timer.start()
        Y = torch.bmm(H, W2m)  # [G, m_pad, d]
        t_bmm += timer.stop()

        # ---- 7) writeback（只寫回原始 m_orig 部分）----
        timer.start()
        for i, e in enumerate(es):
            mo = int(m_orig[e].item())
            if mo > 0:
                base = off_m_cpu[e]
                Ypad[base:base + mo] = Y[i, :mo]
        t_write += timer.stop()

        del Xm, W1m, W2m, H, Y

    return {
        "stack_total_ms": t_stack,
        "bmm_total_ms": t_bmm,
        "act_ms": t_act,
        "writeback_ms": t_write,
    }

@torch.inference_mode()
def benchmark(groups, Xpad, off_m_cpu, W1, W2, act, m_orig, m_pad, device, warmup=5, repeat=20):
    # warmup
    for _ in range(max(0, warmup)):
        _ = run_one_pass_padded(groups, Xpad, off_m_cpu, W1, W2, act, m_orig, m_pad, device)
    # repeat
    sums = defaultdict(float)
    for _ in range(repeat):
        out = run_one_pass_padded(groups, Xpad, off_m_cpu, W1, W2, act, m_orig, m_pad, device)
        for k, v in out.items():
            sums[k] += v
    avg = {k: v / repeat for k, v in sums.items()}
    avg["total_ms"] = avg["stack_total_ms"] + avg["bmm_total_ms"] + avg["act_ms"] + avg["writeback_ms"]
    return avg

# -------------------- CLI --------------------

def human_ms(x): return f"{x:8.3f} ms"
def pct(x, total): return f"{(x/total*100 if total>0 else 0):5.1f}%"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                   choices=["cuda", "cpu"])
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--E", type=int, default=16)
    p.add_argument("--d", type=int, default=1024)
    p.add_argument("--d_ff", type=int, default=4096)
    p.add_argument("--zipf_s", type=float, default=1.2, help="Zipf 指數，越大越不均")
    p.add_argument("--avg_m", type=int, default=32, help="若未指定 tokens_total，使用 avg_m*E 作為總 tokens")
    p.add_argument("--tokens_total", type=int, default=None, help="覆寫總 tokens")
    p.add_argument("--min_m", type=int, default=0, help="每個 expert 至少 tokens（0 或 1 常見）")
    p.add_argument("--pad_multiple", type=int, default=8, help="每個 expert 的 token 進位對齊倍數（例：8）")
    p.add_argument("--act", type=str, default="gelu", choices=["gelu","relu","silu","identity"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeat", type=int, default=20)
    args = p.parse_args()

    case = build_case_zipf_padded(
        E=args.E, d=args.d, d_ff=args.d_ff, zipf_s=args.zipf_s,
        avg_m=args.avg_m, tokens_total=args.tokens_total, min_m=args.min_m,
        pad_multiple=args.pad_multiple, seed=args.seed,
        device_str=args.device, dtype_str=args.dtype, act_name=args.act
    )

    groups = case["groups"]; Xpad = case["Xpad"]; W1 = case["W1"]; W2 = case["W2"]
    m_orig = case["m_orig"]; m_pad = case["m_pad"]; off_m_cpu = case["off_m_cpu"]
    device = case["device"]; act = case["act"]; tokens_total = case["tokens_total"]

    # 量測
    avg = benchmark(groups, Xpad, off_m_cpu, W1, W2, act, m_orig, m_pad, device,
                    warmup=args.warmup, repeat=args.repeat)

    # 報表
    stack_ms = avg["stack_total_ms"]; bmm_ms = avg["bmm_total_ms"]
    act_ms = avg["act_ms"]; write_ms = avg["writeback_ms"]; total_ms = avg["total_ms"]

    print("\n=== Latency Breakdown (Avg over repeats) ===")
    print(f"Stack (pad+Xm/W1m/W2m): {human_ms(stack_ms)}  {pct(stack_ms, total_ms)}")
    print(f"BMM   (2×bmm)        : {human_ms(bmm_ms)}  {pct(bmm_ms, total_ms)}")
    print(f"Act                  : {human_ms(act_ms)}  {pct(act_ms, total_ms)}")
    print(f"Writeback            : {human_ms(write_ms)}  {pct(write_ms, total_ms)}")
    print(f"-------------------------------------------")
    print(f"Total                : {human_ms(total_ms)}")

    # Padding 統計
    pad_tokens = int((m_pad - m_orig).sum().item())
    eff_tokens = int(m_orig.sum().item())
    tot_padded = eff_tokens + pad_tokens
    zeros = int((m_orig == 0).sum().item())
    m_sorted, _ = torch.sort(m_orig, descending=True)
    top1 = int(m_sorted[0].item()) if m_sorted.numel() else 0
    top5 = int(m_sorted[:5].sum().item()) if m_sorted.numel() else 0

    print("\n=== Zipf & Padding Summary ===")
    print(f"E={args.E}, tokens_total={tokens_total}, min_m={args.min_m}, zipf_s={args.zipf_s}, pad_multiple={args.pad_multiple}")
    print(f"nonzero experts={args.E - zeros}, zero experts={zeros}")
    print(f"m_orig:  min={int(m_orig.min().item())}, max={int(m_orig.max().item())}, mean={float(m_orig.float().mean()):.2f}")
    print(f"m_pad :  min={int(m_pad.min().item())},  max={int(m_pad.max().item())},  mean={float(m_pad.float().mean()):.2f}")
    if tokens_total > 0:
        print(f"Top-1 share = {top1/tokens_total:.3f}, Top-5 share = {top5/tokens_total:.3f}")
    print(f"Effective tokens     : {eff_tokens}")
    print(f"Padded extra tokens  : {pad_tokens}  ({(pad_tokens/max(tot_padded,1))*100:.1f}% of computed)")
    sizes = {int(mv): len(es) for mv, es in groups.items()}
    print("Group sizes (m_pad -> #experts):", sizes)
    print(f"Device: {device}, DType: {Xpad.dtype}")

if __name__ == "__main__":
    main()