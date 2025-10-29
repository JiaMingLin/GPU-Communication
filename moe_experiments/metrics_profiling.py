#!/usr/bin/env python3
# metrics_profiling.py
import time
import statistics
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# --- optional deps ---
try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None
    _HAS_TORCH = False

try:
    import pynvml
    _HAS_NVML = True
except Exception:
    pynvml = None
    _HAS_NVML = False


# =========================
# Metrics: 累積與彙整
# =========================
@dataclass
class Metrics:
    lat_ms: List[float] = field(default_factory=list)     # 每次量測的延遲 (ms)
    energy_j: List[float] = field(default_factory=list)   # 每次量測的能耗 (J)；可能為空（不支援）
    pad_added_total: int = 0                              # 累積 padding「多算的行數」總和
    pad_real_total: int = 0                               # 累積真實行數總和
    pad_padded_total: int = 0                             # 累積補後行數總和
    drops_total: int = 0                                  # 累積 drop 數量

    def add_sample(self,
                   latency_ms: float,
                   energy_joule: Optional[float] = None,
                   padding_real_rows: Optional[int] = None,
                   padding_padded_rows: Optional[int] = None,
                   drops: Optional[int] = None):
        self.lat_ms.append(float(latency_ms))
        if energy_joule is not None:
            self.energy_j.append(float(energy_joule))
        if padding_real_rows is not None and padding_padded_rows is not None:
            real = int(padding_real_rows)
            padded = int(padding_padded_rows)
            self.pad_real_total += real
            self.pad_padded_total += padded
            self.pad_added_total += max(0, padded - real)
        if drops is not None:
            self.drops_total += int(drops)

    # 常用統計
    def latency_summary(self) -> dict:
        if not self.lat_ms:
            return {}
        vals = sorted(self.lat_ms)
        def pct(p):
            if not vals: return None
            k = (len(vals)-1) * p
            f = int(k)
            c = min(f+1, len(vals)-1)
            if f == c:
                return vals[f]
            return vals[f] + (vals[c]-vals[f])*(k-f)

        return {
            "count": len(vals),
            "mean_ms": statistics.fmean(vals),
            "min_ms": vals[0],
            "p50_ms": pct(0.50),
            "p90_ms": pct(0.90),
            "p95_ms": pct(0.95),
            "p99_ms": pct(0.99),
            "max_ms": vals[-1],
        }

    def energy_summary(self) -> dict:
        if not self.energy_j:
            return {}
        vals = self.energy_j
        return {
            "count": len(vals),
            "mean_J": statistics.fmean(vals),
            "min_J": min(vals),
            "max_J": max(vals),
        }

    def padding_summary(self) -> dict:
        if self.pad_real_total == 0 and self.pad_padded_total == 0:
            return {}
        overhead_ratio = None
        if self.pad_real_total > 0:
            overhead_ratio = self.pad_added_total / float(self.pad_real_total)
        return {
            "real_rows": self.pad_real_total,
            "padded_rows": self.pad_padded_total,
            "added_rows": self.pad_added_total,
            "overhead_ratio": overhead_ratio,  # (padded-real)/real
            "drops_total": self.drops_total,
        }

    def summary(self) -> dict:
        out = {}
        out["latency"] = self.latency_summary()
        out["energy"] = self.energy_summary()
        out["padding"] = self.padding_summary()
        return out


# =========================
# Profiling: 夾測區段
# =========================
class _EnergySampler:
    """NVML 功率取樣積分（當裝置不支援總能耗計數器時的退路）。
       用在 start()/end() 之間，透過多次讀取功率近似能耗。
    """
    def __init__(self, nvml_handle, sample_interval_s: float = 0.01):
        self.h = nvml_handle
        self.dt = float(sample_interval_s)
        self._running = False
        self._joules = 0.0
        self._last_t = None

    def start(self):
        self._running = True
        self._joules = 0.0
        self._last_t = time.perf_counter()

    def sample_once(self):
        if not self._running:
            return
        now = time.perf_counter()
        dt = now - self._last_t if self._last_t is not None else 0.0
        self._last_t = now
        try:
            mw = pynvml.nvmlDeviceGetPowerUsage(self.h)  # 毫瓦
            w = mw / 1000.0
            self._joules += w * dt
        except Exception:
            pass

    def stop_and_get_joules(self) -> float:
        if self._running:
            # 最後補採一筆
            self.sample_once()
        self._running = False
        return float(self._joules)


class Profiling:
    """使用方式：
       prof = Profiling(metrics, use_cuda_events=True)
       prof.start()
       # ... 量測區段 ...
       prof.end(padding_real_rows=..., padding_padded_rows=..., drops=...)
    """
    def __init__(self,
                 metrics: Metrics,
                 use_cuda_events: bool = True,
                 cuda_device_index: int = 0,
                 nvml_device_index: Optional[int] = None,
                 power_sample_interval_s: float = 0.01):
        self.metrics = metrics
        self.use_cuda_events = bool(use_cuda_events and _HAS_TORCH and torch.cuda.is_available())
        self.cuda_dev = int(cuda_device_index)

        # 時間測量
        self._t0 = None
        self._ev_start = None
        self._ev_end = None

        # NVML / 能耗
        self._nvml_ok = False
        self._nvml_handle = None
        self._energy_start_uJ = None
        self._sampler: Optional[_EnergySampler] = None
        if _HAS_NVML:
            try:
                pynvml.nvmlInit()
                idx = nvml_device_index if nvml_device_index is not None else self.cuda_dev
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(int(idx))
                self._nvml_ok = True
            except Exception:
                self._nvml_ok = False

        # 取樣設定
        self._power_dt = float(power_sample_interval_s)

    # ---- internal helpers ----
    def _nvml_energy_uJ(self) -> Optional[int]:
        if not self._nvml_ok:
            return None
        try:
            return pynvml.nvmlDeviceGetTotalEnergyConsumption(self._nvml_handle)  # microJoules
        except Exception:
            return None

    # ---- public API ----
    def start(self):
        # 延遲計時
        if self.use_cuda_events:
            torch.cuda.set_device(self.cuda_dev)
            self._ev_start = torch.cuda.Event(enable_timing=True)
            self._ev_end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(self.cuda_dev)
            self._ev_start.record()
        else:
            self._t0 = time.perf_counter()

        # 能耗
        self._energy_start_uJ = self._nvml_energy_uJ()
        if self._energy_start_uJ is None and self._nvml_ok:
            # 啟動功率取樣積分
            self._sampler = _EnergySampler(self._nvml_handle, self._power_dt)
            self._sampler.start()

    def end(self,
            padding_real_rows: Optional[int] = None,
            padding_padded_rows: Optional[int] = None,
            drops: Optional[int] = None) -> Tuple[float, Optional[float]]:
        """結束量測並寫入 Metrics。
           回傳：(latency_ms, energy_J or None)
        """
        # 延遲
        if self.use_cuda_events:
            self._ev_end.record()
            torch.cuda.synchronize(self.cuda_dev)
            lat_ms = float(self._ev_start.elapsed_time(self._ev_end))
        else:
            lat_ms = (time.perf_counter() - self._t0) * 1000.0

        # 能耗
        energy_J = None
        if self._energy_start_uJ is not None:
            e_end = self._nvml_energy_uJ()
            if e_end is not None:
                delta_uJ = max(0, int(e_end) - int(self._energy_start_uJ))
                energy_J = float(delta_uJ) / 1e6
        elif self._sampler is not None:
            # 最後補幾次取樣，減少漏積分
            for _ in range(3):
                time.sleep(self._power_dt)
                self._sampler.sample_once()
            energy_J = self._sampler.stop_and_get_joules()

        # 寫入 Metrics
        self.metrics.add_sample(
            latency_ms=lat_ms,
            energy_joule=energy_J,
            padding_real_rows=padding_real_rows,
            padding_padded_rows=padding_padded_rows,
            drops=drops
        )
        return lat_ms, energy_J


# =========================
# Demo / 自我測試
# =========================
if __name__ == "__main__":
    print("== Demo: Metrics + Profiling ==")
    m = Metrics()

    # 嘗試用 CUDA 事件（若不支援會自動退回 perf_counter）
    prof = Profiling(metrics=m, use_cuda_events=True, cuda_device_index=0)

    # 這裡用一次簡單的示範區段；實務中你把 start()/end() 夾在 MoE MLP 計算處
    # 範例一：CPU 模擬工作
    prof.start()
    time.sleep(0.05)  # 模擬 50 ms 工作
    # 假設這次真實行數=10_000、補後行數=12_000、drop=7
    lat_ms, eJ = prof.end(padding_real_rows=10_000, padding_padded_rows=12_000, drops=7)
    print(f"[Run1] latency={lat_ms:.3f} ms, energy={eJ if eJ is not None else 'N/A'} J")

    # 範例二：若有 PyTorch + CUDA，可做一次 GPU 工作
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.set_device(0)
        a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
        b = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
        torch.cuda.synchronize()
        prof.start()
        c = a @ b  # 模擬你的前向計算核心
        torch.cuda.synchronize()
        lat_ms, eJ = prof.end(padding_real_rows=50_000, padding_padded_rows=56_000, drops=0)
        print(f"[Run2] latency={lat_ms:.3f} ms, energy={eJ if eJ is not None else 'N/A'} J")

    # 輸出彙整
    print("\n== Summary ==")
    from pprint import pprint
    pprint(m.summary())
    print("Tips: 若 energy 顯示 N/A，請安裝/啟用 NVML（pynvml）且 GPU 支援能耗計數器。")