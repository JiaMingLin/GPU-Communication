# nvml_energy.py
import time, threading
from typing import List, Tuple
from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage

class NvmlEnergyMeter:
    def __init__(self, index=0, interval_ms=10):
        self.index = index
        self.dt = interval_ms / 1000.0
        self.samples: List[Tuple[float, float]] = []  # (t, power_W)
        self._stop = threading.Event()
        self._thr = None

    def start(self):
        nvmlInit()
        self.h = nvmlDeviceGetHandleByIndex(self.index)
        self.samples.clear()
        self._stop.clear()
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self.t0 = time.perf_counter()
        self._thr.start()

    def _loop(self):
        while not self._stop.is_set():
            p_w = nvmlDeviceGetPowerUsage(self.h) / 1000.0  # mW -> W
            t = time.perf_counter() - self.t0
            self.samples.append((t, p_w))
            time.sleep(self.dt)

    def stop_and_energy_j(self) -> float:
        self._stop.set()
        if self._thr: self._thr.join()
        nvmlShutdown()
        # 梯形積分
        e = 0.0
        for (t0, p0), (t1, p1) in zip(self.samples, self.samples[1:]):
            e += 0.5 * (p0 + p1) * (t1 - t0)
        return e

if __name__ == "__main__":
    meter = NvmlEnergyMeter(index=0, interval_ms=10)
    meter.start()
    time.sleep(10)
    energy_j = meter.stop_and_energy_j()
    print(f"Energy: {energy_j:.4f} J")