"""GPU energy meter for Atlas R2 task evaluation.

Samples GPU power via NVML at ~10Hz, computes total energy (joules)
and peak memory (MB) over a measured interval.
"""

import threading
import time

import pynvml


_TDP_WATTS = 150.0
_SAMPLE_INTERVAL = 0.1


class EnergyMeter:

    def __init__(self, gpu_index=0):
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self._samples = []
        self._peak_mem_bytes = 0
        self._running = False
        self._thread = None

    def start(self):
        self._samples = []
        self._peak_mem_bytes = 0
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _poll(self):
        while self._running:
            try:
                t = time.perf_counter()
                mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                if mw < _TDP_WATTS * 2 * 1000:
                    self._samples.append((t, mw / 1000.0))
                if mem.used > self._peak_mem_bytes:
                    self._peak_mem_bytes = mem.used
            except pynvml.NVMLError:
                pass
            time.sleep(_SAMPLE_INTERVAL)

    @property
    def energy_joules(self):
        if len(self._samples) < 2:
            return 0.0
        energy = 0.0
        for i in range(1, len(self._samples)):
            dt = self._samples[i][0] - self._samples[i - 1][0]
            avg_w = (self._samples[i][1] + self._samples[i - 1][1]) / 2
            energy += avg_w * dt
        return energy

    @property
    def duration_seconds(self):
        if len(self._samples) < 2:
            return 0.0
        return self._samples[-1][0] - self._samples[0][0]

    @property
    def peak_memory_mb(self):
        return self._peak_mem_bytes / (1024 * 1024)

    @property
    def mean_power_watts(self):
        dur = self.duration_seconds
        if dur <= 0:
            return 0.0
        return self.energy_joules / dur

    @property
    def sample_count(self):
        return len(self._samples)

    def summary(self):
        return {
            "energy_joules": round(self.energy_joules, 2),
            "duration_seconds": round(self.duration_seconds, 2),
            "mean_power_watts": round(self.mean_power_watts, 1),
            "peak_memory_mb": round(self.peak_memory_mb, 1),
            "samples": self.sample_count,
        }

    def __del__(self):
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


if __name__ == "__main__":
    import torch

    meter = EnergyMeter()
    print("Idle measurement (3s)...")
    meter.start()
    time.sleep(3)
    meter.stop()
    print(f"  Idle: {meter.summary()}")

    if torch.cuda.is_available():
        meter2 = EnergyMeter()
        print("\nGPU load measurement (matmul 3s)...")
        meter2.start()
        a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 3.0:
            _ = torch.mm(a, a)
            torch.cuda.synchronize()
        meter2.stop()
        print(f"  Load: {meter2.summary()}")
        del a
        torch.cuda.empty_cache()
