"""Peak memory sampling: process RSS (psutil + getrusage high-water mark) and GPU memory (nvidia-smi, when present)."""
from __future__ import annotations

import shutil
import subprocess
import threading
import time

import psutil


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / 2 ** 20


def maxrss_mb() -> float | None:
    try:
        import resource
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss     # KiB on Linux, bytes on macOS
        import sys
        return v / 1024 / 1024 if sys.platform == "darwin" else v / 1024
    except Exception:
        return None


def gpu_memory_mb() -> float | None:
    """Used memory of GPU 0 in MiB through nvidia-smi; None when there is no NVIDIA GPU / tool."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=memory.used", "--format=csv,nounits,noheader"], capture_output=True, text=True, timeout=5)
        return float(out.stdout.strip().splitlines()[0]) if out.returncode == 0 and out.stdout.strip() else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


class PeakSampler:
    """Samples RSS (and GPU memory if enabled) in a background thread; `.peak_*` are valid after stop()."""

    def __init__(self, interval: float = 0.005, gpu: bool = False):
        self.interval, self.gpu = interval, gpu
        self.peak_rss = rss_mb()
        self.base_gpu = gpu_memory_mb() if gpu else None
        self.peak_gpu = self.base_gpu
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        n = 0
        while not self._stop.is_set():
            self.peak_rss = max(self.peak_rss, rss_mb())
            n += 1
            if self.gpu and n % 20 == 0:                      # nvidia-smi is slow: every ~100 ms
                g = gpu_memory_mb()
                if g is not None:
                    self.peak_gpu = max(self.peak_gpu or 0.0, g)
            time.sleep(self.interval)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def stop(self):
        self._stop.set()
        self._t.join(timeout=2)
        self.peak_rss = max(self.peak_rss, rss_mb())
        if self.gpu:
            g = gpu_memory_mb()
            if g is not None:
                self.peak_gpu = max(self.peak_gpu or 0.0, g)
