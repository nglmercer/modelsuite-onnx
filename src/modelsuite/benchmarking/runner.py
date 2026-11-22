"""Benchmark one graph on one provider: load time, first-inference latency, steady-state percentiles, throughput, memory."""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..core.config import BenchmarkConfig
from ..core.errors import ProviderError
from ..core.types import Loose
from ..runtimes.onnx_utils import total_size
from ..runtimes.ort import canonical, create_runner, short
from .memory import PeakSampler, maxrss_mb, rss_mb


class BenchmarkResult(Loose):
    component: str
    variant: str = ""
    provider: str
    status: str                                    # ok | error
    error: str | None = None
    active_providers: list[str] = []
    model_bytes: int | None = None
    load_seconds: float | None = None
    first_inference_ms: float | None = None
    iterations: int = 0
    mean_ms: float | None = None
    median_ms: float | None = None
    std_ms: float | None = None
    min_ms: float | None = None
    max_ms: float | None = None
    p50_ms: float | None = None
    p90_ms: float | None = None
    p95_ms: float | None = None
    p99_ms: float | None = None
    throughput_per_s: float | None = None          # inferences / second
    items_per_s: float | None = None               # batch items / second (batch from the first input's leading dim)
    batch_size: int | None = None
    peak_rss_mb: float | None = None
    rss_delta_mb: float | None = None              # peak RSS minus RSS before the session was created
    peak_gpu_mb: float | None = None
    gpu_delta_mb: float | None = None
    input_source: str = ""
    threads: int | None = None
    isolated_process: bool = False
    note: str = ""


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q))


def measure(path: str, provider: str, inputs: dict[str, np.ndarray], cfg: dict[str, Any], component: str = "", variant: str = "") -> dict[str, Any]:
    """In-process measurement (also what the isolated worker calls)."""
    prov = canonical(provider)
    gpu = prov in ("CUDAExecutionProvider", "TensorrtExecutionProvider")
    base_rss = rss_mb()
    res: dict[str, Any] = {"component": component, "variant": variant, "provider": short(prov), "status": "ok",
                           "model_bytes": total_size(Path(path)), "threads": cfg.get("threads")}
    with PeakSampler(gpu=gpu) as ps:
        try:
            runner = create_runner(path, prov, threads=cfg.get("threads"), allow_partial_fallback=cfg.get("allow_partial_fallback", True))
        except ProviderError as exc:
            res.update(status="error", error=str(exc))
            return res
        res["load_seconds"] = round(runner.load_seconds, 4)
        res["active_providers"] = runner.active_providers
        t = time.perf_counter()
        runner.run(inputs)
        res["first_inference_ms"] = round((time.perf_counter() - t) * 1000, 3)
        for _ in range(int(cfg.get("warmup", 2))):
            runner.run(inputs)
        times: list[float] = []
        started = time.perf_counter()
        for i in range(int(cfg.get("iterations", 10))):
            t = time.perf_counter()
            runner.run(inputs)
            times.append((time.perf_counter() - t) * 1000)
            if i >= 2 and time.perf_counter() - started > float(cfg.get("max_seconds", 120)):
                res["note"] = f"stopped after {i + 1} iterations (max_seconds reached)"
                break
    if times:
        res.update(iterations=len(times), mean_ms=round(statistics.fmean(times), 4), median_ms=round(statistics.median(times), 4),
                   std_ms=round(statistics.pstdev(times), 4), min_ms=round(min(times), 4), max_ms=round(max(times), 4),
                   p50_ms=round(_percentile(times, 50), 4), p90_ms=round(_percentile(times, 90), 4), p95_ms=round(_percentile(times, 95), 4),
                   p99_ms=round(_percentile(times, 99), 4), throughput_per_s=round(1000.0 / statistics.fmean(times), 3))
        first = next(iter(inputs.values()), None)
        if first is not None and first.ndim >= 1 and first.shape[0] >= 1:
            res["batch_size"] = int(first.shape[0])
            res["items_per_s"] = round(res["throughput_per_s"] * first.shape[0], 3)
    res["peak_rss_mb"] = round(max(ps.peak_rss, maxrss_mb() or 0.0) if cfg.get("_isolated") else ps.peak_rss, 1)
    res["rss_delta_mb"] = round(res["peak_rss_mb"] - base_rss, 1)
    if gpu:
        res["peak_gpu_mb"] = ps.peak_gpu
        res["gpu_delta_mb"] = None if ps.peak_gpu is None or ps.base_gpu is None else round(ps.peak_gpu - ps.base_gpu, 1)
    else:
        res["note"] = (res.get("note", "") + " VRAM not measured for non-NVIDIA/CPU providers").strip()
    return res


def benchmark_graph(path: Path, provider: str, inputs: dict[str, np.ndarray], cfg: BenchmarkConfig, component: str = "", variant: str = "",
                    input_source: str = "") -> BenchmarkResult:
    """Run `measure`, in a fresh subprocess by default so peak-RSS belongs to this measurement only."""
    opts = {"warmup": cfg.warmup, "iterations": cfg.iterations, "max_seconds": cfg.max_seconds, "threads": cfg.threads,
            "allow_partial_fallback": cfg.allow_partial_fallback}
    if not cfg.isolate_process:
        out = measure(str(path), provider, inputs, opts, component, variant)
        return BenchmarkResult(**out, input_source=input_source, isolated_process=False)
    with tempfile.TemporaryDirectory(prefix="msbench_") as tmp:
        inp = Path(tmp) / "inputs.npz"
        np.savez(inp, **{k: v for k, v in inputs.items()})
        job = {"path": str(path), "provider": provider, "inputs": str(inp), "cfg": {**opts, "_isolated": True}, "component": component, "variant": variant}
        proc = subprocess.run([sys.executable, "-m", "modelsuite.benchmarking.worker"], input=json.dumps(job), capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-4:]
        return BenchmarkResult(component=component, variant=variant, provider=short(canonical(provider)), status="error",
                               error=f"benchmark worker exited {proc.returncode}: {' | '.join(tail)}", input_source=input_source, isolated_process=True)
    try:
        out = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return BenchmarkResult(component=component, variant=variant, provider=short(canonical(provider)), status="error",
                               error="benchmark worker produced no JSON", input_source=input_source, isolated_process=True)
    return BenchmarkResult(**out, input_source=input_source, isolated_process=True)
