"""Side-by-side tables of benchmark results across variants and providers."""
from __future__ import annotations

from typing import Iterable

from .runner import BenchmarkResult


def pipeline_latency_ms(results: Iterable[BenchmarkResult]) -> float | None:
    """Sum of per-component mean latencies: a proxy for one pass through every graph once (components with no result are ignored)."""
    ok = [r.mean_ms for r in results if r.status == "ok" and r.mean_ms is not None]
    return round(sum(ok), 4) if ok else None


def table(results: list[BenchmarkResult]) -> list[dict]:
    rows = []
    for r in results:
        rows.append({"component": r.component, "variant": r.variant, "provider": r.provider, "status": r.status,
                     "mean_ms": r.mean_ms, "p95_ms": r.p95_ms, "first_ms": r.first_inference_ms, "load_s": r.load_seconds,
                     "items_per_s": r.items_per_s, "peak_rss_mb": r.peak_rss_mb, "size_mb": round((r.model_bytes or 0) / 2 ** 20, 2),
                     "error": r.error})
    return rows
