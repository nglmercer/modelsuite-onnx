"""Accuracy-versus-performance comparison. Recommends only from measurements; abstains when there are none."""
from __future__ import annotations

from typing import Any, Literal

from ..core.types import Loose

Objective = Literal["latency", "size", "balanced"]


class VariantMeasurement(Loose):
    variant: str
    precision: str = "fp32"
    quantization: str = "none"
    size_bytes: int | None = None
    validation_passed: bool | None = None      # None = never validated
    worst_max_abs: float | None = None
    min_cosine: float | None = None
    latency_ms: float | None = None            # mean over components x iterations of the *whole pipeline* proxy (sum of component means)
    peak_rss_mb: float | None = None


class Recommendation(Loose):
    objective: str
    recommended: str | None
    baseline: str
    reasons: list[str] = []
    warnings: list[str] = []
    table: list[dict[str, Any]] = []


def recommend(measurements: list[VariantMeasurement], baseline: str = "fp32", objective: Objective = "balanced") -> Recommendation:
    by_name = {m.variant: m for m in measurements}
    base = by_name.get(baseline)
    rec = Recommendation(objective=objective, recommended=None, baseline=baseline)
    for m in measurements:
        row = m.model_dump()
        if base:
            if base.latency_ms and m.latency_ms:
                row["speedup_vs_baseline"] = round(base.latency_ms / m.latency_ms, 3)
            if base.size_bytes and m.size_bytes:
                row["size_ratio_vs_baseline"] = round(m.size_bytes / base.size_bytes, 3)
        rec.table.append(row)
        if base and m.variant != baseline and m.latency_ms and base.latency_ms and m.latency_ms > base.latency_ms * 1.05 and (m.size_bytes or 0) < (base.size_bytes or 1 << 62):
            rec.warnings.append(f"{m.variant} is smaller but {m.latency_ms / base.latency_ms:.2f}x SLOWER than {baseline} on this hardware: "
                                "lower precision is not automatically faster")
    unvalidated = [m.variant for m in measurements if m.validation_passed is None]
    if unvalidated:
        rec.warnings.append(f"never validated against the original model (excluded from recommendations): {', '.join(unvalidated)}")
    ok = [m for m in measurements if m.validation_passed]
    if not ok:
        rec.reasons.append("no variant has a passing validation result, so nothing can be recommended")
        return rec
    if objective in ("latency", "balanced") and not all(m.latency_ms for m in ok):
        missing = [m.variant for m in ok if not m.latency_ms]
        rec.reasons.append(f"no benchmark for {', '.join(missing)}; run `benchmark` before a {objective} recommendation")
        ok = [m for m in ok if m.latency_ms]
        if not ok:
            return rec
    if objective == "size" and not all(m.size_bytes for m in ok):
        ok = [m for m in ok if m.size_bytes]
    def score(m: VariantMeasurement) -> float:
        lat = (m.latency_ms or 0) / (base.latency_ms if base and base.latency_ms else m.latency_ms or 1)
        siz = (m.size_bytes or 0) / (base.size_bytes if base and base.size_bytes else m.size_bytes or 1)
        return {"latency": lat, "size": siz, "balanced": 0.5 * lat + 0.5 * siz}[objective]
    best = min(ok, key=score)
    rec.recommended = best.variant
    rec.reasons.append(f"lowest {objective} score among variants that passed validation "
                       f"(latency {best.latency_ms and round(best.latency_ms, 2)} ms, size {best.size_bytes}, worst |diff| {best.worst_max_abs})")
    if best.variant != baseline and base and base.validation_passed:
        rec.reasons.append(f"baseline '{baseline}' also passed validation; the quality cost of '{best.variant}' is the numbers in the table")
    return rec
