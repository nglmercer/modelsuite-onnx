"""Provider compatibility: does each execution provider produce the same numbers as the CPU run?

This is where 'the graph loads on GPU and returns near-silence' (the Reverb codec decoder on RADV/WebGPU) is caught:
a provider that loads and runs but disagrees with the CPU reference is reported as incompatible, not as supported."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np

from ..core.config import Tolerance
from ..core.errors import ProviderError
from ..core.plan import Arrays
from ..core.types import Loose, Status
from ..runtimes.ort import create_runner, short
from .metrics import compare_arrays, summarize


class ProviderCheck(Loose):
    provider: str
    component: str
    status: Status
    load_seconds: float | None = None
    active_providers: list[str] = []
    max_abs_vs_cpu: float | None = None
    min_cosine_vs_cpu: float | None = None
    min_rms_ratio_vs_cpu: float | None = None
    detail: str = ""
    known_issue: str | None = None


def check_provider(component: str, path: Path, provider: str, inputs: Arrays, tol: Tolerance, *, known_issue: str | None = None,
                   cpu_outputs: Arrays | None = None, threads: int | None = None, output_kinds: dict[str, str] | None = None,
                   allow_partial_fallback: bool = True) -> ProviderCheck:
    """Load `path` on `provider` (strictly: no silent CPU fallback), run `inputs`, compare with the CPU result."""
    prov = short(provider)
    try:
        runner = create_runner(path, provider, threads=threads, allow_partial_fallback=allow_partial_fallback)
    except ProviderError as exc:
        return ProviderCheck(provider=prov, component=component, status=Status.failed, detail=str(exc), known_issue=known_issue)
    try:
        if cpu_outputs is None:
            cpu_outputs = create_runner(path, "cpu", threads=threads).run(inputs)
        got = runner.run(inputs)
    except Exception as exc:
        return ProviderCheck(provider=prov, component=component, status=Status.error, load_seconds=round(runner.load_seconds, 3),
                             active_providers=runner.active_providers, known_issue=known_issue,
                             detail=f"{type(exc).__name__}: {str(exc).strip().splitlines()[0][:300] if str(exc).strip() else ''}")
    comps = [compare_arrays(n, cpu_outputs[n], got[n], tol, (output_kinds or {}).get(n, "")) for n in cpu_outputs if n in got]
    s = summarize(comps)
    ratios = [(c.rms_got or 0) / c.rms_ref for c in comps if c.rms_ref and c.rms_ref > 1e-8]
    reasons = [r for c in comps for r in c.reasons]
    return ProviderCheck(provider=prov, component=component, status=Status.passed if s["passed"] and comps else Status.failed,
                         load_seconds=round(runner.load_seconds, 3), active_providers=runner.active_providers,
                         max_abs_vs_cpu=float(s["max_abs"]), min_cosine_vs_cpu=s["min_cosine"],
                         min_rms_ratio_vs_cpu=min(ratios) if ratios else None, known_issue=known_issue,
                         detail="; ".join(reasons) if reasons else "matches the CPU reference within tolerance")
