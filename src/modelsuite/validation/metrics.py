"""Tensor comparison. One function, one verdict, with the numbers that justify it."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..core.config import Tolerance
from ..core.types import Loose


class ArrayComparison(Loose):
    name: str
    passed: bool
    shape_ref: list[int]
    shape_got: list[int]
    dtype_ref: str
    dtype_got: str
    max_abs: float | None = None
    mean_abs: float | None = None
    max_rel: float | None = None
    cosine: float | None = None
    equal_fraction: float | None = None
    argmax_agreement: float | None = None
    rms_ref: float | None = None
    rms_got: float | None = None
    nonfinite_got: int = 0
    reasons: list[str] = []


def _f64(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float64) if a.dtype.kind in "fc" or a.dtype.name == "bfloat16" else a.astype(np.float64)


def compare_arrays(name: str, ref: np.ndarray, got: np.ndarray, tol: Tolerance, kind: str = "") -> ArrayComparison:
    """Compare an original-model output with an exported-graph output.

    * integer / bool outputs (and kind="discrete"): the fraction of identical elements must reach tol.min_equal_fraction
    * float outputs: max-abs bound (atol + rtol*max|ref|, or tol.max_abs), cosine similarity, rms ratio (catches silent /
      saturated outputs), and non-finite values (always a failure when the reference is finite)
    * kind="logits": additionally reports argmax agreement along the last axis (informational)"""
    c = ArrayComparison(name=name, passed=True, shape_ref=list(ref.shape), shape_got=list(got.shape),
                        dtype_ref=str(ref.dtype), dtype_got=str(got.dtype))
    if ref.shape != got.shape:
        c.passed = False
        c.reasons.append(f"shape mismatch: reference {tuple(ref.shape)} vs exported {tuple(got.shape)}")
        return c
    if ref.size == 0:
        return c
    discrete = kind == "discrete" or ref.dtype.kind in "iub"
    if discrete:
        eq = float(np.mean(ref == got))
        c.equal_fraction = eq
        if eq < tol.min_equal_fraction:
            c.passed = False
            c.reasons.append(f"only {eq * 100:.3f}% of elements identical (need {tol.min_equal_fraction * 100:.3f}%)")
        return c
    r, g = _f64(ref), _f64(got)
    c.nonfinite_got = int((~np.isfinite(g)).sum())
    if c.nonfinite_got and np.isfinite(r).all():
        c.passed = False
        c.reasons.append(f"{c.nonfinite_got} non-finite values in the exported output")
        return c
    both = np.isfinite(r) & np.isfinite(g)
    if (~np.isfinite(r)).any():  # e.g. -inf masks: they must be in the same places
        if not np.array_equal(~np.isfinite(r), ~np.isfinite(g)):
            c.passed = False
            c.reasons.append("non-finite values differ in position between reference and exported output")
    d = np.abs(r - g)[both]
    scale = float(np.abs(r[both]).max()) if both.any() else 0.0
    c.max_abs = float(d.max()) if d.size else 0.0
    c.mean_abs = float(d.mean()) if d.size else 0.0
    denom = np.maximum(np.abs(r[both]), 1e-6)
    c.max_rel = float((d / denom).max()) if d.size else 0.0
    c.rms_ref = float(math.sqrt(np.mean(r[both] ** 2))) if both.any() else 0.0
    c.rms_got = float(math.sqrt(np.mean(g[both] ** 2))) if both.any() else 0.0
    bound = tol.max_abs if tol.max_abs is not None else tol.atol + tol.rtol * scale
    if c.max_abs > bound:
        c.passed = False
        c.reasons.append(f"max |diff| {c.max_abs:.3g} exceeds bound {bound:.3g}")
    if tol.min_cosine is not None and d.size > 1:
        a, b = r[both].ravel(), g[both].ravel()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 0 and nb > 0:
            c.cosine = float(np.dot(a, b) / (na * nb))
            if c.cosine < tol.min_cosine:
                c.passed = False
                c.reasons.append(f"cosine similarity {c.cosine:.6f} below {tol.min_cosine}")
    if c.rms_ref and c.rms_ref > 1e-8:
        ratio = (c.rms_got or 0.0) / c.rms_ref
        if not (tol.min_rms_ratio <= ratio <= 1.0 / tol.min_rms_ratio):
            c.passed = False
            c.reasons.append(f"output energy ratio {ratio:.3g} (exported/reference) is outside [{tol.min_rms_ratio}, {1 / tol.min_rms_ratio:.3g}]: "
                             "near-silent or saturated output")
    if kind == "logits" and ref.ndim >= 1 and ref.shape[-1] > 1:
        c.argmax_agreement = float(np.mean(ref.argmax(-1) == got.argmax(-1)))
    return c


def summarize(comparisons: list[ArrayComparison]) -> dict[str, Any]:
    return {
        "passed": all(c.passed for c in comparisons),
        "max_abs": max((c.max_abs or 0.0 for c in comparisons), default=0.0),
        "min_cosine": min((c.cosine for c in comparisons if c.cosine is not None), default=None),
        "min_equal_fraction": min((c.equal_fraction for c in comparisons if c.equal_fraction is not None), default=None),
        "min_argmax_agreement": min((c.argmax_agreement for c in comparisons if c.argmax_agreement is not None), default=None),
    }
