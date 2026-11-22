"""Original-vs-exported numerical parity, dynamic-shape coverage and signature checks."""
from __future__ import annotations

import time
import traceback
from typing import Any, Callable

import numpy as np

from ..core.config import Tolerance, ValidationConfig
from ..core.plan import Arrays, ComponentExport, ReferenceCase, Runner
from ..core.types import Loose, Signature, Status
from .metrics import ArrayComparison, compare_arrays, summarize


class CaseResult(Loose):
    name: str
    kind: str
    status: Status
    seconds: float = 0.0
    input_shapes: dict[str, list[int]] = {}
    outputs: list[ArrayComparison] = []
    summary: dict[str, Any] = {}
    error: str | None = None


class ParityReport(Loose):
    component: str
    status: Status
    tolerance: Tolerance
    cases: list[CaseResult] = []
    signature_problems: list[str] = []

    @property
    def passed(self) -> bool:
        return self.status == Status.passed


def effective_tolerance(comp: ComponentExport, case: ReferenceCase | None, cfg: ValidationConfig, precision: str = "fp32") -> Tolerance:
    """Precedence: case > component per-precision > config per-precision > component > config default.

    A lower-precision variant legitimately drifts more, so each precision can carry its own bounds -- but the strict
    fp32 tolerance is never silently reused for fp16, nor the fp16 one for fp32."""
    if case and case.tolerance:
        return case.tolerance
    if precision != "fp32":
        family = precision.split("-")[0]
        for table in (comp.precision_tolerance, cfg.precision_tolerance):   # the component's own bounds beat the config's
            if precision in table:
                return table[precision]
            if family in table:
                return table[family]
    return comp.tolerance or cfg.tolerance


def check_signature(comp: ComponentExport, sig: Signature) -> list[str]:
    """The exported graph must expose exactly the declared inputs and outputs, by name."""
    problems = []
    got_in, got_out = [t.name for t in sig.inputs], [t.name for t in sig.outputs]
    if comp.input_names and set(got_in) != set(comp.input_names):
        problems.append(f"inputs {got_in} != declared {comp.input_names}")
    missing = [n for n in comp.output_names if n not in got_out]
    if missing:   # a graph may expose more outputs (e.g. present K/V) than the adapter chose to compare
        problems.append(f"declared outputs {missing} are missing from the graph (it has {got_out})")
    for n, axes in comp.dynamic_axes.items():
        t = next((t for t in [*sig.inputs, *sig.outputs] if t.name == n), None)
        if t is None:
            continue
        for ax in axes:
            if ax < len(t.shape) and isinstance(t.shape[ax], int):
                problems.append(f"{n} axis {ax} was declared dynamic but is fixed to {t.shape[ax]} in the graph")
    return problems


ReferenceFn = Callable[[ReferenceCase, Arrays], Arrays | None]


def run_case(comp: ComponentExport, case: ReferenceCase, runner: Runner, reference: ReferenceFn, tol: Tolerance) -> CaseResult:
    t0 = time.perf_counter()
    try:
        inputs = case.inputs()
        ref = reference(case, inputs)
        if ref is None:
            return CaseResult(name=case.name, kind=case.kind, status=Status.skipped, seconds=round(time.perf_counter() - t0, 3),
                              error="no reference output available (no bundled golden case and the original model was not provided)")
        got = runner.run(inputs)
    except Exception as exc:
        return CaseResult(name=case.name, kind=case.kind, status=Status.error, seconds=round(time.perf_counter() - t0, 3),
                          error=f"{type(exc).__name__}: {str(exc).strip().splitlines()[0][:300] if str(exc).strip() else ''}")
    comps: list[ArrayComparison] = []
    for name in comp.output_names:
        if name not in got:
            comps.append(ArrayComparison(name=name, passed=False, shape_ref=[], shape_got=[], dtype_ref="", dtype_got="",
                                         reasons=["output missing from the exported graph"]))
            continue
        if name not in ref:
            continue
        comps.append(compare_arrays(name, np.asarray(ref[name]), np.asarray(got[name]), tol, comp.output_kinds.get(name, "")))
    s = summarize(comps)
    return CaseResult(name=case.name, kind=case.kind, status=Status.passed if s["passed"] and comps else Status.failed,
                      seconds=round(time.perf_counter() - t0, 3), input_shapes={k: list(v.shape) for k, v in inputs.items()},
                      outputs=comps, summary=s)


def run_parity(comp: ComponentExport, runner: Runner, reference: ReferenceFn, cfg: ValidationConfig,
               precision: str = "fp32", signature: Signature | None = None, include_dynamic: bool | None = None) -> ParityReport:
    include_dynamic = cfg.check_dynamic_shapes if include_dynamic is None else include_dynamic
    cases = [c for c in comp.cases if include_dynamic or c.kind != "dynamic_shape"]
    results = [run_case(comp, c, runner, reference, effective_tolerance(comp, c, cfg, precision)) for c in cases]
    problems = check_signature(comp, signature) if signature else []
    ran = [r for r in results if r.status != Status.skipped]
    if not ran:
        status = Status.skipped
    else:
        status = Status.passed if all(r.status == Status.passed for r in ran) and not problems else Status.failed
    return ParityReport(component=comp.name, status=status, tolerance=effective_tolerance(comp, None, cfg, precision),
                        cases=results, signature_problems=problems)
