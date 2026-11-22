"""Stateful inference tests: KV caches, recurrent state. The exported graph is stepped with its own state fed
back, and every step is compared with the original model run on the *same* state, so drift cannot hide."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from ..core.config import Tolerance, ValidationConfig
from ..core.plan import Arrays, ComponentExport, Runner, StatefulCase
from ..core.types import Loose, Status
from .metrics import ArrayComparison, compare_arrays, summarize
from .parity import effective_tolerance


class StepResult(Loose):
    step: int
    passed: bool
    max_abs: float
    min_cosine: float | None = None
    state_shapes: dict[str, list[int]] = {}


class StatefulReport(Loose):
    component: str
    case: str
    status: Status
    steps: list[StepResult] = []
    worst_max_abs: float = 0.0
    error: str | None = None
    #: drift of the *exported* trajectory from the *reference* trajectory when each is fed its own state
    trajectory_max_abs: float | None = None


def _advance(state_in: Arrays, outputs: Arrays, case: StatefulCase) -> Arrays:
    nxt = {}
    for link in case.links:
        new = outputs[link.output]
        if link.mode == "concat":
            ax = link.axis if link.axis is not None else new.ndim - 2
            nxt[link.input] = np.concatenate([state_in[link.input], new], axis=ax)
        else:
            nxt[link.input] = new
    return nxt


def run_stateful(comp: ComponentExport, case: StatefulCase, runner: Runner, reference: Callable[[Arrays], Arrays],
                 cfg: ValidationConfig, precision: str = "fp32") -> StatefulReport:
    tol = case.tolerance or effective_tolerance(comp, None, cfg, precision)
    steps: list[StepResult] = []
    try:
        inputs = case.initial()
        # Trajectory A (lockstep): both models always see identical inputs/state -> isolates per-step error.
        # Trajectory B (free-running): the exported graph consumes its own state -> exposes accumulated drift.
        free_inputs = {k: v.copy() for k, v in inputs.items()}
        free_ref_inputs = {k: v.copy() for k, v in inputs.items()}
        worst, traj = 0.0, 0.0
        for i in range(case.steps):
            ref = reference(inputs)
            got = runner.run(inputs)
            comps: list[ArrayComparison] = [compare_arrays(n, np.asarray(ref[n]), np.asarray(got[n]), tol, comp.output_kinds.get(n, ""))
                                            for n in comp.output_names if n in ref and n in got]
            s = summarize(comps)
            steps.append(StepResult(step=i, passed=bool(s["passed"] and comps), max_abs=float(s["max_abs"]), min_cosine=s["min_cosine"],
                                    state_shapes={l.input: list(inputs[l.input].shape) for l in case.links}))
            worst = max(worst, s["max_abs"])
            # free-running comparison of the primary (first) output
            fg = runner.run(free_inputs)
            fr = reference(free_ref_inputs)
            first = comp.output_names[0]
            traj = max(traj, float(np.abs(np.asarray(fg[first], np.float64) - np.asarray(fr[first], np.float64)).max()))
            if i + 1 < case.steps:
                nxt = case.step_inputs(i + 1)
                inputs = {**nxt, **_advance(inputs, got, case)}
                free_inputs = {**nxt, **_advance(free_inputs, fg, case)}
                free_ref_inputs = {**nxt, **_advance(free_ref_inputs, fr, case)}
        status = Status.passed if all(s.passed for s in steps) else Status.failed
        return StatefulReport(component=comp.name, case=case.name, status=status, steps=steps, worst_max_abs=worst, trajectory_max_abs=traj)
    except Exception as exc:
        return StatefulReport(component=comp.name, case=case.name, status=Status.error, steps=steps,
                              error=f"{type(exc).__name__}: {str(exc).strip().splitlines()[0][:300] if str(exc).strip() else ''}")
