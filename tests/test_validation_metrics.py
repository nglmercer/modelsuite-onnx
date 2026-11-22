"""Numerical comparison and tolerance policy."""
from __future__ import annotations

import numpy as np
import pytest

from modelsuite.core.config import Tolerance, ValidationConfig
from modelsuite.core.plan import ComponentExport, ReferenceCase
from modelsuite.validation.metrics import compare_arrays
from modelsuite.validation.parity import check_signature, effective_tolerance, run_parity
from modelsuite.core.types import Signature, Status, TensorSpec

rng = np.random.default_rng(0)
REF = rng.standard_normal((4, 64)).astype(np.float32)


def test_identical_passes_and_reports_numbers():
    c = compare_arrays("y", REF, REF.copy(), Tolerance())
    assert c.passed and c.max_abs == 0 and c.cosine == pytest.approx(1.0)


def test_small_noise_within_bound_big_noise_fails():
    assert compare_arrays("y", REF, REF + 1e-6, Tolerance()).passed
    bad = compare_arrays("y", REF, REF + 0.5, Tolerance())
    assert not bad.passed and any("max |diff|" in r for r in bad.reasons)


def test_shape_mismatch_is_a_failure_not_a_crash():
    c = compare_arrays("y", REF, REF[:2], Tolerance())
    assert not c.passed and "shape mismatch" in c.reasons[0]


def test_nan_in_exported_output_fails():
    got = REF.copy()
    got[0, 0] = np.nan
    c = compare_arrays("y", REF, got, Tolerance())
    assert not c.passed and "non-finite" in c.reasons[0]


def test_near_silent_output_is_caught_even_if_absolute_error_is_small():
    """The Reverb codec-decoder-on-GPU failure mode: tiny absolute error, but the signal is gone."""
    audio = (rng.standard_normal(24000) * 0.017).astype(np.float32)
    silent = audio * 0.008                       # RMS 0.00014 vs 0.017, as in the original study
    c = compare_arrays("audio", audio, silent, Tolerance(atol=1e-1, min_cosine=None))
    assert not c.passed and any("near-silent" in r for r in c.reasons)


def test_discrete_outputs_use_equal_fraction():
    codes = rng.integers(0, 1024, (1, 8, 50))
    off = codes.copy()
    off[0, 0, :3] = 0
    assert compare_arrays("codes", codes, codes.copy(), Tolerance()).passed
    assert not compare_arrays("codes", codes, off, Tolerance(min_equal_fraction=1.0)).passed
    assert compare_arrays("codes", codes, off, Tolerance(min_equal_fraction=0.99)).passed


def test_logits_report_argmax_agreement():
    lg = rng.standard_normal((5, 10)).astype(np.float32)
    c = compare_arrays("logits", lg, lg + 1e-7, Tolerance(), kind="logits")
    assert c.argmax_agreement == 1.0


def test_tolerance_precedence_case_beats_precision_beats_default():
    cfg = ValidationConfig()
    comp = ComponentExport(name="c", build=lambda: None, example_inputs=lambda: {}, input_names=[], output_names=[], tolerance=Tolerance(atol=7e-4))
    case = ReferenceCase("k", lambda: {}, tolerance=Tolerance(atol=1.0))
    assert effective_tolerance(comp, case, cfg, "int8").atol == 1.0
    assert effective_tolerance(comp, None, cfg, "fp32").atol == 7e-4                      # strict one never silently loosened for fp32
    assert effective_tolerance(comp, None, cfg, "fp16").atol == cfg.precision_tolerance["fp16"].atol
    assert effective_tolerance(comp, None, cfg, "fp16-weights").atol == cfg.precision_tolerance["fp16"].atol   # family lookup
    assert effective_tolerance(comp, None, cfg, "int8-dynamic").min_cosine == cfg.precision_tolerance["int8"].min_cosine
    comp.precision_tolerance["int8"] = Tolerance(atol=9.0)
    assert effective_tolerance(comp, None, cfg, "int8-dynamic").atol == 9.0               # component's own bounds win


def test_signature_check_flags_missing_names_and_fixed_dynamic_axes():
    comp = ComponentExport(name="c", build=lambda: None, example_inputs=lambda: {}, input_names=["x"], output_names=["y"], dynamic_axes={"x": {0: "b"}})
    ok = Signature(inputs=[TensorSpec(name="x", dtype="float32", shape=["b", 4])], outputs=[TensorSpec(name="y", dtype="float32", shape=["b"])])
    assert check_signature(comp, ok) == []
    wrong = Signature(inputs=[TensorSpec(name="x", dtype="float32", shape=[4, 4])], outputs=[TensorSpec(name="z", dtype="float32", shape=[1])])
    problems = " ".join(check_signature(comp, wrong))
    assert "fixed to 4" in problems and "missing" in problems


def test_run_parity_statuses():
    class Runner:
        def __init__(self, f): self.f = f
        def run(self, inputs): return {"y": self.f(inputs["x"])}
    comp = ComponentExport(name="c", build=lambda: None, example_inputs=lambda: {}, input_names=["x"], output_names=["y"],
                           cases=[ReferenceCase("a", lambda: {"x": REF}), ReferenceCase("b", lambda: {"x": REF[:1]}, "dynamic_shape")])
    ref = lambda case, inp: {"y": inp["x"] * 2}
    cfg = ValidationConfig()
    assert run_parity(comp, Runner(lambda x: x * 2), ref, cfg).status == Status.passed
    assert run_parity(comp, Runner(lambda x: x * 2.5), ref, cfg).status == Status.failed
    only_static = run_parity(comp, Runner(lambda x: x * 2), ref, cfg, include_dynamic=False)
    assert [c.name for c in only_static.cases] == ["a"]
    boom = run_parity(comp, Runner(lambda x: 1 / 0), ref, cfg)
    assert boom.status == Status.failed and boom.cases[0].status == Status.error and "ZeroDivisionError" in boom.cases[0].error
    no_ref = run_parity(comp, Runner(lambda x: x), lambda c, i: None, cfg)
    assert no_ref.status == Status.skipped                                           # nothing to compare with: never "passed"


# ---- stateful validation must be able to fail ----------------------------------------------------------------------------------------
from modelsuite.core.plan import StatefulCase, StateLink
from modelsuite.validation.stateful import run_stateful


def _running_sum_case(mode="replace", steps=6):
    """Reference: y = x + h; h' = y  (replace)  or  cache' = [cache, x]; y = sum(cache')  (concat)."""
    if mode == "replace":
        links = [StateLink("h_out", "h_in", "replace")]
        init = lambda: {"x": np.ones((1, 3), np.float32), "h_in": np.zeros((1, 3), np.float32)}
        ref = lambda i: {"y": i["x"] + i["h_in"], "h_out": i["x"] + i["h_in"]}
    else:
        links = [StateLink("kv_out", "kv_in", "concat", axis=1)]
        init = lambda: {"x": np.ones((1, 1, 2), np.float32), "kv_in": np.zeros((1, 0, 2), np.float32)}
        ref = lambda i: {"y": np.concatenate([i["kv_in"], i["x"]], 1).sum(1), "kv_out": i["x"]}
    comp = ComponentExport(name="rnn", build=lambda: None, example_inputs=lambda: {}, input_names=list(init()), output_names=["y"] + [l.output for l in links],
                           stateful_component=True)
    case = StatefulCase("seq", links, steps, init, lambda i: {"x": np.full_like(init()["x"], float(i + 1))})
    return comp, case, ref


class _FnRunner:
    def __init__(self, f): self.f, self.calls = f, 0
    def run(self, inputs):
        self.calls += 1
        return self.f(inputs)


def test_stateful_replace_mode_passes_for_a_correct_graph_and_checks_every_step():
    comp, case, ref = _running_sum_case("replace")
    rep = run_stateful(comp, case, _FnRunner(ref), lambda i: ref(i), ValidationConfig())
    assert rep.status == Status.passed and [s.step for s in rep.steps] == list(range(6)) and rep.trajectory_max_abs == 0.0


def test_stateful_validation_catches_a_graph_that_goes_wrong_only_from_step_three():
    comp, case, ref = _running_sum_case("replace")
    n = {"calls": 0}

    def buggy(inputs):                                  # right at first, then the state handling silently breaks
        n["calls"] += 1
        out = ref(inputs)
        if n["calls"] > 5:                               # (each step makes two calls: lock-step + free-running)
            out = {k: v * 1.01 for k, v in out.items()}
        return out
    rep = run_stateful(comp, case, _FnRunner(buggy), lambda i: ref(i), ValidationConfig())
    assert rep.status == Status.failed
    ok = [s.passed for s in rep.steps]
    assert ok[:2] == [True, True] and not all(ok) and rep.trajectory_max_abs > 0


def test_stateful_concat_mode_grows_the_cache_each_step():
    comp, case, ref = _running_sum_case("concat", steps=5)
    seen = []

    def runner(inputs):
        seen.append(inputs["kv_in"].shape[1])
        return ref(inputs)
    rep = run_stateful(comp, case, _FnRunner(runner), lambda i: ref(i), ValidationConfig())
    assert rep.status == Status.passed and seen[::2] == [0, 1, 2, 3, 4]      # lock-step calls see a cache that grows by one each step


def test_stateful_runner_exception_is_an_error_status_with_progress_kept():
    comp, case, ref = _running_sum_case("replace")
    calls = {"n": 0}

    def flaky(inputs):
        calls["n"] += 1
        if calls["n"] > 6:
            raise RuntimeError("provider crashed")
        return ref(inputs)
    rep = run_stateful(comp, case, _FnRunner(flaky), lambda i: ref(i), ValidationConfig())
    assert rep.status == Status.error and "provider crashed" in rep.error and len(rep.steps) >= 1
