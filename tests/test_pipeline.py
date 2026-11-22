"""End-to-end pipeline behaviour on a small custom PyTorch module."""
from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from modelsuite.core import pipeline, steps
from modelsuite.core.config import JobConfig
from modelsuite.core.errors import ConfigError
from modelsuite.core.golden import GoldenStore
from modelsuite.core.manifest import Artifact


@pytest.fixture(scope="module")
def converted(tmp_path_factory, toy_file):
    out = tmp_path_factory.mktemp("conv") / "art"
    cfg = JobConfig.model_validate({"model": f"{toy_file}:build", "output": str(out),
                                    "optimization": {"precision": ["fp16", "fp16-weights"], "quantization": ["int8-dynamic", "int4"], "graph_optimization": "extended"},
                                    "benchmark": {"iterations": 3, "warmup": 1, "isolate_process": False}})
    events = []
    ctx = pipeline.convert(cfg, emit=lambda s, m: events.append((s, m)))
    return out, cfg, events


def test_artifact_contents_and_manifest_fields(converted):
    out, cfg, _ = converted
    art = Artifact(out)
    m = art.read()
    # every field the plan requires
    assert m.source.reference.endswith("toy_model.py:build") and m.source.revision and m.source.framework == "pytorch"
    assert m.adapter.name == "torch-module" and m.exporters and m.job_fingerprint == cfg.fingerprint()
    c = m.component("toy")
    assert [t.name for t in c.signature.inputs] == ["x"] and c.signature.inputs[0].shape == ["batch", 16] and c.exporter == "torch-legacy"
    assert set(m.variants) >= {"fp32", "fp16", "fp16-weights", "int8-dynamic", "int4", "opt-extended"}
    assert m.variants["int8-dynamic"].quantization.method == "int8-dynamic" and m.variants["fp16"].precision == "fp16"
    assert m.licenses["model"]["id"] == "mit" and m.redistribution.allowed is True and "onnx" in m.environment["packages"]
    assert (out / "environment.json").is_file() and (out / "reports" / "summary.txt").is_file() and m.reference_cases["toy"]
    assert art.verify() == []                               # every recorded checksum matches


def test_every_variant_was_validated_against_the_original_with_numbers(converted):
    m = Artifact(converted[0]).read()
    for name in ("fp32", "fp16", "fp16-weights", "opt-extended", "int4"):
        v = m.variants[name].validation
        assert v["status"] == "passed", (name, v["status"])
    parity = m.variants["fp32"].validation["components"]["toy"]["parity"]
    assert {c["name"] for c in parity["cases"]} == {"example", "batch1", "batch9"}          # includes dynamic-shape cases
    assert parity["cases"][0]["summary"]["max_abs"] < 1e-5
    assert m.variants["fp32"].validation["structural"]["toy"]["status"] == "passed"


def test_a_lower_precision_failure_is_reported_not_hidden(converted):
    """int8 on this tiny random net misses its cosine bound: the variant must be 'failed' with the measured numbers."""
    v = Artifact(converted[0]).read().variants["int8-dynamic"].validation
    assert v["status"] == "failed"
    reasons = [r for c in v["components"]["toy"]["parity"]["cases"] for o in c["outputs"] for r in o["reasons"]]
    assert any("cosine" in r for r in reasons)


def test_recommendation_only_from_validated_and_measured_variants(converted):
    rec = Artifact(converted[0]).read().benchmark["recommendation"]
    assert rec["recommended"] in {"fp32", "fp16", "fp16-weights", "opt-extended", "int4"} and rec["recommended"] != "int8-dynamic"
    assert all("latency_ms" in row for row in rec["table"])


def test_golden_references_are_bundled_and_inputs_replayable(converted):
    g = GoldenStore(Artifact(converted[0]))
    assert set(g.cases("toy")) == {"example", "batch1", "batch9"}
    ins, outs = g.load("toy", "batch9")
    assert ins["x"].shape == (9, 16) and outs["y"].shape == (9, 8)


def test_resume_skips_finished_heavy_steps(converted):
    out, cfg, _ = converted
    events = []
    pipeline.convert(cfg, emit=lambda s, m: events.append((s, m)))
    msgs = " | ".join(m for s, m in events)
    assert "toy: resumed" in msgs and "fp16: resumed" in msgs and "already validated" in msgs
    assert not any("exporting with" in m or "building variant" in m for _, m in events)      # nothing was redone


def test_resume_keeps_the_recorded_exporter_versions(converted):
    out, cfg, _ = converted
    before = Artifact(out).read().exporters
    assert before                                                      # torch / onnx versions of the run that exported
    pipeline.convert(cfg)                                              # everything resumed: nothing re-exported
    assert Artifact(out).read().exporters == before


def test_resume_after_a_crash_between_components_reuses_finished_work(tmp_path, toy_file):
    cfg = JobConfig.model_validate({"model": f"{toy_file}:build", "output": str(tmp_path / "a"), "steps": ["discover", "inspect", "analyze", "export"]})
    pipeline.convert(cfg)
    art = Artifact(tmp_path / "a")
    (art.graph_path("fp32", "toy")).unlink()                         # simulate a lost file
    events = []
    pipeline.convert(cfg, emit=lambda s, m: events.append(m))
    assert any("exporting with" in m for m in events) and art.graph_path("fp32", "toy").is_file() and art.verify() == []


def test_different_config_in_the_same_output_directory_is_refused(converted, toy_file):
    out, cfg, _ = converted
    other = cfg.model_copy(update={"export": cfg.export.model_copy(update={"opset": 17})})
    with pytest.raises(ConfigError, match="different configuration"):
        pipeline.convert(other)
    pipeline.convert(other.model_copy(update={"output": str(out) + "_force"}))        # a fresh directory is fine


def test_validation_detects_a_corrupted_graph(tmp_path, toy_file):
    """A validator that cannot fail is worthless: damage the weights and expect 'failed' with a parity reason."""
    import onnx
    from onnx import numpy_helper
    out = tmp_path / "art"
    pipeline.convert(JobConfig.model_validate({"model": f"{toy_file}:build", "output": str(out), "steps": ["discover", "inspect", "analyze", "export", "package"]}))
    art = Artifact(out)
    g = art.graph_path("fp32", "toy")
    model = onnx.load(str(g))
    for init in model.graph.initializer:
        if len(init.dims) == 2:
            a = numpy_helper.to_array(init).copy()
            a[0] += 0.5
            init.CopyFrom(numpy_helper.from_array(a, init.name))
            break
    onnx.save(model, str(g))
    ctx = pipeline.open_artifact(out)                                  # golden-only mode: no original model needed
    res = steps.validate(ctx, variants=["fp32"])
    assert res["fp32"]["status"] == "failed"
    assert ctx.artifact.verify()                                       # and the checksum no longer matches the manifest either


def test_golden_only_validation_works_without_the_original_model(converted):
    ctx = pipeline.open_artifact(converted[0])
    assert ctx.live is False
    res = steps.validate(ctx, variants=["fp32"])
    assert res["fp32"]["status"] == "passed" and res["fp32"]["original_model_used"] is False


def test_optimize_on_an_existing_artifact_adds_variants(tmp_path, toy_file):
    out = tmp_path / "art"
    pipeline.convert(JobConfig.model_validate({"model": f"{toy_file}:build", "output": str(out), "steps": ["discover", "inspect", "analyze", "export", "validate", "package"]}))
    ctx = pipeline.open_artifact(out)
    ctx.config = ctx.config.model_copy(update={"optimization": ctx.config.optimization.model_copy(update={"precision": ["fp16"]})})
    assert steps.optimize(ctx) == ["fp16"]
    res = steps.validate(ctx, variants=["fp16"])
    assert res["fp16"]["status"] == "passed" and "fp16" in Artifact(out).read().variants


def test_a_component_that_cannot_be_converted_keeps_the_parent_graph_and_says_so(tmp_path, toy_file, monkeypatch):
    from modelsuite.optimization import passes

    def boom(*a, **k):
        raise passes.OptimizationError("synthetic converter failure")
    monkeypatch.setattr(passes, "convert_fp16", boom)
    out = tmp_path / "art"
    cfg = JobConfig.model_validate({"model": f"{toy_file}:build", "output": str(out), "optimization": {"precision": ["fp16", "fp16-weights"]},
                                    "steps": ["discover", "inspect", "analyze", "export", "optimize", "package"]})
    pipeline.convert(cfg)
    m = Artifact(out).read()
    assert "fp16" not in m.variants                                    # every component failed -> no pretend variant
    assert "fp16-weights" in m.variants
    report = json.loads((out / "reports" / "optimize.json").read_text())
    assert report["fp16"]["status"] == "failed" and "synthetic" in json.dumps(report["fp16"])


def test_unknown_step_name_is_rejected(toy_file, tmp_path):
    with pytest.raises(ConfigError, match="unknown step"):
        pipeline.convert(JobConfig.model_validate({"model": f"{toy_file}:build", "output": str(tmp_path / "x"), "steps": ["fly"]}))
