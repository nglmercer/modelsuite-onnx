"""Reverb-TTS as the first custom adapter and the regression test for the generic interfaces.

Uses a tiny, randomly initialised checkpoint with the real Reverb layout (see reverb_fixture.py) so it runs offline in seconds.
The real 297M checkpoint is covered by tests/test_reverb_real.py (marked slow)."""
from __future__ import annotations

import numpy as np
import pytest

for m in ("torch", "transformers", "reverb_tts", "safetensors", "onnx", "onnxruntime"):
    pytest.importorskip(m)

from modelsuite.core import pipeline
from modelsuite.core.config import JobConfig
from modelsuite.core.manifest import Artifact
from modelsuite.core.types import Support

OPTS = {"include_codec": False}


def cfg(model, out, **kw):
    return JobConfig.model_validate({"model": str(model), "output": str(out), "adapter_options": OPTS, **kw})


@pytest.fixture(scope="module")
def converted(tiny_reverb, tmp_path_factory):
    out = tmp_path_factory.mktemp("rv") / "art"
    pipeline.convert(cfg(tiny_reverb, out, optimization={"precision": ["fp16-weights"], "quantization": ["int8-dynamic"]}))
    m = Artifact(out).read()
    m._path = out
    return m


def test_reverb_is_detected_over_the_generic_adapters(tiny_reverb, tmp_path):
    ctx = pipeline.inspect_only(cfg(tiny_reverb, tmp_path / "o"))
    assert ctx.adapter.name == "reverb-tts" and ctx.detection.confidence >= 0.95
    assert [c.name for c in ctx.info.components] == ["core", "embed_prefix", "embed_target", "heads"]
    assert ctx.info.components[0].stateful and ctx.info.task == "text-to-speech"
    others = {n: d.confidence for n, d in ctx.candidates}
    assert others["reverb-tts"] > others["hf-transformers"]                      # specific beats generic


def test_analysis_carries_the_reverb_specific_knowledge(tiny_reverb, tmp_path):
    ctx = pipeline.analyze_only(JobConfig.model_validate({"model": str(tiny_reverb), "output": str(tmp_path / "o")}))      # codec included
    codes = {i.code for i in ctx.compat.issues}
    assert ctx.compat.support == Support.supported and {"sampler-not-exported", "gpu-codec-decoder", "license"} <= codes
    assert [c.name for c in ctx.info.components][-2:] == ["codec_decoder", "codec_encoder"]
    lic = next(i for i in ctx.compat.issues if i.code == "license" and "Model license" in i.message)
    assert "non-commercial" in lic.message


def test_four_graphs_with_the_documented_signatures(converted):
    core = converted.component("core")
    assert [t.name for t in core.signature.inputs] == ["x", "past_k", "past_v", "bias"]
    assert [t.name for t in core.signature.outputs] == ["hidden", "new_k", "new_v"]
    assert core.signature.inputs[1].shape[3] == "Pp" and core.stateful
    assert {c.name for c in converted.components} == {"core", "embed_prefix", "embed_target", "heads"}
    assert {(e["from"], e["to"]) for e in converted.topology} >= {("embed_prefix", "core"), ("core", "heads"), ("core", "core")}
    assert any(a.role == "postprocessor" and a.path == "sampler" for a in converted.assets)        # the host-side sampler is documented, not exported


def test_fp32_graphs_match_the_original_including_the_prefix_to_target_state_handoff(converted):
    v = converted.variants["fp32"].validation
    assert v["status"] == "passed"
    core = v["components"]["core"]
    assert {c["name"] for c in core["parity"]["cases"]} == {"target-pass", "prefix-pass", "polish-pass", "longer-target"}
    st = core["stateful"][0]
    assert st["case"] == "prefix-then-target" and st["status"] == "passed" and len(st["steps"]) == 2 and st["worst_max_abs"] < 1e-4


def test_the_official_sampler_through_onnx_reproduces_100_percent_of_the_codes(converted):
    """The headline claim of the original study ('100 % equal codes at all 8 codebooks'), now as an automated regression."""
    q = converted.variants["fp32"].validation["quality"][0]
    assert q["name"] == "sampler-code-agreement" and q["passed"] is True
    assert q["metrics"]["code_agreement"] == 1.0 and q["metrics"]["per_codebook_percent"] == [100.0] * 8


def test_lower_precision_is_measured_not_assumed(converted):
    fp16w = converted.variants["fp16-weights"].validation
    assert fp16w["status"] == "passed" and fp16w["quality"][0]["metrics"]["code_agreement"] >= 0.9
    int8 = converted.variants["int8-dynamic"].validation
    assert int8["status"] == "failed"                                           # dynamic int8 wrecks this model: that is the finding
    assert int8["quality"][0]["metrics"]["code_agreement"] < 0.9
    rec = converted.benchmark["recommendation"]
    assert rec["recommended"] != "int8-dynamic"


def test_license_gates_the_redistribution_verdict(converted):
    assert converted.source.license.id == "cc-by-nc-4.0" and converted.source.license.category == "non-commercial"
    r = converted.redistribution
    assert r.allowed is None and r.status == "conditional" and any("Non-commercial" in n for n in r.required_notices)   # never silently 'allowed'


def test_artifact_validates_against_bundled_goldens_without_the_checkpoint(converted):
    """Delete nothing, pass no model: the artifact must still verify itself (parity cases only; stateful/quality need the original)."""
    from modelsuite.core import steps
    ctx = pipeline.open_artifact(converted._path)
    assert ctx.live is False and ctx.plan is None
    res = steps.validate(ctx, variants=["fp32", "fp16-weights"])
    assert res["fp32"]["status"] == "passed" and res["fp32"]["original_model_used"] is False and res["fp32"]["quality"] == []
    assert res["fp16-weights"]["status"] == "passed"
    core = res["fp32"]["components"]["core"]
    assert {c["name"] for c in core["parity"]["cases"]} == {"target-pass", "prefix-pass", "polish-pass", "longer-target"}
    assert core["stateful"][0]["status"] == "skipped" and "original model" in core["stateful"][0]["reason"]


def test_missing_variant_is_a_clear_error(tiny_reverb, tmp_path):
    from modelsuite.core.errors import UnsupportedModelError
    with pytest.raises(UnsupportedModelError, match="no 'int4' weights"):
        pipeline.convert(JobConfig.model_validate({"model": str(tiny_reverb), "output": str(tmp_path / "o"), "adapter_options": {**OPTS, "variant": "int4"}}))
