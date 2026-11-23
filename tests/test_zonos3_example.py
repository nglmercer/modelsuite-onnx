"""examples/zonos3-mini: a real-world adapter written as a plugin, tested on a tiny randomly initialised release.

Needs Zyphra's reference code (the numerical oracle): git clone https://github.com/Zyphra/ZONOS3-Mini and
    ZONOS3_SRC=<clone>/model/src pytest tests/test_zonos3_example.py
Skipped when ZONOS3_SRC is not set. The real 800M checkpoint is covered by tests/test_zonos3_real.py (pytest -m slow)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

EX = Path(__file__).resolve().parents[1] / "examples" / "zonos3-mini"
SRC = os.environ.get("ZONOS3_SRC")
pytestmark = pytest.mark.skipif(not SRC or not (Path(SRC) / "zonos3").is_dir(), reason="set ZONOS3_SRC=<ZONOS3-Mini clone>/model/src")
import importlib.util
for m in ("torch", "onnx", "onnxruntime"):
    pytest.importorskip(m)
for m in ("einops", "dac"):                       # do not *import* dac: its __init__ needs audiotools; reference_env shims the two submodules we use
    if importlib.util.find_spec(m) is None:
        pytest.skip(f"{m} is not installed (pip install einops; pip install --no-deps descript-audio-codec==1.0.0)", allow_module_level=True)

sys.path.insert(0, str(EX))

from modelsuite.core import pipeline, registry, steps
from modelsuite.core.config import JobConfig
from modelsuite.core.manifest import Artifact
from modelsuite.core.types import Support


@pytest.fixture(scope="module")
def plugin():
    registry.load_plugin(str(EX / "zonos3_plugin.py"))
    yield
    registry.adapters._items.pop("zonos3-mini", None)


@pytest.fixture(scope="module")
def release(plugin, tmp_path_factory):
    from reference_env import ensure_reference
    from tiny_checkpoint import make_tiny_release
    ensure_reference(SRC, codec=True)
    return make_tiny_release(tmp_path_factory.mktemp("z3") / "release")


def cfg(model, out, **kw):
    return JobConfig.model_validate({"model": str(model), "output": str(out), "adapter_options": {"source_dir": SRC}, **kw})


@pytest.fixture(scope="module")
def converted(release, tmp_path_factory):
    out = tmp_path_factory.mktemp("z3out") / "art"
    pipeline.convert(cfg(release, out, optimization={"precision": ["fp16-weights"], "quantization": ["int8-dynamic"]}))
    m = Artifact(out).read()
    m._path = out
    return m


def test_detected_over_the_generic_adapters_and_analysed(release, tmp_path):
    ctx = pipeline.analyze_only(cfg(release, tmp_path / "o"))
    assert ctx.adapter.name == "zonos3-mini" and ctx.detection.confidence >= 0.95
    assert ctx.info.license == "apache-2.0" and [c.name for c in ctx.info.components] == ["lm", "codec_decoder"]
    assert ctx.compat.support == Support.supported
    assert {"stateful-lm", "decode-speed", "not-exported"} <= {i.code for i in ctx.compat.issues}


def test_lm_signature_and_state_links(converted):
    lm = converted.component("lm")
    assert [t.name for t in lm.signature.inputs] == ["ids", "speaker_vec", "speaker_flag", "cond_index", "past_k", "past_v", "conv_state", "gdn_state"]
    assert lm.stateful and {(l["output"], l["input"]) for l in lm.state_links} == {("new_k", "past_k"), ("new_v", "past_v"), ("new_conv", "conv_state"), ("new_state", "gdn_state")}
    shapes = {t.name: t.shape for t in lm.signature.inputs}
    assert shapes["past_k"][1] == "P" and shapes["ids"] == ["T", 11] and shapes["gdn_state"][1:] == [4, 128, 128]      # GDN: 4 heads x 128 x 128
    assert {a.path for a in converted.assets if a.role == "postprocessor"} == {"host-pipeline"}


def test_gdn_and_moe_became_onnx_loops(converted):
    import onnx
    m = onnx.load(str(Artifact(converted._path).graph_path("fp32", "lm")), load_external_data=False)
    assert sum(n.op_type == "Loop" for n in m.graph.node) == 4 + 4               # tiny fixture: 4 GDN layers + 4 MoE layers (layers 1..4)


def test_lm_matches_zyphras_reference_including_state_and_conditioning_rows(converted):
    v = converted.variants["fp32"].validation
    assert v["status"] == "passed"
    lm = v["components"]["lm"]
    assert {c["name"] for c in lm["parity"]["cases"]} == {"prefill", "decode-step", "chunk", "long-prefill"}
    assert max(c["summary"]["max_abs"] for c in lm["parity"]["cases"]) < 1e-4
    st = lm["stateful"][0]
    assert st["case"] == "prefill-then-decode" and st["status"] == "passed" and len(st["steps"]) == 6 and st["trajectory_max_abs"] < 1e-4


def test_greedy_rollout_reproduces_the_reference_tokens(converted):
    q = {x["name"]: x for x in converted.variants["fp32"].validation["quality"]}["greedy-rollout"]
    assert q["passed"] and q["metrics"]["token_agreement"] == 1.0


def test_codec_decoder_matches_the_reference_for_any_frame_count(converted):
    comp = converted.variants["fp32"].validation["components"]["codec_decoder"]["parity"]
    assert {c["name"] for c in comp["cases"]} == {"example", "longer", "short"}
    assert all(c["summary"]["max_abs"] < 1e-6 and 0.99 < c["outputs"][0]["rms_got"] / c["outputs"][0]["rms_ref"] < 1.01 for c in comp["cases"])
    sig = converted.component("codec_decoder").signature
    assert sig.inputs[0].shape == [1, 10, "T"] and sig.outputs[0].shape[-1] == "N"


def test_codec_output_length_is_exactly_2940_samples_per_frame(converted):
    from modelsuite.runtimes.ort import create_runner
    r = create_runner(Artifact(converted._path).graph_path("fp32", "codec_decoder"), "cpu")
    for frames in (1, 2, 9, 31):
        assert r.run({"codes": np.zeros((1, 10, frames), np.int64)})["audio"].shape == (1, 1, 2940 * frames)


def test_lower_precision_is_measured_and_the_failure_is_visible(converted):
    assert converted.variants["fp16-weights"].validation["status"] == "passed"
    int8 = converted.variants["int8-dynamic"].validation
    assert int8["status"] == "failed"                                              # random tiny weights + dynamic int8: the report says so
    assert int8["components"]["lm"]["structural"]["status"] == "passed"            # (and the graph is valid: sorted despite the Loop bodies)


def test_golden_only_revalidation_needs_neither_the_checkpoint_nor_the_reference_code(converted):
    ctx = pipeline.open_artifact(converted._path)
    res = steps.validate(ctx, variants=["fp32"])["fp32"]
    assert res["status"] == "passed" and res["original_model_used"] is False


def test_a_wrong_state_hand_off_is_caught(converted):
    """The stateful test must fail if the exported cache semantics were wrong: swap k and v."""
    from modelsuite.core.plan import StateLink
    from modelsuite.runtimes.ort import create_runner
    from modelsuite.validation.stateful import run_stateful
    ctx = pipeline.open_artifact(converted._path, model=converted.source.reference)
    comp = ctx.plan.component("lm")
    case = comp.stateful[0]
    good = run_stateful(comp, case, create_runner(ctx.artifact.graph_path("fp32", "lm"), "cpu"), comp.reference, ctx.config.validation)
    assert good.status.value == "passed"
    # a graph that returns the old state unchanged (a plausible bug) would diverge from step 1 on: emulate it with a runner that ignores new state
    class Frozen:
        def __init__(self, inner): self.inner, self.first = inner, None
        def run(self, inputs):
            out = self.inner.run(inputs)
            if self.first is None:
                self.first = {k: out[k] for k in ("new_conv", "new_state")}
            return {**out, **self.first}                                           # recurrent state never advances
    bad = run_stateful(comp, case, Frozen(create_runner(ctx.artifact.graph_path("fp32", "lm"), "cpu")), comp.reference, ctx.config.validation)
    assert bad.status.value == "failed" and any(not s.passed for s in bad.steps)


def test_without_the_reference_code_the_adapter_says_how_to_get_it(release, tmp_path, monkeypatch):
    monkeypatch.delenv("ZONOS3_SRC", raising=False)
    from modelsuite.core.errors import UnsupportedModelError
    import importlib
    ref_mod = importlib.import_module("reference_env")
    monkeypatch.setattr(ref_mod, "find_source", lambda explicit=None: (_ for _ in ()).throw(ref_mod.ReferenceUnavailable("needs the reference: set ZONOS3_SRC")))
    j = JobConfig.model_validate({"model": str(release), "output": str(tmp_path / "o")})
    with pytest.raises(UnsupportedModelError, match="ZONOS3_SRC"):
        pipeline.convert(j)


def test_meta_built_codec_equals_the_normally_built_one(release):
    """The real checkpoint is loaded through build_empty_codec (the normal constructor needs ~8 GB for parts it discards): same function, tiny model."""
    import torch
    from reference_env import ensure_reference
    ensure_reference(SRC, codec=True)
    from tiny_checkpoint import tiny_codec_config
    from codec_graph import build_empty_codec
    from zonos3.codec.istft_head_a_20261002.model import ISTFTCodec
    cfg, head = tiny_codec_config()
    blob = torch.load(release / "tiny_codec.pt", map_location="cpu", weights_only=True)
    normal = ISTFTCodec(blob["cfg"], blob["head"])
    normal.load_state_dict(blob["state"], strict=True)
    lean = build_empty_codec(blob["cfg"], blob["head"])
    missing = lean.load_state_dict(blob["state"], strict=False)
    assert not missing.unexpected_keys and all(k.startswith("head.window_") or k == "" for k in missing.missing_keys) or not missing.missing_keys
    codes = torch.from_numpy(np.random.default_rng(3).integers(0, 4096, (1, 10, 9)))
    with torch.inference_mode():
        a, b = normal.eval()(codes), lean.eval()(codes)
    assert torch.allclose(a, b, atol=1e-7) and not any(p.is_meta for p in lean.parameters())
