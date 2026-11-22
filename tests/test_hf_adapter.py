"""Generic Hugging Face adapter on real tiny checkpoints: text head, KV-cache LM, and honest reports for what it cannot do."""
from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from modelsuite.core import pipeline
from modelsuite.core.config import JobConfig
from modelsuite.core.errors import UnsupportedModelError
from modelsuite.core.manifest import Artifact
from modelsuite.core.types import Support


def cfg(model, out, **kw):
    return JobConfig.model_validate({"model": str(model), "output": str(out), **kw})


def test_bert_classifier_detect_inspect_analyze(tiny_bert, tmp_path):
    ctx = pipeline.analyze_only(cfg(tiny_bert, tmp_path / "o"))
    assert ctx.adapter.name == "hf-transformers" and ctx.info.task == "text-classification" and ctx.info.parameters > 1000
    assert ctx.info.license == "apache-2.0" and ctx.compat.support == Support.supported
    assert ctx.compat.strategy["model"] == ["torch-legacy", "torch-dynamo"] and ctx.compat.estimates["fits_in_ram"] is True


def test_bert_converts_with_padding_masks_and_dynamic_shapes(tiny_bert, tmp_path, run_convert):
    ctx = run_convert(tiny_bert, tmp_path / "bert", optimization={"precision": ["fp16"]})
    m = Artifact(tmp_path / "bert").read()
    assert m.variants["fp32"].validation["status"] == "passed" and m.variants["fp16"].validation["status"] == "passed"
    cases = {c["name"]: c for c in m.variants["fp32"].validation["components"]["model"]["parity"]["cases"]}
    assert set(cases) == {"example", "longer-batch1", "wide-padded"}
    assert cases["wide-padded"]["input_shapes"]["attention_mask"] == [3, 11] and cases["wide-padded"]["summary"]["max_abs"] < 1e-5
    assert cases["example"]["outputs"][0]["argmax_agreement"] == 1.0
    assert [t.name for t in m.component("model").signature.inputs] == ["input_ids", "attention_mask", "token_type_ids"]
    assert any(a.path == "assets/config.json" for a in m.assets)


def test_llama_exports_one_graph_with_a_kv_cache_and_passes_stateful_and_quality(tiny_llama, tmp_path, run_convert):
    ctx = run_convert(tiny_llama, tmp_path / "llama")
    m = Artifact(tmp_path / "llama").read()
    comp = m.component("model")
    assert comp.stateful and [t.name for t in comp.signature.inputs] == ["input_ids", "attention_mask", "past_keys", "past_values"]
    v = m.variants["fp32"].validation
    assert v["status"] == "passed"
    st = v["components"]["model"]["stateful"][0]
    assert st["status"] == "passed" and len(st["steps"]) == 8 and st["worst_max_abs"] < 1e-5
    q = {x["name"]: x for x in v["quality"]}
    assert q["kv-cache-consistency"]["passed"] and q["greedy-generation"]["passed"] and q["greedy-generation"]["metrics"]["agreement"] == 1.0
    names = {c["name"] for c in v["components"]["model"]["parity"]["cases"]}
    assert "prefill-empty-past" in names                    # past length 0 is a distinct code path


def test_greedy_check_fails_when_the_graph_is_wrong(tiny_llama, tmp_path, run_convert):
    import onnx
    from onnx import numpy_helper
    run_convert(tiny_llama, tmp_path / "llama3", steps=["discover", "inspect", "analyze", "export", "package"])
    g = Artifact(tmp_path / "llama3").graph_path("fp32", "model")
    model = onnx.load(str(g))
    for init in model.graph.initializer:                    # damage the output projection
        if len(init.dims) == 2 and init.dims[0] == 100 or (len(init.dims) == 2 and init.dims[1] == 100):
            a = numpy_helper.to_array(init).copy()
            init.CopyFrom(numpy_helper.from_array(a[::-1].copy() if a.shape[0] == 100 else a[:, ::-1].copy(), init.name))
    onnx.save(model, str(g))
    from modelsuite.core import steps
    ctx = pipeline.open_artifact(tmp_path / "llama3", model=str(tiny_llama))
    res = steps.validate(ctx, variants=["fp32"])["fp32"]
    assert res["status"] == "failed"
    assert any(q["passed"] is False for q in res["quality"]) or res["components"]["model"]["parity"]["status"] == "failed"


# ---- honest reports for what is not supported ------------------------------------------------------------------------------------------------
def _fake_model(tmp_path, config, name="m"):
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps(config))
    return d


def test_unknown_model_type_is_reported_as_unsupported_with_the_reason(tmp_path):
    d = _fake_model(tmp_path, {"model_type": "totally_new_arch_v9", "architectures": ["TotallyNewForCausalLM"], "hidden_size": 8})
    ctx = pipeline.analyze_only(cfg(d, tmp_path / "o"))
    assert ctx.compat.support == Support.unsupported
    codes = {i.code for i in ctx.compat.issues}
    assert {"unknown-model-type", "unknown-architecture"} <= codes
    with pytest.raises(UnsupportedModelError, match="cannot be converted"):
        pipeline.convert(cfg(d, tmp_path / "o2"))
    assert (tmp_path / "o2" / "reports" / "analyze.json").is_file()        # the report is the deliverable


def test_remote_code_and_quantized_and_seq2seq_and_audio_are_flagged(tmp_path):
    cases = {"remote-code": {"model_type": "bert", "architectures": ["BertModel"], "auto_map": {"AutoModel": "x.Y"}},
             "quantized-checkpoint": {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "quantization_config": {"quant_method": "gptq"}},
             "unsupported-generic-task": {"model_type": "t5", "architectures": ["T5ForConditionalGeneration"], "is_encoder_decoder": True},
             "audio-modality": {"model_type": "wav2vec2", "architectures": ["Wav2Vec2ForCTC"]}}
    for code, conf in cases.items():
        ctx = pipeline.analyze_only(cfg(_fake_model(tmp_path, conf, code), tmp_path / "o"))
        assert code in {i.code for i in ctx.compat.issues}, code
        assert ctx.compat.support == Support.unsupported, code


def test_remote_code_can_be_allowed_explicitly(tmp_path):
    d = _fake_model(tmp_path, {"model_type": "bert", "architectures": ["BertModel"], "auto_map": {"AutoModel": "x.Y"}})
    ctx = pipeline.analyze_only(cfg(d, tmp_path / "o", export={"trust_remote_code": True}))
    assert "remote-code" not in {i.code for i in ctx.compat.issues}


def test_non_uniform_kv_models_fall_back_to_a_no_cache_graph_with_a_warning(tmp_path):
    d = _fake_model(tmp_path, {"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"], "vocab_size": 50, "n_layer": 2, "n_head": 2, "n_embd": 8})
    ctx = pipeline.analyze_only(cfg(d, tmp_path / "o"))
    nk = next(i for i in ctx.compat.issues if i.code == "no-kv-cache")
    assert "not in the verified uniform-KV set" in nk.message and ctx.compat.support == Support.partial


def test_ram_estimate_blocks_a_model_that_cannot_fit(tmp_path, monkeypatch):
    from modelsuite.adapters import base
    d = _fake_model(tmp_path, {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 8, "num_hidden_layers": 1,
                               "num_attention_heads": 2, "vocab_size": 10})
    import psutil
    monkeypatch.setattr(base, "memory_estimate", lambda info: {"parameters": 70_000_000_000, "fp32_bytes": 280e9, "export_peak_ram_bytes_estimate": int(700e9),
                                                              "available_ram_bytes": int(8e9), "fits_in_ram": False, "needs_external_data": True})
    monkeypatch.setattr("modelsuite.adapters.hf_transformers.memory_estimate", base.memory_estimate)
    ctx = pipeline.analyze_only(cfg(d, tmp_path / "o"))
    assert "ram" in {i.code for i in ctx.compat.errors} and ctx.compat.support == Support.unsupported


# ---- Optimum, in an isolated interpreter ------------------------------------------------------------------------------------------------------
def _optimum_python():
    import os
    from modelsuite.exporters.subprocess_tools import OptimumExporter
    ok, why = OptimumExporter().availability()
    return ok, why


def test_optimum_unavailable_is_a_clear_error_not_a_crash(tiny_llama, tmp_path, monkeypatch):
    monkeypatch.setenv("MODELSUITE_OPTIMUM_PYTHON", "/nonexistent/python")
    from modelsuite.core.errors import UnsupportedModelError
    with pytest.raises(UnsupportedModelError, match="optimum: cannot run /nonexistent/python"):      # caught by analyze, before any export work
        pipeline.convert(cfg(tiny_llama, tmp_path / "o", adapter_options={"mode": "optimum"}, export={"exporter": "optimum"}, steps=["discover", "inspect", "export"]))


def test_optimum_export_runs_in_its_own_interpreter_and_validates_against_the_original(tiny_llama, tmp_path):
    """Needs a second environment (optimum-onnx pins an older transformers):  MODELSUITE_OPTIMUM_PYTHON=venv-optimum/bin/python pytest"""
    ok, why = _optimum_python()
    if not ok:
        pytest.skip(f"optimum not available ({why}); set MODELSUITE_OPTIMUM_PYTHON to an interpreter that has optimum-onnx")
    out = tmp_path / "opt"
    pipeline.convert(cfg(tiny_llama, out, adapter_options={"mode": "optimum"}, export={"exporter": "optimum"}))
    m = Artifact(out).read()
    c = m.component("model")
    assert c.exporter == "optimum" and any(t.name.startswith("past_key_values.0") for t in c.signature.inputs)
    assert any(e.name == "optimum:optimum-onnx" or e.name.startswith("optimum") for e in m.exporters)          # its own version, not ours
    assert m.variants["fp32"].validation["status"] == "passed"
    assert any(a.path.endswith("config.json") for a in m.assets)
