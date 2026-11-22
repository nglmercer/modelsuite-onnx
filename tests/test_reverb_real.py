"""The real 297M Reverb-TTS checkpoint + Higgs codec (downloads ~1.4 GB, ~3 GB RAM, several minutes). Run with:  pytest -m slow

Asserts the claims of the original study (docs/feasibility.md) as an automated regression."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow
for m in ("torch", "transformers", "reverb_tts", "onnx", "onnxruntime", "huggingface_hub"):
    pytest.importorskip(m)

from modelsuite.core import pipeline
from modelsuite.core.config import JobConfig
from modelsuite.core.manifest import Artifact

MB = 2 ** 20


@pytest.fixture(scope="module")
def real(tmp_path_factory):
    out = tmp_path_factory.mktemp("real") / "reverb"
    cfg = JobConfig.model_validate({"model": "software-mansion/Reverb-TTS", "revision": "v1.0.0", "output": str(out),
                                    "adapter_options": {"variant": "fp16"}, "optimization": {"precision": ["fp16-weights"]},
                                    "steps": ["discover", "inspect", "analyze", "export", "optimize", "validate", "package"]})
    pipeline.convert(cfg)
    return Artifact(out).read()


def test_graph_sizes_match_the_original_table(real):
    size = {c: real.variants["fp32"].sizes_bytes[c] / MB for c in real.variants["fp32"].sizes_bytes}
    # README table (MB, 1e6) converted to MiB, +-5 %
    for name, mb in {"core": 1058, "embed_prefix": 98, "embed_target": 34, "heads": 34, "codec_decoder": 87, "codec_encoder": 654}.items():
        assert size[name] == pytest.approx(mb * 1e6 / MB, rel=0.05), name


def test_official_sampler_through_onnx_gives_100_percent_equal_codes(real):
    q = real.variants["fp32"].validation["quality"][0]
    assert q["passed"] and q["metrics"]["per_codebook_percent"] == [100.0] * 8


def test_codec_numbers_match_the_original_study(real):
    comps = real.variants["fp32"].validation["components"]
    dec = comps["codec_decoder"]["parity"]["cases"]
    assert max(c["summary"]["max_abs"] for c in dec) < 1e-5                       # original: 7e-7
    enc = comps["codec_encoder"]["parity"]["cases"]
    assert all(c["outputs"][0]["equal_fraction"] >= 0.995 for c in enc)           # original: 99.5-100 %


def test_prefix_to_target_state_handoff_and_every_variant_validated(real):
    core = real.variants["fp32"].validation["components"]["core"]
    assert core["stateful"][0]["status"] == "passed"
    assert all(v.validation["status"] == "passed" for v in real.variants.values())
    assert real.redistribution.status == "conditional" and real.source.license.id == "cc-by-nc-4.0"
    assert real.licenses["components"]["codec_decoder"]["id"] == "llama3"
