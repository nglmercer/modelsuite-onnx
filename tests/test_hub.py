"""Hugging Face Hub discovery against real, tiny public models. Run with:  pytest -m network"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.network
pytest.importorskip("huggingface_hub")
pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("onnxruntime")

from modelsuite.core import pipeline
from modelsuite.core.config import JobConfig
from modelsuite.core.discovery import resolve
from modelsuite.core.manifest import Artifact

REPO = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def test_metadata_only_resolution_downloads_no_weights_but_knows_the_size(tmp_path):
    s = resolve(REPO, cache_dir=str(tmp_path / "fresh-cache"))          # a fresh cache: what resolve() itself fetches
    assert s.kind == "huggingface" and len(s.revision) == 40 and s.revision_kind == "commit" and not s.weights_complete
    assert not any(p.suffix in (".safetensors", ".bin") for p in s.files())            # nothing heavy fetched yet
    assert (s.path / "config.json").is_file() and s.config["model_type"] == "llama"
    s.fetch_weights()
    assert s.weights_complete and any(p.suffix == ".safetensors" for p in s.files())   # ...until an export asks for them
    w = s.weight_summary()
    assert w["parameters"] == 1_032_272 and w["dtypes"] == ["F32"]                      # read from remote safetensors headers


def test_hub_model_converts_validates_and_records_the_commit(tmp_path):
    cfg = JobConfig.model_validate({"model": REPO, "output": str(tmp_path / "a")})
    pipeline.convert(cfg)
    m = Artifact(tmp_path / "a").read()
    assert m.source.repo_id == REPO and len(m.source.revision) == 40 and m.source.revision_kind == "commit"
    assert m.variants["fp32"].validation["status"] == "passed"
    assert m.redistribution.status == "unknown"                                          # this repo declares no license: never assumed free


def test_pinning_a_revision_is_honoured():
    first = resolve(REPO)
    again = resolve(REPO, revision=first.revision)
    assert again.revision == first.revision


def test_unknown_repo_is_a_discovery_error():
    from modelsuite.core.errors import DiscoveryError
    with pytest.raises(DiscoveryError):
        resolve("hf-internal-testing/this-repo-does-not-exist-xyz")
