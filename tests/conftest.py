"""Shared fixtures. Heavy fixtures are session-scoped and built from tiny, randomly initialised real models (no downloads)."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

pytest.importorskip("numpy")


def need(*mods: str):
    for m in mods:
        pytest.importorskip(m)


@pytest.fixture(scope="session")
def toy_file(tmp_path_factory) -> Path:
    """file.py:build -> a small MLP with dropout (training-mode hazard), dynamic batch and two shape variants."""
    need("torch", "onnx", "onnxruntime")
    p = tmp_path_factory.mktemp("toy") / "toy_model.py"
    p.write_text(textwrap.dedent('''
        import numpy as np, torch
        from modelsuite.adapters.torch_module import TorchModuleSpec

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.Linear(16, 64); self.b = torch.nn.Linear(64, 8); self.drop = torch.nn.Dropout(0.5)
            def forward(self, x):
                return self.b(self.drop(torch.relu(self.a(x))))

        def build():
            torch.manual_seed(0)
            ex = lambda b=4, seed=0: {"x": np.random.default_rng(seed).standard_normal((b, 16)).astype(np.float32)}
            # NOTE: the module is deliberately left in training mode: the suite must force eval() itself
            return TorchModuleSpec(module=Net(), example_inputs=ex, output_names=["y"],
                                   dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}},
                                   shape_variants={"batch1": lambda: ex(1, 1), "batch9": lambda: ex(9, 2)}, license="mit", name="toy")
    '''))
    return p


@pytest.fixture(scope="session")
def tiny_bert(tmp_path_factory) -> Path:
    need("torch", "transformers", "onnx", "onnxruntime")
    import torch
    from transformers import BertConfig, BertForSequenceClassification
    torch.manual_seed(0)
    d = tmp_path_factory.mktemp("hf") / "tiny_bert"
    BertForSequenceClassification(BertConfig(vocab_size=100, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                                             intermediate_size=64, max_position_embeddings=64, num_labels=3)).save_pretrained(d)
    (d / "README.md").write_text("---\nlicense: apache-2.0\n---\n# tiny bert\n")
    return d


@pytest.fixture(scope="session")
def tiny_llama(tmp_path_factory) -> Path:
    need("torch", "transformers", "onnx", "onnxruntime")
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    d = tmp_path_factory.mktemp("hf") / "tiny_llama"
    LlamaForCausalLM(LlamaConfig(vocab_size=100, hidden_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                 intermediate_size=64, max_position_embeddings=64)).save_pretrained(d)
    (d / "README.md").write_text("---\nlicense: mit\n---\n")
    return d


@pytest.fixture(scope="session")
def tiny_reverb(tmp_path_factory) -> Path:
    need("torch", "transformers", "reverb_tts", "safetensors", "onnx", "onnxruntime")
    from reverb_fixture import make_tiny_reverb
    return make_tiny_reverb(tmp_path_factory.mktemp("reverb") / "tiny_reverb")


def quiet(step: str, msg: str) -> None:
    pass


@pytest.fixture
def run_convert():
    """convert(model, out, **overrides) -> JobContext, with a collecting emitter available as ctx.events."""
    from modelsuite.core import pipeline
    from modelsuite.core.config import JobConfig

    def go(model: str, out: Path, force=None, **overrides):
        events: list[tuple[str, str]] = []
        cfg = JobConfig.model_validate({"model": str(model), "output": str(out), **overrides})
        ctx = pipeline.convert(cfg, emit=lambda s, m: events.append((s, m)), force=force)
        ctx.events = events
        return ctx
    return go
