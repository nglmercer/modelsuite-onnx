"""Builds a tiny, randomly initialised checkpoint with the exact Reverb layout (config.json with llm_config +
model.safetensors, real reverb_tts.Network keys) so the Reverb adapter can be tested quickly and offline."""
from __future__ import annotations

import json
from pathlib import Path

LLM = {"vocab_size": 200, "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4,
       "num_key_value_heads": 2, "head_dim": 16, "max_position_embeddings": 256, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
       "hidden_act": "silu", "tie_word_embeddings": False, "attention_bias": False}


def make_tiny_reverb(root: Path, variant: str = "fp16", rank: int | None = 16, seed: int = 0) -> Path:
    import torch
    from reverb_tts.model import Network
    from safetensors.torch import save_file
    torch.manual_seed(seed)
    net = Network(LLM, rank).eval()
    for p in net.parameters():            # unit-scale weights keep logits sensible
        torch.nn.init.normal_(p, std=0.08) if p.ndim > 1 else None
    d = root / "model" / "safetensors" / variant
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"llm_config": LLM}))
    state = {k: v.contiguous() for k, v in net.state_dict().items()}
    save_file(state, str(d / "model.safetensors"))
    (root / "README.md").write_text("---\nlicense: cc-by-nc-4.0\n---\n# tiny reverb fixture\n")
    return root
