"""A tiny, randomly initialised ZONOS3 LM checkpoint with the real parameter layout (params.json + model.pth).

The reference loader (`Zonos3LM.load`) accepts it unchanged, so adapter tests can run in seconds without the 1.8 GB release.
Sizes follow the release architecture (4 GDN heads x 128, kernel 4, 10 codebooks) but with few layers / experts / a small width."""
from __future__ import annotations

import json
from pathlib import Path

TINY_PARAMS = {
    "model": {
        "n_layers": 5, "dim": 64, "head_dim": 16, "n_heads": 4, "n_kv_heads": 2, "ffn_dim_multiplier": 2.0, "multiple_of": 64,
        "moe_ffn_dim_multiplier": 1.5, "moe_multiple_of": 64, "norm_eps": 1e-05, "rope_theta": 10000.0, "max_seqlen": 512,
        "n_codebooks": 10, "codebook_size": 32, "codebook_sizes": [32] * 10, "eoa_id": 32, "audio_pad_id": 33, "text_vocab": 100,
        "speaker_enabled": True, "speaker_embedding_dim": 16, "attention_pattern": "gdn3_global1", "use_xsa": False,
        "residual_type": "full_attnres", "block_attnres_block_size": 4, "moe_router_topk": 2, "moe_n_experts": 8, "moe_alpha": 0.5,
        "moe_router_dim": 128, "moe_router_type": "linear", "moe_router_use_eda": False, "moe_balancing_strategy": "quantile",
        "moe_start_from_layer": 1, "moe_end_from_layer": 0, "loss_softcap": 15.0, "condition_tokens_enabled": True,
        "condition_n_buckets": 10, "condition_count": 3, "condition_bucket_counts": [4, 4, 6],
        "condition_names": ["loudness", "fmax", "speaking_rate"], "audio_codec": "tiny", "speaker_reference_prefix_enabled": False,
    }
}


def make_tiny_lm(root: Path, seed: int = 0) -> Path:
    """Write params.json + model.pth under `root`; needs zonos3's reference package on sys.path."""
    import torch
    from zonos3.reference.lm import Zonos3LM, Zonos3LMConfig
    torch.manual_seed(seed)
    config = Zonos3LMConfig.from_params(TINY_PARAMS)
    model = Zonos3LM(config)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("balancing_biases"):
                p.copy_(torch.randn(p.shape) * 0.3)
            elif name.endswith(("A_log", "dt_bias", "temp", "out_weight")):
                p.copy_(torch.randn(p.shape) * 0.3 + (0.5 if name.endswith(("temp", "out_weight")) else 0.0))
            elif p.ndim <= 1 or name.endswith("rmsnorm_eda.weight") or name.endswith("norm.weight"):
                p.copy_(1.0 + 0.1 * torch.randn(p.shape))
            else:
                p.copy_(torch.randn(p.shape) * (0.9 / max(p.shape[-1], 1) ** 0.5))
    root.mkdir(parents=True, exist_ok=True)
    (root / "params.json").write_text(json.dumps(TINY_PARAMS))
    torch.save({k: v.detach().clone() for k, v in model.state_dict().items()}, root / "model.pth")
    return root


# ---- tiny codec ---------------------------------------------------------------------------------------------------------------------
def tiny_codec_config():
    """(cfg, head_cfg) of a tiny `ISTFTCodec`. The stage recipe is the one in Zyphra's own test-suite (model/tests/test_codec_stream.py,
    MIT, (c) 2026 Zyphra, Inc.); only the construction is repeated here so that tests need neither the 12 GB of codec checkpoints nor
    the reference repository's tests directory."""
    def moss(inp, out, ctx, rate):
        return {"type": "MossProjectedTransformer", "input_dimension": inp, "output_dimension": out, "d_model": 32, "num_heads": 2,
                "num_layers": 1, "dim_feedforward": 64, "context_frames": ctx, "frame_rate": rate, "layer_scale": 1.0,
                "attention_dropout": 0.0, "ffn_dropout": 0.0}

    def up(size):
        return {"type": "PatchedPretransform", "patch_size": size, "direction": "up"}

    stages = [moss(1024, 64, 6, 15), up(2), moss(32, 64, 12, 30), up(2), moss(32, 64, 24, 60), up(2), moss(32, 64, 48, 120), up(2),
              moss(32, 0, 96, 240), up(23)]
    cfg = {"decoder_stages": stages, "encoder_stages": [], "dual_encoding": {"width": 32, "hidden": 64, "depth": 4}, "residual_codebooks": 9,
           "residual_codebook_size": 4096, "semantic_codebook_size": 4096, "codebook_dim": 8, "quantizer_dropout": 0.0,
           "checkpoint_transformers": False, "internal_sample_rate": 44160, "frame_rate": 15, "internal_hop_length": 2944,
           "external_samples_per_frame": 2940}
    head = {"type": "mr_istft", "depth": 2, "hidden": 64, "n_ffts": [736, 2208], "fir_kernel": 127}
    return cfg, head


def make_tiny_codec(root: Path, seed: int = 0) -> Path:
    """Write `tiny_codec.pt` ({cfg, head, state}) under `root`; needs the reference package and the dac shim."""
    import torch
    from zonos3.codec.istft_head_a_20261002.model import ISTFTCodec
    torch.manual_seed(seed)
    cfg, head = tiny_codec_config()
    model = ISTFTCodec(cfg, head).eval()
    with torch.no_grad():
        model.head.mixer.weight.normal_(0, 0.05)       # both branches through every tap (the init is a delta on branch 0)
    root.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": cfg, "head": head, "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}, root / "tiny_codec.pt")
    return root


def make_tiny_release(root: Path, with_codec: bool = True) -> Path:
    """A directory shaped like the HF release (lm/zonos3_mini/{params.json,model.pth}, README with the license) plus an optional tiny codec."""
    make_tiny_lm(root / "lm" / "zonos3_mini")
    if with_codec:
        make_tiny_codec(root)
    (root / "README.md").write_text("---\nlicense: apache-2.0\nlibrary_name: zonos3\n---\n# tiny zonos3 fixture\n")
    return root
