"""ONNX-friendly wrappers for Gradium Phonon TTS (a modelsuite plugin, nothing here is in the core).

Three graphs reproduce the model without PyTorch at inference time:

* ``core``          text/voice/latent embeddings + causal attention with explicit KV cache: one graph for prefill and decode;
* ``flow``          the LSD flow head, one Euler step (the checkpoint's ``num_flow_steps`` is 1; the host loops otherwise);
* ``codec_decoder`` the Mimi-style continuous codec: latents [B,T,32] -> 24 kHz waveform [B,1,T*1920].

The host owns the tokenizer/prompt assembly, the noise draw (zero for greedy), denormalisation and the EOS countdown.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.models.pocket_tts.modeling_pocket_tts import apply_rotary_pos_emb


class CoreGraph(torch.nn.Module):
    """Backbone + EOS head with explicit KV cache.

    Voice and generated-latent embeddings arrive as *dense* tensors of the same sequence length as ``input_ids``
    (zeros where the token is not a placeholder). The graph scatters them with ``torch.where`` instead of the
    data-dependent ``masked_scatter`` the original uses, which makes a single graph serve prefill and decode.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model.model                  # GradiumPhononTTSModel
        self.eos_head = model.eos_head
        cfg = model.config
        self.L = cfg.num_hidden_layers
        self.KV = cfg.num_key_value_heads
        self.D = cfg.head_dim
        self.H = cfg.num_attention_heads
        self.bos_id = cfg.audio_bos_token_id
        self.prompt_id = cfg.audio_prompt_token_id
        self.latent_id = cfg.audio_latent_token_id

    def forward(self, input_ids, prompt_latents, audio_latents, attention_mask, position_ids, past_keys, past_values):
        B, S = input_ids.shape
        P = past_keys.shape[3]
        embeds = self.model.embed_tokens(input_ids)
        dtype = embeds.dtype

        bos = self.model.embed_audio_latents(self.model.bos_latent).to(dtype)
        embeds = torch.where((input_ids == self.bos_id).unsqueeze(-1), bos, embeds)

        prompt = self.model.audio_prompt_proj(prompt_latents.to(self.model.audio_prompt_proj.weight.dtype)).to(dtype)
        embeds = torch.where((input_ids == self.prompt_id).unsqueeze(-1), prompt, embeds)

        latent = self.model.embed_audio_latents(
            audio_latents.to(self.model.audio_latent_proj.weight.dtype)
        ).to(dtype)
        embeds = torch.where((input_ids == self.latent_id).unsqueeze(-1), latent, embeds)

        cos, sin = self.model.rotary_emb(embeds, position_ids=position_ids)
        # (B, S, P+S) -> (B, 1, S, P+S) additive causal + padding mask
        key_pos = torch.arange(P + S, device=input_ids.device)
        allowed = (key_pos[None, None, :] <= position_ids[:, :, None]) & attention_mask[:, None, :].bool()
        additive = torch.where(allowed.unsqueeze(1), torch.zeros((), dtype=dtype, device=input_ids.device),
                               torch.full((), -float("inf"), dtype=dtype, device=input_ids.device))

        new_k, new_v = [], []
        hidden = embeds
        for i, layer in enumerate(self.model.layers):
            attn = layer.self_attn
            x = layer.input_layernorm(hidden)
            q = attn.q_proj(x).view(B, S, self.H, self.D).transpose(1, 2)
            k = attn.k_proj(x).view(B, S, self.KV, self.D).transpose(1, 2)
            v = attn.v_proj(x).view(B, S, self.KV, self.D).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            k = torch.cat([past_keys[i], k], dim=2)
            v = torch.cat([past_values[i], v], dim=2)
            if self.KV != self.H:                      # GQA (this checkpoint has 8 == 8)
                k = k.repeat_interleave(self.H // self.KV, dim=1)
                v = v.repeat_interleave(self.H // self.KV, dim=1)
            scores = torch.matmul(q, k.transpose(-1, -2)) * attn.scaling + additive
            probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(dtype)
            out = torch.matmul(probs, v).transpose(1, 2).reshape(B, S, -1)
            hidden = hidden + attn.o_proj(out)
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
            if self.KV != self.H:
                k, v = k[:, ::self.H // self.KV], v[:, ::self.H // self.KV]
            new_k.append(k)
            new_v.append(v)
        hidden = self.model.norm(hidden)
        eos = self.eos_head(hidden)[..., 0]
        return hidden, eos, torch.stack(new_k), torch.stack(new_v)


class FlowGraph(torch.nn.Module):
    """One LSD Euler step: velocity = flow_head(latent, backbone_hidden, [start, end])."""

    def __init__(self, model):
        super().__init__()
        self.flow_head = model.flow_head
        # baked as initializers so the host never needs the original checkpoint's buffers
        self.register_buffer("latent_mean", model.latent_mean.clone())
        self.register_buffer("latent_std", model.latent_std.clone())

    def forward(self, latents, condition, start_time, end_time):
        velocity = self.flow_head(latents, condition, times=[start_time, end_time])
        # velocity stays in the NORMALISED space (that is what the backbone consumes next step);
        # audio_latent is the denormalised value the codec and the caller need.
        return velocity, velocity * self.latent_std + self.latent_mean


class CodecGraph(torch.nn.Module):
    """Whole-sequence decode with the processor's audio tokenizer.

    An empty DynamicCache keeps the decoder's mask builder off the packed-sequence path (which uses ``aten::diff``,
    unsupported by the legacy exporter) and is numerically identical to decoding without a cache.
    """

    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def forward(self, latents):
        from transformers.cache_utils import DynamicCache
        cache = DynamicCache(config=self.tokenizer.decoder.config)
        return self.tokenizer.decode(latents, decoder_cache=cache).audio_values
