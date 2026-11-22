"""torch modules that get exported for Reverb-TTS. Ported from scripts/export_net.py and scripts/export_codec.py
(the original repo scripts stay as the reference); the maths is unchanged.

Imports torch / reverb_tts at module level, so import this module only from inside adapter build() callbacks."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from reverb_tts.model import CODEBOOKS, CODES, MASK


def rms(x, w, eps):
    x32 = x.float()
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return w * x32.to(x.dtype)


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


class Core(torch.nn.Module):
    """Hand-written Qwen3 stack (explicit RoPE / GQA / bias), mirroring reverb_tts.sampler.Batch for batch size 1."""

    def __init__(self, network):
        super().__init__()
        self.n = network
        cfg = network.llm.config
        self.heads, self.kv, self.hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.eps = cfg.rms_norm_eps
        theta = cfg.rope_parameters["rope_theta"] if getattr(cfg, "rope_parameters", None) else cfg.rope_theta
        self.register_buffer("inv_freq", 1.0 / (theta ** (torch.arange(0, self.hd, 2).float() / self.hd)), persistent=False)

    def rope(self, positions):
        freqs = positions.float()[:, None] * self.inv_freq[None]
        emb = torch.cat([freqs, freqs], -1)
        return emb.cos()[None, None], emb.sin()[None, None]

    def layer(self, i, x, cos, sin, past, bias):
        L = self.n.llm.layers[i]
        a = L.self_attn
        B, S, _ = x.shape
        h = rms(x, L.input_layernorm.weight, self.eps)
        q = rms(a.q_proj(h).view(B, S, self.heads, self.hd), a.q_norm.weight, self.eps).transpose(1, 2)
        k = rms(a.k_proj(h).view(B, S, self.kv, self.hd), a.k_norm.weight, self.eps).transpose(1, 2)
        v = a.v_proj(h).view(B, S, self.kv, self.hd).transpose(1, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        new_k, new_v = k, v
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        rep = self.heads // self.kv
        kk = k.repeat_interleave(rep, dim=1)
        vv = v.repeat_interleave(rep, dim=1)
        scores = (q @ kk.transpose(-1, -2)) / math.sqrt(self.hd)
        if bias is not None:
            scores = scores + bias
        att = torch.softmax(scores.float(), dim=-1).to(q.dtype) @ vv
        att = att.transpose(1, 2).reshape(B, S, self.heads * self.hd)
        x = x + a.o_proj(att)
        h = rms(x, L.post_attention_layernorm.weight, self.eps)
        m = L.mlp
        x = x + m.down_proj(torch.nn.functional.silu(m.gate_proj(h)) * m.up_proj(h))
        return x, new_k, new_v


class CoreGraph(Core):
    """One graph for both prefix and target passes.

    x [1,S,H]; past_k/past_v [L,1,KV,Pp,hd] (Pp may be 0); bias [S, Pp+S] additive attention bias.
    Positions are Pp..Pp+S-1. Returns final-normed hidden [1,S,H] and this call's rotated keys/values."""

    def forward(self, x, past_k, past_v, bias):
        S = x.shape[1]
        Pp = past_k.shape[3]
        cos, sin = self.rope(Pp + torch.arange(S))
        mask = bias[None, None]
        ks, vs = [], []
        for i in range(len(self.n.llm.layers)):
            x, k, v = self.layer(i, x, cos, sin, (past_k[i], past_v[i]), mask)
            ks.append(k)
            vs.append(v)
        return rms(x, self.n.llm.norm.weight, self.eps), torch.stack(ks), torch.stack(vs)


class EmbedPrefix(torch.nn.Module):
    def __init__(self, network):
        super().__init__()
        self.n = network

    def forward(self, text_ids, ref_codes):
        offsets = CODES * torch.arange(CODEBOOKS)[None, :, None]
        return torch.cat([self.n.llm.embed_tokens(text_ids), self.n.audio_embeddings(ref_codes + offsets).sum(1)], dim=1)


class EmbedTarget(torch.nn.Module):
    def __init__(self, network):
        super().__init__()
        self.n = network

    def forward(self, tokens):
        offsets = CODES * torch.arange(CODEBOOKS)[None, :, None]
        return self.n.audio_embeddings(tokens + offsets).sum(1)


class Heads(torch.nn.Module):
    """hidden [1,F,H] -> log-probs [1,8,F,1024] fp32 (the mask token is never a prediction)."""

    def __init__(self, network):
        super().__init__()
        self.n = network

    def forward(self, hidden):
        Fr = hidden.shape[1]
        logits = (hidden @ self.n.audio_heads.weight.T).float().view(1, Fr, CODEBOOKS, CODES).transpose(1, 2)
        return logits.log_softmax(-1)[..., :MASK]


# ---------------------------------------------------------------------------- Higgs Audio v2 codec
class CodecDecoder(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, codes):  # [1,8,T] int64 -> [1,1,T*960]
        q = 0
        for i, qz in enumerate(self.m.quantizer.quantizers):
            q = q + qz.decode(codes[:, i])
        return self.m.acoustic_decoder(self.m.fc2(q.transpose(1, 2)).transpose(1, 2))


class CodecEncoder(torch.nn.Module):
    """wav24 [1,1,N] + wav16 [1,N16] (the 24 -> 16 kHz resample is done by the caller) -> codes [1,8,T]."""

    def __init__(self, m, pad_acoustic: bool):
        super().__init__()
        self.m, self.pad_acoustic = m, pad_acoustic

    def forward(self, wav24, wav16):
        m = self.m
        x = F.pad(wav16, (160, 160))
        hs = m.semantic_model(x, output_hidden_states=True).hidden_states
        sem = torch.stack(list(hs), dim=1).mean(dim=1)
        if m.config.semantic_downsample_factor > 1:
            sem = sem[:, ::m.config.semantic_downsample_factor, :]
        e_sem = m.encoder_semantic(sem.transpose(1, 2))
        e_ac = m.acoustic_encoder(F.pad(wav24, (m.pad, m.pad)) if self.pad_acoustic else wav24)
        e = torch.cat([e_ac, e_sem], dim=1)
        e = m.fc(e.transpose(1, 2)).transpose(1, 2)
        return m.quantizer.encode(e, m.config.target_bandwidths[-1]).transpose(0, 1)


def resample_24_to_16(w24: torch.Tensor) -> torch.Tensor:
    """The resampler the encoder graph's wav16 input must come from (torchaudio's, as in the original export)."""
    import torchaudio
    return torchaudio.functional.resample(w24, 24000, 16000)[:, 0]
