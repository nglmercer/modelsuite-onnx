"""Exportable Reverb network: `prefix` (text + reference -> KV) and `step` (target tokens + KV -> log-probs).

Mirrors reverb_tts.sampler.Batch for batch size 1 (no padding, so no padding masks are needed).
"""
import math
import torch
from reverb_tts.model import CODEBOOKS, CODES, MASK


def rms(x, w, eps):
    x32 = x.float()
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return w * x32.to(x.dtype)


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


class Core(torch.nn.Module):
    def __init__(self, network):
        super().__init__()
        self.n = network
        cfg = network.llm.config
        self.heads, self.kv, self.hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.eps = cfg.rms_norm_eps
        theta = cfg.rope_parameters["rope_theta"] if getattr(cfg, "rope_parameters", None) else cfg.rope_theta
        self.register_buffer("inv_freq", 1.0 / (theta ** (torch.arange(0, self.hd, 2).float() / self.hd)), persistent=False)

    def rope(self, positions):  # positions [S] -> cos/sin [1,1,S,hd]
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


class Prefix(Core):
    """text_ids [1,Pt], ref_codes [1,8,R] -> keys, values [24,1,6,P,128] (rotated keys, as HF caches them)."""

    def forward(self, text_ids, ref_codes):
        n = self.n
        offsets = CODES * torch.arange(CODEBOOKS)[None, :, None]
        text = n.llm.embed_tokens(text_ids)
        ref = n.audio_embeddings(ref_codes + offsets).sum(1)
        x = torch.cat([text, ref], dim=1)
        S = x.shape[1]
        cos, sin = self.rope(torch.arange(S))
        ks, vs = [], []
        for i in range(len(self.n.llm.layers)):
            x, k, v = self.layer(i, x, cos, sin, None, None)
            ks.append(k)
            vs.append(v)
        return torch.stack(ks), torch.stack(vs)


class Step(Core):
    """tokens [1,8,F], keys/values [24,1,6,P,128], prefix_bias [P] -> log-probs [1,8,F,1024] (fp32)."""

    def forward(self, tokens, keys, values, prefix_bias):
        n = self.n
        F = tokens.shape[2]
        P = keys.shape[3]
        offsets = CODES * torch.arange(CODEBOOKS)[None, :, None]
        x = n.audio_embeddings(tokens + offsets).sum(1)
        cos, sin = self.rope(P + torch.arange(F))
        bias = torch.cat([prefix_bias, torch.zeros(F, dtype=prefix_bias.dtype)])[None, None, None, :]
        for i in range(len(self.n.llm.layers)):
            x, _, _ = self.layer(i, x, cos, sin, (keys[i], values[i]), bias)
        x = rms(x, n.llm.norm.weight, self.eps)
        logits = (x @ n.audio_heads.weight.T).float().view(1, F, CODEBOOKS, CODES).transpose(1, 2)
        return logits.log_softmax(-1)[..., :MASK]


class CoreGraph(Core):
    """One graph for both prefix and target passes.

    x [1,S,1024]; past_k/past_v [24,1,6,Pp,128] (Pp may be 0); bias [S, Pp+S] additive attention bias.
    Positions are Pp..Pp+S-1. Returns final-normed hidden [1,S,1024] and this call's rotated keys/values.
    """

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
    """hidden [1,F,1024] -> log-probs [1,8,F,1024] fp32."""

    def __init__(self, network):
        super().__init__()
        self.n = network

    def forward(self, hidden):
        F = hidden.shape[1]
        logits = (hidden @ self.n.audio_heads.weight.T).float().view(1, F, CODEBOOKS, CODES).transpose(1, 2)
        return logits.log_softmax(-1)[..., :MASK]
