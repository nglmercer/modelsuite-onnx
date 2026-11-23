"""Export-friendly ZONOS3-Mini language model.

`reference/lm.py` in Zyphra's repository is a *numerical oracle*: it loops over tokens in Python, picks experts with
`int(choices[t, r])` and keeps its caches in Python dicts. None of that can be traced. This module computes exactly the same
function from the same parameters, but with

* every state tensor an explicit graph input/output (so the host owns the cache, like Reverb's KV hand-off),
* the Gated DeltaNet recurrence and the MoE expert dispatch written as `torch.jit.script` loops over tokens
  (they export to ONNX `Loop`; the trip count is the dynamic number of rows),
* RMSNorm / softmax / interleaved RoPE spelled out (ONNX opset 18 has no RMSNormalization).

State layout (all float32; the reference keeps GDN state in float32 as well):
    past_k / past_v   [n_attn_layers, P, n_kv_heads, head_dim]   rotated keys, as the reference caches them
    conv_state        [n_gdn_layers, 3 * inner, kernel - 1]      causal depthwise-conv history
    gdn_state         [n_gdn_layers, heads, head_dim, head_dim]  delta-rule matrix, [heads, value, key]
Rows may be replaced by a speaker vector or a condition-table row (what `Zonos3LM._replace_rows` does) through
`speaker_vec/speaker_flag` and `cond_index` (index into the concatenation of all condition tables)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

GDN_HEADS, GDN_HEAD_DIM, GDN_KERNEL = 4, 128, 4   # constants of the release architecture (reference/lm.py)


def rms(x, weight=None, eps: float = 1e-5):
    y = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype)
    return y if weight is None else y * weight


@torch.jit.script
def gdn_scan(state: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, decay: torch.Tensor, beta: torch.Tensor, scale: float):
    """Delta-rule recurrence, token by token. state [H, hd(value), hd(key)]; q/k/v [T, H, hd]; decay/beta [T, H]."""
    outs = torch.zeros_like(v)
    for t in range(q.shape[0]):
        state = state * decay[t].unsqueeze(-1).unsqueeze(-1)
        prediction = (state * k[t].unsqueeze(1)).sum(-1)
        delta = beta[t].unsqueeze(-1) * (v[t] - prediction)
        state = state + delta.unsqueeze(-1) * k[t].unsqueeze(1)
        outs[t] = (state * (q[t] * scale).unsqueeze(1)).sum(-1)
    return outs, state


@torch.jit.script
def moe_dispatch(latent: torch.Tensor, choices: torch.Tensor, weights: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor, half: int):
    """Per token: gather its top-k experts' weights and apply them. latent [T, R]; choices [T, k]; weights [T, k];
    gate_up [E, 2*half, R]; down [E, R, half]. Only the selected experts are read (a dense 64-expert pass would read 32x more)."""
    out = torch.zeros_like(latent)
    k = choices.shape[1]
    for t in range(latent.shape[0]):
        idx = choices[t]
        y = torch.bmm(gate_up[idx], latent[t].reshape(1, -1, 1).expand(k, -1, -1)).squeeze(-1)      # [k, 2*half]
        h = F.silu(y[:, :half]) * y[:, half:]
        o = torch.bmm(down[idx], h.unsqueeze(-1)).squeeze(-1)                                          # [k, R]
        out[t] = (o * weights[t].unsqueeze(-1)).sum(0)
    return out


def rope_interleaved(x, positions, theta: float):
    hd = x.shape[-1]
    frequencies = theta ** (-torch.arange(0, hd, 2, dtype=torch.float32) / hd)
    angles = positions.float()[:, None] * frequencies[None, :]
    cos, sin = angles.cos()[:, None, :], angles.sin()[:, None, :]
    pairs = x.float().reshape(x.shape[0], x.shape[1], hd // 2, 2)
    a, b = pairs[..., 0], pairs[..., 1]
    return torch.stack((a * cos - b * sin, a * sin + b * cos), -1).flatten(-2)


class LMGraph(torch.nn.Module):
    """Wraps a loaded `zonos3.reference.lm.Zonos3LM` (float32) and runs its forward as a pure function of explicit state."""

    def __init__(self, ref):
        super().__init__()
        self.r = ref
        c = ref.config
        self.c = c
        self.gdn_layers = [i for i in range(c.n_layers) if c.uses_gdn(i)]
        self.attn_layers = [i for i in range(c.n_layers) if not c.uses_gdn(i)]

    # -- sublayers ---------------------------------------------------------------------------------------------------
    def _route(self, sources, query):
        bank = torch.stack(sources)
        keys = rms(bank, eps=self.c.norm_eps)
        weights = (keys * query).sum(-1).float().softmax(0).to(bank.dtype)
        return (bank * weights.unsqueeze(-1)).sum(0)

    def _gdn(self, x, layer, conv_state, gdn_state):
        a = self.r.layers[layer].attention
        heads, hd, kernel = GDN_HEADS, GDN_HEAD_DIM, GDN_KERNEL
        inner, channels, n = heads * hd, 3 * heads * hd, x.shape[0]
        qkv, z, aa, b = F.linear(x, a.in_proj.weight).split((3 * inner, inner, heads, heads), -1)
        sequence = torch.cat((conv_state, qkv.T), -1)                                  # [channels, kernel-1+T]
        conv = F.silu(F.conv1d(sequence.unsqueeze(0), a.conv.weight, groups=channels).squeeze(0).T)
        new_conv = sequence[:, sequence.shape[1] - (kernel - 1):]
        q, k, v = [part.reshape(n, heads, hd).float() for part in conv.split(inner, -1)]
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        decay = torch.exp(-a.A_log.float().exp() * F.softplus(aa.float() + a.dt_bias.float()))
        beta = b.float().sigmoid()
        outs, new_state = gdn_scan(gdn_state, q, k, v, decay, beta, float(hd ** -0.5))
        normalized = rms(outs.to(x.dtype), a.out_weight, eps=1e-5)
        gated = (normalized.float() * F.silu(z.reshape(n, heads, hd).float())).to(x.dtype).reshape(n, inner)
        return F.linear(gated, a.wo.weight), new_conv, new_state

    def _attention(self, x, layer, past_k, past_v):
        c, a = self.c, self.r.layers[layer].attention
        n, nh, nkv, hd = x.shape[0], c.n_heads, c.n_kv_heads, c.head_dim
        P = past_k.shape[0]
        gate = F.linear(x, a.gater.weight).sigmoid()
        q = F.linear(x, a.wq.weight).reshape(n, nh, hd)
        kv = F.linear(x, a.wkv.weight.reshape(2 * nkv * hd, c.dim))
        k, v = kv[:, : nkv * hd].reshape(n, nkv, hd), kv[:, nkv * hd:].reshape(n, nkv, hd)
        q = rms(q, eps=1e-6) * a.temp.abs().reshape(1, nh, 1)
        k = rms(k, eps=1e-6)
        positions = P + torch.arange(n)
        q = rope_interleaved(q, positions, c.rope_theta)
        k = rope_interleaved(k, positions, c.rope_theta)
        k_all, v_all = torch.cat((past_k, k), 0), torch.cat((past_v, v), 0)
        rep = nh // nkv
        key = k_all.repeat_interleave(rep, 1).transpose(0, 1)                          # [nh, P+T, hd]
        value = v_all.repeat_interleave(rep, 1).transpose(0, 1)
        scores = (q.transpose(0, 1) @ key.transpose(-1, -2)) * (hd ** -0.5)            # [nh, T, P+T]
        allowed = torch.arange(P + n)[None, :] <= positions[:, None]
        scores = scores.masked_fill(~allowed[None], float("-inf"))
        out = (scores.softmax(-1) @ value).transpose(0, 1)                             # [T, nh, hd]
        return F.linear((out * gate.unsqueeze(-1)).reshape(n, nh * hd), a.wo.weight), k_all, v_all

    def _ffn(self, x, layer):
        c, f = self.c, self.r.layers[layer].feed_forward
        if layer == 0:
            hidden = c.ffn_hidden
            flat = F.linear(x, f.w_in.weight.reshape(2 * hidden, -1))
            return F.linear(flat[:, :hidden] * F.silu(flat[:, hidden:]), f.w_out.weight)
        logits = F.linear(rms(x, f.router.rmsnorm_eda.weight), f.router.router_linear.weight).float()
        choices = torch.topk(logits - f.router.balancing_biases.float(), c.moe_topk, -1).indices
        weights = logits.gather(-1, choices).softmax(-1)
        latent = F.linear(x, f.latent_down.weight)
        gate_up, down = f.experts.gate_up_proj, f.experts.down_proj
        out = moe_dispatch(latent, choices, weights, gate_up, down, gate_up.shape[1] // 2)
        return F.linear(out, f.latent_up.weight)

    # -- forward -------------------------------------------------------------------------------------------------------
    def forward(self, ids, speaker_vec, speaker_flag, cond_index, past_k, past_v, conv_state, gdn_state):
        r, c = self.r, self.c
        embedders = r.multi_embedder.embedders
        x = sum(F.embedding(ids[:, cb], embedders[cb].weight) for cb in range(c.n_codebooks + 1))
        speaker = F.linear(speaker_vec, r.speaker_projection.weight, r.speaker_projection.bias)
        x = torch.where(speaker_flag.unsqueeze(-1) > 0, speaker, x)
        table = torch.cat([e.weight for e in r.condition_embeddings], 0)
        x = torch.where(cond_index.unsqueeze(-1) >= 0, table[cond_index.clamp(min=0)], x)
        x = rms(x, eps=c.norm_eps)
        sources = [x]
        queries = r.block_attnres_queries
        new_k, new_v, new_conv, new_state = [], [], [], []
        gi = ai = 0
        for layer in range(c.n_layers):
            block = r.layers[layer]
            attn_in = rms(self._route(sources, queries[2 * layer]), block.attention_norm.weight, c.norm_eps)
            if c.uses_gdn(layer):
                out, cs, gs = self._gdn(attn_in, layer, conv_state[gi], gdn_state[gi])
                new_conv.append(cs)
                new_state.append(gs)
                gi += 1
            else:
                out, k_all, v_all = self._attention(attn_in, layer, past_k[ai], past_v[ai])
                new_k.append(k_all)
                new_v.append(v_all)
                ai += 1
            sources.append(out)
            ffn_in = rms(self._route(sources, queries[2 * layer + 1]), block.ffn_norm.weight, c.norm_eps)
            sources.append(self._ffn(ffn_in, layer))
        normalized = rms(self._route(sources, queries[-1]), r.out_norm.weight, c.norm_eps)
        logits = F.linear(normalized, r.multi_output.weight).reshape(ids.shape[0], c.n_codebooks, c.audio_vocab)
        logits = c.loss_softcap * torch.tanh(logits / c.loss_softcap)
        # explicit reshapes pin the state dimensions that are really static, so the signature carries no exporter-invented symbols
        n_attn, n_gdn = len(self.attn_layers), len(self.gdn_layers)
        return (logits, torch.stack(new_k), torch.stack(new_v),
                torch.stack(new_conv).reshape(n_gdn, 3 * GDN_HEADS * GDN_HEAD_DIM, GDN_KERNEL - 1),
                torch.stack(new_state).reshape(n_gdn, GDN_HEADS, GDN_HEAD_DIM, GDN_HEAD_DIM))
