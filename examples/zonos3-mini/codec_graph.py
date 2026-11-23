"""Export-friendly ZONOS3 codec *decoder*: codes [1, 10, T] -> 44.1 kHz audio [1, 1, 2940 * T].

Equivalent to `ISTFTCodec.forward` (the whole-sequence decode that Zyphra's streaming decoder reproduces to 1e-5), with the three things a
tracer cannot express replaced by exact equivalents:

* `MossLayer.attention` loops over 512-row blocks in Python and builds biases in place -> one banded dense attention (same mask: a query
  sees the `context_frames` most recent keys, itself included). Memory is O(T^2) at the 240 Hz stage: decode long audio in chunks.
* `torch.istft` (complex tensors) -> inverse real DFT as two matmuls with constant bases + overlap-add + window-envelope division + centre trim.
* torchaudio `Resample` computes its output length in Python (it would be frozen at the example length) -> same polyphase kernel, dynamic length."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def _moss_layer(layer, x):
    b, t, d = x.shape
    heads, hd = layer.heads, layer.head_dim
    q, k, v = layer.qkv(layer.norm1(x)).reshape(b, t, 3, heads, hd).unbind(2)
    frequency = torch.exp(torch.arange(hd // 2, dtype=torch.float32) * (-math.log(10000.0) * 2 / hd))
    phase = torch.arange(t, dtype=torch.float32)[:, None] * frequency[None, :]
    cos, sin = phase.cos()[None, :, None, :], phase.sin()[None, :, None, :]

    def rope(z):
        zf = z.float().reshape(b, t, heads, hd // 2, 2)
        r, i = zf.unbind(-1)
        return torch.stack((r * cos - i * sin, r * sin + i * cos), -1).flatten(-2).to(z.dtype)

    q, k = rope(q), rope(k)
    q, k, v = (z.transpose(1, 2) for z in (q, k, v))                         # [b, heads, t, hd]
    scores = (q @ k.transpose(-1, -2)) * (hd ** -0.5)
    pos = torch.arange(t)
    delta = pos[:, None] - pos[None, :]
    valid = (delta >= 0) & (delta < layer.context)
    att = scores.masked_fill(~valid, float("-inf")).softmax(-1) @ v
    x = x + layer.attention_scale * layer.attention_out(att.transpose(1, 2).reshape(b, t, d))
    return x + layer.ffn_scale * layer.ffn(layer.norm2(x))


def _stage(stage, x):
    from zonos3.codec.hybrid10_dual_encoding.base_model import MossStage
    if not isinstance(stage, MossStage):
        return stage(x)                                                       # Patch: reshape/permute only
    h = stage.input_proj(x.transpose(1, 2))
    for layer in stage.layers:
        h = _moss_layer(layer, h)
    return stage.output_proj(h).transpose(1, 2)


def _dft_bases(n: int):
    """C, S such that frames = Re @ C + Im @ S equals irfft(Re + i Im, n) per frame (torch.istft's onesided inverse)."""
    k = np.arange(n // 2 + 1, dtype=np.float64)[:, None]
    m = np.arange(n, dtype=np.float64)[None, :]
    w = np.full((n // 2 + 1, 1), 2.0)
    w[0, 0] = w[-1, 0] = 1.0                                                  # DC and Nyquist appear once; the imaginary parts of both are ignored
    ang = 2.0 * np.pi * k * m / n
    return torch.tensor(w * np.cos(ang) / n, dtype=torch.float32), torch.tensor(-w * np.sin(ang) / n, dtype=torch.float32)


def _overlap_add(frames, hop: int):
    """frames [T, n] with n = r * hop -> [(T + r - 1) * hop]: frame t starts at t * hop."""
    t, n = frames.shape
    r = n // hop
    parts = frames.reshape(t, r, hop)
    out = 0
    for s in range(r):
        out = out + F.pad(parts[:, s, :], (0, 0, s, r - 1 - s))              # rows s .. s+T-1
    return out.reshape(-1)


class ISTFT(torch.nn.Module):
    def __init__(self, n: int, hop: int, window: torch.Tensor):
        super().__init__()
        self.n, self.hop = n, hop
        c, s = _dft_bases(n)
        self.register_buffer("c", c)
        self.register_buffer("s", s)
        self.register_buffer("window", window.float().clone())

    def forward(self, re, im, length: int):
        """re, im [T, n/2+1] -> [length] samples, centred, window-envelope normalised (== torch.istft(center=True, length=length))."""
        frames = (re @ self.c + im @ self.s) * self.window
        y = _overlap_add(frames, self.hop)
        t = re.shape[0]
        env = _overlap_add((self.window ** 2).unsqueeze(0).expand(t, -1), self.hop)
        start = self.n // 2
        return y[start:start + length] / env[start:start + length]


class CodecDecoderGraph(torch.nn.Module):
    """Wraps a loaded `ISTFTCodec` (eval, float32, mr_istft head)."""

    def __init__(self, model):
        super().__init__()
        self.m = model
        head = model.head
        if type(head).__name__ != "MultiResISTFTHead":
            raise NotImplementedError(f"only the multi-resolution iSTFT head is supported, got {type(head).__name__}")
        self.istfts = torch.nn.ModuleList(ISTFT(n, head.hop, getattr(head, f"window_{i}")) for i, n in enumerate(head.n_ffts))
        rs = model.to_external
        self.rs_orig, self.rs_new = int(rs.orig_freq) // int(rs.gcd), int(rs.new_freq) // int(rs.gcd)
        self.register_buffer("rs_kernel", rs.kernel.detach().clone())
        self.rs_width = int(rs.width)

    def forward(self, codes):
        m, head = self.m, self.m.head
        frames = codes.shape[-1]
        x = m.latent(codes)
        for stage in m.stages:
            x = _stage(stage, x)
        h = head.norm(head.blocks(x).transpose(1, 2))                         # [1, T', dim]
        T = h.shape[1]
        branches = []
        for lin, istft in zip(head.outs, self.istfts):
            log_magnitude, phase = lin(h).float().chunk(2, dim=-1)
            magnitude = torch.exp(log_magnitude.clamp(max=head.max_log_magnitude))[0]
            branches.append(istft(magnitude * torch.cos(phase[0]), magnitude * torch.sin(phase[0]), T * head.hop))
        internal = head.mixer(torch.stack(branches, 0).unsqueeze(0))        # [1, 1, T' * hop]
        # torchaudio's polyphase resampler, with the output length computed from the (dynamic) frame count
        w = F.pad(internal.reshape(1, -1), (self.rs_width, self.rs_width + self.rs_orig))
        res = F.conv1d(w.unsqueeze(1), self.rs_kernel, stride=self.rs_orig).transpose(1, 2).reshape(1, -1)
        return res[0, : frames * self.m.external_hop][None, None]


def build_empty_codec(cfg: dict, head_cfg: dict):
    """`ISTFTCodec(cfg, head)` without ever materialising the parts it throws away.

    ISTFTCodec.__init__ first builds a *complete* MossCodec (encoder + decoder, ~8 GB of randomly initialised fp32 for the release) and then keeps only
    decoder stages 0-8, the quantizer, the semantic head and the resampler. Constructing it on the meta device costs no memory; `to_empty` then allocates
    only what is kept, and the checkpoint fills it. Non-persistent buffers (the Hann windows) are not in any state dict, so they are recomputed here."""
    import torch
    import torchaudio.transforms as T
    from zonos3.codec.istft_head_a_20261002.model import ISTFTCodec
    # torchaudio's Resample computes its sinc kernel numerically when built, so it cannot live on the meta device; its (small) kernel is part of the
    # checkpoint (`to_external.kernel`) and is overwritten on load anyway. Build only the resamplers on the real device.
    original = T.Resample.__init__

    def real_device_init(self, *a, **k):
        with torch.device("cpu"):
            original(self, *a, **k)
    T.Resample.__init__ = real_device_init
    try:
        with torch.device("meta"):
            model = ISTFTCodec(cfg, head_cfg)
    finally:
        T.Resample.__init__ = original
    model.to_empty(device="cpu")
    for i, n in enumerate(model.head.n_ffts):
        getattr(model.head, f"window_{i}").copy_(torch.hann_window(n, periodic=True))
    return model
