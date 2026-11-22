"""Reverb-TTS adapter: the first custom (audio, non-autoregressive, multi-graph, KV-carrying) adapter and the
regression test for the generic interfaces. See docs/adapters.md and docs/reverb.md."""
from .adapter import ReverbAdapter, ReverbOptions

__all__ = ["ReverbAdapter", "ReverbOptions"]
