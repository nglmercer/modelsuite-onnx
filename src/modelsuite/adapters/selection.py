"""Pick the adapter for a model: explicit choice wins, otherwise the most confident detector (ties -> priority)."""
from __future__ import annotations

from dataclasses import dataclass

from ..core import registry
from ..core.discovery import ModelSource
from ..core.errors import NoAdapterError
from ..core.types import Detection
from .base import Adapter


@dataclass
class Selection:
    adapter: Adapter
    detection: Detection
    candidates: list[tuple[str, Detection]]


def ensure_builtin_adapters() -> None:
    """Kept for callers that want to force registration; the registry bootstraps built-ins on first lookup anyway."""
    registry.adapters.load_entry_points()


def instantiate(entry) -> Adapter:
    return entry() if isinstance(entry, type) else entry


def select_adapter(source: ModelSource, forced: str | None = None) -> Selection:
    ensure_builtin_adapters()
    if forced:
        adapter = instantiate(registry.adapters.get(forced))
        det = adapter.detect(source)
        return Selection(adapter, det, [(forced, det)])
    scored: list[tuple[str, Detection, Adapter]] = []
    for name, entry in registry.adapters.items():
        try:
            adapter = instantiate(entry)
            det = adapter.detect(source)
        except Exception as exc:  # a faulty plugin must not take down detection for the others
            scored.append((name, Detection(confidence=0.0, reason=f"detect() raised {type(exc).__name__}: {exc}"), None))  # type: ignore[arg-type]
            continue
        scored.append((name, det, adapter))
    usable = [s for s in scored if s[2] is not None and s[1].confidence > 0]
    candidates = [(n, d) for n, d, _ in sorted(scored, key=lambda s: -s[1].confidence)]
    if not usable:
        lines = "\n".join(f"  - {n}: {d.reason or 'not recognised'}" for n, d in candidates)
        raise NoAdapterError(f"no adapter recognises '{source.reference}'. Registered adapters said:\n{lines}\n"
                             "Write an adapter (see docs/adapters.md) and load it with --plugin.")
    usable.sort(key=lambda s: (-s[1].confidence, -s[2].priority))
    name, det, adapter = usable[0]
    return Selection(adapter, det, candidates)
