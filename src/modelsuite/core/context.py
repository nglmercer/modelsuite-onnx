"""Job context and state shared by pipeline steps."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .config import JobConfig
from .discovery import ModelSource
from .manifest import Artifact, Manifest, utcnow
from .plan import ExportPlan
from .types import CompatibilityReport, ModelInfo

Emit = Callable[[str, str], None]


def _silent(step: str, msg: str) -> None:
    pass


@dataclass
class JobContext:
    config: JobConfig
    artifact: Artifact
    emit: Emit = _silent
    force: set[str] = field(default_factory=set)       # step names (or "all") to redo even if resumable
    source: ModelSource | None = None
    adapter: Any = None
    adapter_options: Any = None
    detection: Any = None
    candidates: list = field(default_factory=list)
    info: ModelInfo | None = None
    compat: CompatibilityReport | None = None
    plan: ExportPlan | None = None
    manifest: Manifest | None = None
    live: bool = True                                   # False = golden-only mode (original model not supplied)
    #: True only inside `convert`, where the goldens were produced from this very model by this very job. When a user hands an
    #: existing artifact a model to validate against, the live model is authoritative and the goldens are not trusted over it.
    trust_golden: bool = False

    def license(self):
        """Source-file license, or the one the adapter declared when the source carries none (python specs)."""
        from .licenses import classify
        lic = self.source.license()
        if lic.id is None and self.info is not None and self.info.license:
            return classify(self.info.license, "adapter")
        return lic

    def forced(self, step: str) -> bool:
        return "all" in self.force or step in self.force

    def say(self, step: str, msg: str) -> None:
        self.emit(step, msg)

    def save_report(self, name: str, data: Any) -> Path:
        self.artifact.reports_dir.mkdir(parents=True, exist_ok=True)
        p = self.artifact.reports_dir / name
        payload = data.model_dump(mode="json") if hasattr(data, "model_dump") else data
        p.write_text(json.dumps(payload, indent=2, default=str) + "\n")
        return p


class JobState:
    """job/state.json: per-step status and timing. The manifest is the source of truth for *results*; this is for humans and resume logic."""

    def __init__(self, artifact: Artifact):
        self.path = artifact.job_dir / "state.json"
        self.data: dict[str, Any] = {"steps": {}}
        if self.path.is_file():
            try:
                self.data = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                self.data = {"steps": {}}

    @property
    def fingerprint(self) -> str | None:
        return self.data.get("fingerprint")

    def start(self, fingerprint: str, config: dict) -> None:
        self.data["fingerprint"] = fingerprint
        self.data["config"] = config
        self.data.setdefault("created", utcnow())
        self.save()

    def mark(self, step: str, status: str, seconds: float = 0.0, message: str = "") -> None:
        self.data["steps"][step] = {"status": status, "seconds": round(seconds, 3), "finished": utcnow(), "message": message}
        self.save()

    def status(self, step: str) -> str | None:
        return self.data["steps"].get(step, {}).get("status")

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        tmp.replace(self.path)


class Timer:
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.seconds = time.perf_counter() - self.t
