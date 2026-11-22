"""Exporter contract. Exporters wrap existing tools; they do not reimplement them."""
from __future__ import annotations

import abc
import time
from pathlib import Path
from typing import Any

from pydantic import Field

from ..core.config import ExportConfig
from ..core.plan import ComponentExport
from ..core.types import Loose


class ExporterCapabilities(Loose):
    name: str
    description: str
    frameworks: list[str] = Field(default_factory=list)
    formats: list[str] = Field(default_factory=lambda: ["onnx"])
    scope: str = "component"                      # "component": one graph from a callable module; "model": a whole checkpoint
    dynamic_shapes: bool = True
    stateful_models: bool = False                 # KV cache / recurrent state handled natively
    external_data: bool = True
    custom_op_symbolics: bool = False
    requires: list[str] = Field(default_factory=list)        # pip distributions
    limitations: list[str] = Field(default_factory=list)
    config: dict[str, str] = Field(default_factory=dict)     # option -> meaning


class ExportResult(Loose):
    component: str
    exporter: str
    strategy: str
    path: str
    seconds: float
    tool_versions: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    extra_components: dict[str, str] = Field(default_factory=dict)   # for model-scope exporters: more graphs it produced


class Exporter(abc.ABC):
    name: str = ""

    @abc.abstractmethod
    def capabilities(self) -> ExporterCapabilities: ...

    @abc.abstractmethod
    def availability(self) -> tuple[bool, str]:
        """(usable, reason). Must be cheap and must not raise."""

    def supports(self, component: ComponentExport) -> tuple[bool, str]:
        """Can this exporter handle this component? Default: any component whose module the exporter understands."""
        return True, ""

    @abc.abstractmethod
    def export(self, component: ComponentExport, out_path: Path, cfg: ExportConfig) -> ExportResult: ...

    @staticmethod
    def _timed():
        return time.perf_counter()


def order_exporters(component: ComponentExport, requested: str, default_order: list[str]) -> list[str]:
    """The exporters to try for a component, in order. An explicit user choice wins and is never expanded."""
    if requested not in ("auto", ""):
        return [requested]
    return list(component.exporters or default_order)


def describe(exporter_name: str) -> dict[str, Any]:
    from ..core.registry import exporters
    cls = exporters.get(exporter_name)
    e = cls() if isinstance(cls, type) else cls
    ok, why = e.availability()
    return {"capabilities": e.capabilities().model_dump(), "available": ok, "reason": why}
