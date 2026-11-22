"""The adapter plugin contract.

An adapter owns everything architecture-specific; the core owns everything else. To support a new
architecture, subclass Adapter, decorate it with @register_adapter (or expose it through the
`modelsuite.adapters` entry-point group, or load the file with --plugin) and implement:

    detect()         -> how confident the adapter is that it understands this model
    inspect()        -> architecture, size, components (cheap: no weights loaded)
    plan()           -> ExportPlan: components, example inputs, dynamic axes, reference runners,
                        stateful cases, assets, quality checks   (lazy: still no weights loaded)

and optionally override compatibility() / the option schema. Nothing in core/ imports an adapter.
"""
from __future__ import annotations

import abc
from typing import Any, ClassVar

from pydantic import BaseModel

from ..core.config import JobConfig
from ..core.discovery import ModelSource
from ..core.plan import ExportPlan
from ..core.types import CompatibilityReport, Detection, Issue, ModelInfo, Severity, Support


class AdapterOptions(BaseModel):
    """Base for an adapter's option schema (validated from JobConfig.adapter_options)."""
    model_config = {"extra": "forbid"}


class Adapter(abc.ABC):
    name: ClassVar[str] = ""
    version: ClassVar[str] = "0.1.0"
    priority: ClassVar[int] = 0                 # tie-breaker: specific adapters outrank generic ones
    description: ClassVar[str] = ""
    frameworks: ClassVar[list[str]] = []
    options_model: ClassVar[type[AdapterOptions]] = AdapterOptions

    # -- options -----------------------------------------------------------------------
    def parse_options(self, raw: dict[str, Any] | None) -> AdapterOptions:
        return self.options_model.model_validate(raw or {})

    # -- required ------------------------------------------------------------------------
    @abc.abstractmethod
    def detect(self, source: ModelSource) -> Detection:
        """Cheap and side-effect free: look at file names, config.json, the reference string. Never load weights."""

    @abc.abstractmethod
    def inspect(self, source: ModelSource, options: AdapterOptions) -> ModelInfo: ...

    @abc.abstractmethod
    def plan(self, source: ModelSource, options: AdapterOptions, config: JobConfig) -> ExportPlan:
        """Build the export plan. Must not load weights: put that behind ComponentExport.build()."""

    # -- optional ---------------------------------------------------------------------------
    def compatibility(self, source: ModelSource, info: ModelInfo, options: AdapterOptions, config: JobConfig) -> CompatibilityReport:
        """Feasibility of converting to config.export.format. The default only reports size-related facts;
        adapters add architecture-specific findings (unsupported ops, trust_remote_code, ...)."""
        issues: list[Issue] = []
        est = memory_estimate(info)
        return CompatibilityReport(target_format=config.export.format, support=Support.unknown,
                                   summary="No architecture-specific analysis available.", issues=issues, estimates=est)

    def weight_patterns(self, source: ModelSource, options: AdapterOptions) -> list[str] | None:
        """Hub file patterns to download for the export (None = everything that looks like weights). Override for
        repositories that hold several variants of one model."""
        return None

    def runtime_requirements(self, plan: ExportPlan) -> list[str]:
        return list(plan.runtime_requirements)


def memory_estimate(info: ModelInfo) -> dict[str, Any]:
    """Rough peak-RAM / size numbers from the parameter count. Labelled as estimates everywhere they are shown."""
    import psutil
    est: dict[str, Any] = {}
    if info.parameters:
        p = info.parameters
        est["parameters"] = p
        est["fp32_bytes"] = p * 4
        est["fp16_bytes"] = p * 2
        est["int8_bytes"] = p
        est["int4_bytes"] = p // 2
        est["export_peak_ram_bytes_estimate"] = int(p * 4 * 2.5)   # fp32 module + traced graph + serialised copy
        avail = psutil.virtual_memory().available
        est["available_ram_bytes"] = int(avail)
        est["fits_in_ram"] = est["export_peak_ram_bytes_estimate"] < avail
        est["needs_external_data"] = p * 4 >= 1536 * 1024 * 1024
    return est


def feasibility_issue(code: str, message: str, severity: Severity = Severity.warning, hint: str | None = None) -> Issue:
    return Issue(severity=severity, code=code, message=message, hint=hint)
