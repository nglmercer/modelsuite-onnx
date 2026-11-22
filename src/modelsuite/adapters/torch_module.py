"""Generic adapter for custom PyTorch modules.

Reference syntax:   path/to/file.py:factory   or   package.module:factory

`factory()` is *your* code (it is executed, also by `inspect`). It returns a TorchModuleSpec, a ComponentExport,
or a ready ExportPlan. This is the fastest way to plug an arbitrary architecture into the pipeline: you supply
the module and example inputs, the suite does export, validation, optimisation and benchmarking."""
from __future__ import annotations

import ast
import importlib
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..core import registry
from ..core.config import JobConfig, Tolerance
from ..core.discovery import ModelSource, parse_python_spec
from ..core.errors import ConfigError, UnsupportedModelError
from ..core.plan import Arrays, AssetSpec, ComponentExport, ExportPlan, ReferenceCase, StatefulCase
from ..core.registry import register_adapter
from ..core.types import CompatibilityReport, ComponentInfo, Detection, Issue, ModelInfo, Severity, Support
from .base import Adapter, AdapterOptions, memory_estimate


@dataclass
class TorchModuleSpec:
    """What a user factory returns for a single-graph model."""
    module: Any                                   # torch.nn.Module
    example_inputs: Arrays | Callable[[], Arrays]  # name -> ndarray; also defines the input names (dict order = call order)
    output_names: list[str] = field(default_factory=lambda: ["output"])
    dynamic_axes: dict[str, dict[int, str]] = field(default_factory=dict)
    name: str = "model"
    #: extra input sets for dynamic-shape testing: name -> inputs
    shape_variants: dict[str, Arrays | Callable[[], Arrays]] = field(default_factory=dict)
    stateful: list[StatefulCase] = field(default_factory=list)
    tolerance: Tolerance | None = None
    license: str | None = None
    description: str = ""
    exporters: list[str] | None = None
    assets: list[AssetSpec] = field(default_factory=list)


def _as_callable(x):
    return x if callable(x) else (lambda x=x: x)


def load_factory(source: ModelSource):
    """Import the user's module and return its factory (executes user code)."""
    assert source.entry
    file = source.path
    name = f"modelsuite_user_{file.stem}"
    if name in sys.modules and getattr(sys.modules[name], "__file__", None) == str(file):
        module = sys.modules[name]
    else:
        spec = importlib.util.spec_from_file_location(name, file)
        if spec is None or spec.loader is None:
            raise ConfigError(f"cannot import {file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        sys.path.insert(0, str(file.parent))
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path.remove(str(file.parent))
    try:
        return getattr(module, source.entry)
    except AttributeError:
        raise ConfigError(f"{file} has no attribute '{source.entry}'") from None


def spec_to_plan(spec: TorchModuleSpec) -> ExportPlan:
    ex = _as_callable(spec.example_inputs)
    names = list(ex().keys())
    cases = [ReferenceCase("example", ex, "parity", "the tracing example")]
    cases += [ReferenceCase(n, _as_callable(v), "dynamic_shape", "different shapes from the export example")
              for n, v in spec.shape_variants.items()]
    comp = ComponentExport(
        name=spec.name, build=lambda: spec.module.eval(), example_inputs=ex, input_names=names,
        output_names=list(spec.output_names), dynamic_axes=spec.dynamic_axes, description=spec.description,
        cases=cases, stateful=spec.stateful, tolerance=spec.tolerance, exporters=spec.exporters,
        stateful_component=bool(spec.stateful), role="model")
    return ExportPlan(components=[comp], assets=list(spec.assets))


def to_plan(obj: Any) -> ExportPlan:
    if isinstance(obj, ExportPlan):
        return obj
    if isinstance(obj, ComponentExport):
        return ExportPlan(components=[obj])
    if isinstance(obj, TorchModuleSpec):
        return spec_to_plan(obj)
    if isinstance(obj, (list, tuple)) and obj and all(isinstance(o, (TorchModuleSpec, ComponentExport)) for o in obj):
        comps: list[ComponentExport] = []
        for o in obj:
            comps += (spec_to_plan(o) if isinstance(o, TorchModuleSpec) else ExportPlan([o])).components
        return ExportPlan(components=comps)
    raise ConfigError(f"the factory returned {type(obj).__name__}; expected TorchModuleSpec, ComponentExport, ExportPlan or a list of specs")


def _defines(file: Path, attr: str) -> bool:
    """Static check (no execution): does the file define a top-level function/class/assignment called attr?"""
    try:
        tree = ast.parse(file.read_text())
    except (OSError, SyntaxError):
        return False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == attr:
            return True
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == attr for t in node.targets):
            return True
    return False


@register_adapter
class TorchModuleAdapter(Adapter):
    name = "torch-module"
    description = "Custom PyTorch module from a Python factory (file.py:factory)"
    frameworks = ["pytorch"]
    priority = 10

    def detect(self, source: ModelSource) -> Detection:
        if source.kind != "python-spec":
            return Detection(confidence=0.0, reason="reference is not a 'file.py:factory' python spec")
        if not _defines(source.path, source.entry or ""):
            return Detection(confidence=0.0, reason=f"{source.path.name} defines no top-level '{source.entry}'")
        return Detection(confidence=1.0, reason=f"python spec {source.path.name}:{source.entry}")

    def _spec(self, source: ModelSource):
        factory = load_factory(source)
        built = factory()
        return built

    def inspect(self, source: ModelSource, options: AdapterOptions) -> ModelInfo:
        built = self._spec(source)
        plan = to_plan(built)
        params, comps = 0, []
        for c in plan.components:
            n = _count_params(c.module())
            params += n or 0
            comps.append(ComponentInfo(name=c.name, role=c.role or "model", description=c.description, parameters=n,
                                       stateful=c.stateful_component))
        lic = built.license if isinstance(built, TorchModuleSpec) else None
        return ModelInfo(reference=source.reference, source_kind=source.kind, revision=source.revision, framework="pytorch",
                         architecture=type(plan.components[0].module()).__name__, parameters=params or None,
                         weight_bytes=(params * 4) if params else None, license=lic, components=comps)

    def plan(self, source: ModelSource, options: AdapterOptions, config: JobConfig) -> ExportPlan:
        return to_plan(self._spec(source))

    def compatibility(self, source, info, options, config) -> CompatibilityReport:
        issues = [Issue(severity=Severity.info, code="user-factory",
                        message="The module comes from a user factory: only a real export + parity run proves it converts.")]
        if info.license is None:
            issues.append(Issue(severity=Severity.warning, code="license-unknown",
                                message="The factory did not declare a license; the artifact will be marked redistribution=unknown."))
        return CompatibilityReport(target_format=config.export.format, support=Support.partial,
                                   summary="Exportability of custom modules is established by running the export; see `convert`.",
                                   issues=issues, estimates=memory_estimate(info))


def _count_params(module: Any) -> int | None:
    try:
        return sum(int(p.numel()) for p in module.parameters())
    except Exception:
        return None
