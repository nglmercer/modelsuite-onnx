"""The adapter ⇄ pipeline contract.

An Adapter turns a model into an ExportPlan: a list of independently exportable components plus everything
needed to verify them. The plan holds *callables*, so building a plan is cheap and nothing heavy (weights,
torch) is touched until a pipeline step actually needs it. The core never imports a framework; adapters and
exporters do."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ContextManager, Protocol

import numpy as np

from .config import Tolerance
from .types import StateMode

Arrays = dict[str, np.ndarray]


@dataclass
class ReferenceCase:
    """One deterministic input set. `kind="dynamic_shape"` cases deliberately differ in shape from the export example."""
    name: str
    inputs: Callable[[], Arrays]
    kind: str = "parity"                      # parity | dynamic_shape
    description: str = ""
    tolerance: Tolerance | None = None


@dataclass
class StateLink:
    """How an output feeds a state input on the next step (KV cache, RNN hidden state...)."""
    output: str
    input: str
    mode: StateMode = "replace"               # replace: next_in = out; concat: next_in = concat(prev_in, out, axis)
    axis: int | None = None


@dataclass
class StatefulCase:
    """A multi-step run that feeds state back in. Compared step by step against the original model."""
    name: str
    links: list[StateLink]
    steps: int
    initial: Callable[[], Arrays]             # every input of step 0 (state inputs included)
    step_inputs: Callable[[int], Arrays]      # the non-state inputs for step i >= 1
    tolerance: Tolerance | None = None
    description: str = ""


@dataclass
class ComponentExport:
    """One independently exportable graph."""
    name: str
    build: Callable[[], Any]                  # -> framework object (torch.nn.Module); called at most once (memoised)
    example_inputs: Callable[[], Arrays]      # inputs for tracing; keys == input_names
    input_names: list[str]
    output_names: list[str]
    dynamic_axes: dict[str, dict[int, str]] = field(default_factory=dict)
    role: str = ""
    description: str = ""
    cases: list[ReferenceCase] = field(default_factory=list)
    stateful: list[StatefulCase] = field(default_factory=list)
    #: runs the ORIGINAL model; default is derived from build() by the framework support module
    reference: Callable[[Arrays], Arrays] | None = None
    exporters: list[str] | None = None        # preference order; None = suite default for the framework
    export_options: dict[str, Any] = field(default_factory=dict)
    export_context: Callable[[], ContextManager] | None = None   # e.g. register custom-op symbolics, patch an op
    tolerance: Tolerance | None = None
    #: per-precision override, e.g. {"fp16": Tolerance(...)}
    precision_tolerance: dict[str, Tolerance] = field(default_factory=dict)
    known_issues: list[str] = field(default_factory=list)
    #: provider (short name) -> reason it is known not to work for this graph; reported, never silently worked around
    provider_notes: dict[str, str] = field(default_factory=dict)
    #: output name -> hint for metrics ("discrete" = compare exactly, "audio" = also report rms ratio)
    output_kinds: dict[str, str] = field(default_factory=dict)
    #: ops (ONNX op types) that quantization must not touch for this component
    quantize_exclude_ops: list[str] = field(default_factory=list)
    stateful_component: bool = False
    #: called by release(): lets an adapter drop resources it shares between components (a loaded checkpoint, a reference model)
    on_release: Callable[[], None] | None = None
    _module: Any = field(default=None, repr=False, compare=False)
    _built: bool = field(default=False, repr=False, compare=False)

    def module(self) -> Any:
        if not self._built:
            self._module = self.build()
            self._built = True
        return self._module

    def release(self) -> None:
        """Drop the memoised module so its weights can be garbage-collected between components."""
        self._module, self._built = None, False
        if self.on_release is not None:
            self.on_release()


@dataclass
class AssetSpec:
    """A non-graph file the runtime needs next to the ONNX files (tokenizer, config, voices, ...)."""
    role: str                                  # tokenizer | config | preprocessor | postprocessor | voice | other
    source: Path | None                        # file to copy; None = description-only (e.g. 'sampler lives in the host app')
    dest: str                                  # relative path under assets/
    required: bool = True
    description: str = ""
    license_note: str | None = None


@dataclass
class Edge:
    src: str
    output: str
    dst: str
    input: str


@dataclass
class QualityCheck:
    """A task-level check (WER, sampler agreement, perplexity...) that needs more than per-tensor parity."""
    name: str
    run: Callable[["QualityContext"], "QualityResult"]
    description: str = ""
    expensive: bool = False


class Runner(Protocol):
    def run(self, inputs: Arrays) -> Arrays: ...


@dataclass
class QualityContext:
    variant: str
    runner: Callable[[str], Runner]            # component name -> runner for the variant under test
    seed: int = 0


@dataclass
class QualityResult:
    passed: bool | None
    metrics: dict[str, Any] = field(default_factory=dict)
    message: str = ""


@dataclass
class ExportPlan:
    components: list[ComponentExport]
    topology: list[Edge] = field(default_factory=list)
    assets: list[AssetSpec] = field(default_factory=list)
    quality_checks: list[QualityCheck] = field(default_factory=list)
    runtime_requirements: list[str] = field(default_factory=list)
    custom_op_libraries: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    #: component name -> license id, for components whose weights do not come under the model's own license
    component_licenses: dict[str, str] = field(default_factory=dict)

    def component(self, name: str) -> ComponentExport:
        for c in self.components:
            if c.name == name:
                return c
        raise KeyError(name)
