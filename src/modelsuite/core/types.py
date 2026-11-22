"""Small typed value objects shared by every module. All are pydantic models, so they serialise to JSON."""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

Dim = Union[int, str, None]  # int = static, str = symbolic name, None = unknown


class Strict(BaseModel):
    """Base for configuration models: unknown keys are errors, so typos do not pass silently."""

    model_config = ConfigDict(extra="forbid")


class Loose(BaseModel):
    """Base for records that must survive schema growth when read back from disk."""

    model_config = ConfigDict(extra="allow")


class Severity(str, Enum):
    info = "info"
    warning = "warning"
    error = "error"


class Support(str, Enum):
    supported = "supported"        # a known path exists and was exercised
    partial = "partial"            # exportable with documented limits
    unsupported = "unsupported"    # known not to work; reasons listed
    unknown = "unknown"            # nothing recognises the model


class Issue(Loose):
    severity: Severity
    code: str
    message: str
    hint: str | None = None


class TensorSpec(Loose):
    name: str
    dtype: str                      # numpy-style: float32, float16, int64, bool, bfloat16...
    shape: list[Dim] = Field(default_factory=list)

    def is_dynamic(self) -> bool:
        return any(not isinstance(d, int) for d in self.shape)


class Signature(Loose):
    inputs: list[TensorSpec] = Field(default_factory=list)
    outputs: list[TensorSpec] = Field(default_factory=list)


class Detection(Loose):
    """An adapter's answer to 'is this model yours?'."""
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""


class ComponentInfo(Loose):
    """A model component as found by inspection (before anything is exported)."""
    name: str
    role: str = ""
    description: str = ""
    parameters: int | None = None
    signature: Signature | None = None
    stateful: bool = False


class ModelInfo(Loose):
    """What `modelsuite inspect` reports."""
    reference: str
    source_kind: str
    revision: str | None = None
    framework: str = "unknown"
    architecture: str = "unknown"
    model_type: str | None = None
    task: str | None = None
    parameters: int | None = None
    weight_bytes: int | None = None
    weight_dtypes: list[str] = Field(default_factory=list)
    license: str | None = None
    components: list[ComponentInfo] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class CompatibilityReport(Loose):
    """What `modelsuite analyze` reports for one target format."""
    target_format: str
    support: Support
    summary: str
    issues: list[Issue] = Field(default_factory=list)
    exporters: dict[str, str] = Field(default_factory=dict)       # exporter -> "available" | reason it is not
    strategy: dict[str, list[str]] = Field(default_factory=dict)  # component -> exporters in the order they will be tried
    estimates: dict[str, Any] = Field(default_factory=dict)       # memory / size estimates

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == Severity.error]


class Status(str, Enum):
    passed = "passed"
    failed = "failed"
    skipped = "skipped"
    error = "error"


StateMode = Literal["replace", "concat"]
