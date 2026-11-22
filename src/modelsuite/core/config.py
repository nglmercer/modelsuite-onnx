"""Validated configuration schemas. A job is fully described by a JobConfig, so it can be replayed from the manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, field_validator

from .errors import ConfigError
from .types import Strict

Precision = Literal["fp32", "fp16", "fp16-weights", "bf16-weights"]
Quantization = Literal["none", "int8-dynamic", "int8-static", "int4"]
GraphLevel = Literal["none", "basic", "extended", "all"]


class Tolerance(Strict):
    """Numerical-parity thresholds. A comparison passes when every enabled check passes."""
    atol: float = 1e-4
    rtol: float = 1e-3
    max_abs: float | None = None          # absolute cap on the max |ref - got| (defaults to atol + rtol*max|ref|)
    min_cosine: float | None = 0.9999     # cosine similarity of the flattened outputs (float outputs only)
    min_equal_fraction: float = 1.0       # integer / bool outputs: share of identical elements
    min_rms_ratio: float = 0.5            # got.rms / ref.rms must stay within [r, 1/r]: catches silent / saturated outputs


class ExportConfig(Strict):
    format: Literal["onnx"] = "onnx"
    exporter: str = "auto"                 # auto | torch-legacy | torch-dynamo | optimum | ort-genai | <plugin>
    opset: int = 18
    external_data: Literal["auto", "always", "never"] = "auto"
    external_data_threshold_mb: int = 1536  # protobuf's hard limit is 2 GiB; stay clear of it
    trust_remote_code: bool = False
    fetch_weights: bool = True
    golden_max_mb: int = 96                # reference inputs/outputs bundled with the artifact (0 disables)
    copy_assets: bool = True


class OptimizationConfig(Strict):
    precision: list[Precision] = Field(default_factory=list)
    quantization: list[Quantization] = Field(default_factory=list)
    graph_optimization: GraphLevel = "none"
    int4_block_size: int = 32
    fp16_keep_io_types: bool = True
    fp16_op_block_list: list[str] = Field(default_factory=list)
    calibration_dir: str | None = None     # directory of .npz files, one input dict each (int8-static)
    calibration_samples: int = 16
    exclude_components: list[str] = Field(default_factory=list)


def _default_precision_tolerances() -> dict[str, Tolerance]:
    """Documented defaults for lower precisions. They are *policy*, not physics: cosine similarity is the primary
    criterion because absolute error scales with the output magnitude. Override per model (adapter or config) when
    the task needs tighter or looser bounds; the measured numbers are always in the report either way."""
    return {
        "fp16": Tolerance(atol=1e-2, rtol=1e-2, min_cosine=0.9995, min_equal_fraction=0.99),
        "bf16": Tolerance(atol=5e-2, rtol=2e-2, min_cosine=0.998, min_equal_fraction=0.97),
        "int8": Tolerance(atol=1e-1, rtol=1e-1, min_cosine=0.99, min_equal_fraction=0.9),
        "int4": Tolerance(atol=5e-1, rtol=2e-1, min_cosine=0.95, min_equal_fraction=0.8),
    }


class ValidationConfig(Strict):
    tolerance: Tolerance = Field(default_factory=Tolerance)
    #: per-precision tolerances keyed by precision / quantisation family (fp16, bf16, int8, int4) or exact variant precision
    precision_tolerance: dict[str, Tolerance] = Field(default_factory=_default_precision_tolerances)
    providers: list[str] = Field(default_factory=lambda: ["cpu"])
    check_dynamic_shapes: bool = True
    check_stateful: bool = True
    run_quality_checks: bool = True
    seed: int = 0
    threads: int | None = None


class BenchmarkConfig(Strict):
    providers: list[str] = Field(default_factory=lambda: ["cpu"])
    warmup: int = 2
    iterations: int = 10
    max_seconds: float = 120.0             # stop early (after >= 3 iterations) for slow graphs
    threads: int | None = None
    isolate_process: bool = True           # one subprocess per measurement: clean peak-RSS numbers
    allow_partial_fallback: bool = True    # nodes the provider lacks may run on CPU; the active-provider list is still checked
    seed: int = 0


class JobConfig(Strict):
    """Everything needed to reproduce a conversion."""
    model: str
    revision: str | None = None
    adapter: str | None = None
    adapter_options: dict[str, Any] = Field(default_factory=dict)
    plugins: list[str] = Field(default_factory=list)
    output: str = "artifacts"
    offline: bool = False
    export: ExportConfig = Field(default_factory=ExportConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    #: pipeline steps to run; default is the full pipeline
    steps: list[str] | None = None
    seed: int = 0

    @field_validator("model")
    @classmethod
    def _model_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("model must not be empty")
        return v

    def fingerprint(self, exclude: tuple[str, ...] = ("output", "steps", "offline")) -> str:
        """Stable hash of everything that affects the produced bytes (used to invalidate resumed jobs)."""
        data = self.model_dump(mode="json", exclude=set(exclude))
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


def load_config_file(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    text = p.read_text()
    try:
        data = json.loads(text) if p.suffix == ".json" else yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot parse {p}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{p}: top level must be a mapping")
    return data


def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out
