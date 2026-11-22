"""ONNX structural validation: checker, full shape inference, op census and a custom-op audit."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core import deps
from ..core.types import Loose, Status
from ..runtimes.onnx_utils import graph_stats

# domains whose ops ORT implements natively
_NATIVE_DOMAINS = {"", "ai.onnx", "ai.onnx.ml", "com.microsoft", "com.ms.internal.nhwc", "ai.onnx.contrib"}


class StructuralResult(Loose):
    status: Status
    path: str
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict[str, Any] = {}

    @property
    def ok(self) -> bool:
        return self.status == Status.passed


def check_graph(path: str | Path, full_check: bool = True) -> StructuralResult:
    """onnx.checker (path-based, so >2 GB models and external data work) + strict shape inference."""
    onnx = deps.require("onnx", "validating ONNX graphs", "onnx")
    p = Path(path)
    errors: list[str] = []
    warnings: list[str] = []
    stats: dict[str, Any] = {}
    if not p.is_file():
        return StructuralResult(status=Status.failed, path=str(p), errors=[f"file not found: {p}"])
    try:
        onnx.checker.check_model(str(p), full_check=full_check)
    except Exception as exc:  # onnx.checker.ValidationError / shape inference errors
        errors.append(f"{type(exc).__name__}: {str(exc).strip().splitlines()[0][:400]}")
    try:
        stats = graph_stats(p)
    except Exception as exc:
        errors.append(f"cannot read graph: {exc}")
    if stats:
        if stats.get("custom_domains"):
            warnings.append(f"custom operator domains {stats['custom_domains']}: need a runtime that registers them")
        for loc in stats.get("external_data_files", []):
            if not (p.parent / loc).is_file():
                errors.append(f"external data file '{loc}' referenced by the graph is missing")
        if stats.get("opset") is None:
            errors.append("graph has no default-domain opset import")
    return StructuralResult(status=Status.failed if errors else Status.passed, path=str(p), errors=errors, warnings=warnings, stats=stats)
