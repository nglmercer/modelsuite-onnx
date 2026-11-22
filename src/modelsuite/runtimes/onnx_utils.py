"""ONNX-file level helpers: signatures, op histograms, size accounting. Needs `onnx`."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ..core import deps
from ..core.types import Signature, TensorSpec

# ONNX TensorProto elem types -> numpy-style names (only names; no numpy dtype needed for bfloat16)
_ELEM = {1: "float32", 2: "uint8", 3: "int8", 4: "uint16", 5: "int16", 6: "int32", 7: "int64", 8: "string", 9: "bool",
         10: "float16", 11: "float64", 12: "uint32", 13: "uint64", 14: "complex64", 15: "complex128", 16: "bfloat16"}


def _dims(t) -> list:
    out: list[Any] = []
    for d in t.tensor_type.shape.dim:
        if d.HasField("dim_value"):
            out.append(int(d.dim_value))
        elif d.HasField("dim_param") and d.dim_param:
            out.append(d.dim_param)
        else:
            out.append(None)
    return out


def load_header(path: Path):
    """Load a model *without* its external tensors (cheap even for GB-sized graphs)."""
    onnx = deps.require("onnx", "reading ONNX graphs", "onnx")
    return onnx.load(str(path), load_external_data=False)


def graph_signature(path: Path) -> Signature:
    model = load_header(path)
    init = {i.name for i in model.graph.initializer}
    ins = [TensorSpec(name=i.name, dtype=_ELEM.get(i.type.tensor_type.elem_type, "unknown"), shape=_dims(i.type))
           for i in model.graph.input if i.name not in init]
    outs = [TensorSpec(name=o.name, dtype=_ELEM.get(o.type.tensor_type.elem_type, "unknown"), shape=_dims(o.type))
            for o in model.graph.output]
    return Signature(inputs=ins, outputs=outs)


def graph_stats(path: Path) -> dict[str, Any]:
    model = load_header(path)
    ops = Counter((n.domain or "ai.onnx", n.op_type) for n in model.graph.node)
    domains = {d.domain: d.version for d in model.opset_import}
    external = sorted({e.value for t in model.graph.initializer for e in t.external_data if e.key == "location"})
    return {
        "ir_version": model.ir_version,
        "opset": domains.get("", domains.get("ai.onnx")),
        "opset_imports": domains,
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
        "ops": {f"{d}::{o}" if d != "ai.onnx" else o: c for (d, o), c in sorted(ops.items())},
        "custom_domains": sorted(d for d in domains if d not in ("", "ai.onnx", "ai.onnx.ml", "com.microsoft")),
        "external_data_files": external,
        "producer": f"{model.producer_name} {model.producer_version}".strip(),
    }


def total_size(path: Path) -> int:
    """Bytes of a graph plus its external data files."""
    size = path.stat().st_size
    for pattern in (path.name + ".data", path.stem + ".onnx_data"):
        f = path.parent / pattern
        if f.is_file():
            size += f.stat().st_size
    for loc in graph_stats(path)["external_data_files"]:
        f = path.parent / loc
        if f.is_file() and f.name not in (path.name + ".data", path.stem + ".onnx_data"):
            size += f.stat().st_size
    return size


def numpy_dtype(name: str):
    if name == "bfloat16":
        ml = deps.require("ml_dtypes", "bfloat16 tensors", package="ml_dtypes")
        return np.dtype(ml.bfloat16)
    return np.dtype(name)


def synth_inputs(sig: Signature, seed: int = 0, dynamic_default: int = 8) -> dict[str, np.ndarray]:
    """Last-resort random inputs from a signature. Integer inputs are small non-negative values, which is *not*
    guaranteed to be valid for every graph (e.g. embedding indices) -- golden inputs are always preferred."""
    rng = np.random.default_rng(seed)
    out = {}
    for t in sig.inputs:
        shape = [d if isinstance(d, int) else dynamic_default for d in t.shape]
        dt = numpy_dtype(t.dtype)
        if dt.kind == "f":
            out[t.name] = rng.standard_normal(shape).astype(dt)
        elif dt.kind in "iu":
            out[t.name] = rng.integers(0, 2, shape).astype(dt)
        elif dt.kind == "b":
            out[t.name] = rng.integers(0, 2, shape).astype(bool)
        else:
            out[t.name] = np.zeros(shape, dt)
    return out
