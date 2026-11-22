"""ONNX Runtime integration: provider discovery that *actually tries* each provider, strict session creation
that never silently runs on CPU when another provider was requested, and a numpy-in/numpy-out runner."""
from __future__ import annotations

import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..core import deps
from ..core.errors import ProviderError
from ..core.types import Loose, Signature, TensorSpec
from .onnx_utils import _ELEM

log = logging.getLogger("modelsuite.ort")

SHORT_NAMES = {
    "cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider", "tensorrt": "TensorrtExecutionProvider",
    "dml": "DmlExecutionProvider", "directml": "DmlExecutionProvider", "openvino": "OpenVINOExecutionProvider",
    "coreml": "CoreMLExecutionProvider", "rocm": "ROCMExecutionProvider", "migraphx": "MIGraphXExecutionProvider",
    "webgpu": "WebGpuExecutionProvider", "qnn": "QNNExecutionProvider", "xnnpack": "XnnpackExecutionProvider",
    "azure": "AzureExecutionProvider", "cann": "CANNExecutionProvider", "dnnl": "DnnlExecutionProvider",
    "vitisai": "VitisAIExecutionProvider", "nnapi": "NnapiExecutionProvider",
}
_LONG_TO_SHORT = {}
for _s, _l in SHORT_NAMES.items():
    _LONG_TO_SHORT.setdefault(_l, _s)
_LONG_TO_SHORT["DmlExecutionProvider"] = "dml"


def canonical(name: str) -> str:
    """'cuda' / 'CUDAExecutionProvider' -> 'CUDAExecutionProvider'."""
    if name in SHORT_NAMES.values():
        return name
    try:
        return SHORT_NAMES[name.lower()]
    except KeyError:
        raise ProviderError(f"unknown execution provider '{name}'. Known: {', '.join(sorted(SHORT_NAMES))}") from None


def short(name: str) -> str:
    return _LONG_TO_SHORT.get(name, name)


class ProviderStatus(Loose):
    name: str                       # canonical ORT name
    short: str
    listed: bool                    # reported by onnxruntime.get_available_providers()
    usable: bool                    # a probe session really ran on it
    reason: str = ""
    probe_ms: float | None = None
    devices: list[str] = []


def ort_module():
    return deps.require("onnxruntime", "running ONNX models", "onnx")


def _probe_model_bytes() -> bytes:
    onnx = deps.require("onnx", "building a provider probe graph", "onnx")
    h = onnx.helper
    X = h.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 4])
    Y = h.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 4])
    W = h.make_tensor("w", onnx.TensorProto.FLOAT, [4, 4], np.eye(4, dtype=np.float32).flatten().tolist())
    g = h.make_graph([h.make_node("MatMul", ["x", "w"], ["m"]), h.make_node("Relu", ["m"], ["y"])], "probe", [X], [Y], [W])
    m = h.make_model(g, opset_imports=[h.make_opsetid("", 13)])
    m.ir_version = 8
    return m.SerializeToString()


def make_session_options(threads: int | None = None, graph_level: str = "all", log_level: int = 3, profile: bool = False):
    ort = ort_module()
    so = ort.SessionOptions()
    so.log_severity_level = log_level
    if threads:
        so.intra_op_num_threads = threads
    so.graph_optimization_level = {
        "none": ort.GraphOptimizationLevel.ORT_DISABLE_ALL, "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED, "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }[graph_level]
    so.enable_profiling = profile
    return so


def probe_providers(deep: bool = True) -> list[ProviderStatus]:
    """Every provider ORT lists, plus whether a tiny graph really runs on it.

    Merely appearing in get_available_providers() is not proof: a CUDA build without the CUDA libraries lists CUDA
    and then fails (or falls back) at session creation. Deep probing creates and runs a probe session per provider."""
    ort = ort_module()
    listed = ort.get_available_providers()
    probe = _probe_model_bytes() if deep else None
    out: list[ProviderStatus] = []
    for name in listed:
        st = ProviderStatus(name=name, short=short(name), listed=True, usable=False)
        if not deep:
            st.usable = True
            st.reason = "listed (not probed)"
            out.append(st)
            continue
        t = time.perf_counter()
        try:
            sess = _strict_session_from(probe, name, None, make_session_options(1, "none"))
            res = sess.run(None, {"x": np.ones((1, 4), np.float32)})[0]
            st.usable = bool(np.allclose(res, 1.0))
            st.reason = "ok" if st.usable else "probe graph produced wrong numbers on this provider"
        except Exception as exc:  # any failure means "not usable"; the reason is what the user needs
            st.reason = f"{type(exc).__name__}: {str(exc).strip().splitlines()[0][:200] if str(exc).strip() else ''}"
        st.probe_ms = round((time.perf_counter() - t) * 1000, 1)
        out.append(st)
    try:
        import onnxruntime as o
        dev = getattr(o, "get_device", lambda: "")()
        for st in out:
            if st.short in ("cuda", "tensorrt", "rocm", "dml", "openvino", "webgpu"):
                st.devices = [dev] if dev else []
    except Exception:
        pass
    return out


def usable_providers() -> list[str]:
    return [p.short for p in probe_providers() if p.usable]


def _strict_session_from(model: str | bytes, provider: str, provider_options: dict | None, so):
    """InferenceSession pinned to `provider`; raises ProviderError rather than quietly using another one."""
    ort = ort_module()
    name = canonical(provider)
    if name not in ort.get_available_providers():
        raise ProviderError(f"provider '{short(name)}' is not available in this onnxruntime build "
                            f"(available: {', '.join(short(p) for p in ort.get_available_providers())})")
    prov: list[Any] = [(name, provider_options)] if provider_options else [name]
    try:
        sess = ort.InferenceSession(model if isinstance(model, (bytes, bytearray)) else str(model), so, providers=prov)
    except Exception as exc:
        raise ProviderError(f"could not create a session on '{short(name)}': {exc}") from exc
    active = sess.get_providers()
    if not active or active[0] != name:
        raise ProviderError(f"requested '{short(name)}' but onnxruntime is running on {[short(a) for a in active]} "
                            "(it fell back); fix the provider installation instead of benchmarking the wrong device")
    return sess


@dataclass
class OnnxRunner:
    """numpy-in / numpy-out wrapper around an InferenceSession."""
    path: Path | None
    session: Any
    provider: str
    load_seconds: float = 0.0
    inputs: Signature = field(default_factory=Signature)
    _in_dtype: dict[str, np.dtype] = field(default_factory=dict)

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        feed = {}
        for name, arr in inputs.items():
            want = self._in_dtype.get(name)
            feed[name] = arr if want is None or arr.dtype == want else arr.astype(want)
        names = [o.name for o in self.session.get_outputs()]
        outs = self.session.run(names, feed)
        return dict(zip(names, outs))

    @property
    def signature(self) -> Signature:
        return self.inputs

    @property
    def active_providers(self) -> list[str]:
        return [short(p) for p in self.session.get_providers()]


_ORT_TYPES = {"tensor(float)": "float32", "tensor(float16)": "float16", "tensor(double)": "float64", "tensor(int64)": "int64",
              "tensor(int32)": "int32", "tensor(int16)": "int16", "tensor(int8)": "int8", "tensor(uint8)": "uint8",
              "tensor(bool)": "bool", "tensor(bfloat16)": "bfloat16"}


def create_runner(path: str | Path, provider: str = "cpu", *, threads: int | None = None, graph_level: str = "all",
                  provider_options: dict | None = None, custom_op_libraries: list[str] | None = None,
                  allow_partial_fallback: bool = True) -> OnnxRunner:
    """Create a runner on exactly the requested provider (see _strict_session_from).

    allow_partial_fallback=False additionally sets session.disable_cpu_ep_fallback, so a graph that has even one node
    the provider cannot take fails loudly instead of running partly on CPU."""
    so = make_session_options(threads, graph_level)
    for lib in custom_op_libraries or []:
        so.register_custom_ops_library(lib)
    name = canonical(provider)
    if name != "CPUExecutionProvider" and not allow_partial_fallback:
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    t = time.perf_counter()
    sess = _strict_session_from(path, name, provider_options, so)
    load = time.perf_counter() - t
    sig = Signature(
        inputs=[TensorSpec(name=i.name, dtype=_ORT_TYPES.get(i.type, i.type), shape=list(i.shape)) for i in sess.get_inputs()],
        outputs=[TensorSpec(name=o.name, dtype=_ORT_TYPES.get(o.type, o.type), shape=list(o.shape)) for o in sess.get_outputs()])
    in_dtype = {}
    for i in sess.get_inputs():
        n = _ORT_TYPES.get(i.type)
        if n and n != "bfloat16":
            in_dtype[i.name] = np.dtype(n)
    return OnnxRunner(Path(path), sess, name, load, sig, in_dtype)
