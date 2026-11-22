"""Independently selectable optimisation passes. Each takes a graph path and writes a new one; none mutates its input.

    graph_optimize   ORT offline graph optimisation (constant folding, fusion) at basic / extended / all
    convert_fp16     full float16 conversion (onnxconverter-common), I/O kept fp32 by default
    cast_weights     weights-only fp16 / bf16 storage (Cast nodes restore fp32 at load): size win, runtime is still fp32
    quantize_int8_dynamic / quantize_int8_static / quantize_int4_weights   onnxruntime.quantization

Nothing here claims a pass is a good idea; quality and speed are measured by validation / benchmarking and compared
in optimization.compare."""
from __future__ import annotations

import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from ..core import deps
from ..core.config import OptimizationConfig
from ..core.errors import ConfigError, DependencyMissingError, ModelSuiteError
from ..core.types import Loose
from ..runtimes.onnx_utils import total_size


class PassResult(Loose):
    path: str
    size_bytes: int
    notes: list[str] = []
    details: dict[str, Any] = {}


class OptimizationError(ModelSuiteError):
    pass


def _ext_name(dst: Path) -> str:
    return dst.name + ".data"


def _clean(dst: Path) -> None:
    for f in [dst, dst.with_name(_ext_name(dst))]:
        if f.exists():
            f.unlink()


def _save(model, dst: Path, force_external: bool = False, threshold: int = 1536 * 1024 * 1024) -> None:
    onnx = deps.require("onnx")
    _clean(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    big = model.ByteSize() >= threshold if not force_external else True
    if big:
        onnx.save_model(model, str(dst), save_as_external_data=True, all_tensors_to_one_file=True, location=_ext_name(dst), size_threshold=1024)
    else:
        onnx.save_model(model, str(dst))


def _load_full(src: Path):
    onnx = deps.require("onnx")
    return onnx.load(str(src), load_external_data=True)


def _has_external(src: Path) -> bool:
    return src.with_name(_ext_name(src)).exists() or any(src.parent.glob(src.stem + ".onnx_data"))


# --- graph optimisation ----------------------------------------------------------------------------------------
def graph_optimize(src: Path, dst: Path, level: str = "extended") -> PassResult:
    """Offline ORT optimisation. `all` adds hardware-specific layout transforms that are NOT portable across machines."""
    ort = deps.require("onnxruntime", "graph optimisation", "onnx")
    if level not in ("basic", "extended", "all"):
        raise ConfigError(f"graph optimisation level must be basic|extended|all, got '{level}'")
    _clean(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    so = ort.SessionOptions()
    so.graph_optimization_level = {"basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC, "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
                                   "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL}[level]
    so.optimized_model_filepath = str(dst)
    so.log_severity_level = 3
    if _has_external(src) or total_size(src) > 1024 ** 3:
        so.add_session_config_entry("session.optimized_model_external_initializers_file_name", _ext_name(dst))
        so.add_session_config_entry("session.optimized_model_external_initializers_min_size_in_bytes", "1024")
    try:
        ort.InferenceSession(str(src), so, providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise OptimizationError(f"ORT graph optimisation failed: {exc}") from exc
    if not dst.is_file():
        raise OptimizationError("ORT did not write an optimised model")
    notes = ["optimised for the CPU execution provider"]
    if level == "all":
        notes.append("level 'all' inserts hardware-specific layout nodes: the result is only guaranteed to load in ORT on similar hardware")
    return PassResult(path=str(dst), size_bytes=total_size(dst), notes=notes, details={"level": level})


# --- precision ----------------------------------------------------------------------------------------------------
def topological_sort(graph) -> bool:
    """Reorder graph.node so every node follows the producers of its inputs (including names captured by sub-graphs).
    Returns True if the order changed. Raises OptimizationError on a cycle / missing producer."""
    def free_names(node) -> set[str]:
        names = {i for i in node.input if i}
        for a in node.attribute:
            for sub in ([a.g] if a.HasField("g") else []) + list(a.graphs):
                local = {i.name for i in sub.input} | {t.name for t in sub.initializer} | {o for n in sub.node for o in n.output}
                for n in sub.node:
                    names |= {x for x in free_names(n) if x not in local}
        return names
    avail = {i.name for i in graph.input} | {t.name for t in graph.initializer} | {t.name for t in graph.sparse_initializer}
    nodes = list(graph.node)
    needs = [free_names(n) for n in nodes]
    produced_by = {o: k for k, n in enumerate(nodes) for o in n.output if o}
    done, order, changed = [False] * len(nodes), [], False
    ready_queue = [k for k in range(len(nodes))]
    progress = True
    while len(order) < len(nodes) and progress:
        progress = False
        for k in ready_queue:
            if done[k]:
                continue
            if all(x in avail or x not in produced_by for x in needs[k]):
                done[k] = True
                order.append(k)
                avail.update(o for o in nodes[k].output if o)
                progress = True
        ready_queue = [k for k in ready_queue if not done[k]]
    if len(order) < len(nodes):
        raise OptimizationError("graph has a cycle or an input with no producer; cannot topologically sort")
    changed = order != list(range(len(nodes)))
    if changed:
        del graph.node[:]
        graph.node.extend(nodes[k] for k in order)
    return changed


def _structurally_valid(path: Path) -> str | None:
    """None if the saved graph passes the strict ONNX checker, else the error text."""
    onnx = deps.require("onnx")
    try:
        onnx.checker.check_model(str(path), full_check=True)
        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {str(exc).strip().splitlines()[0][:300]}"


def convert_fp16(src: Path, dst: Path, keep_io_types: bool = True, op_block_list: Iterable[str] = ()) -> PassResult:
    """Full fp16 graph (weights and compute). Both converters available are tried in turn (ORT's maintained fork of the fp16
    converter, then onnxconverter-common) and the first output that passes the strict ONNX checker is kept. Many CPU kernels run fp16 by
    casting internally, so expect a size win and possibly *slower* CPU inference: benchmark before adopting."""
    blocked = sorted(set(op_block_list))
    attempts: list[str] = []

    def ort_converter(model):
        mod = deps.require("onnxruntime.transformers.float16", "float16 conversion", "onnx", package="onnxruntime")
        out = mod.convert_float_to_float16(model, keep_io_types=keep_io_types, op_block_list=blocked or None)
        topological_sort(out.graph)
        return out

    def onnxconverter(model):
        conv = deps.require("onnxconverter_common.float16", "float16 conversion", "fp16", package="onnxconverter-common")
        kwargs: dict[str, Any] = {"keep_io_types": keep_io_types}
        if blocked:
            kwargs["op_block_list"] = blocked
        original = conv.remove_unnecessary_cast_node     # its optional Cast-pair cleanup crashes on multi-consumer Casts; it is only a tidy-up
        conv.remove_unnecessary_cast_node = lambda graph: None
        try:
            out = conv.convert_float_to_float16(model, **kwargs)
        finally:
            conv.remove_unnecessary_cast_node = original
        topological_sort(out.graph)
        return out

    for name, fn in (("onnxruntime.transformers.float16", ort_converter), ("onnxconverter-common", onnxconverter)):
        try:
            model = _load_full(src)
            out = fn(model)
        except DependencyMissingError as exc:
            attempts.append(f"{name}: not installed")
            continue
        except Exception as exc:
            attempts.append(f"{name}: {type(exc).__name__}: {str(exc)[:200]}")
            continue
        # both leave value_info typed float32 on tensors they converted: stale annotations that strict shape inference rejects; inference recomputes them
        del out.graph.value_info[:]
        _save(out, dst, force_external=_has_external(src))
        bad = _structurally_valid(dst)
        if bad is None:
            return PassResult(path=str(dst), size_bytes=total_size(dst), details={"keep_io_types": keep_io_types, "converter": name, "rejected": attempts},
                              notes=["float16 compute graph" + (" with fp32 inputs/outputs" if keep_io_types else "") + f" (converter: {name})"]
                              + [f"rejected converter output: {a}" for a in attempts])
        attempts.append(f"{name}: produced an invalid graph ({bad})")
    _clean(dst)
    raise OptimizationError("fp16 conversion failed with every available converter: " + " | ".join(attempts))


def cast_weights(src: Path, dst: Path, dtype: str = "float16", min_elements: int = 128) -> PassResult:
    """Store float32 initialisers as fp16/bf16 and restore them with a Cast at the start of the graph.

    Shrinks the file about 2x while compute stays fp32 and runs on any provider. ORT's constant folding turns the
    Casts back into fp32 initialisers at load time, so *runtime RAM does not shrink*; this is storage compression."""
    onnx = deps.require("onnx")
    from onnx import TensorProto, helper, numpy_helper
    if dtype not in ("float16", "bfloat16"):
        raise ConfigError("cast_weights dtype must be float16 or bfloat16")
    if dtype == "bfloat16":
        deps.require("ml_dtypes", "bfloat16 weight storage", package="ml_dtypes")
    model = _load_full(src)
    g = model.graph
    graph_inputs = {i.name for i in g.input}
    converted, casts, kept = [], [], 0
    kept_inits = []
    n_cast_elems = 0
    for init in g.initializer:
        if init.data_type != TensorProto.FLOAT or init.name in graph_inputs or int(np.prod(init.dims or [1])) < min_elements:
            kept_inits.append(init)
            kept += 1
            continue
        arr = numpy_helper.to_array(init)
        if dtype == "float16":
            with np.errstate(over="ignore"):      # overflow is checked for explicitly on the next line
                small = arr.astype(np.float16)
            if not np.isfinite(small).all() and np.isfinite(arr).all():
                kept_inits.append(init)   # would overflow fp16: keep that tensor in fp32 rather than corrupt it
                kept += 1
                continue
        else:
            import ml_dtypes
            small = arr.astype(ml_dtypes.bfloat16)
        packed = numpy_helper.from_array(small, init.name + "__" + dtype)
        converted.append(packed)
        casts.append(helper.make_node("Cast", [packed.name], [init.name], to=TensorProto.FLOAT, name=f"cast_w_{len(casts)}"))
        n_cast_elems += arr.size
    del g.initializer[:]
    g.initializer.extend(kept_inits + converted)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(casts + nodes)           # casts first keeps the graph topologically sorted
    _save(model, dst, force_external=_has_external(src))
    return PassResult(path=str(dst), size_bytes=total_size(dst), details={"dtype": dtype, "tensors_converted": len(converted), "tensors_kept_fp32": kept,
                                                                      "elements_converted": n_cast_elems},
                      notes=[f"weights stored as {dtype}; compute remains float32; ORT folds the casts at load (file size shrinks, runtime RAM does not)"])


# --- quantisation ---------------------------------------------------------------------------------------------------------
def _exclude_nodes(src: Path, exclude_ops: Iterable[str]) -> list[str]:
    ops = set(exclude_ops)
    if not ops:
        return []
    from ..runtimes.onnx_utils import load_header
    return [n.name for n in load_header(src).graph.node if n.op_type in ops and n.name]


def quantize_int8_dynamic(src: Path, dst: Path, op_types: tuple[str, ...] = ("MatMul", "Gemm"), per_channel: bool = True,
                          exclude_ops: Iterable[str] = ()) -> PassResult:
    q = deps.require("onnxruntime.quantization", "int8 quantisation", "onnx")
    _clean(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        q.quantize_dynamic(str(src), str(dst), op_types_to_quantize=list(op_types), per_channel=per_channel, weight_type=q.QuantType.QInt8,
                           nodes_to_exclude=_exclude_nodes(src, exclude_ops) or None,
                           use_external_data_format=_has_external(src) or total_size(src) > 1024 ** 3)
    except Exception as exc:
        raise OptimizationError(f"int8 dynamic quantisation failed: {type(exc).__name__}: {exc}") from exc
    return PassResult(path=str(dst), size_bytes=total_size(dst), details={"op_types": list(op_types), "per_channel": per_channel},
                      notes=["weights int8, activations quantised on the fly; accuracy is NOT guaranteed: validate"])


def load_calibration(directory: str | Path, limit: int) -> list[dict[str, np.ndarray]]:
    files = sorted(Path(directory).glob("*.npz"))[:limit]
    if not files:
        raise ConfigError(f"no .npz calibration samples in {directory}")
    return [{k: v for k, v in np.load(f).items()} for f in files]


def quantize_int8_static(src: Path, dst: Path, samples: list[dict[str, np.ndarray]], op_types: tuple[str, ...] = ("MatMul", "Gemm", "Conv"),
                         per_channel: bool = True, exclude_ops: Iterable[str] = ()) -> PassResult:
    q = deps.require("onnxruntime.quantization", "int8 static quantisation", "onnx")
    if not samples:
        raise OptimizationError("static quantisation needs calibration samples (calibration_dir, or golden inputs in the artifact)")

    class Reader(q.CalibrationDataReader):
        def __init__(self, data):
            self.data, self._i = data, iter(data)

        def get_next(self):
            return next(self._i, None)

        def rewind(self):
            self._i = iter(self.data)
    _clean(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        q.quantize_static(str(src), str(dst), Reader(samples), quant_format=q.QuantFormat.QDQ, per_channel=per_channel,
                          weight_type=q.QuantType.QInt8, activation_type=q.QuantType.QInt8, op_types_to_quantize=list(op_types),
                          nodes_to_exclude=_exclude_nodes(src, exclude_ops) or None,
                          use_external_data_format=_has_external(src) or total_size(src) > 1024 ** 3)
    except Exception as exc:
        raise OptimizationError(f"int8 static quantisation failed: {type(exc).__name__}: {exc}") from exc
    return PassResult(path=str(dst), size_bytes=total_size(dst), details={"calibration_samples": len(samples), "op_types": list(op_types)},
                      notes=[f"calibrated on {len(samples)} sample(s); calibration quality decides accuracy: validate"])


def gemm_to_matmul(model) -> int:
    """Rewrite Gemm(A, W_const[, C]) into MatMul(A, W') [+ Add]. torch exports 2-D nn.Linear as Gemm, which MatMulNBits skips.
    Only the exactly-equivalent case is rewritten: alpha == beta == 1, transA == 0, constant 2-D B. Returns the number rewritten."""
    from onnx import helper, numpy_helper
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    new_nodes, count = [], 0
    for n in g.node:
        if n.op_type != "Gemm" or n.domain not in ("", "ai.onnx"):
            new_nodes.append(n)
            continue
        a = {x.name: helper.get_attribute_value(x) for x in n.attribute}
        b_name = n.input[1]
        if (a.get("alpha", 1.0) != 1.0 or a.get("beta", 1.0) != 1.0 or a.get("transA", 0) != 0 or b_name not in inits
                or len(inits[b_name].dims) != 2):
            new_nodes.append(n)
            continue
        w = numpy_helper.to_array(inits[b_name])
        if a.get("transB", 0):
            w = np.ascontiguousarray(w.T)
        wname = f"{b_name}__mm"
        g.initializer.append(numpy_helper.from_array(w, wname))
        has_c = len(n.input) > 2 and n.input[2]
        mm_out = n.output[0] + "__mm" if has_c else n.output[0]
        new_nodes.append(helper.make_node("MatMul", [n.input[0], wname], [mm_out], name=(n.name or n.output[0]) + "__matmul"))
        if has_c:
            new_nodes.append(helper.make_node("Add", [mm_out, n.input[2]], [n.output[0]], name=(n.name or n.output[0]) + "__bias"))
        count += 1
    if count:
        del g.node[:]
        g.node.extend(new_nodes)
    return count


def quantize_int4_weights(src: Path, dst: Path, block_size: int = 32, symmetric: bool = True, exclude_ops: Iterable[str] = ()) -> PassResult:
    """Weight-only int4 (MatMulNBits) for MatMul weights. Everything else stays float."""
    mod = deps.require("onnxruntime.quantization.matmul_nbits_quantizer", "int4 quantisation", "onnx")
    from ..runtimes.onnx_utils import load_header
    excl = _exclude_nodes(src, exclude_ops)
    model = _load_full(src)
    rewritten = gemm_to_matmul(model)
    try:
        quant = mod.MatMulNBitsQuantizer(model, bits=4, block_size=block_size, is_symmetric=symmetric, nodes_to_exclude=excl or None)
        quant.process()
        _clean(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        quant.model.save_model_to_file(str(dst), use_external_data_format=_has_external(src) or total_size(src) > 1024 ** 3)
    except Exception as exc:
        raise OptimizationError(f"int4 quantisation failed: {type(exc).__name__}: {exc}") from exc
    n = sum(1 for nd in load_header(dst).graph.node if nd.op_type == "MatMulNBits")
    if n == 0:
        raise OptimizationError("int4 quantisation changed nothing: no MatMul with a constant weight was found (nothing to quantise)")
    return PassResult(path=str(dst), size_bytes=total_size(dst), details={"block_size": block_size, "symmetric": symmetric, "matmul_nbits_nodes": n,
                                                                      "gemm_rewritten_to_matmul": rewritten},
                      notes=([f"{rewritten} Gemm node(s) rewritten to MatMul+Add first (MatMulNBits only takes MatMul)"] if rewritten else []) + ["MatMul weights int4 (block-wise); other weights and all compute stay float; expect measurable quality loss on small models"])
