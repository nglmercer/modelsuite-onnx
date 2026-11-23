"""PyTorch -> ONNX through torch.onnx.export, in two strategies:

* torch-legacy : the TorchScript-based exporter (dynamo=False). Mature for traceable, control-flow-free modules; this is what
                 the Reverb conversion was produced with.
* torch-dynamo : the torch.export / FX based exporter (dynamo=True). Handles more Python, needs onnxscript.

Both are wrapped here with the same output conventions (named inputs/outputs, dynamic axes, external data)."""
from __future__ import annotations

import contextlib
import io
import logging
import os
import warnings
from pathlib import Path

from ..core import deps
from ..core.config import ExportConfig
from ..core.errors import ExportError
from ..core.plan import ComponentExport
from ..core.registry import register_exporter
from .base import ExportResult, Exporter, ExporterCapabilities
from .torch_support import to_tensor, torch_mod

log = logging.getLogger("modelsuite.export")


def consolidate_external_data(path: Path, min_bytes: int = 1024) -> int:
    """Rewrite a graph's weights into ONE sidecar `<name>.onnx.data`, streaming tensor by tensor.

    Handles every layout an exporter may leave behind (weights embedded in the proto, one external file per tensor, a differently
    named sidecar) without ever holding more than the largest tensor in memory -- a naive `onnx.load` + `save_model` needs the
    whole model in RAM twice, which is what kills multi-GB exports. Returns the number of tensors moved."""
    onnx = deps.require("onnx", "post-processing exported graphs", "onnx")
    from onnx import TensorProto
    model = onnx.load(str(path), load_external_data=False)
    target = path.name + ".data"
    tmp = path.parent / (target + ".tmp")
    old_files: set[Path] = set()
    moved = 0
    pos = 0
    with open(tmp, "wb") as out:
        for t in model.graph.initializer:
            if t.data_location == TensorProto.EXTERNAL:
                info = {e.key: e.value for e in t.external_data}
                src = path.parent / info["location"]
                offset = int(info.get("offset", 0))
                with open(src, "rb") as fh:
                    fh.seek(offset)
                    data = fh.read(int(info["length"])) if "length" in info else fh.read()
                old_files.add(src)
            elif len(t.raw_data) >= min_bytes:
                data = bytes(t.raw_data)
                t.ClearField("raw_data")
            else:
                continue
            out.write(data)
            del t.external_data[:]
            t.data_location = TensorProto.EXTERNAL
            for k, v in (("location", target), ("offset", str(pos)), ("length", str(len(data)))):
                e = t.external_data.add()
                e.key, e.value = k, v
            pos += len(data)
            moved += 1
            if pos % 64:                                  # keep every tensor 64-byte aligned (mmap friendly)
                pad = 64 - pos % 64
                out.write(b"\0" * pad)
                pos += pad
    final = path.parent / target
    old_files.discard(final)
    if final.exists():
        final.unlink()                                    # the previous sidecar was read in full above
    tmp.replace(final)
    path.write_bytes(model.SerializeToString())
    for f in old_files:
        if f.exists() and f.parent == path.parent:
            f.unlink()
    return moved


def _isolate_external_data(path: Path, threshold_bytes: int, mode: str) -> None:
    """Normalise how weights are stored: <name>.onnx + <name>.onnx.data when large (or when asked), one file otherwise."""
    size = path.stat().st_size
    sidecars = [p for p in path.parent.iterdir() if p.is_file() and p.name != path.name and p.name.startswith(path.stem)
                and p.suffix != ".onnx"]
    external = any(s.name.endswith((".data", "_data")) for s in sidecars) or _references_external(path)
    need = mode == "always" or (mode == "auto" and (size >= threshold_bytes or external))
    if need:
        consolidate_external_data(path)


def _references_external(path: Path) -> bool:
    onnx = deps.require("onnx")
    from onnx import TensorProto
    m = onnx.load(str(path), load_external_data=False)
    return any(t.data_location == TensorProto.EXTERNAL for t in m.graph.initializer)


class _TorchExporter(Exporter):
    strategy = ""
    dynamo = False

    def availability(self) -> tuple[bool, str]:
        missing = [m for m in ("torch", "onnx") if not deps.have(m)]
        if self.dynamo and not deps.have("onnxscript"):
            missing.append("onnxscript")
        if missing:
            return False, f"missing: {', '.join(missing)}"
        if not self.dynamo:
            # torch >= 2.9 keeps dynamo=False but newer releases may drop it; ask the signature rather than assume
            import inspect
            torch = torch_mod()
            if "dynamo" not in inspect.signature(torch.onnx.export).parameters:
                return False, "this torch version has no legacy (dynamo=False) exporter switch"
        return True, "available"

    def supports(self, component):
        torch = torch_mod()
        if not isinstance(component.module(), torch.nn.Module):
            return False, "component.build() did not return a torch.nn.Module"
        return True, ""

    def export(self, component: ComponentExport, out_path: Path, cfg: ExportConfig) -> ExportResult:
        torch = torch_mod()
        t0 = self._timed()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        module = component.module().eval()
        example = component.example_inputs()
        args = tuple(to_tensor(example[n]) for n in component.input_names)
        opset = int(component.export_options.get("opset", cfg.opset))
        kwargs = dict(input_names=component.input_names, output_names=component.output_names,
                      dynamic_axes=component.dynamic_axes or None, opset_version=opset)
        kwargs.update({k: v for k, v in component.export_options.items() if k not in ("opset",)})
        ctx = component.export_context() if component.export_context else contextlib.nullcontext()
        captured = io.StringIO()
        with warnings.catch_warnings(record=True) as caught, ctx, torch.inference_mode(), \
                contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            warnings.simplefilter("always")
            try:
                if self.dynamo:
                    # dynamic_axes is accepted by the dynamo exporter too; it is translated to dynamic_shapes.
                    torch.onnx.export(module, args, str(out_path), dynamo=True, **kwargs)
                else:
                    torch.onnx.export(module, args, str(out_path), dynamo=False, **kwargs)
            except Exception as exc:
                raise ExportError(f"{self.name} failed for '{component.name}': {type(exc).__name__}: {exc}") from exc
        module.eval()  # torch.onnx.export restores the original training flag recursively; undo that
        if not out_path.is_file():
            raise ExportError(f"{self.name} reported success but {out_path} was not written")
        _isolate_external_data(out_path, cfg.external_data_threshold_mb * 1024 * 1024, cfg.external_data)
        notes = sorted({str(w.message).splitlines()[0][:200] for w in caught if "Tracer" in str(w.category) or "export" in str(w.message).lower()})
        return ExportResult(component=component.name, exporter=self.name, strategy=self.strategy, path=str(out_path),
                            seconds=round(self._timed() - t0, 3), warnings=notes[:10],
                            tool_versions={"torch": torch.__version__, "onnx": deps.dist_version("onnx") or "?",
                                           "onnxscript": deps.dist_version("onnxscript") or "-"})


@register_exporter
class TorchLegacyExporter(_TorchExporter):
    name = "torch-legacy"
    strategy = "legacy"
    dynamo = False

    def capabilities(self) -> ExporterCapabilities:
        return ExporterCapabilities(
            name=self.name, description="torch.onnx.export with the TorchScript tracer (dynamo=False)",
            frameworks=["pytorch"], dynamic_shapes=True, stateful_models=True, custom_op_symbolics=True,
            requires=["torch", "onnx"],
            limitations=["Tracing bakes in Python control flow and shapes that were not marked dynamic",
                         "Data-dependent branching is not captured", "Some very new torch ops lack symbolics at old opsets"],
            config={"opset": "ONNX opset (default 18)", "export_options": "extra torch.onnx.export kwargs from the adapter"})


@register_exporter
class TorchDynamoExporter(_TorchExporter):
    name = "torch-dynamo"
    strategy = "dynamo"
    dynamo = True

    def capabilities(self) -> ExporterCapabilities:
        return ExporterCapabilities(
            name=self.name, description="torch.onnx.export through torch.export / FX (dynamo=True)",
            frameworks=["pytorch"], dynamic_shapes=True, stateful_models=True, custom_op_symbolics=False,
            requires=["torch", "onnx", "onnxscript"],
            limitations=["Needs graph-capturable code (torch.export constraints)", "Dynamic axes become dynamic_shapes; "
                         "unconstrained dims can specialise", "Slower to export than the legacy tracer"],
            config={"opset": "ONNX opset (default 18)"})
