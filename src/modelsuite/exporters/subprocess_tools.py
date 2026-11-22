"""Exporters that wrap *other projects' CLIs* (Optimum ONNX, ONNX Runtime GenAI model builder).

They run in a subprocess against a configurable interpreter because these tools pin their own versions of
transformers / torch; running them in-process would force the suite's environment to match. Point them at a
dedicated environment with MODELSUITE_OPTIMUM_PYTHON / MODELSUITE_ORTGENAI_PYTHON."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from ..core.config import ExportConfig
from ..core.errors import ExportError
from ..core.plan import ComponentExport
from ..core.registry import register_exporter
from .base import ExportResult, Exporter, ExporterCapabilities


def _probe(python: str, module: str, timeout: int = 60) -> tuple[bool, str]:
    try:
        r = subprocess.run([python, "-c", f"import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('{module}') else 3)"],
                           capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"cannot run {python}: {exc}"
    return (r.returncode == 0), ("available" if r.returncode == 0 else f"'{module}' is not installed in {python}")


def _dist_version(python: str, dist: str) -> str:
    try:
        r = subprocess.run([python, "-c", f"import importlib.metadata as m; print(m.version('{dist}'))"],
                           capture_output=True, text=True, timeout=60)
        return r.stdout.strip() if r.returncode == 0 else "?"
    except (OSError, subprocess.TimeoutExpired):
        return "?"


def _keep_log(directory: Path, name: str, cmd: list[str], proc: subprocess.CompletedProcess) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text("$ " + " ".join(cmd) + f"\n(exit {proc.returncode})\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}\n")


def _error_line(proc: subprocess.CompletedProcess) -> str:
    """The most informative line of a failed tool run: the last 'SomeError: message' line, else the last non-empty line."""
    lines = [l.strip() for l in (proc.stderr or proc.stdout).splitlines() if l.strip()]
    for l in reversed(lines):
        head = l.split(":", 1)[0]
        if ("Error" in head or "Exception" in head) and ":" in l:
            return l[:500]
    return lines[-1][:500] if lines else "(no output)"


def _move_graph(src_dir: Path, name: str, dest: Path) -> Path:
    """Move <name>.onnx and its external-data sidecars so that they sit at `dest`."""
    src = src_dir / f"{name}.onnx"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
    for sc in src_dir.glob(f"{name}.onnx_data*"):
        shutil.move(str(sc), str(dest.with_name(dest.name + ".data") if sc.name.endswith("_data") else dest.parent / sc.name))
    for sc in src_dir.glob(f"{name}.onnx.data"):
        shutil.move(str(sc), str(dest.parent / (dest.name + ".data")))
    return dest


@register_exporter
class OptimumExporter(Exporter):
    """Hugging Face Optimum ONNX (whole-model export, KV-cache aware for decoder LMs)."""
    name = "optimum"
    ENV = "MODELSUITE_OPTIMUM_PYTHON"

    def python(self) -> str:
        return os.environ.get(self.ENV) or sys.executable

    def capabilities(self) -> ExporterCapabilities:
        return ExporterCapabilities(
            name=self.name, description="optimum-onnx CLI (python -m optimum.exporters.onnx), run in a subprocess",
            frameworks=["transformers", "diffusers", "timm", "sentence-transformers"], scope="model", dynamic_shapes=True,
            stateful_models=True, requires=["optimum-onnx"],
            limitations=["Only architectures registered in optimum's TasksManager export; others fail with a clear message",
                         "Pins its own transformers range: use a dedicated environment via " + self.ENV,
                         "Exports the whole checkpoint, so per-component control is limited to what optimum splits itself"],
            config={"task": "optimum task name (default: auto-detected)", "extra_args": "list of extra CLI arguments"})

    def availability(self) -> tuple[bool, str]:
        return _probe(self.python(), "optimum.exporters.onnx")

    def supports(self, component):
        if "source" not in component.export_options:
            return False, "the adapter did not provide export_options['source'] (model path / id) for a whole-model export"
        return True, ""

    def export(self, component: ComponentExport, out_path: Path, cfg: ExportConfig) -> ExportResult:
        t0 = self._timed()
        opts = component.export_options
        py = self.python()
        ok, why = self.availability()
        if not ok:
            raise ExportError(f"optimum exporter unavailable: {why}")
        tmp = Path(tempfile.mkdtemp(prefix="optimum_", dir=out_path.parent.parent if out_path.parent.parent.exists() else None))
        cmd = [py, "-m", "optimum.exporters.onnx", "--model", str(opts["source"]), "--opset", str(opts.get("opset", cfg.opset)),
               "--device", "cpu"]
        if opts.get("task"):
            cmd += ["--task", opts["task"]]
        if cfg.trust_remote_code:
            cmd += ["--trust-remote-code"]
        cmd += list(opts.get("extra_args", [])) + [str(tmp)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        _keep_log(out_path.parent, "optimum_export.log", cmd, proc)
        if proc.returncode != 0:
            shutil.rmtree(tmp, ignore_errors=True)
            raise ExportError(f"optimum export failed (exit {proc.returncode}): {_error_line(proc)}  [full log: {out_path.parent / 'optimum_export.log'}]")
        graphs = sorted(tmp.glob("*.onnx"))
        if not graphs:
            shutil.rmtree(tmp, ignore_errors=True)
            raise ExportError("optimum finished but wrote no .onnx file")
        main = next((g for g in graphs if g.stem == "model"), graphs[0])
        _move_graph(tmp, main.stem, out_path)
        extras = {}
        for g in graphs:
            if g is not main and g.exists():
                extras[g.stem] = str(_move_graph(tmp, g.stem, out_path.with_name(g.name)))
        # tokenizer / config files optimum wrote beside the graphs are kept as assets by the pipeline
        keep = out_path.parent / "_optimum_files"
        keep.mkdir(exist_ok=True)
        for f in tmp.iterdir():
            if f.is_file():
                shutil.move(str(f), str(keep / f.name))
        shutil.rmtree(tmp, ignore_errors=True)
        return ExportResult(component=component.name, exporter=self.name, strategy="optimum-cli", path=str(out_path),
                            seconds=round(self._timed() - t0, 3), extra_components=extras,
                            tool_versions={"optimum-onnx": _dist_version(py, "optimum-onnx"),
                                           "optimum": _dist_version(py, "optimum"), "transformers": _dist_version(py, "transformers")})


@register_exporter
class OrtGenAIExporter(Exporter):
    """ONNX Runtime GenAI model builder (generative LLMs with fused KV-cache attention + genai_config.json)."""
    name = "ort-genai"
    ENV = "MODELSUITE_ORTGENAI_PYTHON"

    def python(self) -> str:
        return os.environ.get(self.ENV) or sys.executable

    def capabilities(self) -> ExporterCapabilities:
        return ExporterCapabilities(
            name=self.name, description="onnxruntime-genai model builder (python -m onnxruntime_genai.models.builder)",
            frameworks=["transformers"], scope="model", dynamic_shapes=True, stateful_models=True,
            requires=["onnxruntime-genai"],
            limitations=["Supports a fixed list of decoder LLM architectures (Llama, Mistral, Phi, Qwen, Gemma, ...)",
                         "Output is meant for the onnxruntime-genai runtime (genai_config.json); plain ORT needs the KV-cache wiring by hand",
                         "int4 / fp16 are chosen at build time through export_options['precision']"],
            config={"precision": "fp32|fp16|bf16|int4 (default fp32)", "execution_provider": "cpu|cuda|dml|webgpu"})

    def availability(self) -> tuple[bool, str]:
        return _probe(self.python(), "onnxruntime_genai")

    def supports(self, component):
        if "source" not in component.export_options:
            return False, "no export_options['source']"
        return True, ""

    def export(self, component: ComponentExport, out_path: Path, cfg: ExportConfig) -> ExportResult:
        t0 = self._timed()
        o = component.export_options
        py = self.python()
        ok, why = self.availability()
        if not ok:
            raise ExportError(f"ort-genai exporter unavailable: {why}")
        tmp = Path(tempfile.mkdtemp(prefix="ortgenai_", dir=out_path.parent.parent if out_path.parent.parent.exists() else None))
        cmd = [py, "-m", "onnxruntime_genai.models.builder", "-m", str(o["source"]), "-o", str(tmp),
               "-p", o.get("precision", "fp32"), "-e", o.get("execution_provider", "cpu")]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        _keep_log(out_path.parent, "ort_genai_export.log", cmd, proc)
        if proc.returncode != 0:
            shutil.rmtree(tmp, ignore_errors=True)
            raise ExportError(f"onnxruntime-genai builder failed (exit {proc.returncode}): {_error_line(proc)}  [full log: {out_path.parent / 'ort_genai_export.log'}]")
        graph = next(iter(sorted(tmp.glob("*.onnx"))), None)
        if graph is None:
            shutil.rmtree(tmp, ignore_errors=True)
            raise ExportError("builder finished but wrote no .onnx file")
        _move_graph(tmp, graph.stem, out_path)
        keep = out_path.parent / "_genai_files"
        keep.mkdir(exist_ok=True)
        for f in tmp.iterdir():
            if f.is_file():
                shutil.move(str(f), str(keep / f.name))
        shutil.rmtree(tmp, ignore_errors=True)
        return ExportResult(component=component.name, exporter=self.name, strategy="genai-builder", path=str(out_path),
                            seconds=round(self._timed() - t0, 3),
                            tool_versions={"onnxruntime-genai": _dist_version(py, "onnxruntime-genai")})
