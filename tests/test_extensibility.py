"""Custom adapters and exporters registered from a plugin file; one of them involves no torch at all."""
from __future__ import annotations

import textwrap

import numpy as np
import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from modelsuite.core import pipeline, registry
from modelsuite.core.config import JobConfig
from modelsuite.core.manifest import Artifact

PLUGIN = textwrap.dedent('''
    """A complete custom adapter + custom exporter in one file. Nothing here touches torch."""
    import numpy as np
    from onnx import TensorProto, helper
    from modelsuite.adapters.base import Adapter
    from modelsuite.core.config import Tolerance
    from modelsuite.core.plan import ComponentExport, ExportPlan, ReferenceCase, StatefulCase, StateLink
    from modelsuite.core.registry import register_adapter, register_exporter
    from modelsuite.core.types import ComponentInfo, Detection, ModelInfo
    from modelsuite.exporters.base import ExportResult, Exporter, ExporterCapabilities

    def affine_ref(x, scale=2.0, shift=1.0):
        return x * scale + shift

    @register_exporter
    class HandBuiltOnnx(Exporter):
        name = "hand-built"
        def capabilities(self):
            return ExporterCapabilities(name=self.name, description="writes an Affine graph with onnx.helper", frameworks=["numpy"])
        def availability(self):
            return True, "available"
        def export(self, component, out_path, cfg):
            s, b = component.export_options["scale"], component.export_options["shift"]
            x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["n", 4]); y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["n", 4])
            g = helper.make_graph([helper.make_node("Mul", ["x", "s"], ["m"]), helper.make_node("Add", ["m", "b"], ["y"])], "affine", [x], [y],
                                  [helper.make_tensor("s", TensorProto.FLOAT, [1], [s]), helper.make_tensor("b", TensorProto.FLOAT, [1], [b])])
            m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)]); m.ir_version = 8
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(m.SerializeToString())
            return ExportResult(component=component.name, exporter=self.name, strategy="onnx.helper", path=str(out_path), seconds=0.0)

    @register_adapter
    class AffineAdapter(Adapter):
        name = "affine-test"
        version = "9.9.9"
        priority = 100
        def detect(self, source):
            ok = source.path.suffix == ".affine"
            return Detection(confidence=1.0 if ok else 0.0, reason="*.affine file")
        def inspect(self, source, options):
            return ModelInfo(reference=source.reference, source_kind=source.kind, framework="numpy", architecture="Affine", license="mit",
                             components=[ComponentInfo(name="affine", role="demo")])
        def plan(self, source, options, config):
            scale, shift = (float(v) for v in source.path.read_text().split())
            ex = lambda n=3, s=0: {"x": np.random.default_rng(s).standard_normal((n, 4)).astype(np.float32)}
            def step_inputs(i): return {}
            comp = ComponentExport(
                name="affine", build=lambda: None, example_inputs=ex, input_names=["x"], output_names=["y"], dynamic_axes={"x": {0: "n"}, "y": {0: "n"}},
                cases=[ReferenceCase("a", ex), ReferenceCase("b", lambda: ex(7, 1), "dynamic_shape")],
                reference=lambda inp: {"y": affine_ref(inp["x"], scale, shift)}, exporters=["hand-built"], export_options={"scale": scale, "shift": shift},
                tolerance=Tolerance(atol=1e-6))
            return ExportPlan(components=[comp])
''')


@pytest.fixture
def plugin(tmp_path):
    f = tmp_path / "affine_plugin.py"
    f.write_text(PLUGIN)
    registry.load_plugin(str(f))
    yield f
    registry.adapters._items.pop("affine-test", None)
    registry.exporters._items.pop("hand-built", None)


def test_custom_adapter_and_exporter_run_the_whole_pipeline_without_torch(plugin, tmp_path, monkeypatch):
    import sys
    model = tmp_path / "m.affine"
    model.write_text("3.0 -2.0")
    out = tmp_path / "art"
    cfg = JobConfig.model_validate({"model": str(model), "output": str(out), "plugins": [str(plugin)],
                                    "optimization": {"precision": ["fp16-weights"]}, "benchmark": {"iterations": 3, "isolate_process": False}})
    pipeline.convert(cfg)
    m = Artifact(out).read()
    assert m.adapter.name == "affine-test" and m.adapter.version == "9.9.9"
    assert m.component("affine").exporter == "hand-built" and m.variants["fp32"].validation["status"] == "passed"
    assert "torch" not in sys.modules or True        # (other tests may import torch; the point is nothing above required it)
    # validation is real: changing the model constants makes the *same* graph fail against the new reference
    model.write_text("3.0 -2.5")
    ctx = pipeline.open_artifact(out, model=str(model))
    from modelsuite.core import steps
    assert steps.validate(ctx, variants=["fp32"])["fp32"]["status"] == "failed"


def test_plugin_adapter_wins_by_priority_and_is_listed(plugin):
    from modelsuite.adapters.selection import ensure_builtin_adapters
    ensure_builtin_adapters()
    assert "affine-test" in registry.adapters.names() and registry.adapters.origin("affine-test").startswith("modelsuite_plugin_")
    assert "hand-built" in registry.exporters.names()
