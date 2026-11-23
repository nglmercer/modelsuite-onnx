"""Exporters and optimisation passes on real graphs."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
import onnx

from modelsuite.core.config import ExportConfig
from modelsuite.core.plan import ComponentExport
from modelsuite.exporters.torch_onnx import TorchDynamoExporter, TorchLegacyExporter
from modelsuite.exporters.torch_support import reference_runner
from modelsuite.optimization import passes
from modelsuite.optimization.compare import VariantMeasurement, recommend
from modelsuite.runtimes.ort import create_runner
from modelsuite.runtimes.onnx_utils import total_size
from modelsuite.validation.structural import check_graph


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.b, self.drop = torch.nn.Linear(16, 256), torch.nn.Linear(256, 8), torch.nn.Dropout(0.5)

    def forward(self, x):
        return self.b(self.drop(torch.relu(self.a(x))))


def make_comp(train_mode: bool = True) -> ComponentExport:
    torch.manual_seed(0)
    net = Net()
    net.train(train_mode)
    ex = lambda b=3, s=0: {"x": np.random.default_rng(s).standard_normal((b, 16)).astype(np.float32)}
    return ComponentExport(name="net", build=lambda: net, example_inputs=ex, input_names=["x"], output_names=["y"],
                           dynamic_axes={"x": {0: "b"}, "y": {0: "b"}})


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    comp = make_comp()
    out = tmp_path_factory.mktemp("exp") / "net.onnx"
    TorchLegacyExporter().export(comp, out, ExportConfig())
    return comp, out


@pytest.mark.parametrize("exporter", [TorchLegacyExporter, TorchDynamoExporter])
def test_both_torch_strategies_export_matching_graphs(exporter, tmp_path):
    comp = make_comp()
    out = tmp_path / "net.onnx"
    res = exporter().export(comp, out, ExportConfig())
    assert res.strategy in ("legacy", "dynamo") and check_graph(out).ok
    for b in (1, 3, 11):                                  # dynamic batch
        x = {"x": np.random.default_rng(b).standard_normal((b, 16)).astype(np.float32)}
        got = create_runner(out, "cpu").run(x)["y"]
        assert np.abs(got - reference_runner(comp)(x)["y"]).max() < 1e-5


def test_export_leaves_the_module_in_eval_mode_and_reference_is_deterministic():
    """Regression: torch.onnx.export restores the wrapper's ORIGINAL training flag recursively, which re-enabled dropout in the
    reference model and produced 'parity failures' of ~2.0 on a perfectly good graph."""
    comp = make_comp(train_mode=True)
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        TorchLegacyExporter().export(comp, pathlib.Path(d) / "n.onnx", ExportConfig())
    assert comp.module().training is False
    ref = reference_runner(comp)
    x = comp.example_inputs()
    assert np.array_equal(ref(x)["y"], ref(x)["y"])       # dropout would make these differ


def test_external_data_modes(tmp_path):
    out = tmp_path / "ext" / "net.onnx"
    TorchLegacyExporter().export(make_comp(), out, ExportConfig(external_data="always"))
    assert (out.parent / "net.onnx.data").is_file() and check_graph(out).ok
    assert create_runner(out, "cpu").run({"x": np.zeros((2, 16), np.float32)})["y"].shape == (2, 8)
    out2 = tmp_path / "small" / "net.onnx"
    TorchLegacyExporter().export(make_comp(), out2, ExportConfig(external_data="never"))
    assert not (out2.parent / "net.onnx.data").exists()


def test_export_failure_is_an_ExportError_with_the_component_name(tmp_path):
    from modelsuite.core.errors import ExportError

    class Bad(torch.nn.Module):
        def forward(self, x):
            raise RuntimeError("boom")
    comp = make_comp()
    comp._module, comp._built = Bad(), True
    with pytest.raises(ExportError, match="net"):
        TorchLegacyExporter().export(comp, tmp_path / "x.onnx", ExportConfig())


# ---- optimisation passes ---------------------------------------------------------------------------------------------------
def _max_diff(a, b, x):
    return float(np.abs(create_runner(a, "cpu").run(x)["y"] - create_runner(b, "cpu").run(x)["y"]).max())


X = {"x": np.random.default_rng(5).standard_normal((4, 16)).astype(np.float32)}


@pytest.mark.parametrize("dtype,bound", [("float16", 5e-3), ("bfloat16", 5e-2)])
def test_cast_weights_shrinks_file_keeps_fp32_compute(exported, tmp_path, dtype, bound):
    _, src = exported
    r = passes.cast_weights(src, tmp_path / "w.onnx", dtype)
    assert r.size_bytes < 0.6 * total_size(src) and check_graph(tmp_path / "w.onnx").ok
    assert 0 < _max_diff(src, tmp_path / "w.onnx", X) < bound
    assert r.details["tensors_converted"] >= 2 and "runtime RAM does not" in r.notes[0]


def test_cast_weights_keeps_a_tensor_that_would_overflow_fp16(tmp_path):
    from onnx import TensorProto, helper, numpy_helper
    big = np.full((16, 16), 1e6, np.float32)               # > 65504
    g = helper.make_graph([helper.make_node("MatMul", ["x", "w"], ["y"])], "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 16])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 16])], [numpy_helper.from_array(big, "w")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, str(tmp_path / "o.onnx"))
    r = passes.cast_weights(tmp_path / "o.onnx", tmp_path / "o16.onnx", "float16")
    assert r.details["tensors_converted"] == 0 and r.details["tensors_kept_fp32"] == 1     # kept in fp32 rather than corrupted to inf


def test_fp16_conversion_produces_a_structurally_valid_graph(exported, tmp_path):
    """Regression: onnxconverter-common's output had stale float32 value_info and failed strict shape inference."""
    _, src = exported
    r = passes.convert_fp16(src, tmp_path / "h.onnx")
    assert check_graph(tmp_path / "h.onnx").ok and r.details["converter"]
    assert _max_diff(src, tmp_path / "h.onnx", X) < 5e-3


def test_topological_sort_repairs_out_of_order_nodes():
    from onnx import TensorProto, helper
    n1 = helper.make_node("Relu", ["x"], ["a"], name="n1")
    n2 = helper.make_node("Neg", ["a"], ["y"], name="n2")
    g = helper.make_graph([n2, n1], "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])], [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])])
    assert passes.topological_sort(g) is True and [n.name for n in g.node] == ["n1", "n2"]
    assert passes.topological_sort(g) is False


def test_int8_dynamic_and_int4_and_graph_opt(exported, tmp_path):
    _, src = exported
    q8 = passes.quantize_int8_dynamic(src, tmp_path / "q8.onnx")
    assert q8.size_bytes < 0.45 * total_size(src) and check_graph(tmp_path / "q8.onnx").ok
    q4 = passes.quantize_int4_weights(src, tmp_path / "q4.onnx", block_size=32)
    assert q4.details["matmul_nbits_nodes"] == 2 and q4.details["gemm_rewritten_to_matmul"] == 2      # torch exports Linear as Gemm
    assert q4.size_bytes < 0.4 * total_size(src)          # 4-bit weights + per-block scales
    assert check_graph(tmp_path / "q4.onnx").ok and 0 < _max_diff(src, tmp_path / "q4.onnx", X) < 0.5
    opt = passes.graph_optimize(src, tmp_path / "opt.onnx", "extended")
    assert _max_diff(src, tmp_path / "opt.onnx", X) < 1e-5 and "CPU" in opt.notes[0]
    with pytest.raises(Exception):
        passes.graph_optimize(src, tmp_path / "o2.onnx", "bogus")


def test_int4_on_a_graph_with_nothing_to_quantise_is_an_error_not_a_noop(tmp_path):
    from onnx import TensorProto, helper
    g = helper.make_graph([helper.make_node("Relu", ["x"], ["y"])], "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, str(tmp_path / "r.onnx"))
    with pytest.raises(passes.OptimizationError, match="nothing to quantise"):
        passes.quantize_int4_weights(tmp_path / "r.onnx", tmp_path / "r4.onnx")


def test_static_int8_needs_calibration_data(exported, tmp_path):
    _, src = exported
    with pytest.raises(passes.OptimizationError, match="calibration"):
        passes.quantize_int8_static(src, tmp_path / "s.onnx", [])
    samples = [{"x": np.random.default_rng(i).standard_normal((4, 16)).astype(np.float32)} for i in range(4)]
    r = passes.quantize_int8_static(src, tmp_path / "s.onnx", samples)
    assert r.details["calibration_samples"] == 4 and check_graph(tmp_path / "s.onnx").ok


# ---- recommendation policy -------------------------------------------------------------------------------------------------------------
def M(name, lat, size, ok):
    return VariantMeasurement(variant=name, latency_ms=lat, size_bytes=size, validation_passed=ok, worst_max_abs=0.0)


def test_recommend_abstains_without_measurements():
    r = recommend([M("fp32", None, 100, None), M("fp16", None, 50, None)])
    assert r.recommended is None and "never validated" in " ".join(r.warnings)


def test_recommend_never_picks_a_failing_variant_and_flags_slow_low_precision():
    r = recommend([M("fp32", 10, 100, True), M("fp16", 14, 50, True), M("int8", 2, 25, False)], objective="latency")
    assert r.recommended == "fp32"                                        # int8 is fastest but failed validation
    assert any("fp16 is smaller but 1.40x SLOWER" in w for w in r.warnings)
    assert recommend([M("fp32", 10, 100, True), M("fp16", 14, 50, True)], objective="size").recommended == "fp16"
    assert recommend([M("fp32", 10, 100, True)], objective="latency").recommended == "fp32"
    assert recommend([M("fp32", 10, 100, True), M("fp16", None, 50, True)], objective="latency").recommended == "fp32"


# ---- streaming external-data consolidation ------------------------------------------------------------------------------------------------
def _model_with_weights(n=3):
    from onnx import TensorProto, helper, numpy_helper
    inits = [numpy_helper.from_array(np.random.default_rng(i).standard_normal((32, 32)).astype(np.float32), f"w{i}") for i in range(n)]
    nodes, prev = [], "x"
    for i in range(n):
        nodes.append(helper.make_node("MatMul", [prev, f"w{i}"], [f"h{i}"]))
        prev = f"h{i}"
    g = helper.make_graph(nodes, "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 32])], [helper.make_tensor_value_info(prev, TensorProto.FLOAT, [1, 32])], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    return m, prev


@pytest.mark.parametrize("layout", ["embedded", "one-file-per-tensor", "wrong-sidecar-name"])
def test_consolidate_external_data_handles_every_layout_exporters_leave_behind(tmp_path, layout):
    """Regression for multi-GB exports: torch writes one file per tensor above 2 GiB; the old code loaded everything into RAM."""
    from modelsuite.exporters.torch_onnx import consolidate_external_data
    m, out = _model_with_weights()
    p = tmp_path / "m.onnx"
    x = {"x": np.random.default_rng(9).standard_normal((1, 32)).astype(np.float32)}
    onnx.save(m, str(p))
    want = create_runner(p, "cpu").run(x)[out]
    if layout == "one-file-per-tensor":
        for t in m.graph.initializer:
            onnx.external_data_helper.set_external_data(t, location=f"{t.name}.bin")
            (tmp_path / f"{t.name}.bin").write_bytes(t.raw_data)
            t.ClearField("raw_data")
        onnx.save(m, str(p))
    elif layout == "wrong-sidecar-name":
        onnx.save_model(m, str(p), save_as_external_data=True, all_tensors_to_one_file=True, location="weights.bin", size_threshold=0)
    assert consolidate_external_data(p) == 3
    assert sorted(f.name for f in tmp_path.iterdir()) == ["m.onnx", "m.onnx.data"]               # stray per-tensor files are gone
    assert p.stat().st_size < 2000 and check_graph(p).ok
    assert np.allclose(create_runner(p, "cpu").run(x)[out], want, atol=1e-6)
    assert consolidate_external_data(p) == 3                                                     # idempotent: re-running keeps it valid
    assert np.allclose(create_runner(p, "cpu").run(x)[out], want, atol=1e-6)


def test_topological_sort_respects_values_captured_by_loop_bodies():
    """Regression: the int8 quantiser left a Loop-containing graph unsorted; a node that only a Loop *body* reads must still come first."""
    from onnx import TensorProto, helper
    body = helper.make_graph([helper.make_node("Add", ["carried", "outer_const"], ["carried_out"]), helper.make_node("Identity", ["cond"], ["cond_out"])], "body",
                             [helper.make_tensor_value_info("i", TensorProto.INT64, []), helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
                              helper.make_tensor_value_info("carried", TensorProto.FLOAT, [1])],
                             [helper.make_tensor_value_info("cond_out", TensorProto.BOOL, []), helper.make_tensor_value_info("carried_out", TensorProto.FLOAT, [1])])
    loop = helper.make_node("Loop", ["trip", "", "x"], ["y"], body=body, name="loop")
    make_const = helper.make_node("Neg", ["x"], ["outer_const"], name="make_const")             # only the loop body uses outer_const
    g = helper.make_graph([loop, make_const], "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1]), helper.make_tensor_value_info("trip", TensorProto.INT64, [])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])])
    assert passes.topological_sort(g) is True and [n.name for n in g.node] == ["make_const", "loop"]


def test_quantised_graph_with_a_loop_is_valid_after_the_pass(tmp_path):
    """A MatMul feeding a Loop: quantisation inserts nodes; the pass must hand back a graph the strict checker accepts."""
    from onnx import TensorProto, helper, numpy_helper
    w = numpy_helper.from_array(np.random.default_rng(0).standard_normal((32, 32)).astype(np.float32), "w")
    body = helper.make_graph([helper.make_node("Add", ["c", "h"], ["c_out"]), helper.make_node("Identity", ["cond"], ["cond_out"])], "body",
                             [helper.make_tensor_value_info("i", TensorProto.INT64, []), helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
                              helper.make_tensor_value_info("c", TensorProto.FLOAT, [1, 32])],
                             [helper.make_tensor_value_info("cond_out", TensorProto.BOOL, []), helper.make_tensor_value_info("c_out", TensorProto.FLOAT, [1, 32])])
    g = helper.make_graph([helper.make_node("MatMul", ["x", "w"], ["h"]), helper.make_node("Loop", ["trip", "", "h"], ["y"], body=body)], "g",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 32]), helper.make_tensor_value_info("trip", TensorProto.INT64, [])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 32])], [w])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, str(tmp_path / "l.onnx"))
    passes.quantize_int8_dynamic(tmp_path / "l.onnx", tmp_path / "q.onnx")
    assert check_graph(tmp_path / "q.onnx").ok
