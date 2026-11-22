"""Provider discovery and the no-silent-fallback guarantee."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")
import onnxruntime as ort

from modelsuite.core.errors import ProviderError
from modelsuite.runtimes import ort as R
from modelsuite.runtimes.onnx_utils import graph_signature, graph_stats, synth_inputs, total_size
from modelsuite.validation.structural import check_graph


@pytest.fixture(scope="module")
def probe_graph(tmp_path_factory):
    import onnx
    p = tmp_path_factory.mktemp("g") / "p.onnx"
    p.write_bytes(R._probe_model_bytes())
    return p


def test_cpu_is_probed_usable_and_every_status_has_a_reason():
    st = {p.short: p for p in R.probe_providers()}
    assert st["cpu"].usable and st["cpu"].reason == "ok"
    assert all(p.reason for p in st.values())


def test_names_canonicalise_and_unknown_is_rejected():
    assert R.canonical("cuda") == "CUDAExecutionProvider" and R.canonical("CPUExecutionProvider") == "CPUExecutionProvider"
    assert R.short("DmlExecutionProvider") == "dml"
    with pytest.raises(ProviderError, match="unknown execution provider"):
        R.canonical("quantum")


@pytest.mark.parametrize("name", ["cuda", "tensorrt", "dml", "openvino", "coreml", "rocm"])
def test_requesting_an_unavailable_provider_raises_instead_of_using_cpu(probe_graph, name):
    if R.canonical(name) in ort.get_available_providers():
        pytest.skip(f"{name} exists on this machine")
    with pytest.raises(ProviderError, match="not available"):
        R.create_runner(probe_graph, name)


def test_a_provider_ort_silently_dropped_is_detected(probe_graph, monkeypatch):
    """Simulate ORT quietly falling back: the session reports CPU although another provider was requested."""
    real = ort.InferenceSession

    class Fallback(real):
        def get_providers(self):
            return ["CPUExecutionProvider"]
    monkeypatch.setattr(ort, "InferenceSession", Fallback)
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    # ORT would raise on the missing CUDA lib itself; either way the caller must see a ProviderError, never a CPU runner
    with pytest.raises(ProviderError):
        R.create_runner(probe_graph, "cuda")


def test_runner_runs_and_coerces_dtype(probe_graph):
    r = R.create_runner(probe_graph, "cpu")
    out = r.run({"x": np.ones((1, 4), np.float64)})        # float64 in -> coerced to the graph's float32
    assert out["y"].dtype == np.float32 and np.allclose(out["y"], 1)
    assert r.active_providers[0] == "cpu" and r.load_seconds > 0


def test_graph_helpers_and_structural_check(probe_graph, tmp_path):
    sig = graph_signature(probe_graph)
    assert [t.name for t in sig.inputs] == ["x"] and sig.outputs[0].dtype == "float32"
    st = graph_stats(probe_graph)
    assert st["ops"] == {"MatMul": 1, "Relu": 1} and st["opset"] == 13
    assert total_size(probe_graph) == probe_graph.stat().st_size
    assert check_graph(probe_graph).ok
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"not an onnx file")
    res = check_graph(bad)
    assert not res.ok and res.errors
    assert not check_graph(tmp_path / "missing.onnx").ok
    assert synth_inputs(sig)["x"].shape == (1, 4)


def test_structural_check_flags_missing_external_data(tmp_path):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    w = numpy_helper.from_array(np.ones((64, 64), np.float32), "w")
    g = helper.make_graph([helper.make_node("MatMul", ["x", "w"], ["y"])], "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 64])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 64])], [w])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    p = tmp_path / "e.onnx"
    onnx.save_model(m, str(p), save_as_external_data=True, all_tensors_to_one_file=True, location="e.onnx.data", size_threshold=0)
    assert check_graph(p).ok and total_size(p) > p.stat().st_size
    (tmp_path / "e.onnx.data").unlink()
    assert not check_graph(p).ok
