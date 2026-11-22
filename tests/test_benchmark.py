"""Benchmark runner: statistics, memory, process isolation and honest failure for missing providers."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("onnxruntime")
from modelsuite.benchmarking.compare import pipeline_latency_ms
from modelsuite.benchmarking.runner import benchmark_graph
from modelsuite.core.config import BenchmarkConfig
from modelsuite.runtimes import ort as R


@pytest.fixture(scope="module")
def graph(tmp_path_factory):
    p = tmp_path_factory.mktemp("b") / "p.onnx"
    p.write_bytes(R._probe_model_bytes())
    return p


X = {"x": np.ones((1, 4), np.float32)}


@pytest.mark.parametrize("isolate", [False, True])
def test_statistics_are_complete_and_consistent(graph, isolate):
    r = benchmark_graph(graph, "cpu", X, BenchmarkConfig(warmup=1, iterations=20, isolate_process=isolate), "c", "fp32", "unit")
    assert r.status == "ok" and r.iterations == 20 and r.isolated_process is isolate
    assert r.min_ms <= r.p50_ms <= r.p90_ms <= r.p95_ms <= r.p99_ms <= r.max_ms
    assert r.mean_ms > 0 and r.first_inference_ms > 0 and r.load_seconds > 0
    assert r.throughput_per_s == pytest.approx(1000 / r.mean_ms, rel=0.01) and r.items_per_s == pytest.approx(r.throughput_per_s * r.batch_size, rel=0.01)
    assert r.peak_rss_mb > 10 and r.model_bytes == graph.stat().st_size and r.active_providers[0] == "cpu"
    assert "VRAM" in r.note                                  # honest about what was not measured on CPU


def test_missing_provider_is_an_error_result_never_a_cpu_measurement(graph):
    import onnxruntime as ort
    if "CUDAExecutionProvider" in ort.get_available_providers():
        pytest.skip("CUDA present")
    for isolate in (False, True):
        r = benchmark_graph(graph, "cuda", X, BenchmarkConfig(isolate_process=isolate), "c", "fp32")
        assert r.status == "error" and "not available" in r.error and r.mean_ms is None


def test_max_seconds_stops_long_runs_early(graph):
    r = benchmark_graph(graph, "cpu", X, BenchmarkConfig(iterations=1_000_000, max_seconds=0.2, warmup=0, isolate_process=False), "c")
    assert 3 <= r.iterations < 1_000_000 and "max_seconds" in r.note


def test_pipeline_latency_sums_only_successful_components(graph):
    ok = benchmark_graph(graph, "cpu", X, BenchmarkConfig(iterations=3, isolate_process=False), "a")
    bad = benchmark_graph(graph, "cuda", X, BenchmarkConfig(isolate_process=False), "b")
    assert pipeline_latency_ms([ok, bad]) == pytest.approx(ok.mean_ms, abs=1e-3)
    assert pipeline_latency_ms([bad]) is None
