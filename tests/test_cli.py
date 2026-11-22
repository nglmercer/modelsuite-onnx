"""Command line behaviour: output formats, exit codes, error handling."""
from __future__ import annotations

import json

import pytest

from modelsuite.cli.main import EXIT_ERROR, EXIT_OK, EXIT_PROBLEM, main


def run(capsys, *argv):
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_providers_json_lists_cpu_as_really_usable(capsys):
    pytest.importorskip("onnxruntime")
    code, out, _ = run(capsys, "providers", "--json")
    data = json.loads(out)
    assert code == EXIT_OK and any(p["short"] == "cpu" and p["usable"] for p in data)


def test_adapters_and_exporters_commands(capsys):
    code, out, _ = run(capsys, "adapters", "--json")
    names = {a["name"] for a in json.loads(out)["adapters"]}
    assert code == EXIT_OK and {"torch-module", "hf-transformers", "reverb-tts"} <= names
    code, out, _ = run(capsys, "exporters", "--json")
    rows = {r["name"]: r for r in json.loads(out)}
    assert {"torch-legacy", "torch-dynamo", "optimum", "ort-genai"} <= set(rows) and rows["optimum"]["capabilities"]["limitations"]


def test_missing_model_is_a_clean_error_not_a_traceback(capsys):
    code, _, err = run(capsys, "inspect", "/definitely/not/here", "-q")
    assert code == EXIT_ERROR and err.startswith("error:") and "Traceback" not in err


def test_bad_adapter_option_syntax(capsys, tiny_bert):
    code, _, err = run(capsys, "inspect", str(tiny_bert), "--adapter-option", "novalue", "-q")
    assert code == EXIT_ERROR and "key=value" in err


def test_inspect_analyze_json_and_unsupported_exit_code(capsys, tiny_bert, tmp_path):
    pytest.importorskip("transformers")
    code, out, _ = run(capsys, "inspect", str(tiny_bert), "--json", "-q")
    d = json.loads(out)
    assert code == EXIT_OK and d["adapter"]["name"] == "hf-transformers" and d["model"]["task"] == "text-classification"
    code, out, _ = run(capsys, "analyze", str(tiny_bert), "--json", "-q")
    assert code == EXIT_OK and json.loads(out)["compatibility"]["support"] == "supported"
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "config.json").write_text(json.dumps({"model_type": "nope_v9", "architectures": ["NopeModel"]}))
    code, out, _ = run(capsys, "analyze", str(bad), "-q")
    assert code == EXIT_PROBLEM and "UNSUPPORTED" in out and "unknown-model-type" in out


def test_convert_validate_benchmark_roundtrip_via_cli(capsys, toy_file, tmp_path):
    pytest.importorskip("torch")
    art = tmp_path / "art"
    code, out, _ = run(capsys, "convert", f"{toy_file}:build", "-o", str(art), "--precision", "fp16-weights", "-q")
    assert code == EXIT_OK and "overall validation: passed" in out and (art / "manifest.json").is_file()
    code, out, _ = run(capsys, "validate", str(art), "-q")                        # golden-only
    assert code == EXIT_OK and "golden-only" in out
    code, out, _ = run(capsys, "validate", f"{toy_file}:build", str(art), "-q")   # live
    assert code == EXIT_OK
    code, out, _ = run(capsys, "benchmark", str(art), "--iterations", "3", "--warmup", "1", "--no-isolate", "-q")
    assert code == EXIT_OK and "per-component results" in out and "mean" in out
    code, out, _ = run(capsys, "optimize", str(art), "--quantize", "int4", "-q")
    assert code in (EXIT_OK, EXIT_PROBLEM) and "int4" in out
    code, _, err = run(capsys, "optimize", str(art), "-q")
    assert code == EXIT_ERROR and "nothing to do" in err
    code, _, err = run(capsys, "benchmark", str(tmp_path / "nothing"), "-q")
    assert code == EXIT_ERROR and "manifest.json" in err


def test_unavailable_provider_fails_the_benchmark_command(capsys, toy_file, tmp_path):
    pytest.importorskip("torch")
    import onnxruntime as ort
    if "CUDAExecutionProvider" in ort.get_available_providers():
        pytest.skip("CUDA present")
    art = tmp_path / "art"
    run(capsys, "convert", f"{toy_file}:build", "-o", str(art), "-q")
    code, out, _ = run(capsys, "benchmark", str(art), "--providers", "cuda", "--iterations", "2", "--no-isolate", "-q")
    assert code == EXIT_PROBLEM and "ERROR" in out and "not available" in out          # reported, with a non-zero exit; never measured on CPU
