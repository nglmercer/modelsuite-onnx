"""Core: configuration, registry/plugins, discovery, licenses, manifest. No torch needed."""
from __future__ import annotations

import json
import struct
import textwrap

import pytest
from pydantic import ValidationError

from modelsuite.core import licenses, registry
from modelsuite.core.config import JobConfig, Tolerance, merge
from modelsuite.core.discovery import parse_python_spec, resolve, safetensors_header_summary
from modelsuite.core.errors import ArtifactError, ConfigError, DiscoveryError
from modelsuite.core.manifest import Artifact, Manifest, SourceInfo, ToolVersion, VariantRecord, collect_files, sha256_file


# ---- config -------------------------------------------------------------------------------------------------
def test_config_rejects_unknown_keys_and_bad_values():
    with pytest.raises(ValidationError):
        JobConfig(model="x", export={"exportr": "auto"})                 # typo must not pass silently
    with pytest.raises(ValidationError):
        JobConfig(model="  ")
    with pytest.raises(ValidationError):
        JobConfig(model="x", optimization={"precision": ["fp8"]})


def test_fingerprint_ignores_output_but_tracks_behaviour():
    a = JobConfig(model="m", output="a")
    assert a.fingerprint() == JobConfig(model="m", output="b").fingerprint()
    assert a.fingerprint() != JobConfig(model="m", export={"opset": 17}).fingerprint()
    assert a.fingerprint() != JobConfig(model="m", adapter_options={"k": 1}).fingerprint()


def test_merge_is_deep():
    assert merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 9}}) == {"a": {"b": 9, "c": 2}}


# ---- registry / plugins ---------------------------------------------------------------------------------------
def test_registry_duplicate_and_replace():
    r = registry.Registry("thing", "modelsuite.nonexistent")
    r.register("a", 1)
    r.register("a", 1)                                  # same object: idempotent
    with pytest.raises(ConfigError):
        r.register("a", 2)
    r.register("a", 2, replace=True)
    assert r.get("a") == 2 and r.names() == ["a"]
    with pytest.raises(Exception, match="unknown thing 'zzz'"):
        r.get("zzz")


def test_plugin_file_registers_adapter_without_touching_core(tmp_path):
    plug = tmp_path / "my_plugin.py"
    plug.write_text(textwrap.dedent('''
        from modelsuite.adapters.base import Adapter
        from modelsuite.core.registry import register_adapter
        from modelsuite.core.types import Detection, ModelInfo

        @register_adapter
        class DotMyAdapter(Adapter):
            name = "dot-my-test"
            def detect(self, source):
                return Detection(confidence=0.99 if source.path.suffix == ".my" else 0.0, reason="suffix .my")
            def inspect(self, source, options):
                return ModelInfo(reference=source.reference, source_kind=source.kind, framework="custom")
            def plan(self, source, options, config):
                raise NotImplementedError
    '''))
    registry.load_plugin(str(plug))
    registry.load_plugin(str(plug))                      # idempotent
    f = tmp_path / "model.my"
    f.write_text("x")
    from modelsuite.adapters.selection import select_adapter
    sel = select_adapter(resolve(str(f)))
    assert sel.adapter.name == "dot-my-test" and sel.detection.confidence == 0.99
    registry.adapters._items.pop("dot-my-test", None)


def test_broken_plugin_path_is_a_clear_error(tmp_path):
    with pytest.raises(ConfigError, match="plugin file not found"):
        registry.load_plugin(str(tmp_path / "nope.py"))
    with pytest.raises(ConfigError, match="cannot import plugin module"):
        registry.load_plugin("definitely_not_a_module_xyz")


def test_no_adapter_error_lists_what_each_adapter_said(tmp_path):
    from modelsuite.adapters.selection import select_adapter
    from modelsuite.core.errors import NoAdapterError
    d = tmp_path / "mystery"
    d.mkdir()
    (d / "weights.bin").write_bytes(b"\0" * 16)
    with pytest.raises(NoAdapterError) as e:
        select_adapter(resolve(str(d)))
    assert "hf-transformers" in str(e.value) and "--plugin" in str(e.value)


# ---- discovery -------------------------------------------------------------------------------------------------------------------
def test_resolve_local_and_errors(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "bert"}))
    s = resolve(str(d))
    assert s.kind == "local-dir" and s.config["model_type"] == "bert" and s.revision_kind in ("git", "fingerprint")
    assert s.revision == resolve(str(d)).revision        # stable
    (d / "config.json").write_text(json.dumps({"model_type": "other"}))
    assert resolve(str(d)).revision != s.revision or s.revision_kind == "git"
    with pytest.raises(DiscoveryError):
        resolve("not a path or id")


def test_python_spec_parsing(tmp_path):
    f = tmp_path / "x.py"
    f.write_text("def build(): pass\n")
    assert parse_python_spec(f"{f}:build") == (f.resolve(), "build")
    assert parse_python_spec(f"{f}::build") == (f.resolve(), "build")
    assert parse_python_spec(f"{tmp_path}/missing.py:build") is None
    assert parse_python_spec("org/name") is None


def test_safetensors_header_summary_reads_header_only(tmp_path):
    header = {"w": {"dtype": "F16", "shape": [3, 4], "data_offsets": [0, 24]}, "b": {"dtype": "F32", "shape": [5], "data_offsets": [24, 44]},
              "__metadata__": {"format": "pt"}}
    h = json.dumps(header).encode()
    p = tmp_path / "m.safetensors"
    p.write_bytes(struct.pack("<Q", len(h)) + h)          # no tensor bytes at all: only the header may be read
    params, nbytes, dtypes = safetensors_header_summary(p)
    assert (params, nbytes, dtypes) == (17, 3 * 4 * 2 + 5 * 4, {"F16", "F32"})


# ---- licenses -----------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("raw,cat,redis", [("apache-2.0", "permissive", "allowed"), ("CC-BY-NC-4.0", "non-commercial", "conditional"),
                                           ("cc-by-nc-nd-4.0", "non-commercial", "no"), ("llama3", "restricted", "conditional"),
                                           ("some-weird-license", "unknown", "unknown"), (None, "unknown", "unknown")])
def test_license_policy(raw, cat, redis):
    lic = licenses.classify(raw)
    assert (lic.category, lic.redistribution) == (cat, redis)


def test_license_from_model_card_then_file(tmp_path):
    (tmp_path / "LICENSE").write_text("Apache License\nVersion 2.0, January 2004")
    assert licenses.detect_model_license(tmp_path).id == "apache-2.0"
    assert licenses.detect_model_license(tmp_path, "cc-by-nc-4.0").id == "cc-by-nc-4.0"      # card wins over the file
    (tmp_path / "LICENSE").write_text("totally custom terms")
    assert licenses.detect_model_license(tmp_path).id == "custom"
    assert licenses.detect_model_license(None).id is None


def test_model_card_front_matter(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "README.md").write_text("---\nlicense: cc-by-nc-4.0\ntags: [tts]\n---\n# hi")
    assert resolve(str(tmp_path)).license().category == "non-commercial"


# ---- manifest / artifact -----------------------------------------------------------------------------------------------------------------
def _manifest():
    return Manifest(suite_version="t", source=SourceInfo(reference="r", kind="local-dir"), adapter=ToolVersion(name="a", version="1"))


def test_manifest_roundtrip_and_integrity(tmp_path):
    art = Artifact(tmp_path / "art")
    g = art.graph_path("fp32", "c")
    g.parent.mkdir(parents=True)
    g.write_bytes(b"onnx-bytes")
    (g.parent / "c.onnx.data").write_bytes(b"weights")
    m = _manifest()
    m.variants["fp32"] = VariantRecord(name="fp32", files={"c": collect_files(art.root, g)})
    art.write(m)
    assert {f.path for f in art.read().variant("fp32").files["c"]} == {"models/fp32/c.onnx", "models/fp32/c.onnx.data"}   # external data tracked
    assert art.verify() == []
    (g.parent / "c.onnx.data").write_bytes(b"tampered")
    assert any("checksum" in p for p in art.verify())
    g.unlink()
    assert any("missing" in p for p in art.verify())


def test_manifest_rejects_newer_schema_and_garbage(tmp_path):
    art = Artifact(tmp_path)
    with pytest.raises(ArtifactError, match="no manifest.json"):
        art.read()
    art.manifest_path.write_text("{not json")
    with pytest.raises(ArtifactError, match="not valid JSON"):
        art.read()
    art.manifest_path.write_text(json.dumps({**_manifest().model_dump(mode="json"), "schema_version": 99}))
    with pytest.raises(ArtifactError, match="newer"):
        art.read()
