"""The artifact manifest: one JSON file that makes a conversion self-describing and portable.

Layout of an artifact directory:

    manifest.json            this schema
    environment.json         exact package versions the job ran with
    models/<variant>/<component>.onnx[.data]
    assets/                  tokenizer / config / adapter-declared files
    reference/<component>/   golden inputs + outputs of the *original* model (small cases only)
    reports/                 JSON + text reports of every step
    job/                     resumable job state and intermediate files (safe to delete after a successful run)
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import Field

from .errors import ArtifactError
from .licenses import LicenseInfo
from .types import Loose, Signature, Status

SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


class FileEntry(Loose):
    path: str                       # relative to the artifact root, posix separators
    size: int
    sha256: str


class SourceInfo(Loose):
    reference: str
    kind: str
    repo_id: str | None = None
    revision: str | None = None
    revision_kind: str = ""
    framework: str = "unknown"
    architecture: str = "unknown"
    model_type: str | None = None
    parameters: int | None = None
    license: LicenseInfo | None = None


class ToolVersion(Loose):
    name: str
    version: str


class ComponentRecord(Loose):
    name: str
    role: str = ""
    description: str = ""
    signature: Signature
    stateful: bool = False
    state_links: list[dict[str, Any]] = Field(default_factory=list)
    exporter: str | None = None
    strategy: str | None = None
    opset: int | None = None
    known_issues: list[str] = Field(default_factory=list)
    provider_notes: dict[str, str] = Field(default_factory=dict)   # provider -> why it must not be used for this graph
    quantize_exclude_ops: list[str] = Field(default_factory=list)
    output_kinds: dict[str, str] = Field(default_factory=dict)
    tolerance: dict[str, Any] | None = None
    export_attempts: list[dict[str, Any]] = Field(default_factory=list)


class QuantizationRecord(Loose):
    method: str = "none"            # none | int8-dynamic | int8-static | int4
    details: dict[str, Any] = Field(default_factory=dict)


class VariantRecord(Loose):
    name: str                                     # "fp32", "fp16", "fp32-int8-dynamic", ...
    precision: str = "fp32"
    quantization: QuantizationRecord = Field(default_factory=QuantizationRecord)
    graph_optimization: str = "none"
    parent: str | None = None
    files: dict[str, list[FileEntry]] = Field(default_factory=dict)      # component -> files (graph + external data)
    sizes_bytes: dict[str, int] = Field(default_factory=dict)            # component -> total bytes
    notes: list[str] = Field(default_factory=list)
    component_precision: dict[str, str] = Field(default_factory=dict)    # component -> precision actually stored
    failed_components: dict[str, str] = Field(default_factory=dict)      # component -> why it kept the parent's graph
    validation: dict[str, Any] | None = None
    benchmark: dict[str, Any] | None = None
    recommended_for: list[str] = Field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(self.sizes_bytes.values())


class AssetRecord(Loose):
    role: str
    path: str
    required: bool = True
    description: str = ""
    sha256: str | None = None
    license_note: str | None = None


class Redistribution(Loose):
    allowed: bool | None            # True / False, or None when the answer depends on conditions or is unknown
    status: str = "unknown"         # allowed | conditional | no | unknown   (never "allowed" unless the license says so)
    reason: str
    required_notices: list[str] = Field(default_factory=list)


class Manifest(Loose):
    schema_version: int = SCHEMA_VERSION
    created: str = Field(default_factory=utcnow)
    updated: str = Field(default_factory=utcnow)
    suite_version: str
    source: SourceInfo
    adapter: ToolVersion
    adapter_options: dict[str, Any] = Field(default_factory=dict)
    exporters: list[ToolVersion] = Field(default_factory=list)
    target_format: str = "onnx"
    job_config: dict[str, Any] = Field(default_factory=dict)
    job_fingerprint: str = ""
    components: list[ComponentRecord] = Field(default_factory=list)
    topology: list[dict[str, Any]] = Field(default_factory=list)          # edges: {from, output, to, input}
    variants: dict[str, VariantRecord] = Field(default_factory=dict)
    default_variant: str = "fp32"
    assets: list[AssetRecord] = Field(default_factory=list)
    runtime_requirements: list[str] = Field(default_factory=list)
    reference_cases: dict[str, list[str]] = Field(default_factory=dict)   # component -> golden case names bundled
    validation: dict[str, Any] = Field(default_factory=dict)
    benchmark: dict[str, Any] = Field(default_factory=dict)
    licenses: dict[str, Any] = Field(default_factory=dict)
    redistribution: Redistribution | None = None
    environment: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    def component(self, name: str) -> ComponentRecord:
        for c in self.components:
            if c.name == name:
                return c
        raise ArtifactError(f"component '{name}' is not in the manifest (has: {[c.name for c in self.components]})")

    def variant(self, name: str | None = None) -> VariantRecord:
        name = name or self.default_variant
        try:
            return self.variants[name]
        except KeyError:
            raise ArtifactError(f"variant '{name}' is not in the manifest (has: {list(self.variants)})") from None


def collect_files(root: Path, graph: Path) -> list[FileEntry]:
    """FileEntry for a graph and every external-data file stored beside it."""
    files = [graph, *sorted(graph.parent.glob(graph.name + ".data")), *sorted(graph.parent.glob(graph.stem + ".onnx_data"))]
    out, seen = [], set()
    for f in files:
        if f.is_file() and f not in seen:
            seen.add(f)
            out.append(FileEntry(path=f.relative_to(root).as_posix(), size=f.stat().st_size, sha256=sha256_file(f)))
    return out


class Artifact:
    """Helper around an artifact directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_NAME

    def models_dir(self, variant: str) -> Path:
        return self.root / "models" / variant

    def graph_path(self, variant: str, component: str) -> Path:
        return self.models_dir(variant) / f"{component}.onnx"

    @property
    def reference_dir(self) -> Path:
        return self.root / "reference"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def assets_dir(self) -> Path:
        return self.root / "assets"

    @property
    def job_dir(self) -> Path:
        return self.root / "job"

    def exists(self) -> bool:
        return self.manifest_path.is_file()

    def read(self) -> Manifest:
        if not self.manifest_path.is_file():
            raise ArtifactError(f"no {MANIFEST_NAME} in {self.root} (is it an artifact directory?)")
        try:
            data = json.loads(self.manifest_path.read_text())
        except json.JSONDecodeError as exc:
            raise ArtifactError(f"{self.manifest_path} is not valid JSON: {exc}") from exc
        version = data.get("schema_version", 0)
        if version > SCHEMA_VERSION:
            raise ArtifactError(f"manifest schema {version} is newer than this suite understands ({SCHEMA_VERSION}); upgrade modelsuite")
        return Manifest.model_validate(data)

    def write(self, manifest: Manifest) -> None:
        manifest.updated = utcnow()
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=False) + "\n")
        tmp.replace(self.manifest_path)

    def verify(self, variant: str | None = None) -> list[str]:
        """Re-hash every file listed in the manifest; returns human-readable problems (empty = intact)."""
        m = self.read()
        problems = []
        names = [variant] if variant else list(m.variants)
        for vn in names:
            for comp, files in m.variant(vn).files.items():
                for fe in files:
                    p = self.root / fe.path
                    if not p.is_file():
                        problems.append(f"{vn}/{comp}: missing {fe.path}")
                    elif p.stat().st_size != fe.size or sha256_file(p) != fe.sha256:
                        problems.append(f"{vn}/{comp}: {fe.path} does not match its recorded checksum")
        return problems

    def resolve_graph(self, manifest: Manifest, variant: str, component: str) -> Path:
        files = manifest.variant(variant).files.get(component)
        if not files:
            raise ArtifactError(f"variant '{variant}' has no component '{component}'")
        graph = next((f for f in files if f.path.endswith(".onnx")), files[0])
        return self.root / graph.path


def status_of(passed: bool | None) -> Status:
    return Status.skipped if passed is None else (Status.passed if passed else Status.failed)
