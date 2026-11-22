"""Model discovery: turn a user-supplied reference (local path or Hugging Face id) into a ModelSource.

`inspect` and `analyze` only need metadata (config, model card, tokenizer files, safetensors headers), so
discovery downloads *metadata only* by default and `fetch_weights()` is called explicitly before an export."""
from __future__ import annotations

import hashlib
import json
import os
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import deps
from .errors import DiscoveryError
from .licenses import LicenseInfo, detect_model_license

METADATA_PATTERNS = ["*.json", "*.md", "*.txt", "LICENSE*", "NOTICE*", "*.model", "*.tiktoken", "*.py"]
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".onnx", ".gguf")
_SAFETENSORS_DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
                            "F8_E4M3": 1, "F8_E5M2": 1, "U16": 2, "U32": 4, "U64": 8}
_HF_ID = re.compile(r"^[\w.\-]+/[\w.\-]+$")


@dataclass
class ModelSource:
    reference: str                       # what the user typed
    kind: str                            # "local-dir" | "local-file" | "huggingface"
    path: Path                           # local directory (or the file itself for local-file)
    repo_id: str | None = None
    revision: str | None = None          # commit sha for hub models; git HEAD / content fingerprint for local ones
    revision_kind: str = ""              # "commit" | "git" | "fingerprint"
    offline: bool = False
    weights_complete: bool = True        # False while only metadata has been fetched
    entry: str | None = None             # python specs only: the factory name in `path` (file.py:factory)
    _config: dict[str, Any] | None = field(default=None, repr=False)
    _card: dict[str, Any] | None = field(default=None, repr=False)
    _remote_summary: dict[str, Any] | None = field(default=None, repr=False)

    # -- files ---------------------------------------------------------------------------
    @property
    def root(self) -> Path:
        return self.path if self.path.is_dir() else self.path.parent

    def files(self) -> list[Path]:
        if self.path.is_file():
            return [self.path]
        return sorted(p for p in self.path.rglob("*") if p.is_file() and ".git" not in p.parts)

    def weight_files(self) -> list[Path]:
        files = [p for p in self.files() if p.suffix in WEIGHT_SUFFIXES]
        if any(p.suffix == ".safetensors" for p in files):  # prefer safetensors when both exist
            files = [p for p in files if p.suffix == ".safetensors"]
        return files

    # -- metadata ---------------------------------------------------------------------------
    @property
    def config(self) -> dict[str, Any]:
        if self._config is None:
            cfg = self.root / "config.json"
            self._config = _read_json(cfg) if cfg.is_file() else {}
        return self._config

    @property
    def model_card(self) -> dict[str, Any]:
        """YAML front matter of README.md (the Hugging Face model-card metadata)."""
        if self._card is None:
            self._card = {}
            readme = self.root / "README.md"
            if readme.is_file():
                m = re.match(r"^---\s*\n(.*?)\n---", readme.read_text(errors="ignore"), re.S)
                if m:
                    try:
                        data = yaml.safe_load(m.group(1))
                        self._card = data if isinstance(data, dict) else {}
                    except yaml.YAMLError:
                        self._card = {}
        return self._card

    def license(self) -> LicenseInfo:
        card = self.model_card.get("license")
        card = card[0] if isinstance(card, list) and card else card
        return detect_model_license(self.root, card if isinstance(card, str) else None, self.config.get("license"))

    def weight_summary(self) -> dict[str, Any]:
        """Parameter count, bytes and dtypes read from safetensors headers (no tensor data is loaded)."""
        if self._remote_summary is not None and not self.weights_complete:
            return self._remote_summary
        params, nbytes, dtypes = 0, 0, set()
        unknown = False
        for f in self.weight_files():
            if f.suffix == ".safetensors" and f.exists():
                try:
                    p, b, d = safetensors_header_summary(f)
                except (OSError, ValueError):
                    unknown = True
                    continue
                params, nbytes = params + p, nbytes + b
                dtypes |= d
            else:
                unknown = unknown or not f.exists()
                if f.exists():
                    nbytes += f.stat().st_size
        total_params = params or None
        return {"parameters": total_params, "bytes": nbytes or None, "dtypes": sorted(dtypes), "partial": unknown}

    # -- weights -----------------------------------------------------------------------------
    def fetch_weights(self, allow_patterns: list[str] | None = None) -> None:
        """Download the weights of a hub model that was resolved metadata-only (all files, or only `allow_patterns`,
        which an adapter supplies when a repo holds several variants). No-op for local models."""
        if self.kind != "huggingface" or self.weights_complete:
            return
        if self.offline:
            raise DiscoveryError(f"{self.reference}: weights are not cached and offline mode is on")
        hub = deps.require("huggingface_hub", "downloading model weights", "hf")
        try:
            kw: dict[str, Any] = {"allow_patterns": [*METADATA_PATTERNS, *allow_patterns]} if allow_patterns else \
                {"ignore_patterns": ["*.msgpack", "*.h5", "*.ot", "*.onnx", "*.onnx_data", "*.tflite", "*.gguf"]}
            local = hub.snapshot_download(self.repo_id, revision=self.revision, **kw)
        except Exception as exc:
            raise DiscoveryError(f"could not download {self.repo_id}@{self.revision}: {exc}") from exc
        self.path = Path(local)
        self.weights_complete = True

    # -- identity ---------------------------------------------------------------------------
    def fingerprint(self) -> str:
        """Cheap content identity: file names + sizes + config.json bytes (weights are not hashed: they can be GBs)."""
        h = hashlib.sha256()
        for f in self.files():
            if f.suffix == ".py":
                continue
            rel = f.relative_to(self.root) if self.path.is_dir() else Path(f.name)
            h.update(f"{rel}:{f.stat().st_size}\n".encode())
        cfg = self.root / "config.json"
        if cfg.is_file():
            h.update(cfg.read_bytes())
        return h.hexdigest()[:16]


def safetensors_header_summary(path: Path) -> tuple[int, int, set[str]]:
    """(parameter count, tensor bytes, dtype names) from a safetensors file header."""
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) < 8:
            raise ValueError("truncated safetensors file")
        (n,) = struct.unpack("<Q", raw)
        if n > 100 * 1024 * 1024:
            raise ValueError("implausible safetensors header")
        header = json.loads(fh.read(n))
    params = nbytes = 0
    dtypes: set[str] = set()
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        count = 1
        for d in meta["shape"]:
            count *= d
        dt = meta["dtype"]
        dtypes.add(dt)
        params += count
        nbytes += count * _SAFETENSORS_DTYPE_BYTES.get(dt, 4)
    return params, nbytes, dtypes


def _read_json(p: Path) -> dict[str, Any]:
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _git_head(path: Path) -> str | None:
    d = path if path.is_dir() else path.parent
    for parent in [d, *d.parents]:
        head = parent / ".git" / "HEAD"
        if head.is_file():
            try:
                ref = head.read_text().strip()
                if ref.startswith("ref:"):
                    rf = parent / ".git" / ref.split(" ", 1)[1]
                    return rf.read_text().strip() if rf.is_file() else None
                return ref
            except OSError:
                return None
    return None


def _file_hash(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


_PY_SPEC = re.compile(r"^(?P<target>.+?)::?(?P<attr>[A-Za-z_]\w*)$")


def parse_python_spec(ref: str) -> tuple[Path, str] | None:
    """'path/to/model.py:factory' or 'package.module:factory' -> (source file, attribute). None if not a spec."""
    m = _PY_SPEC.match(ref)
    if not m or ref.startswith(("http:", "https:")):
        return None
    target, attr = m["target"], m["attr"]
    f = Path(target).expanduser()
    if target.endswith(".py"):
        return (f.resolve(), attr) if f.is_file() else None
    if re.fullmatch(r"[A-Za-z_][\w.]*", target) and not f.exists() and "/" not in ref.split(":")[0]:
        import importlib.util
        try:
            found = importlib.util.find_spec(target)
        except (ImportError, ValueError):
            found = None
        if found and found.origin and found.origin.endswith(".py"):
            return Path(found.origin).resolve(), attr
    return None


def looks_like_hub_id(ref: str) -> bool:
    return bool(_HF_ID.match(ref)) and not Path(ref).exists()


def resolve(reference: str, revision: str | None = None, *, offline: bool = False, fetch_weights: bool = False,
            cache_dir: str | None = None) -> ModelSource:
    """Resolve a local path or a Hugging Face model id to a ModelSource."""
    spec = parse_python_spec(reference)
    if spec is not None:
        file, attr = spec
        return ModelSource(reference, "python-spec", file, revision=_git_head(file) or _file_hash(file),
                           revision_kind="git" if _git_head(file) else "fingerprint", offline=offline, entry=attr)
    path = Path(reference).expanduser()
    if path.exists():
        kind = "local-dir" if path.is_dir() else "local-file"
        rev = _git_head(path)
        src = ModelSource(reference, kind, path.resolve(), revision=rev, revision_kind="git" if rev else "", offline=offline)
        if not rev:
            src.revision, src.revision_kind = src.fingerprint(), "fingerprint"
        return src

    if not looks_like_hub_id(reference):
        raise DiscoveryError(f"'{reference}' is neither an existing path nor a Hugging Face model id (org/name)")

    hub = deps.require("huggingface_hub", "resolving Hugging Face models", "hf")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    try:
        if offline:
            local = hub.snapshot_download(reference, revision=revision, local_files_only=True, cache_dir=cache_dir)
            sha = Path(local).name
            complete = True
        else:
            info = hub.HfApi().model_info(reference, revision=revision)
            sha = info.sha
            local = hub.snapshot_download(reference, revision=sha, allow_patterns=METADATA_PATTERNS, cache_dir=cache_dir)
            complete = False
    except Exception as exc:
        raise DiscoveryError(f"cannot resolve Hugging Face model '{reference}': {exc}") from exc
    src = ModelSource(reference, "huggingface", Path(local), repo_id=reference, revision=sha, revision_kind="commit",
                      offline=offline, weights_complete=complete)
    if not complete:
        src._remote_summary = _remote_weight_summary(hub, reference, sha)
    if fetch_weights:
        src.fetch_weights()
    return src


def _remote_weight_summary(hub: Any, repo_id: str, sha: str) -> dict[str, Any] | None:
    """Parameter count / bytes / dtypes of a hub repo from its safetensors headers, without downloading weights."""
    try:
        meta = hub.get_safetensors_metadata(repo_id, revision=sha)
    except Exception:
        return None
    sizes = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
    by_dtype = meta.parameter_count
    return {"parameters": sum(by_dtype.values()) or None,
            "bytes": sum(n * sizes.get(dt, 4) for dt, n in by_dtype.items()) or None,
            "dtypes": sorted(by_dtype), "partial": False}
