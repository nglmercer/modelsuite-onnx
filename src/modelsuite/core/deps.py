"""Optional dependency handling. Nothing here imports a heavy package at module import time."""
from __future__ import annotations

import importlib
import importlib.metadata as md
import importlib.util
import platform
import sys
from typing import Any

from .errors import DependencyMissingError

#: distribution names recorded in every manifest (a package that is absent is recorded as null).
TRACKED_DISTRIBUTIONS = (
    "modelsuite", "numpy", "pydantic", "onnx", "onnxruntime", "onnxruntime-gpu", "onnxruntime-directml",
    "onnxruntime-openvino", "onnxruntime-genai", "onnxscript", "onnxconverter-common", "torch", "torchaudio",
    "transformers", "huggingface_hub", "safetensors", "tokenizers", "optimum", "optimum-onnx", "reverb-tts",
    "faster-whisper", "scipy", "soundfile",
)


def have(module: str) -> bool:
    """True when the module can be imported (without importing it)."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def dist_version(distribution: str) -> str | None:
    try:
        return md.version(distribution)
    except md.PackageNotFoundError:
        return None


def require(module: str, purpose: str = "", extra: str | None = None, package: str | None = None) -> Any:
    """Import and return `module`, or raise a DependencyMissingError that tells the user what to install."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        # A broken install of a present package should not be reported as "missing".
        if have(module.split(".")[0]):
            raise DependencyMissingError(package or module, f"{purpose} (import failed: {exc})", extra) from exc
        raise DependencyMissingError(package or module.split(".")[0], purpose, extra) from exc


def environment_snapshot() -> dict[str, Any]:
    """Interpreter, platform and the versions of every tracked package, for the manifest."""
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {d: dist_version(d) for d in TRACKED_DISTRIBUTIONS if dist_version(d) is not None},
    }
