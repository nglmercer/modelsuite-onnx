"""Imports (and thereby registers) the built-in adapters. A built-in whose module fails to import is recorded as a load error
and skipped, so one broken optional dependency cannot take the others down."""
from __future__ import annotations

import importlib

from ..core import registry

for _mod in ("torch_module", "hf_transformers", "reverb"):
    try:
        importlib.import_module(f"{__package__}.{_mod}")
    except ImportError as exc:
        registry.adapters._load_errors[_mod] = f"{type(exc).__name__}: {exc}"
