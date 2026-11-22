"""Golden references: inputs and outputs of the ORIGINAL model, bundled with the artifact.

They make an artifact self-verifying: `optimize`, `validate` and `benchmark` can compare any variant with what the
original produced, long after (and without) the original weights. bfloat16 arrays are stored as float32."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .manifest import Artifact

_INDEX = "index.json"


def _plain(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float32) if a.dtype.name == "bfloat16" else np.ascontiguousarray(a)


class GoldenStore:
    def __init__(self, artifact: Artifact):
        self.dir = artifact.reference_dir

    def _index(self) -> dict:
        f = self.dir / _INDEX
        return json.loads(f.read_text()) if f.is_file() else {}

    def _write_index(self, idx: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / _INDEX).write_text(json.dumps(idx, indent=1))

    def path(self, component: str, case: str) -> Path:
        return self.dir / component / f"{case}.npz"

    def save(self, component: str, case: str, kind: str, inputs: dict[str, np.ndarray], outputs: dict[str, np.ndarray]) -> int:
        p = self.path(component, case)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, **{f"in::{k}": _plain(v) for k, v in inputs.items()}, **{f"out::{k}": _plain(v) for k, v in outputs.items()})
        idx = self._index()
        idx.setdefault(component, {})[case] = {"kind": kind, "bytes": p.stat().st_size}
        self._write_index(idx)
        return p.stat().st_size

    def cases(self, component: str) -> dict[str, dict]:
        return self._index().get(component, {})

    def has(self, component: str, case: str) -> bool:
        return case in self.cases(component) and self.path(component, case).is_file()

    def load(self, component: str, case: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        with np.load(self.path(component, case)) as z:
            ins = {k[4:]: z[k] for k in z.files if k.startswith("in::")}
            outs = {k[5:]: z[k] for k in z.files if k.startswith("out::")}
        return ins, outs

    def bytes(self, component: str) -> int:
        return sum(v["bytes"] for v in self.cases(component).values())

    def first_inputs(self, component: str) -> tuple[str, dict[str, np.ndarray]] | None:
        """Inputs of the primary (first 'parity') case: used for benchmarking and calibration."""
        cases = self.cases(component)
        for name, meta in cases.items():
            if meta.get("kind") == "parity":
                return name, self.load(component, name)[0]
        for name in cases:
            return name, self.load(component, name)[0]
        return None

    def all_inputs(self, component: str) -> list[dict[str, np.ndarray]]:
        return [self.load(component, n)[0] for n in self.cases(component)]
