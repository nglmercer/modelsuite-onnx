"""torch helpers shared by adapters and exporters: numpy <-> tensor conversion and the default 'original model' runner."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..core import deps
from ..core.plan import Arrays, ComponentExport


def torch_mod():
    return deps.require("torch", "PyTorch models", "torch")


def to_tensor(a: np.ndarray):
    torch = torch_mod()
    if a.dtype.name == "bfloat16":
        return torch.from_numpy(a.astype(np.float32)).to(torch.bfloat16)
    return torch.from_numpy(np.ascontiguousarray(a))


def to_numpy(t) -> np.ndarray:
    torch = torch_mod()
    t = t.detach().cpu()
    if t.dtype == torch.bfloat16:
        ml = deps.require("ml_dtypes", "bfloat16 outputs", package="ml_dtypes")
        return t.float().numpy().astype(ml.bfloat16)
    return t.numpy()


def flatten_outputs(out: Any, names: list[str]) -> Arrays:
    """Normalise a module's return value to {output_name: ndarray}, in `names` order."""
    torch = torch_mod()
    if isinstance(out, dict):
        vals = [out[n] for n in names]
    elif isinstance(out, (tuple, list)):
        vals = list(out)
    else:
        vals = [out]
    if len(vals) != len(names):
        raise ValueError(f"module returned {len(vals)} outputs but {len(names)} output names were declared ({names})")
    return {n: to_numpy(v) for n, v in zip(names, vals) if isinstance(v, torch.Tensor)}


def module_reference(comp: ComponentExport) -> Callable[[Arrays], Arrays]:
    """Default reference runner: call the (eval-mode) torch module on the inputs, in declared input order."""
    def run(inputs: Arrays) -> Arrays:
        torch = torch_mod()
        m = comp.module()
        # torch.onnx.export restores the wrapper's *original* training flag recursively, which can silently turn
        # dropout back on for the inner model. Force eval on every reference run.
        if hasattr(m, "eval"):
            m.eval()
        with torch.inference_mode():
            out = m(*[to_tensor(inputs[n]) for n in comp.input_names])
        return flatten_outputs(out, comp.output_names)
    return run


def reference_runner(comp: ComponentExport) -> Callable[[Arrays], Arrays]:
    return comp.reference or module_reference(comp)


def set_seed(seed: int) -> None:
    torch = torch_mod()
    torch.manual_seed(seed)
    np.random.seed(seed)
