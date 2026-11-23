"""Makes Zyphra's reference code importable *without installing it*.

Why not `pip install zonos3`? It pins `transformers<=4.57.3`, `torchaudio==2.9.1` and Python >= 3.12, which would downgrade this
environment (the same trap as Optimum). The reference LM only needs torch/numpy, and the codec decoder only needs two tiny pieces of
`descript-audio-codec` (`dac.nn.layers`, `dac.nn.quantize`), whose package `__init__` drags in `audiotools`. So:

* `ZONOS3_SRC` (or `--adapter-option source_dir=...`) names `<ZONOS3-Mini clone>/model/src`; it is put on sys.path;
* the `dac` package is registered as an empty namespace pointing at the installed files, so only the two submodules import;
* `audiotools.ml.BaseModel` and `dac.model.base.CodecMixin` (base classes `modded_dac` subclasses) are stubbed with trivial bases.
  Neither carries parameters or behaviour the decoder uses."""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path

ENV = "ZONOS3_SRC"


class ReferenceUnavailable(RuntimeError):
    pass


def find_source(explicit: str | None = None) -> Path:
    cand = explicit or os.environ.get(ENV)
    if cand:
        p = Path(cand).expanduser()
        for q in (p, p / "model" / "src"):
            if (q / "zonos3").is_dir():
                return q
        raise ReferenceUnavailable(f"{cand} does not contain a 'zonos3' package (expected <clone>/model/src)")
    spec = importlib.util.find_spec("zonos3")
    if spec is not None and spec.submodule_search_locations:
        return Path(list(spec.submodule_search_locations)[0]).parent
    raise ReferenceUnavailable("Zyphra's reference code is needed (it is the numerical oracle and defines the module layout). "
                               f"git clone https://github.com/Zyphra/ZONOS3-Mini and set {ENV}=<clone>/model/src (or pass --adapter-option source_dir=...)")


def ensure_reference(explicit: str | None = None, codec: bool = False) -> Path:
    src = find_source(explicit)
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if codec:
        _shim_dac()
    return src


def _shim_dac() -> None:
    if "dac.nn.layers" in sys.modules and getattr(sys.modules.get("dac"), "_modelsuite_shim", False):
        return
    spec = importlib.util.find_spec("dac")
    if spec is None or not spec.submodule_search_locations:
        raise ReferenceUnavailable("the codec decoder needs `dac.nn.layers` / `dac.nn.quantize`: pip install --no-deps descript-audio-codec==1.0.0 einops")
    root = Path(list(spec.submodule_search_locations)[0])
    dac = types.ModuleType("dac")
    dac.__path__ = [str(root)]
    dac._modelsuite_shim = True
    nn_pkg = types.ModuleType("dac.nn")
    nn_pkg.__path__ = [str(root / "nn")]
    model_pkg = types.ModuleType("dac.model")
    model_pkg.__path__ = []
    base = types.ModuleType("dac.model.base")
    base.CodecMixin = type("CodecMixin", (), {})
    model_pkg.base = base
    import torch
    audiotools = types.ModuleType("audiotools")
    ml = types.ModuleType("audiotools.ml")
    ml.BaseModel = torch.nn.Module
    audiotools.ml = ml
    sys.modules.update({"dac": dac, "dac.nn": nn_pkg, "dac.model": model_pkg, "dac.model.base": base, "audiotools": audiotools, "audiotools.ml": ml})
    importlib.import_module("dac.nn.layers")
    importlib.import_module("dac.nn.quantize")
