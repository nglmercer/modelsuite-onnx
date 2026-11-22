# Writing an adapter

An adapter owns everything architecture-specific. You never edit `modelsuite/core`.

## 1. The smallest useful adapter: a PyTorch module from a factory

```python
# my_model.py
import numpy as np, torch
from modelsuite.adapters.torch_module import TorchModuleSpec

def build():
    net = torch.nn.Sequential(torch.nn.Linear(16, 8), torch.nn.ReLU())
    ex = lambda b=4, seed=0: {"x": np.random.default_rng(seed).standard_normal((b, 16)).astype(np.float32)}
    return TorchModuleSpec(module=net, example_inputs=ex, output_names=["y"], name="mlp", license="mit",
                           dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}},
                           shape_variants={"batch1": lambda: ex(1, 1), "batch9": lambda: ex(9, 2)})
```

```sh
modelsuite convert my_model.py:build -o out --precision fp16 --quantize int8-dynamic --benchmark
```

Export, golden references, parity on all three shapes, per-variant validation, benchmark and manifest follow. `build()` is user code and is
executed (also by `inspect`).

## 2. A full adapter (plugin file)

```python
from modelsuite.adapters.base import Adapter, AdapterOptions
from modelsuite.core.registry import register_adapter
from modelsuite.core.plan import ComponentExport, ExportPlan, ReferenceCase, StatefulCase, StateLink, AssetSpec, QualityCheck, Edge
from modelsuite.core.types import Detection, ModelInfo

@register_adapter
class MyAdapter(Adapter):
    name, version, priority = "my-arch", "1.0.0", 20       # higher priority wins ties; generic adapters use 5-10

    def detect(self, source):            # cheap, no weights: look at file names / config.json
        ...  # -> Detection(confidence=0..1, reason="...")
    def inspect(self, source, options):  # -> ModelInfo(architecture, parameters, components=[...], license=...)
        ...
    def plan(self, source, options, config):   # -> ExportPlan; every heavy thing is behind a callable
        ...
    # optional
    def compatibility(self, source, info, options, config): ...    # unsupported ops, RAM, licences -> CompatibilityReport
    def weight_patterns(self, source, options): ...                  # Hub files to download (repos that hold several variants)
```

Load it without touching the suite: `modelsuite convert <model> --plugin my_adapter.py`, `MODELSUITE_PLUGINS=a.py:b.py`, the `plugins:` key of a job
config, or the `modelsuite.adapters` / `modelsuite.exporters` entry-point groups of an installed package.

### What a `ComponentExport` declares

| Field | Meaning |
|---|---|
| `build()` | returns the framework object (torch module); called lazily, memoised, `release()`d between components |
| `example_inputs()`, `input_names`, `output_names`, `dynamic_axes` | the tracing example and the signature contract (checked against the exported graph) |
| `cases` | `ReferenceCase(name, inputs, kind="parity"\|"dynamic_shape")`; inputs must be deterministic and *valid* (indices in range, masks that exercise padding) |
| `reference` | runs the **original** model; default is `build()` called on the inputs in eval mode |
| `stateful` | `StatefulCase(links=[StateLink(out, in, "replace"\|"concat")], steps, initial, step_inputs)` for KV caches / recurrent state |
| `tolerance`, `precision_tolerance`, `output_kinds` | bounds; `"discrete"` = exact match, `"logits"` = also report argmax agreement, `"audio"` = energy check |
| `exporters` / `export_options` / `export_context` | exporter preference, extra exporter arguments, a context manager (register custom-op symbolics, patch an op) |
| `provider_notes` | provider → why the graph must not run there (reported in the manifest and by `validate`) |
| `quantize_exclude_ops`, `known_issues` | recorded in the manifest and honoured by `optimize` |

`ExportPlan` adds `topology` (`Edge`s between components), `assets` (tokenizer / config / voices, or `source=None` to *document* something the host must supply,
e.g. a sampler), `quality_checks`, `runtime_requirements`, `custom_op_libraries` (ORT custom-op `.so` paths registered on every session) and `component_licenses`.

### Custom exporters

Subclass `Exporter` (`capabilities()`, `availability()`, `export()`), decorate with `@register_exporter`, and name it in `ComponentExport.exporters`. An exporter that
only needs `onnx.helper` is a few lines (see the test plugin in `tests/test_extensibility.py`).

## 3. Test it like the built-in ones

1. Build a **tiny, randomly initialised real checkpoint** in the architecture's real layout (see `tests/reverb_fixture.py`): fast, offline, exercises the real loader.
2. Convert it, then assert on the manifest: signatures, parity cases present, `validation.status == "passed"`.
3. Add a test that makes validation *fail* (damage a weight, break a state link). A validator that cannot fail is worthless.
4. Mark anything that needs a large download `@pytest.mark.slow` or `network`.

## Gotchas learnt the hard way (each has a regression test)

* `torch.onnx.export` restores the wrapper's *original* training flag recursively → dropout silently comes back in the reference run. The suite calls `.eval()` after every export and before every reference run; do the same in your own `reference`.
* A trace with an all-ones attention mask can bake in the "no mask" path. Put padding in your example/cases.
* `nn.Linear` on 2-D input exports as `Gemm`, which ORT's int4 quantiser ignores (the suite rewrites constant-weight `Gemm` first).
* Exporters that "succeed" can still be numerically wrong: keep ≥2 strategies in `exporters` and let the parity smoke test choose.
