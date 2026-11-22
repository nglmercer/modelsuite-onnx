# modelsuite architecture

`modelsuite` converts, optimises, validates and benchmarks ML models. It is a framework of small
contracts (adapters, exporters, a manifest) plus a resumable pipeline. It does **not** claim to
convert every checkpoint; it makes conversion *measurable* and extends to new architectures by
adding an adapter, not by editing the core.

```
src/modelsuite/
  core/          types, config, registry, discovery, licenses, manifest, golden store, plan contract, pipeline + steps, reports
  adapters/      Adapter contract + torch-module, hf-transformers, reverb-tts (plugins live outside this tree)
  exporters/     Exporter contract + torch-legacy, torch-dynamo, optimum (subprocess), ort-genai (subprocess)
  optimization/  passes (graph opt, fp16, weight-cast, int8 dynamic/static, int4), variant comparison / recommendation
  validation/    structural, metrics, parity (+ dynamic shapes), stateful, provider compatibility
  benchmarking/  runner (+ isolated worker), memory sampling, comparison
  runtimes/      ONNX Runtime sessions, provider probing, ONNX file helpers
  cli/           `modelsuite ...`
```

**Independence rules (enforced by tests)**

* `core/` imports no model architecture, no torch, no transformers, no onnx at import time. A custom adapter +
  custom exporter that use only `onnx.helper` run the whole pipeline (`tests/test_extensibility.py`).
* Optional dependencies are imported inside the function that needs them (`core/deps.require` raises a
  `DependencyMissingError` that names the pip extra). Tools with conflicting pins (Optimum, ORT GenAI) run in a **subprocess
  against a configurable interpreter** (`MODELSUITE_OPTIMUM_PYTHON`, `MODELSUITE_ORTGENAI_PYTHON`), because
  `optimum-onnx` pins `transformers<4.58` while the Reverb stack needs transformers 5.x.
* Registries bootstrap their built-ins on first lookup, so library use and CLI use see the same adapters/exporters.

## The pipeline

```
discover -> inspect -> analyze -> export -> optimize -> validate -> benchmark -> package
```

| Step | What it does | Resume behaviour |
|---|---|---|
| discover | resolve a local path / Hub id / `file.py:factory`; pick the adapter (explicit, else highest `detect()` confidence, ties by `priority`) | always re-run (cheap) |
| inspect | architecture, size (from safetensors headers: no weights loaded), components, license | always re-run |
| analyze | compatibility report: support level, issues with hints, exporter availability, RAM/size estimates. `unsupported` stops `convert` and the report is kept | always re-run |
| export | per component: try exporters in order; with `exporter=auto` and several candidates, the first whose output passes a **parity smoke test** wins (a strategy that exports but is numerically wrong is rejected, and that is recorded). Golden reference cases are captured from the original while its module is loaded. Assets are copied | per component: skipped when the graph exists and its size matches the manifest |
| optimize | build variants from the fp32 graphs (below). A component whose conversion fails keeps the parent graph and is listed in `failed_components`; a variant where *every* component failed is not created | per variant |
| validate | structural check, parity incl. dynamic-shape cases, stateful cases, provider compatibility, adapter quality checks, for every variant | per variant |
| benchmark | load / first inference / mean / p50-p99 / throughput / peak RAM(/VRAM) per variant × component × provider, then a recommendation | per variant |
| package | environment snapshot, integrity re-hash of every file, human summary | always |

State: `job/state.json` (timings, status) and the manifest (results). Re-running the identical command continues
where it stopped; a different configuration in the same output directory is refused (config fingerprint), not merged.

## Artifact layout

```
<out>/manifest.json            the schema in core/manifest.py (schema_version 1)
<out>/environment.json         python/platform + every tracked package version
<out>/models/<variant>/<component>.onnx[.data]
<out>/assets/                  tokenizer / config / adapter-declared files (+ descriptions of things that live in the host)
<out>/reference/<component>/   golden inputs + OUTPUTS OF THE ORIGINAL MODEL (size-capped, export.golden_max_mb)
<out>/reports/                 inspect/analyze/export/optimize/validation/benchmark .json + summary.txt
<out>/job/                     resumable state; safe to delete after a successful run
```

The manifest records: original model + revision (`commit` for Hub models, `git` HEAD or a content `fingerprint` for
local ones), framework/architecture, adapter name+version, exporter versions, component signatures (names, dtypes,
symbolic shapes), state links, topology, per-variant precision / quantisation / files with SHA-256 / sizes /
notes / failed components, assets, runtime requirements, validation and benchmark results, model + dependency
licenses, a redistribution verdict, the environment, and the full job configuration (so a conversion can be replayed).

**Golden references make an artifact self-verifying:** `validate <artifact>` and `optimize <artifact>` work years later without the
original weights. When the original *is* supplied (`validate <model> <artifact>`), the live model is authoritative and the
goldens are not trusted over it.

## Variants and optimisation

| Variant name | Pass | Notes |
|---|---|---|
| `fp32` | the export | base; golden references come from here |
| `opt-<level>` | ORT offline graph optimisation (`basic`/`extended`/`all`) | `all` adds hardware-specific layout nodes: not portable |
| `fp16` | full fp16 graph | tries ORT's converter, then onnxconverter-common; **keeps the first output that passes the strict ONNX checker** |
| `fp16-weights` / `bf16-weights` | weights stored in 16 bit, `Cast` back to fp32 | file size ~½; ORT folds the casts at load so **runtime RAM does not shrink**; runs on any provider |
| `int8-dynamic` / `int8-static` | ORT quantisation (static uses calibration `.npz` or the golden inputs) | |
| `int4` | MatMulNBits weight-only; constant-weight `Gemm` is first rewritten to `MatMul+Add` | errors if there is nothing to quantise |

Every variant is validated with a documented per-precision tolerance (`ValidationConfig.precision_tolerance`: fp16 / bf16 / int8 / int4, overridable
per component or in the config). The strict fp32 bounds are never silently loosened for fp32 and the loose ones are never applied to fp32.
`optimization.compare.recommend` only considers variants that **passed validation**, abstains without benchmark data, and warns when a smaller variant is slower.

## Validation

* **structural**: `onnx.checker` (path based, handles external data) with full shape inference; missing external-data files; custom-op domains.
* **parity** (`validation/metrics.py`): max-abs against `atol + rtol*scale`, cosine similarity, non-finite check (and that infinities sit in the same places),
  energy ratio (catches *near-silent* or saturated outputs even when absolute error is tiny), exact-equal fraction for discrete outputs, argmax agreement for logits.
* **dynamic shapes**: adapters declare `ReferenceCase(kind="dynamic_shape")` with different batch/length/past values (and the check that axes declared dynamic really are).
* **stateful** (`validation/stateful.py`): `StateLink(output → input, mode=replace|concat)`; the graph is stepped with its own state; each step is compared with the original
  (lock-step) and a free-running trajectory is compared too, so accumulated drift is reported.
* **providers** (`validation/providers.py`): the same inputs on each requested provider vs CPU. A provider that loads but returns different numbers is `failed`, with the known issue noted.
* **quality checks**: adapter-supplied task-level tests (`kv-cache-consistency`, `greedy-generation`, Reverb's `sampler-code-agreement`).

## Execution providers

`runtimes/ort.probe_providers()` lists what ORT reports **and runs a probe graph on each**: a CUDA build without CUDA libraries is "listed" but not "usable".
Sessions are pinned to the requested provider and verified with `get_providers()`; a drop to CPU raises `ProviderError`
(`allow_partial_fallback=False` additionally sets `session.disable_cpu_ep_fallback`). Benchmarks of an unavailable provider are reported as errors with a non-zero exit code, never as CPU numbers.
CPU is the only provider exercised in this repository's CI environment; the others are implemented through the same strict path but **untested here** (see `limitations.md`).
