# Limitations, unsupported cases and what is untested

Read this before trusting an artifact. The suite's design goal is that *every* limitation below shows up in a report
(`analyze`, the manifest, or a validation result) instead of being discovered in production.

## Scope: what "supported" means

A model is **supported** only when a real export + numerical parity run has passed. `analyze` predicts feasibility from config and
metadata; `convert` is what proves it. There is no promise to convert an arbitrary checkpoint.

| Area | Status |
|---|---|
| Hugging Face encoder / head models on **text** (feature extraction, sequence/token classification, masked LM, QA) and **images** (classification) | works; tested on tiny BERT-class checkpoints |
| Decoder-only causal LMs with a **uniform KV cache** (`llama, mistral, qwen2, qwen3, phi3, gemma, olmo, olmo2, granite, smollm3` model types) | works as one prefill+decode graph; tested on a tiny Llama (stateful decode, cache consistency, greedy generation vs `generate()`). **Other families are not verified** |
| Other causal LMs (GPT-2, sliding-window / hybrid layers, MoE) | exported as a **no-KV-cache logits graph** with an explicit warning (`partial`), or via `mode=optimum` |
| Encoder-decoder (T5/BART/Whisper), multiple choice, audio-input HF models (wav2vec2, CTC) | **reported unsupported** by the generic adapter; route to `mode=optimum` or write an adapter |
| Checkpoints already quantised (GPTQ/AWQ/bnb) | **rejected**: the torch exporters cannot export packed weights. Convert the full-precision original, then `optimize` |
| `trust_remote_code` models | rejected unless `export.trust_remote_code=true`; the code is then executed |
| Diffusion pipelines, TensorFlow / JAX models, GGUF output | **not implemented** (roadmap phase 7) |
| ONNX Runtime GenAI | wrapper implemented (`ort-genai` exporter, subprocess); **not exercised here** (`onnxruntime-genai` has no wheel for this environment) |
| Microsoft Olive | **not integrated** |

## Numerical caveats

* Tolerances for lower precisions are **policy defaults** (`ValidationConfig.precision_tolerance`), not guarantees. Cosine similarity is the primary criterion. Tighten them for your task.
* Parity covers the *cases the adapter supplies*. A graph can match on those and still differ elsewhere (rope scaling beyond tested lengths, unusual masks). The generic adapter tests batch/length/padding/past-length variations only.
* `int8-dynamic` on tiny, randomly initialised models loses a lot (cosine 0.95-0.96; on the Reverb fixture the official sampler agrees on only ~43 % of codes). That is the model, not a bug; real checkpoints behave differently. **Always read the validation numbers of a quantised variant.**
* `fp16-weights` / `bf16-weights` are *storage* compression: ORT folds the Casts at load, so RAM does not shrink. Full `fp16` shrinks compute too but many CPU kernels run fp16 by internal casting, so it can be **slower** on CPU (the recommendation step flags this when measured).
* `graph_optimization=all` writes hardware-specific layout nodes; the result may not load on other CPUs.
* Exported output signatures may carry odd symbolic names for dimensions the exporter could not resolve (e.g. `Concatnew_k_dim_4` for a head dimension that is really 16). They are annotations, not constraints.
* `golden` references are capped (`export.golden_max_mb`, default 96 MiB per component); cases that do not fit are not bundled and are then only checked when the original model is supplied.
* int8-static calibration defaults to the golden test inputs, which is a *small, synthetic* set: supply `--calibration-dir` with representative data for real use.

## Operators and exporters

* torch-legacy bakes in Python control flow and any shape not marked dynamic; torch-dynamo needs graph-capturable code. When `exporter=auto` and the first strategy produces a numerically wrong graph, the next is tried; if none passes, the first export is kept and validation reports how it differs.
* Custom operators: provide `export_context` (symbolic registration) and `custom_op_libraries` (ORT custom-op libraries). The structural check lists non-standard domains as warnings. **No custom-op model was converted in this repository.**
* Models whose fp32 weights exceed ~1.5 GiB are stored with external data (`.onnx.data`); the 2 GiB protobuf limit is handled for export but **no multi-GB model was exported here** (largest tested: Reverb `core`, ~1 GiB).
* Optimum and ORT GenAI pin their own `transformers`; they only run through a separate interpreter (`MODELSUITE_OPTIMUM_PYTHON`). Optimum-exported causal LMs are validated on **prefill with an empty cache only**.

## Execution providers: what is and is not verified

Provider handling is implemented once (strict session creation, post-creation provider check, parity vs CPU, no silent CPU fallback) and exercised **only on CPU**
in this repository: the machine it was built on has ONNX Runtime with `CPUExecutionProvider` and `AzureExecutionProvider` only.

* CUDA, TensorRT, DirectML, OpenVINO, CoreML, ROCm, WebGPU: the code paths are covered by unit tests with simulated fallbacks and by the "unavailable provider ⇒ error, never CPU numbers" tests, **not by real runs on those devices**.
* VRAM is measured only through `nvidia-smi` for CUDA/TensorRT; other providers report `null` and say so.
* Peak RSS is accurate only with process isolation (default); `--no-isolate` includes earlier work in the same process.
* The Reverb codec decoder is known to emit near-silence on AMD RADV GFX1013 through ORT's WebGPU EP (original study). The adapter records that as a provider note and the provider-parity check detects the symptom (energy ratio); it was **not re-measured on that hardware here**.

## Code execution

`--plugin`, `MODELSUITE_PLUGINS`, `file.py:factory` model specs, job-config `plugins:` entries and `export.trust_remote_code=true` all **execute Python from the
named source with your privileges** (so does `modelsuite inspect` on a python spec, because the factory builds the model). Only point them at code you have read.
The suite never executes code found inside a *downloaded model repository* unless `trust_remote_code` is set. Hub downloads use safetensors / config files only for analysis.

## ZONOS3-Mini example

Verified on the real checkpoint: the stateful LM graph and the codec decoder match Zyphra's reference (numbers in `examples/zonos3-mini/README.md`). Not verified: an
`fp16-weights` variant (ORT expands it to fp32 on load: out of memory on the 15 GB development machine), int8/int4 on the real weights, any GPU provider, and a full
text-to-audio run (the tokenizer, prompt assembly, sampler and speaker encoder are the host's job). Converting needs ~9-10 GB of RAM at peak.

## Licenses

The suite never uploads or publishes anything. It records the model license, per-component licenses (e.g. the Reverb codec) and
dependency licenses, and computes a conservative redistribution verdict (`allowed` / `conditional` / `no` / `unknown`; the most restrictive component wins).
`unknown` means *do not redistribute the converted files*. License detection is heuristic (model-card front matter, `LICENSE` text signatures) and is not legal advice.

## Not implemented yet (roadmap, see README)

Phase 7-8: ORT GenAI end-to-end, diffusion adapters, TensorFlow/JAX exporters, GGUF, Olive workflows, a graphical model explorer / comparison UI.
Supertonic and Haru adapters (phase 6) are not written; Reverb is the only specialised adapter.
