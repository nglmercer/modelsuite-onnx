# modelsuite: model conversion, optimisation, validation and benchmarking

A plugin-based Python framework that **discovers, analyses, converts, optimises, validates and benchmarks** ML models, starting with ONNX.
It supports many architectures through reusable exporters and custom *adapters*; it does **not** promise to convert every checkpoint, and
it tells you, with numbers, when a conversion is wrong. The first reference implementation is [Reverb-TTS](#reverb-tts-to-onnx-the-original-reference-conversion)
(the scripts that used to be this repo's whole content now live behind a Reverb adapter and a regression test).

```sh
python -m venv venv && . venv/bin/activate
pip install -e '.[onnx,torch,hf,reverb]' --extra-index-url https://download.pytorch.org/whl/cpu   # the core alone needs only numpy/pydantic/pyyaml/psutil

modelsuite inspect  hf-internal-testing/tiny-random-LlamaForCausalLM        # what is it? (no weights downloaded)
modelsuite analyze  hf-internal-testing/tiny-random-LlamaForCausalLM        # can it be converted, how, with which caveats?
modelsuite convert  hf-internal-testing/tiny-random-LlamaForCausalLM -o out --precision fp16-weights --quantize int8-dynamic --benchmark
modelsuite validate out                      # re-check any artifact against its bundled golden references
modelsuite validate ./my_checkpoint out      # ...or against the live original model (also runs stateful + quality checks)
modelsuite optimize out --precision fp16     # add variants to an existing artifact
modelsuite benchmark out --providers cpu,cuda
modelsuite providers                         # which execution providers *actually run* here
modelsuite adapters ; modelsuite exporters
```

`<model>` is a local directory, a Hugging Face id (`org/name`, optional `--revision`) or a Python spec (`file.py:factory`) for a custom PyTorch module.
Add `--json` for machine-readable output; every step also writes JSON reports into the artifact. Exit codes: `0` ok, `1` error, `2` completed but validation failed / model unsupported.

### What you get

* **A manifest** (`manifest.json`) with the source model and revision, adapter/exporter versions, component graphs and signatures, precision and quantisation per variant, assets,
  runtime requirements, validation and benchmark results, model + dependency licenses and a redistribution verdict, plus the exact environment and job configuration.
  Files are checksummed; multi-graph models, external ONNX weights and tokenizer/config assets are first-class.
* **Validation as a first-class feature:** structural ONNX checks, original-vs-exported numerical parity (including *dynamic-shape* cases and padding masks),
  step-by-step **stateful** tests (KV caches, recurrent state), per-provider agreement with CPU, and task-level quality checks. Golden references of the *original* model travel with the artifact.
* **Measured optimisation:** fp16 / bf16-weights / int8 dynamic+static / int4 / graph optimisation, each validated and benchmarked; the recommendation only considers variants that passed, and says when a smaller variant is slower.
* **Honest providers:** a provider counts only if a probe graph ran on it; a session that silently falls back to CPU is an error, not a benchmark.
* **Resumable, reproducible jobs:** re-run the same command to continue; a different configuration in the same output directory is refused.

### Verified in this repository (CPU, Python 3.14, torch 2.15, onnxruntime 1.32)

| Case | Result |
|---|---|
| Tiny BERT classifier (HF) | exported, parity incl. padded/dynamic shapes, fp16 variant passes |
| Tiny Llama (HF, local and from the Hub) | one prefill+decode graph with KV cache; 8-step stateful decode, cache consistency and greedy generation == `transformers.generate()` |
| Custom PyTorch module (`file.py:factory`) | full pipeline; the training-mode hazard of `torch.onnx.export` is guarded |
| Custom adapter **and** custom exporter from a plugin file, no torch | full pipeline |
| **Reverb-TTS, real 297M checkpoint + Higgs codec** (`pytest -m slow`) | all 6 graphs exported; official sampler through ONNX reproduces **100 %** of the codes at all 8 codebooks; codec decoder diff 6e-7, encoder codes 100 % identical; sizes match the table below (see [`docs/reverb.md`](docs/reverb.md)) |
| **ZONOS3-Mini** adapter, a plugin in [`examples/zonos3-mini`](examples/zonos3-mini) (hybrid GDN/attention + MoE LM) | stateful LM graph with ONNX `Loop`s; real 800M weights: logits within 4e-5 of Zyphra's reference, 100 % argmax agreement; full pipeline on a tiny release. **Real codec weights and the full real-weights pipeline were not verified** (out of memory on the dev machine); see its README |
| **Reverb-TTS** adapter on a tiny fixture (every test run) | official sampler through the ONNX graphs reproduces **100 %** of the original codes; prefix→target KV hand-off passes; dynamic-int8 correctly **fails** (43 % agreement) |
| Optimum exporter (separate interpreter) | exported + validated against the original (prefill) |

`pytest` runs 108 offline tests in ~25 s; `pytest -m network` adds the Hub tests. What is **not** verified (non-CPU providers, other LLM families, multi-GB models, ORT GenAI, Olive) is listed in
[`docs/limitations.md`](docs/limitations.md); the roadmap items not yet built are at the end of that file.

### Documentation

* [`docs/architecture.md`](docs/architecture.md): modules, pipeline, artifact + manifest, validation, providers
* [`docs/adapters.md`](docs/adapters.md): writing an adapter / exporter, plugin loading, how to test one, gotchas
* [`docs/reverb.md`](docs/reverb.md): how the original scripts map onto the adapter, regression results
* [`docs/limitations.md`](docs/limitations.md): unsupported cases, numerical caveats, what is untested

### Roadmap

| Phase | Status |
|---|---|
| 1 Foundation: interfaces, registry, discovery, config, manifests | done |
| 2 Exporters: torch (legacy + dynamo), Optimum, generic Transformers | done (Optimum via a separate interpreter) |
| 3 Validation: structural, runtime, parity, dynamic shapes, stateful, reports | done |
| 4 Optimisation: precision, quantisation, graph optimisation, recommendation | done |
| 5 Benchmarking: provider discovery, latency/throughput/memory | done on CPU; other providers implemented, untested here |
| 6 Specialised adapters | Reverb done; Supertonic, Haru **not started** |
| 7 Extended ecosystem: ORT GenAI, diffusion, TF/JAX, GGUF | ORT GenAI wrapper only; rest not started |
| 8 User interface | not started |

---

## Reverb-TTS to ONNX: the original reference conversion

Scripts, checks and notes to convert [software-mansion/Reverb-TTS](https://huggingface.co/software-mansion/Reverb-TTS)
(a masked-diffusion TTS with Qwen3 layers and the Higgs Audio v2 codec) to ONNX, so it can run in
ONNX Runtime (CPU, or Vulkan through the WebGPU EP) without PyTorch. Built for the
SonicBoom TTS server; nothing here depends on it except `rust-probe/`.

> **License.** The Reverb weights are **CC BY-NC 4.0** (non-commercial, attribution required),
> so ONNX files made from them are too. The scripts in this repo are MIT. The Higgs Audio v2
> tokenizer has its own license: check it before redistributing the codec graphs.
> This repo does not contain any weights.

### What gets exported

Reverb is not autoregressive: it fills a frames x 8-codebook grid in 8 masked passes plus 2
polish passes over the whole target, attending to a cached prefix (text + reference codes).
So the transformer is exported once as a shared graph that serves both prefix and target passes,
and the sampler stays in your host language.

| Graph | Inputs -> outputs | fp32 size |
|---|---|---|
| `core.onnx` | `x[1,S,1024]`, `past_k/past_v[24,1,6,Pp,128]` (Pp may be 0), `bias[S,Pp+S]` -> `hidden`, `new_k`, `new_v` | ~1.06 GB |
| `embed_prefix.onnx` | `text_ids[1,Pt]`, `ref_codes[1,8,R]` -> `x[1,P,1024]` | 98 MB |
| `embed_target.onnx` | `tokens[1,8,F]` -> `x[1,F,1024]` | 34 MB |
| `heads.onnx` | `hidden[1,F,1024]` -> `log_probs[1,8,F,1024]` | 34 MB |
| `codec_decoder.onnx` | `codes[1,8,T]` -> `audio[1,1,T*960]` (24 kHz) | 87 MB |
| `codec_encoder.onnx` | `wav24[1,1,N]`, `wav16[1,N16]` (caller resamples to 16 kHz) -> `codes[1,8,T]` | 654 MB |

Prefix pass: `embed_prefix` -> `core` with empty past and a zero `bias`. Each of the 10 sampler
passes: `embed_target` -> `core` (past = prefix keys/values; `bias` is 0, or `polish_bias` on the
reference columns for the two polish passes) -> `heads`. Discard `core`'s `new_k/new_v` for target
passes. Sampling logic is `reverb_tts/sampler.py`.

### Reproduce

```sh
python -m venv venv && . venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
cd scripts

python export_onnx2.py fp16          # core, embeddings, heads -> onnx_fp16/
REF_WAV=reference.wav python export_codec.py   # codec encoder + decoder -> onnx_fp16/

python parity_pt.py                  # hand-written network vs reverb_tts (PyTorch)
python parity_onnx.py onnx_fp16      # official sampler through the ONNX graphs: expects 100 % equal codes
REF_WAV=reference.wav python parity_codec.py   # codec encoder/decoder vs transformers
```

`REF_WAV` is any clean 5-10 s speech clip. `export_onnx2.py fp32` exports from the fp32
checkpoint (1.2 GB download); `fp16` weights are upcast on load and gave identical codes in testing.
Exports use the legacy TorchScript exporter (`dynamo=False`, opset 18).

Quality check (Whisper small word error rate over 4 Spanish texts x 6 seeds, three voices):

```sh
REF_WAV=reference.wav REF_TEXT="exact transcript of the clip" python gen_eval.py
python score_dir.py gen/preset1 gen/preset2 gen/stref    # needs faster-whisper
```

### Results (CPU, BC-250 APU, 2026-11-20)

* ONNX graphs reproduce the official codes **100 %** (all 8 codebooks) and a bit-identical waveform.
* Codec decoder max waveform diff 7e-7; encoder 99.5-100 % identical codes with the same
  resampler (97-98 % with SciPy `resample_poly`).
* GPU (ORT WebGPU/Vulkan, RADV GFX1013): `core`/`heads` match CPU (argmax 100 %) and are 4-6x
  faster; **`codec_decoder` outputs near-silence on that GPU, keep it on CPU.**
* Spanish word error rate: 0.000 on both preset voices, 0.031 cloned from a clean clip.

Details and caveats: [`docs/feasibility.md`](docs/feasibility.md). `rust-probe/` holds the Rust
program used for the CPU-vs-GPU comparison (it uses SonicBoom's `onnx_runtime` module, so it is a
reference, not a standalone crate).

### Download the converted files

* Reverb transformer graphs (CC BY-NC 4.0): <https://huggingface.co/melser/Reverb-TTS-onnx>
* Higgs Audio v2 codec graphs (Boson Higgs Audio 2 Community License, Meta Llama 3 terms): <https://huggingface.co/melser/Higgs-Audio-2-Tokenizer-onnx>

Both list `SHA256SUMS`. Pin a commit hash when downloading from your app.
