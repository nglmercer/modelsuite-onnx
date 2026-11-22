# Reverb-TTS as the first adapter

Reverb-TTS (a masked-diffusion TTS: Qwen3 layers + the Higgs Audio v2 codec) was the reason this suite exists. Its conversion needed things no generic exporter
provides, so it is the regression test for the adapter contract. The original scripts are kept in `scripts/` as the reference; the logic now lives in
`src/modelsuite/adapters/reverb/`.

## Script → generic interface

| Original | Now | What was generalised |
|---|---|---|
| `scripts/export_net.py` (`CoreGraph`, `EmbedPrefix`, `EmbedTarget`, `Heads`) | `adapters/reverb/network.py` (verbatim maths) | modules are built lazily behind `ComponentExport.build()`; one shared loaded `Network` for every component |
| `scripts/export_onnx2.py` | `ReverbAdapter.plan()` + `TorchLegacyExporter` | example shapes derive from `config.json` (`llm_config`), not hard-coded `S=62, Pp=183`; dynamic axes become the component's `dynamic_axes` |
| `scripts/export_codec.py` (`Dec`, `Enc`, `pad_acoustic` probing) | `network.py::CodecDecoder/CodecEncoder`, `adapter.py::_build_encoder` | the `pad_acoustic` probe is kept; the synthetic reference signal replaces the mandatory `REF_WAV` (a real clip is still accepted: `reference_wav` option) |
| `scripts/parity_pt.py` (hand-written net vs `reverb_tts.Batch`) | the sampler-agreement quality check (subsumes it end to end) | |
| `scripts/parity_onnx.py` (`OnnxBatch`, official sampler through ONNX) | `ReverbAdapter._sampler_check` + `_OnnxBatch` → quality check `sampler-code-agreement` | any variant can be checked; thresholds: exactly 1.0 for `fp32` / graph-optimised, `quality_min_agreement_lowprec` (0.9) for lower precision |
| `scripts/parity_codec.py` | codec components' parity cases (decoder vs `codec.decode`, encoder vs `codec.encode`, incl. a longer dynamic-shape case) | the encoder reference is the *official* `encode`, not the export wrapper |
| `scripts/gen_eval.py`, `score_dir.py` (Whisper WER) | not ported into the core | WER needs a speech recogniser (`faster-whisper`); it stays a separate evaluation script. The sampler-agreement check is the in-pipeline criterion |
| `rust-probe/reverb_gpu_probe.rs` | the **provider parity check** + `provider_notes` | the probe's finding (codec decoder silent on RADV/WebGPU) is recorded on the component and detected generically by the energy-ratio check |
| `docs/feasibility.md` | `analyze` output (sampler not exported, GPU caveat, non-commercial license, RAM estimate) | |

What stays Reverb-specific (and out of the core): the Qwen3 hand-written attention (explicit RoPE/GQA/bias), the KV hand-off between the prefix and target passes,
the 8-codebook heads, the codec wrappers, the checkpoint layout (`model/safetensors/<variant>/`), the host-side sampler (documented as an asset, not exported), and the
Higgs codec's separate license.

## Adapter options

`--adapter-option variant=fp16|fp32|int8|int4` (checkpoint to load; quantised ones are dequantised on load and the graphs are fp32), `include_codec=false`,
`codec=<hub id>`, `codec_revision`, `reference_wav=<file>`, `prefix_text_tokens`, `reference_frames`, `target_frames`, `quality_steps`, `quality_min_agreement_lowprec`.
Only the chosen variant's files are downloaded from the Hub (`weight_patterns`).

```sh
modelsuite convert software-mansion/Reverb-TTS --revision v1.0.0 --adapter-option variant=fp16 \
    -o artifacts/reverb-tts-v1.0.0 --precision fp16-weights --benchmark
```

## Components and how the host uses them

| Component | In → out | Notes |
|---|---|---|
| `core` | `x[1,S,H]`, `past_k/past_v[L,1,KV,Pp,hd]` (Pp may be 0), `bias[S,Pp+S]` → `hidden`, `new_k`, `new_v` | **stateful**: the prefix pass's `new_k/new_v` are the target passes' `past_k/past_v` (`StateLink`, replace). Polish passes add the polish bias on the reference columns |
| `embed_prefix` | `text_ids[1,Pt]`, `ref_codes[1,8,R]` → `x[1,P,H]` | |
| `embed_target` | `tokens[1,8,F]` (mask id 1024 allowed) → `x[1,F,H]` | |
| `heads` | `hidden[1,F,H]` → `log_probs[1,8,F,1024]` | |
| `codec_decoder` | `codes[1,8,T]` → `audio[1,1,T*960]` (24 kHz) | provider note: keep on CPU on RADV/WebGPU |
| `codec_encoder` | `wav24[1,1,N]`, `wav16[1,N16]` → `codes[1,8,T]` | caller resamples to 16 kHz with torchaudio's resampler |

The **sampler** (masking, remasking, polish guidance, length estimate) is not exported; the manifest carries an asset record `sampler` describing it (defaults `steps=8`, `remask_ratio=0.25`, `t_shift=0.1`).

## License handling

The Reverb weights are CC BY-NC 4.0; the codec carries the Boson Higgs Audio 2 Community License / Llama 3 terms. The manifest records both
(`licenses.model`, `licenses.components`), and the redistribution verdict is the most restrictive of them (`conditional` → non-commercial, attribution; **the converted files inherit the restriction**).
This repository contains no weights.

## Results on the real checkpoint (2026-11-22, CPU, Python 3.14, torch 2.15, onnxruntime 1.32)

`modelsuite convert software-mansion/Reverb-TTS --revision v1.0.0 --adapter-option variant=fp16 --precision fp16-weights --benchmark` on the Hub
checkpoint (commit `72be799c8fb6d56a206d260b425c48bf3bd5943d`) ended with **overall validation: passed**. These are the original study's claims,
re-measured through the generic pipeline:

| Check | Original study | This run |
|---|---|---|
| Official sampler (`steps=4` on a synthetic 28-frame prompt) driven through the ONNX graphs | 100 % equal codes at all 8 codebooks | **100.00 %** at all 8 codebooks, for `fp32` and `fp16-weights` |
| `core` vs the hand-written PyTorch network, max abs (hidden) | log-prob diff 5e-4, argmax 100 % | 1.4e-4 … 4.9e-4 over target / prefix / polish / longer-target cases |
| Prefix → target KV hand-off (stateful case) | n/a | passed, worst step diff 2.7e-4 |
| `embed_prefix`, `embed_target`, `heads` | n/a | 9e-8, 0, 1.8e-5 |
| Codec decoder vs `transformers`, max waveform diff | 7e-7 | **6.1e-7** (1.2e-6 on a longer input) |
| Codec encoder codes vs `transformers` | 99.5-100 % identical | **100 %** (fp16-weights: 99.2 % on the longer case) |
| Graph sizes (fp32) | 1.06 GB / 98 / 34 / 34 / 87 / 654 MB | 1008 / 93.6 / 32 / 32 / 82.5 / 624 MiB (the same sizes in MiB) |
| `fp16-weights` | n/a | 937 MiB vs 1.83 GiB; same latency (ORT folds the casts at load) |
| CPU time for one pass through every graph | n/a | ~0.79 s (2-core machine; not comparable to the study's BC-250 timings) |

The synthetic prompt is deliberately not a spoken-quality test: the Whisper WER evaluation of the original study (`scripts/gen_eval.py`, `score_dir.py`) is
**not** re-run by the suite. What this proves is that the exported graphs are interchangeable with the PyTorch network under the official sampler.
Re-running the same check against the artifact later, without the original weights: `modelsuite validate artifacts/reverb-tts-v1.0.0` (20 s; parity cases only, stateful and quality checks need the original model).
The slow regression test `pytest -m slow tests/test_reverb_real.py` repeats the conversion and asserts these numbers.

Not re-measured here: GPU behaviour (no RADV/WebGPU on this machine), the Whisper word-error-rate table, and the fp16 / int8 / int4 *checkpoints* other than `fp16`.
