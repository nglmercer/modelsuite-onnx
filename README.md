# Reverb-TTS to ONNX

Scripts, checks and notes to convert [software-mansion/Reverb-TTS](https://huggingface.co/software-mansion/Reverb-TTS)
(a masked-diffusion TTS with Qwen3 layers and the Higgs Audio v2 codec) to ONNX, so it can run in
ONNX Runtime (CPU, or Vulkan through the WebGPU EP) without PyTorch. Built for the
SonicBoom TTS server; nothing here depends on it except `rust-probe/`.

> **License.** The Reverb weights are **CC BY-NC 4.0** (non-commercial, attribution required),
> so ONNX files made from them are too. The scripts in this repo are MIT. The Higgs Audio v2
> tokenizer has its own license: check it before redistributing the codec graphs.
> This repo does not contain any weights.

## What gets exported

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

## Reproduce

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

## Results (CPU, BC-250 APU, 2026-11-20)

* ONNX graphs reproduce the official codes **100 %** (all 8 codebooks) and a bit-identical waveform.
* Codec decoder max waveform diff 7e-7; encoder 99.5-100 % identical codes with the same
  resampler (97-98 % with SciPy `resample_poly`).
* GPU (ORT WebGPU/Vulkan, RADV GFX1013): `core`/`heads` match CPU (argmax 100 %) and are 4-6x
  faster; **`codec_decoder` outputs near-silence on that GPU, keep it on CPU.**
* Spanish word error rate: 0.000 on both preset voices, 0.031 cloned from a clean clip.

Details and caveats: [`docs/feasibility.md`](docs/feasibility.md). `rust-probe/` holds the Rust
program used for the CPU-vs-GPU comparison (it uses SonicBoom's `onnx_runtime` module, so it is a
reference, not a standalone crate).

## Download the converted files

* Reverb transformer graphs (CC BY-NC 4.0): <https://huggingface.co/melser/Reverb-TTS-onnx>
* Higgs Audio v2 codec graphs (Boson Higgs Audio 2 Community License, Meta Llama 3 terms): <https://huggingface.co/melser/Higgs-Audio-2-Tokenizer-onnx>

Both list `SHA256SUMS`. Pin a commit hash when downloading from your app.
