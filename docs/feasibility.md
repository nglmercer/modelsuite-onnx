# Feasibility study: Reverb-TTS on the AMD BC-250 (2026-11-20)

Question: can [software-mansion/Reverb-TTS](https://huggingface.co/software-mansion/Reverb-TTS)
(297M masked-diffusion TTS, Qwen3 layers + Higgs Audio v2 codec) run in the SonicBoom TTS server, on an
APU (BC-250, RADV GFX1013, 14 GB shared RAM)? Upstream ships no ONNX or GGUF, so the
graphs were exported here.

**Verdict: feasible. No hardware blocker.** Weights are CC BY-NC 4.0 (non-commercial).

## Why not llama.cpp
Reverb is not autoregressive. It fills a frames x 8-codebook grid in 8 masked passes plus
2 polish passes over the whole target, attending bidirectionally to a cached prefix
(text + reference codes). That needs custom sampling, not a token loop.

## Export (`scripts/`)
| Graph | Role | fp32 size | Runs on |
|---|---|---|---|
| `core.onnx` | 24 Qwen3 layers, shared by prefix and target passes (`x`, `past_k`, `past_v`, `bias`) | 1058 MB | GPU or CPU |
| `embed_prefix.onnx`, `embed_target.onnx`, `heads.onnx` | factorized text embedding, 8-codebook embedding sum, 8 heads + log-softmax | 98 / 34 / 34 MB | GPU or CPU |
| `codec_decoder.onnx` | codes -> 24 kHz waveform | 87 MB | **CPU only** |
| `codec_encoder.onnx` | reference audio (24 kHz + 16 kHz) -> codes, for cloning | 654 MB | CPU (cached per voice) |

The sampler (masking, remasking, polish guidance, length estimate) stays in Rust.

## Parity (Python, CPU)
* Hand-written `core` vs `reverb_tts` `Batch.log_probs`: max log-prob diff 5e-4, argmax 100 %.
* Official `sample()` driven through the ONNX graphs: **100 % identical codes at all 8
  codebooks** and a bit-identical decoded waveform (2 prompts).
* Codec decoder: max waveform diff 7e-7. Encoder: 99.5-100 % identical codes with the
  same 24->16 kHz resampler, 97-98 % with SciPy `resample_poly` (reference prompt only).

## GPU on the BC-250 (`rust-probe/reverb_gpu_probe.rs`, ORT WebGPU/Vulkan)
| Target frames | CPU pass | GPU pass | log-prob max diff | argmax agreement |
|---|---|---|---|---|
| 62 (~2.5 s) | 0.30 s | 0.08 s | 0.0011 | 100 % |
| 250 (~10 s) | 0.93 s | 0.15 s | 0.0044 | 100 % |

* The transformer is correct and 4-6x faster on the GPU. A 10 s take is about 10 passes
  x 0.15 s + prefix 0.16 s = ~1.7 s, plus the CPU codec decode (~1 s): RTF about 0.3.
  CPU only is roughly RTF 1.1 (Pocket is ~0.5).
* **The codec decoder is silent on the GPU** (RMS 0.00014 vs 0.017) in every buffer-cache
  mode (`bucket`, `disabled`, `simple`, `lazy`). Same RADV/WebGPU conv bug that affects
  Pocket's Mimi decoder and the Qwen codecs. Keep it on CPU.
* Peak RAM in the Python smoke test was 3.4 GB.

## Quality (Whisper small WER, 4 Spanish texts x 6 seeds)
| System / reference | WER | outputs >50 % wrong |
|---|---|---|
| Reverb, preset `es_male_standard_1` | 0.000 | 0/24 |
| Reverb, preset `es_male_standard_2` | 0.000 | 0/24 |
| Reverb, cloned clean 7 s Spanish clip | 0.031 | 1/24 |
| Pocket (current code), same clean clip | 0.173 | 2/24 |
| Pocket (current code), `miku_test` | 0.588 | 15/24 |

`miku_test.wav` (2.4 s) is **Japanese** ("オカイモノ!"), cloned into the Spanish Pocket
model. A cross-language, very short reference explains most of Pocket's poor Spanish.
Reverb supports en, de, fr, es, it, pl, pt only, so it cannot use that clip.

## Remaining work to ship it
1. Rust provider: download pinned ONNX (we must host our export), port the sampler
   (`reverb_tts/sampler.py`, defaults steps=8, remask 0.25, t_shift 0.1), length
   estimation, text splitting, Verbalis-style normalization check.
2. One process-wide lock around every GPU session (Supertonic, Pocket, Reverb).
3. Voice profiles need the reference transcript; preset voices are JSON codes.
4. Decide fp16 weights (halves download/RAM; untested on RADV) and whether to resample the
   reference in Rust (97-98 % code match) or ship the torchaudio kernel.
5. Config, UI language list, docs, and a license warning for the non-commercial weights.
