# ZONOS3-Mini → ONNX (a modelsuite adapter, as a plugin)

[`Zyphra/ZONOS3-Mini`](https://huggingface.co/Zyphra/ZONOS3-Mini) is an 800M-parameter (≈100M active) multilingual TTS with a hybrid
**Gated DeltaNet / attention** backbone, a **latent mixture-of-experts**, and a causal **multi-resolution iSTFT** codec decoder.
Zyphra ships it for ggml / CUDA / MLX / WebGPU runtimes; this example adds **ONNX Runtime**. It is the second custom adapter after Reverb and
the proof that an architecture the core knows nothing about can be added *without touching `src/modelsuite`*.

```sh
git clone https://github.com/Zyphra/ZONOS3-Mini            # the numerical oracle + module layout (MIT); not installed, only put on sys.path
export ZONOS3_SRC=$PWD/ZONOS3-Mini/model/src
pip install einops && pip install --no-deps descript-audio-codec==1.0.0   # two tiny submodules of dac; its audiotools dependency is shimmed

modelsuite analyze Zyphra/ZONOS3-Mini --plugin examples/zonos3-mini/zonos3_plugin.py
modelsuite convert Zyphra/ZONOS3-Mini --plugin examples/zonos3-mini/zonos3_plugin.py -o artifacts/zonos3-mini --precision fp16-weights --benchmark
modelsuite validate artifacts/zonos3-mini                     # later, without the original weights or the reference code
python examples/zonos3-mini/publish_hf.py artifacts/zonos3-mini --repo <you>/ZONOS3-Mini-onnx --dry-run
```

Why the reference is not `pip install`ed: `zonos3` pins `transformers<=4.57.3`, `torchaudio==2.9.1` and Python ≥ 3.12, which would downgrade this environment.
Peak RAM while exporting the LM is ~5.4 GiB (3.2 GiB fp32 reference + the traced graph); the 7.9 GB codec `generator.pt` is memory-mapped.

## What is exported

| Graph | Contract |
|---|---|
| `lm` | `ids[T,11]` (10 codebooks + text column), `speaker_vec[T,192]`, `speaker_flag[T]`, `cond_index[T]`, `past_k/past_v[2,P,4,32]`, `conv_state[9,1536,3]`, `gdn_state[9,4,128,128]` → `logits[T,10,4098]`, `new_k/new_v`, `new_conv`, `new_state`. **One graph for prefill and decode**; feed the four `new_*` back as the matching inputs (start from `P=0` and zeros). `speaker_flag/cond_index` replace rows by the projected speaker embedding / a condition-table row (what `Zonos3LM._replace_rows` does) |
| `codec_decoder` | `codes[1,10,T]` → `audio[1,1,2940·T]` at 44.1 kHz; whole-sequence decode (equals `ISTFTCodec.forward`) |

## How the hard parts were solved

| Problem in the reference | Export-friendly equivalent | Why it is the same function |
|---|---|---|
| GDN recurrence is a Python loop over tokens in fp32 | `torch.jit.script` loop → ONNX `Loop` over rows, state in/out | identical update order; tiny-model GDN state is bit-identical to the reference |
| MoE picks experts with `int(choices[t, r])` | `TopK` + per-row **gather of only the 2 selected experts' weights** inside a `Loop` | same per-(token, expert) arithmetic; reads 2/64 of the expert weights per row |
| Python-dict caches | explicit tensors: rotated K/V, conv history, GDN matrices | the host owns the state; the manifest's `state_links` describe the feedback |
| `F.rms_norm`, `scaled_dot_product_attention`, interleaved RoPE | spelled out with basic ops | opset 18 has no RMSNormalization; numerics match to ~1e-6 |
| `torch.istft` (complex) | inverse real DFT as two matmuls with constant bases + overlap-add + window-envelope division + centre trim | same as `torch.istft(center=True, length=…)`; codec output matches the reference to 5e-8 |
| MOSS attention loops over 512-row blocks, in-place biases | one banded dense attention (query sees the `context_frames` latest keys) | same mask; **memory O(T²) at the 240 Hz stage: decode long audio in chunks** |
| torchaudio `Resample` computes its length in Python (frozen at export) | same polyphase kernel, length from the frame count | output is always exactly `2940·T` samples |

## Validation (all against Zyphra's own reference, same inputs)

* `lm`: prefill with a speaker row and the 3 condition rows, a decode step from non-empty state, a different chunk size, a longer prompt; a **stateful 6-step decode** feeding the state back (lock-step *and* free-running drift); a **greedy frame rollout** (prefill + decode steps) must give the same tokens.
* `codec_decoder`: several frame counts (output length and numbers).
* Variants (`fp16-weights`, `int8-dynamic`, …) get the same checks with documented per-precision tolerances; failures are reported, not hidden.

## NOT migrated (the host must provide it)

Text normalisation (NeMo) and the SuperBPE tokenizer · prompt assembly (speaker row, 3 condition rows, silence prefill, the inter-codebook delay `shear`/`shear_up`) ·
the sampler (temperature, top-k/p, repetition penalty, EOA stop) · the **speaker encoder** (MECT) and the Wav2Vec2-BERT **SSL teacher** (only needed to encode *new*
reference audio; the 27 bundled voices ship as precomputed codes/embeddings) · aligner · edit LoRA / accents / steering / alignment probe · the encoder half of the codec.
Reference for all of these: Zyphra's `zonos3.tts.*`. The optional `text_rows` (phoneme / start / pause rows) are supported through `--adapter-option text_rows=<file.pt>` but **not validated** against a real file.

## Known limits

* Decode on a 2-core CPU is ~0.08 s per 15 Hz frame; real time needs 0.067 s. Prefill is linear in the number of rows (Loops). Zyphra's ggml/CUDA/WebGPU runtimes are the real-time options.
* `int4` does not quantise the MoE experts (they are gathered initialisers, not MatMuls); `fp16-weights` does cover them.
* Only CPU was tested. The decoder-head checkpoint is loaded with `weights_only=True`; if it ever needs full pickle the adapter refuses unless you pass `allow_pickle=true`.

## What is verified, and what is not (read this first)

| | Status |
|---|---|
| **Real 800M LM weights** | Exported to ONNX (3.4 GiB, one stateful graph) and compared with Zyphra's reference in a standalone run: prefill of 40 rows, a decode step and a 7-row chunk continuing from real state. Logits within **4.4e-5** (scale ≈ 15), argmax agreement **100 %**, GDN state within 5e-7. A decode step took ~0.08 s on a 2-core CPU. The full pipeline export of the LM also completed (72 s) |
| **Tiny release with the real layout** (tests, offline, ~20 s) | LM + codec decoder through the whole pipeline: parity, dynamic shapes, stateful decode, greedy rollout, fp16-weights passes; int8-dynamic is reported as failing |
| **Real codec decoder weights** (`generator.pt` 7.9 GB + head 4 GB) | **Not verified.** The checkpoints are downloaded, but the real build did not finish: building the decoder next to the other loaded models exceeded the memory of the development machine (≈5 GB free) and the run was killed. The decoder is verified only on the tiny codec (5e-8 vs the reference) |
| Full real-weights pipeline (validation of every case, quality check, benchmark, fp16-weights variant) | **Not run to completion.** Needs a machine with roughly 12 GB of free RAM |
| Hugging Face upload | **Not done** (no credentials on the development machine, and the real artifact is incomplete) |

To finish the verification on a bigger machine: run the `convert` command above (the pipeline resumes per component), then `publish_hf.py --dry-run`, then publish.

## Tests

`ZONOS3_SRC=<clone>/model/src pytest tests/test_zonos3_example.py` builds a tiny randomly initialised release with the real parameter layout
(`tiny_checkpoint.py`) and runs the whole plugin on it in ~20 s (skipped when `ZONOS3_SRC` is not set).
