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

### End-to-end synthesis from the ONNX graphs

[`run_onnx.py`](run_onnx.py) is the host the two graphs need: the tokenizer, prompt layout, sampler, inter-codebook delay and
EOS handling come from Zyphra's own plain-Python `zonos3.tts.*` (imported from `ZONOS3_SRC`, nothing installed), the generation
loop holds the LM state, and the codec decodes the whole take. `--check` runs Zyphra's PyTorch LM with the same seed first and
compares the generated rows.

```sh
export RELEASE=$PWD/ZONOS3-Mini                         # the upstream clone: lm/zonos3_mini/ + registries/
export VOICE=$RELEASE/assets/default_voices/AmericanFemale.embedding.npy
python examples/zonos3-mini/run_onnx.py \
  --models artifacts/zonos3-mini/models/fp32 --release "$RELEASE" \
  --voice "$VOICE" --text "Hello from ONNX." --max-frames 90 -o hello.wav --check
```

On the real weights: 49 frames (EOS at 38), **49/49 rows identical** to the PyTorch reference, 2.73 s of audio written.
The text must be already normalised; `--max-frames` must be at least the 10 codebooks or the undelayed take is empty.

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

## What is verified, and what is not

Real conversion of `Zyphra/ZONOS3-Mini` @ `2921b88` (CPU, Python 3.14, torch 2.15, onnxruntime 1.32), validated against Zyphra's own reference on the same inputs:

| Check | Result |
|---|---|
| `lm` (800M weights, 3.4 GiB) vs reference: prefill 24 rows with a speaker row + 3 condition rows, a decode step from non-empty state, a 7-row chunk, a 53-row prompt | max abs error **9e-5 … 1.6e-4** on logits of scale ≈ 15, cosine > 0.999999999998, argmax agreement **100 %** |
| `lm` stateful decode (prefill + 5 single-frame steps, state fed back) | passed; worst step 6.9e-5, free-running drift 6.9e-5 |
| Greedy 6-frame rollout through the ONNX LM (prefill + decode, state fed back) | **100 %** of the reference tokens |
| `codec_decoder` (3.8 GiB weights) vs reference, 5 / 24 / 41 frames | max abs error **6e-6 … 1e-5** on audio, energy ratio 1.000 |
| Output length | exactly 2940 samples per frame |
| Speed (2-core CPU, 3 iterations) | LM: 428 ms for a 24-row call; a single decode step ≈ 0.08 s; codec decoder: 0.93 s for 24 frames (1.6 s of audio), peak 4.1 GB RAM |
| Tiny release (tests, offline, ~20 s) | the whole pipeline, plus the int8-dynamic variant correctly reported as failing |

**Not verified / not included**

* An `fp16-weights` variant was built but could not be validated: ORT expands 16-bit weights back to fp32 when it loads the graph, which exceeded the 10 GB budget of the machine used. It is **not** part of the published artifact.
* int4 / int8 variants were not run on the real weights. int4 would not quantise the MoE experts anyway (see limits).
* CPU only; no GPU provider was tested.
* Audio synthesis was run end to end with [`run_onnx.py`](run_onnx.py): Zyphra's own tokenizer/prompt/sampler modules act as the host (they are not exported), producing **49/49 frames identical** to the reference at the same seed and 2.73 s of audio. Text normalisation, new-voice encoding (SSL teacher / speaker encoder) and the codec encoder are still Zyphra-side only.
* `text_rows` (phoneme / start / pause tokens) is supported by an option but was not validated on the real file.

## Tests

`ZONOS3_SRC=<clone>/model/src pytest tests/test_zonos3_example.py` builds a tiny randomly initialised release with the real parameter layout
(`tiny_checkpoint.py`) and runs the whole plugin on it in ~20 s (skipped when `ZONOS3_SRC` is not set). The real-weights numbers above come from running the `convert` command on a machine with ~12 GB free RAM; it is memory-hungry (peak ~9 GB while validating).
