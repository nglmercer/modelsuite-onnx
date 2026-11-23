# Gradium Phonon TTS → ONNX (a modelsuite plugin)

```sh
modelsuite analyze gradium/phonon-40m-multilingual --plugin examples/phonon/phonon_plugin.py
modelsuite convert gradium/phonon-40m-multilingual --plugin examples/phonon/phonon_plugin.py \
    -o artifacts/phonon-40m-multilingual
modelsuite validate artifacts/phonon-40m-multilingual                 # golden-only, no original weights
```

This is a plugin adapter: nothing in `src/modelsuite` was changed. The generic `hf-transformers` adapter
classifies `GradiumPhononTTSForConditionalGeneration` as `seq2seq` and refuses it; this adapter knows the
flow-matching TTS layout.

## Graphs

| Graph | Inputs → outputs | fp32 size |
|---|---|---|
| `core.onnx` | `input_ids[1,S]`, `prompt_latents[1,S,512]`, `audio_latents[1,S,32]`, `attention_mask[1,P+S]`, `position_ids[1,S]`, `past_keys/past_values[9,1,8,P,64]` → `hidden[1,S,512]`, `eos[1,S]`, `present_keys/present_values[9,1,8,P+S,64]` | 117 MiB |
| `flow.onnx` | `latents[1,32]`, `condition[1,512]`, `start_time[1,1]`, `end_time[1,1]` → `velocity[1,32]`, `audio_latent[1,32]` | 36 MiB |
| `codec_decoder.onnx` | `latents[1,T,32]` → `audio[1,1,T*1920]` (24 kHz) | 40 MiB |

`core` is one stateful graph: prefill uses an empty past, each decode step feeds `present_keys/present_values`
back as `past_keys/past_values`. Voice and generated-latent embeddings arrive as **dense** tensors of length
`S`, with values only at the `audio_prompt_token_id` / `audio_latent_token_id` positions (zeros elsewhere);
this replaces the original's data-dependent `masked_scatter` so a single graph handles every sequence length.

`velocity` is the *normalised* latent the backbone consumes at the next step; `audio_latent` is the
denormalised value the codec and the caller need (both are returned by `flow.onnx`, so the host never reads
the original checkpoint's `latent_mean`/`latent_std`).

## What is not exported

Text normalisation + tokenisation + prompt assembly, the Gaussian noise draw (zero for greedy decoding),
the end-of-speech countdown (`eos_threshold`, `min_frames_before_eos`, `frames_after_eos`), and encoding a
new voice from raw audio (voices are 512-d speaker-codec latents already). `run_onnx.py` implements the
first four; it needs only `onnxruntime`, `tokenizers` and `safetensors`.

```sh
venv/bin/python examples/phonon/run_onnx.py artifacts/phonon-40m-multilingual \
    "Hola desde Phonon." -o speech.wav --max-frames 400
```

### Cómo hacer que hable en español

El checkpoint **no tiene un token de idioma**: el idioma/accento lo fijan (a) los tokens del texto y
(b) los latentes de la voz. El parámetro `lang` del runtime oficial (`ptts`) sólo controla la
**normalización del texto** (números, símbolos, fechas); `lang=none` deja el texto tal cual. El
`default-voice.safetensors` del repo multilingüe **no** es una voz española, por eso el ejemplo suena
en inglés aunque el texto esté en español.

```sh
# voz española del catálogo gradium/phonon-voices (Iker es-es, Noa es-es, o un voice_id)
venv/bin/python examples/phonon/run_onnx.py artifacts/phonon-40m-multilingual \
    "Hola desde Phonon. Esto es una prueba en español." \
    -o speech_es.wav --voice Iker --lang es
```

`--voice` acepta un archivo `.safetensors` local, un nombre (`Iker`, `Noa`, `Marcos`, `Rebeca`...) o un
`voice_id` del catálogo; `--lang es` necesita `pip install ptts` y reescribe p. ej. `5 €` → `5 euros`.
Otras voces/es-locale y sus IDs están en https://huggingface.co/gradium/phonon-voices (`voices.jsonl`).
Si se quiere un modelo sólo español, el mismo plugin sirve para `gradium/phonon-40m-es` (repositorio
gated; hay que aceptar su licencia).

The codec weights are **not** in the Phonon checkpoint: `AutoProcessor` fetches them from
`gradium/phonon-mimi` (Apache-2.0) at export time, and the resulting graph is licensed under that repo's terms
(the manifest records `codec_decoder: apache-2.0`). See `run_onnx.py` for the host loop.

## Verification (CPU, fp32, this repository)

All checks passed (`artifacts/phonon-40m-multilingual/reports/validation.json`):

* structural checks on all three graphs, opset 18, torch-legacy exporter;
* original-vs-ONNX parity: `core` ≤ 1.2e-5 (prefill, decode-step, 7-position chunk, long prefill),
  `flow` ≤ 6e-7, `codec_decoder` ≤ 8e-7 at 5, 8 and 17 frames;
* 6-step stateful KV-cache decode (prefill → single-frame steps) passed;
* end-to-end greedy rollout: 8 frames from the real processor prompt and shipped voice, latents within
  1.2e-5 and 24 kHz audio within 3e-5 of `transformers.generate()`.

The tokenizer's exported prompt is exactly `[audio_prompt]*N_voice + text tokens + [audio_bos]`.

## Caveats

* The checkpoint's `num_flow_steps` is 1. For a head with more steps, run `flow.onnx` in a loop with
  `start_time`/`end_time` set to the step's interval and `latents` set to the previous velocity result.
* `codec_decoder` is whole-sequence decode; chunk long audio (its sliding-window attention is dense-banded).
  Its exported graph is traced above the 250-frame attention-window threshold, so shorter sequences work too.
* Non-commercial license: weights are CC BY-NC 4.0; the converted files inherit that, with attribution.
