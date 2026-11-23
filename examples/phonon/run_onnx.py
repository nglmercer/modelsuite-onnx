"""Synthesise speech from a modelsuite Phonon artifact with ONNX Runtime only (no PyTorch model weights).

    # Spanish: use a Spanish voice and normalise the text with the Spanish rules
    venv/bin/python examples/phonon/run_onnx.py artifacts/phonon-40m-multilingual \
        "Hola desde Phonon. Cuesta 5 euros." -o speech.wav --voice Iker --lang es

There is no language token in the checkpoint: the language/accent comes from the voice latents and the text
itself. The `lang` flag only controls text normalisation (numbers, symbols, dates); "none" uses the text as
written. The default voice shipped in the multilingual repo is NOT a Spanish voice -- pass --voice Iker
(es-es, male) or --voice Noa (es-es, female), or a local .safetensors file.

What still needs a library: text tokenisation and reading the voice latents. Those are host responsibilities, not
graphs. The three ONNX graphs do all the neural work; no transformers / PyTorch runtime weights are needed.
"""
from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort
from safetensors.numpy import load_file as load_safetensors
from tokenizers import Tokenizer

F32, I64 = np.float32, np.int64


def resolve_voice(spec: str | None, assets: Path) -> Path:
    """A local file, or a name/voice_id looked up in gradium/phonon-voices/voices.jsonl."""
    if spec is None:
        return assets / "default-voice.safetensors"
    local = Path(spec)
    if local.is_file():
        return local
    from huggingface_hub import hf_hub_download
    catalog = Path(hf_hub_download("gradium/phonon-voices", "voices.jsonl"))
    rows = [json.loads(line) for line in catalog.read_text().splitlines() if line.strip()]
    q = spec.lower()
    matches = [r for r in rows if q in (r["name"].lower(), r["voice_id"].lower(), r["filename"].lower())]
    if not matches:
        matches = [r for r in rows if q in (r["name"].lower(), r["voice_id"].lower())]
    if len(matches) != 1:
        raise SystemExit(f"voice {spec!r}: {len(matches)} matches; use an exact name, voice_id or a local .safetensors file")
    row = matches[0]
    print(f"voice: {row['name']} ({row['locale']}, {row['language']})")
    return Path(hf_hub_download("gradium/phonon-voices", row["filename"]))


def normalize(text: str, lang: str) -> str:
    if lang == "none":
        return text
    try:
        import ptts
    except ImportError as exc:
        raise SystemExit(f"--lang {lang} needs the normalizer: pip install ptts (or use --lang none)") from exc
    return ptts.normalize(text, lang=lang)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("artifact", type=Path)
    ap.add_argument("text")
    ap.add_argument("-o", "--out", type=Path, default=Path("speech.wav"))
    ap.add_argument("--variant", default="fp32")
    ap.add_argument("--voice", default=None,
                    help="local .safetensors, voice name (Iker, Noa...) or voice_id from gradium/phonon-voices; "
                         "default: the multilingual repo's default-voice.safetensors")
    ap.add_argument("--lang", default="none", choices=["none", "en", "fr", "de", "es", "pt"],
                    help="text-normalisation language; 'none' leaves the text as written, 'es' spells numbers/symbols "
                         "the Spanish way (needs 'pip install ptts')")
    ap.add_argument("--max-frames", type=int, default=400, help="hard frame budget (12.5 frames/s)")
    ap.add_argument("--eos-threshold", type=float, default=None)
    ap.add_argument("--min-frames", type=int, default=None)
    args = ap.parse_args()

    assets = args.artifact / "assets"
    models = args.artifact / "models" / args.variant
    cfg = json.loads((assets / "config.json").read_text())
    gen = json.loads((assets / "generation_config.json").read_text())
    pcfg = json.loads((assets / "processor_config.json").read_text())
    eos_threshold = args.eos_threshold if args.eos_threshold is not None else gen.get("eos_threshold", 4.0)
    min_frames = args.min_frames if args.min_frames is not None else gen.get("min_frames_before_eos", 0)
    frames_after_eos = int(pcfg.get("frames_after_eos", 1))

    core = ort.InferenceSession(str(models / "core.onnx"), providers=["CPUExecutionProvider"])
    flow = ort.InferenceSession(str(models / "flow.onnx"), providers=["CPUExecutionProvider"])
    codec = ort.InferenceSession(str(models / "codec_decoder.onnx"), providers=["CPUExecutionProvider"])

    voice_path = resolve_voice(args.voice, assets)
    voice = load_safetensors(str(voice_path))["speaker_wavs"][0].T.astype(F32)
    text = normalize(args.text, args.lang)
    if text != args.text:
        print(f"normalised: {text!r}")
    # Same prompt the chat template builds: voice placeholders, text tokens, then the audio BOS position.
    tokenizer = Tokenizer.from_file(str(assets / "tokenizer.json"))
    text_ids = tokenizer.encode(text, add_special_tokens=False).ids
    ids = np.array([[cfg["audio_prompt_token_id"]] * len(voice) + text_ids + [cfg["audio_bos_token_id"]]], I64)
    prompt = voice[None]
    S = ids.shape[1]

    L, KV, D = cfg["num_hidden_layers"], cfg["num_key_value_heads"], cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
    latent_dim, prompt_dim = cfg["latent_dim"], cfg["audio_prompt_dim"]
    empty = np.zeros((L, 1, KV, 0, D), F32)
    dense_prompt = np.zeros((1, S, prompt_dim), F32)
    dense_prompt[0, ids[0] == cfg["audio_prompt_token_id"]] = prompt[0]

    out = core.run(None, {"input_ids": ids, "prompt_latents": dense_prompt,
                          "audio_latents": np.zeros((1, S, latent_dim), F32),
                          "attention_mask": np.ones((1, S), I64), "position_ids": np.arange(S)[None].astype(I64),
                          "past_keys": empty, "past_values": empty.copy()})
    hidden, eos, past_k, past_v = out
    latents, remaining = [], None
    for step in range(args.max_frames):
        fo = flow.run(None, {"latents": np.zeros((1, latent_dim), F32), "condition": hidden[:, -1],
                             "start_time": np.zeros((1, 1), F32), "end_time": np.ones((1, 1), F32)})
        velocity, audio_latent = fo
        latents.append(audio_latent[0])
        if step >= min_frames and float(eos[0, -1]) > eos_threshold and remaining is None:
            remaining = frames_after_eos
        if remaining is not None:
            remaining -= 1
            if remaining < 0:
                break
        out = core.run(None, {"input_ids": np.full((1, 1), cfg["audio_latent_token_id"], I64),
                              "prompt_latents": np.zeros((1, 1, prompt_dim), F32),
                              "audio_latents": velocity[:, None, :], "attention_mask": np.ones((1, S + step + 1), I64),
                              "position_ids": np.array([[S + step]], I64), "past_keys": past_k, "past_values": past_v})
        hidden, eos, past_k, past_v = out

    LAT = np.stack(latents)
    padded = np.zeros((1, len(latents), latent_dim), F32)
    padded[0] = LAT
    frame_size = int(pcfg["feature_extractor"]["frame_size"])
    audio = codec.run(None, {"latents": padded})[0][0, 0, :len(latents) * frame_size]
    fade = int(pcfg.get("fade_in_duration", 0.0) * 24000)
    if fade:
        audio[:fade] *= np.linspace(0, 1, fade, dtype=F32)
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(args.out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(pcm.tobytes())
    print(f"wrote {args.out}: {len(latents)} frames, {len(audio) / 24000:.2f} s")


if __name__ == "__main__":
    main()
