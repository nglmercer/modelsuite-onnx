"""Text -> speech with the two ZONOS3-Mini ONNX graphs (lm.onnx + codec_decoder.onnx), end to end.

    export ZONOS3_SRC=<ZONOS3-Mini clone>/model/src
    python run_onnx.py --models artifacts/zonos3-mini/models/fp32 --release <HF snapshot or local release dir> \\
           --voice <...>/web/voices/English/Imogen_en-British_F_1.embedding.npy --text "Hello from ONNX." -o hello.wav

What this script is: the *host* the published graphs need. The graphs are the model; everything around them comes from Zyphra's own plain-Python modules,
imported from your clone (nothing is installed): the prompt layout (`zonos3.tts.prompt`), SuperBPE tokenizer (`zonos3.tts.superbpe`), the sampler
(`zonos3.tts.sampler`), the inter-codebook delay and EOS handling (`zonos3.tts.delay` / `zonos3.tts.stream`). The generation loop below follows
`zonos3.reference.synth.generate_rows` (MIT, (c) 2026 Zyphra, Inc.) for the plain case: speaker-embedding conditioning, no CFG / steering / accents / LACI.

`--check` additionally runs Zyphra's PyTorch reference LM with the SAME seed and compares the generated frames with the ONNX run.

Not done here (be aware): text normalisation (NeMo/zyphra-textnorm): give already-normalised text (no digits, symbols or abbreviations);
voice cloning from audio (only precomputed `.embedding.npy` voices); streaming (the codec decodes the whole take at once: memory grows with length)."""
from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference_env import ensure_reference

GDN_HEADS, GDN_HEAD_DIM, GDN_KERNEL = 4, 128, 4


def load_embedding(path: str, dim: int) -> np.ndarray:
    arr = np.squeeze(np.load(path, allow_pickle=False).astype(np.float32))
    vec = arr if arr.ndim == 1 else arr.mean(axis=0)
    if vec.shape[-1] != dim:
        raise SystemExit(f"{path}: embedding has {vec.shape[-1]} values, the model expects {dim}")
    return np.ascontiguousarray(vec)


def build_prompt(cfg, builder, text: str, silence_frames: int, silence_frame, conditions: list[int]):
    """[speaker row][condition rows][text rows][silence prefill], the layout of zonos3.tts.plan.build_prompt for the embedding-conditioned case."""
    import torch
    from zonos3.tts.prompt import silence_rows_from_codes
    rows = builder.build(text)
    if silence_frames > 0:
        rows = torch.cat((rows, silence_rows_from_codes(silence_frame, silence_frames, builder.config).to(rows.dtype)), 0)
    prefill = None if silence_frames <= 0 else silence_rows_from_codes(silence_frame, silence_frames, builder.config).to(torch.int32)
    rows = torch.cat((builder.condition_rows(len(conditions), dtype=rows.dtype), rows), 0)
    rows = torch.cat((builder.speaker_slot(dtype=rows.dtype), rows), 0)
    return rows.to(torch.int32), prefill                                            # speaker row = 0, condition rows = 1 .. len(conditions)


class OnnxLM:
    """lm.onnx with its state (KV cache, GDN matrices, conv history) held here, the way a host must."""

    def __init__(self, path: str, cfg):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.session = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.cfg = cfg
        n_attn = len(cfg.attention_layers)
        n_gdn = cfg.n_layers - n_attn
        f32 = np.float32
        self.state = {"past_k": np.zeros((n_attn, 0, cfg.n_kv_heads, cfg.head_dim), f32), "past_v": np.zeros((n_attn, 0, cfg.n_kv_heads, cfg.head_dim), f32),
                      "conv_state": np.zeros((n_gdn, 3 * GDN_HEADS * GDN_HEAD_DIM, GDN_KERNEL - 1), f32),
                      "gdn_state": np.zeros((n_gdn, GDN_HEADS, GDN_HEAD_DIM, GDN_HEAD_DIM), f32)}
        self.offsets = np.cumsum([0] + [n + 1 for n in cfg.condition_bucket_counts])      # start of each condition table in the concatenated table

    def __call__(self, ids, speaker=None, cond=None):
        """ids [T, 11] int; speaker: {row: vec}; cond: {row: bucket id} (rows relative to this call; the i-th condition row reads table i)."""
        T = len(ids)
        spk = np.zeros((T, self.cfg.speaker_embedding_dim), np.float32)
        flag = np.zeros(T, np.int64)
        for r, v in (speaker or {}).items():
            spk[r], flag[r] = v, 1
        cidx = -np.ones(T, np.int64)
        for k, (r, bucket) in enumerate(sorted((cond or {}).items())):
            cidx[r] = self.offsets[k] + bucket
        out = self.session.run(None, {"ids": np.asarray(ids, np.int64), "speaker_vec": spk, "speaker_flag": flag, "cond_index": cidx, **self.state})
        self.state = {"past_k": out[1], "past_v": out[2], "conv_state": out[3], "gdn_state": out[4]}
        return out[0]


class TorchLM:
    """Zyphra's reference LM (the oracle), for --check."""

    def __init__(self, release_lm: Path):
        import torch
        from zonos3.reference.lm import LMCache, Zonos3LM
        self.torch, self.lm, self.cache = torch, Zonos3LM.load(release_lm, dtype=torch.float32).eval(), LMCache()

    def __call__(self, ids, speaker=None, cond=None):
        P = self.cache.position
        sp = {P + r: self.torch.from_numpy(v) for r, v in (speaker or {}).items()}
        cd = {P + r: b for r, b in (cond or {}).items()}
        with self.torch.inference_mode():
            return self.lm(self.torch.as_tensor(np.asarray(ids), dtype=self.torch.long), self.cache, speaker_rows=sp or None, condition_rows=cd or None).logits.numpy()


def sample_row(cfg, logits, history, total_generated, stop_id, generator, sampling):
    """One sampled row (zonos3.reference.synth.sample_row, plain case)."""
    import torch
    from zonos3.tts.sampler import active_repetition_window, clamp_sampling, sample_tts
    c, n_cb = cfg, cfg.n_codebooks
    s = clamp_sampling(sampling, n_codebooks=n_cb, codebook_size=c.codebook_size)
    active = active_repetition_window(s, min(total_generated, len(history)))
    rep_ids = rep_pen = None
    if active > 0:
        recent = history[len(history) - active:, :n_cb].to(torch.int64).transpose(0, 1)
        valid = (recent >= 0) & (recent < c.codebook_size)
        valid[s.repetition_codebooks:] = False
        rep_ids = torch.where(valid, recent, torch.full_like(recent, -1)).unsqueeze(0)
        rep_pen = torch.tensor([s.repetition_penalty], dtype=torch.float32)
    return sample_tts(
        torch.from_numpy(np.asarray(logits, np.float32)).unsqueeze(0), torch.tensor([s.temperature], dtype=torch.float32), torch.tensor([s.topk], dtype=torch.int32),
        torch.tensor([s.top_p], dtype=torch.float32), torch.tensor([s.min_p], dtype=torch.float32), codebook_size=c.codebook_size, audio_pad_id=c.audio_pad_id,
        stop_token_ids=torch.tensor([stop_id], dtype=torch.int64), repetition_token_ids=rep_ids, repetition_penalties=rep_pen, codebook_sizes=c.codebook_sizes,
        text_vocab=c.text_vocab, generators=[generator], stop_blocked=None, stop_bias=None)[0].cpu()


def generate(cfg, lm, prompt, speaker_vec, conditions, sampling, seed: int, max_frames: int, log=print):
    """Autoregressive generation (zonos3.reference.synth.generate_rows, plain case). Returns (rows, eos_frame)."""
    import torch
    from zonos3.tts.delay import apply_cold_start_delay
    n_cb, pad, stop_id = cfg.n_codebooks, cfg.audio_pad_id, cfg.eoa_id
    generator = torch.Generator().manual_seed(int(seed))
    has_audio_prefix = bool((prompt[-1, :n_cb] != pad).any())
    t0 = time.perf_counter()
    logits = lm(prompt.numpy(), speaker={0: speaker_vec}, cond={1 + i: int(b) for i, b in enumerate(conditions)})
    log(f"prefill {len(prompt)} rows: {time.perf_counter() - t0:.1f}s")
    rows, history, eos_frame, countdown, remain = [], prompt[:, :n_cb].clone(), -1, -1, max_frames
    t0 = time.perf_counter()
    while True:
        total = len(rows)
        token = sample_row(cfg, logits[-1], history, total, stop_id, generator, sampling)
        apply_cold_start_delay(token, total, n_codebooks=n_cb, pad_id=pad, has_audio_prefix=has_audio_prefix)
        rows.append(token)
        history = torch.cat((history, token[None, :n_cb]), 0)
        remain -= 1
        finished = remain <= 0
        if eos_frame < 0:
            cols = [i for i, tok in enumerate(token[:n_cb].tolist()) if tok == stop_id]
            if cols:
                eos_frame, countdown = max(0, total - max(cols)), n_cb + 1
        if countdown > 0:
            countdown -= 1
            finished = finished or countdown == 0
        if finished:
            break
        logits = lm(token[None].numpy())
    dt = time.perf_counter() - t0
    log(f"decoded {len(rows)} frames in {dt:.1f}s ({dt / max(len(rows), 1):.3f} s/frame; real time = 0.067), eos_frame={eos_frame if eos_frame >= 0 else None}")
    return rows, (eos_frame if eos_frame >= 0 else None)


def decode_audio(codec_path: str, cfg, rows, eos_frame, prefill, headroom_db: float, fade_ms: float = 20.0) -> np.ndarray:
    import onnxruntime as ort
    from zonos3.generation import SAMPLE_RATE
    from zonos3.tts.audio import output_pcm
    from zonos3.tts.stream import StreamDecoder
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(codec_path, so, providers=["CPUExecutionProvider"])

    def decode(codes: np.ndarray) -> np.ndarray:                                    # undelayed [frames, n_codebooks] -> float32 [frames * hop]
        return sess.run(None, {"codes": np.ascontiguousarray(codes.T[None], dtype=np.int64)})[0][0, 0].astype(np.float32)
    whole = StreamDecoder(decode, n_codebooks=cfg.n_codebooks, audio_pad_id=cfg.audio_pad_id, hop=2940, prefill_rows=prefill, context_codes=None, mode="whole")
    for row in rows:
        whole.push(row, eos_frame)
    pcm = output_pcm(whole.finish(), float(10.0 ** (-headroom_db / 20.0)))
    n = min(len(pcm), int(SAMPLE_RATE * fade_ms / 1000))
    if n:
        pcm = pcm.copy()
        pcm[-n:] *= 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, n, dtype=np.float32)))
    return pcm.astype(np.float32)


def write_wav(path: str, pcm: np.ndarray, rate: int) -> None:
    pcm16 = np.rint(np.clip(np.nan_to_num(pcm), -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(rate)
        w.writeframes(pcm16.tobytes())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", required=True, help="directory with lm.onnx and codec_decoder.onnx (e.g. artifacts/zonos3-mini/models/fp32)")
    ap.add_argument("--release", required=True, help="release directory with lm/zonos3_mini/{params.json,tokenizer/...} and registries/")
    ap.add_argument("--voice", required=True, help="a precomputed speaker embedding (.embedding.npy)")
    ap.add_argument("--text", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=90, help="frame budget (15 frames = 1 s); the codec decodes the whole take, memory grows with it")
    ap.add_argument("--conditions", default=None, help="loudness,fmax,speaking_rate bucket ids (default: unknown = 10,10,30)")
    ap.add_argument("--source-dir", default=None, help="<ZONOS3-Mini clone>/model/src (else $ZONOS3_SRC)")
    ap.add_argument("--check", action="store_true", help="also run Zyphra's PyTorch reference LM with the same seed and compare the generated frames (needs ~3.5 GB more RAM)")
    a = ap.parse_args()

    ensure_reference(a.source_dir, codec=False)
    import torch
    from types import SimpleNamespace
    from zonos3.generation import SAMPLE_RATE, SAMPLING, SILENCE_PREFILL_DEFAULT_FRAMES
    from zonos3.reference.lm import Zonos3LMConfig
    from zonos3.tts.prompt import TTSPromptBuilder, TTSPromptConfig
    from zonos3.tts.superbpe import load_spec

    release = Path(a.release).expanduser()
    lm_dir = release / "lm" / "zonos3_mini"
    cfg = Zonos3LMConfig.from_params(json.loads((lm_dir / "params.json").read_text()))
    spec = load_spec(lm_dir)
    builder = TTSPromptBuilder(TTSPromptConfig(n_codebooks=cfg.n_codebooks, codebook_sizes=cfg.codebook_sizes, audio_pad_id=cfg.audio_pad_id,
                                               text_vocab=cfg.text_vocab, text_tokenizer_spec=spec))
    reg = json.loads((release / "registries" / "codec_registry.json").read_text())
    entry = next(e for e in reg.values() if e["codec"] == cfg.audio_codec)
    silence = json.loads(next((release / "registries").glob("silence_frame*.json")).read_text())
    conditions = [int(x) for x in a.conditions.split(",")] if a.conditions else [int(n) for n in cfg.condition_bucket_counts]    # default: every control "unknown"
    prompt, prefill = build_prompt(cfg, builder, a.text, SILENCE_PREFILL_DEFAULT_FRAMES, torch.tensor(silence["frame"], dtype=torch.int32), conditions)
    speaker = load_embedding(a.voice, cfg.speaker_embedding_dim)
    sampling = SimpleNamespace(**SAMPLING)
    print(f"prompt: {len(prompt)} rows (1 speaker + {len(conditions)} condition + text + {SILENCE_PREFILL_DEFAULT_FRAMES} silence), conditions {conditions}, seed {a.seed}")

    models = Path(a.models)
    results = {}
    if a.check:
        print("-- reference (Zyphra's PyTorch LM)")
        ref = TorchLM(lm_dir)
        results["torch"] = generate(cfg, ref, prompt, speaker, conditions, sampling, a.seed, a.max_frames)
        del ref
        import gc
        gc.collect()
    print("-- ONNX")
    onnx_lm = OnnxLM(str(models / "lm.onnx"), cfg)
    results["onnx"] = generate(cfg, onnx_lm, prompt, speaker, conditions, sampling, a.seed, a.max_frames)
    del onnx_lm
    rows, eos = results["onnx"]
    if a.check:
        rr, ro = results["torch"][0], rows
        n = min(len(rr), len(ro))
        same = [bool(torch.equal(rr[k], ro[k])) for k in range(n)]
        first_diff = next((k for k, ok in enumerate(same) if not ok), None)
        print(f"CHECK: reference {len(rr)} frames vs ONNX {len(ro)} frames; identical rows {sum(same)}/{n}"
              + (f"; first difference at frame {first_diff}" if first_diff is not None else "; all compared rows identical")
              + f"; eos frame {results['torch'][1]} vs {eos}")
    pcm = decode_audio(str(models / "codec_decoder.onnx"), cfg, rows, eos, prefill, float(entry["decoder"].get("headroom_db", 0.0)))
    if pcm.size == 0:
        raise SystemExit(f"no audio decoded: {len(rows)} generated frames cannot complete the {cfg.n_codebooks}-codebook "
                         f"undelay (at least {cfg.n_codebooks} are needed); raise --max-frames")
    write_wav(a.out, pcm, SAMPLE_RATE)
    print(f"wrote {a.out}: {len(pcm) / SAMPLE_RATE:.2f} s, peak {float(np.abs(pcm).max()):.3f}, rms {float(np.sqrt((pcm ** 2).mean())):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
