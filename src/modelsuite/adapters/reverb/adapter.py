"""Reverb-TTS (software-mansion/Reverb-TTS): masked-diffusion TTS, Qwen3 backbone + Higgs Audio v2 codec.

Why it needs a custom adapter (none of this fits a generic exporter):
  * not autoregressive: the sampler fills an 8-codebook x frames grid over 8 masked + 2 polish passes, so the *sampler stays in
    the host language* and only five tensor graphs are exported;
  * ONE transformer graph (`core`) serves both the prefix pass (empty past) and every target pass (past = prefix K/V) -- the
    KV tensors are carried between calls by the host (a stateful link) and the attention bias is an explicit input;
  * the Hugging Face codec is wrapped by hand (decoder over quantizer.decode; encoder takes a pre-resampled 16 kHz input);
  * the codec decoder returns near-silence on some GPU stacks (RADV/Vulkan WebGPU) -- recorded as a provider note and caught by the
    provider-parity check;
  * the quality criterion is *the official sampler producing identical codes* through the ONNX graphs, not tensor parity alone.
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from ...core import deps
from ...core.config import JobConfig, Tolerance
from ...core.discovery import ModelSource, safetensors_header_summary
from ...core.errors import UnsupportedModelError
from ...core.plan import (Arrays, AssetSpec, ComponentExport, Edge, ExportPlan, QualityCheck, QualityContext, QualityResult, ReferenceCase,
                          StatefulCase, StateLink)
from ...core.registry import register_adapter
from ...core.types import CompatibilityReport, ComponentInfo, Detection, Issue, ModelInfo, Severity, Support
from ..base import Adapter, AdapterOptions, memory_estimate

REPOSITORY = "software-mansion/Reverb-TTS"
CODEC_ID, CODEC_REVISION = "eustlb/higgs-audio-v2-tokenizer", "0e3a9177c2991016b8e5e17d23aa474ee665bef2"
VARIANTS = ("fp32", "fp16", "int8", "int4")
CODEBOOKS, CODES, MASK = 8, 1025, 1024
FRAME_RATE, HOP = 25, 960      # codec frames per second; samples per frame at 24 kHz
WEBGPU_NOTE = ("Near-silent output on AMD RADV/Vulkan through the ORT WebGPU EP (RMS 0.00014 vs 0.017) in every buffer-cache mode; keep this "
               "graph on CPU. The provider-parity check detects it.")


class ReverbOptions(AdapterOptions):
    variant: str = "fp16"                       # checkpoint variant to load; quantised ones are dequantised on load, the graphs are fp32
    include_codec: bool = True                  # export codec_decoder / codec_encoder
    codec: str = CODEC_ID
    codec_revision: str | None = None           # default: the revision reverb_tts pins
    reference_wav: str | None = None            # optional real clip for the encoder cases (any rate); synthetic speech-like signal otherwise
    prefix_text_tokens: int = 12                # sizes of the synthetic cases (the real model works for any size)
    reference_frames: int = 20
    target_frames: int = 10
    quality_steps: int = 4                      # sampler main passes in the sampler-agreement check
    quality_min_agreement_lowprec: float = 0.9  # required code agreement for non-fp32 variants (fp32 / graph-optimised must be exactly 1.0)


# ----------------------------------------------------------------------------------------------------------- locating files
def _weights_dir(root: Path, variant: str) -> Path | None:
    for f in (root / "model" / "safetensors" / variant, root / "model" / variant, root / variant, root):
        if (f / "config.json").is_file():
            return f
    return None


def _tokenizer_dir(weights: Path, root: Path) -> Path | None:
    for f in (weights, weights.parent, weights.parent.parent, root / "model"):
        if (f / "tokenizer.json").is_file():
            return f
    return None


def _read_cfg(d: Path) -> dict[str, Any]:
    return json.loads((d / "config.json").read_text())


def _dims(llm: dict[str, Any]) -> dict[str, int]:
    heads = llm["num_attention_heads"]
    return {"layers": llm["num_hidden_layers"], "hidden": llm["hidden_size"], "kv": llm.get("num_key_value_heads", heads),
            "hd": llm.get("head_dim", 128), "vocab": llm["vocab_size"]}


class _Shared:
    """The loaded network / codec, shared by every component and by the quality check (weights load once)."""

    def __init__(self, weights_dir: Path, opts: ReverbOptions):
        self.weights_dir, self.opts = weights_dir, opts
        self._net = None
        self._codec = None

    @property
    def network(self):
        if self._net is None:
            from reverb_tts.model import Network
            self._net = Network.load(self.weights_dir).eval()
        return self._net

    @property
    def codec(self):
        if self._codec is None:
            tr = deps.require("transformers", "the Higgs Audio codec", "hf")
            rev = self.opts.codec_revision or (CODEC_REVISION if self.opts.codec == CODEC_ID else None)
            self._codec = tr.HiggsAudioV2TokenizerModel.from_pretrained(self.opts.codec, revision=rev).eval()
        return self._codec


@register_adapter
class ReverbAdapter(Adapter):
    name = "reverb-tts"
    description = "Reverb-TTS masked-diffusion TTS (Qwen3 + Higgs Audio v2 codec) -> 4 transformer graphs + 2 codec graphs"
    frameworks = ["pytorch", "transformers", "reverb-tts"]
    options_model = ReverbOptions
    priority = 50

    # -- detection / inspection -----------------------------------------------------------------------------------------------
    def detect(self, source: ModelSource) -> Detection:
        if source.kind not in ("local-dir", "huggingface"):
            return Detection(confidence=0.0, reason="not a directory or Hub repository")
        if source.reference == REPOSITORY:
            return Detection(confidence=1.0, reason=f"the official repository {REPOSITORY}")
        for v in VARIANTS:
            d = _weights_dir(source.path, v)
            if d is not None:
                try:
                    cfg = _read_cfg(d)
                except (OSError, json.JSONDecodeError):
                    continue
                if "llm_config" in cfg:
                    return Detection(confidence=0.95, reason=f"{d.relative_to(source.path) if d != source.path else '.'}/config.json has an 'llm_config' "
                                     "(Reverb checkpoint layout)")
        return Detection(confidence=0.0, reason="no Reverb checkpoint (config.json with 'llm_config') found")

    def _locate(self, source: ModelSource, opts: ReverbOptions) -> Path:
        if opts.variant not in VARIANTS:
            raise UnsupportedModelError(f"unknown variant '{opts.variant}', expected one of {VARIANTS}")
        d = _weights_dir(source.path, opts.variant)
        if d is None:
            raise UnsupportedModelError(f"no '{opts.variant}' weights (config.json + model.safetensors) under {source.path}")
        return d

    def weight_patterns(self, source: ModelSource, options: ReverbOptions) -> list[str] | None:
        # only the chosen variant + tokenizer + preset voices, not all four variants of the release
        return [f"model/safetensors/{options.variant}/*", "model/tokenizer*", "model/chat_template.jinja", "voices/*.json"]

    def inspect(self, source: ModelSource, options: ReverbOptions) -> ModelInfo:
        wd = self._locate(source, options)
        cfg = _read_cfg(wd)
        d = _dims(cfg["llm_config"])
        params = nbytes = None
        st = wd / "model.safetensors"
        if st.is_file():
            params, nbytes, dtypes = safetensors_header_summary(st)
        else:
            dtypes = set()
        comps = [ComponentInfo(name="core", role="transformer (prefix + target passes)", stateful=True,
                               description=f"{d['layers']} Qwen3 layers, hidden {d['hidden']}, KV {d['kv']}x{d['hd']}, explicit attention bias"),
                 ComponentInfo(name="embed_prefix", role="embedding", description="text ids + reference codes -> prefix embeddings"),
                 ComponentInfo(name="embed_target", role="embedding", description="8-codebook tokens -> target embeddings"),
                 ComponentInfo(name="heads", role="output heads", description="hidden -> log-probs [1,8,F,1024]")]
        if options.include_codec:
            comps += [ComponentInfo(name="codec_decoder", role="codec", description="codes -> 24 kHz audio (keep on CPU on some GPU stacks)"),
                      ComponentInfo(name="codec_encoder", role="codec", description="reference audio -> codes (voice cloning)")]
        return ModelInfo(reference=source.reference, source_kind=source.kind, revision=source.revision, framework="pytorch",
                         architecture="Reverb-TTS (masked-diffusion TTS)", model_type="reverb-tts", task="text-to-speech", parameters=params,
                         weight_bytes=nbytes, weight_dtypes=sorted(dtypes), license=source.license().id, components=comps,
                         extra={"variant": options.variant, "llm": d, "quantization": cfg.get("quantization"),
                                "sampler": "NOT exported: lives in the host (reverb_tts/sampler.py)"})

    def compatibility(self, source, info, options: ReverbOptions, config: JobConfig) -> CompatibilityReport:
        from ...core.registry import exporters
        issues: list[Issue] = []
        add = lambda sev, code, msg, hint=None: issues.append(Issue(severity=sev, code=code, message=msg, hint=hint))
        support = Support.supported
        if not deps.have("reverb_tts"):
            add(Severity.error, "reverb-tts-missing", "The reverb-tts package is required (network loader and the official sampler).", "pip install reverb-tts")
            support = Support.unsupported
        for mod in ("torch", "transformers", "onnx", "onnxruntime"):
            if not deps.have(mod):
                add(Severity.error, f"{mod}-missing", f"'{mod}' is not installed.", f"pip install {mod}")
                support = Support.unsupported
        q = (info.extra.get("quantization") or {})
        if q.get("rotation") or q.get("activations"):
            add(Severity.error, "unsupported-quantization", "Checkpoints with rotated projections or quantised activations are rejected by reverb_tts itself.",
                "use the fp32 / fp16 / int8 / int4 weight-only variants")
            support = Support.unsupported
        add(Severity.info, "sampler-not-exported", "The sampler (masking, remasking, polish guidance, length estimate) is not an ONNX graph; "
            "the host must implement it (reference: reverb_tts/sampler.py, defaults steps=8, remask 0.25, t_shift 0.1).")
        if options.include_codec:
            add(Severity.warning, "gpu-codec-decoder", WEBGPU_NOTE)
        add(Severity.warning, "license", "Reverb weights are CC BY-NC 4.0 (non-commercial, attribution); the converted graphs inherit it. The Higgs Audio v2 "
            "codec has its own license (Boson Higgs Audio 2 Community License / Llama 3 terms): check it before redistributing codec graphs.")
        est = memory_estimate(info)
        measured = int(3.4 * 2 ** 30) if options.include_codec else int(2.2 * 2 ** 30)           # peak seen with the real checkpoint in the original study
        est["export_peak_ram_bytes_estimate"] = min(measured, int((info.parameters or 0) * 4 * 2.5) + 400 * 2 ** 20) if info.parameters else measured
        import psutil
        est["fits_in_ram"] = est["export_peak_ram_bytes_estimate"] < psutil.virtual_memory().available
        if not est["fits_in_ram"]:
            add(Severity.warning, "ram", f"Export peaked at ~3.4 GiB in the original study; {psutil.virtual_memory().available / 2 ** 30:.1f} GiB is available now.")
        est["note"] = "core fp32 ~1.06 GB, codec encoder ~0.65 GB, codec decoder ~87 MB (original study)"
        status = {}
        for n in exporters.names():
            e = exporters.get(n)
            e = e() if isinstance(e, type) else e
            ok, why = e.availability()
            status[n] = "available" if ok else why
        if status.get("torch-legacy") != "available":
            add(Severity.error, "no-exporter", f"torch-legacy exporter unavailable: {status.get('torch-legacy')}")
            support = Support.unsupported
        return CompatibilityReport(target_format=config.export.format, support=support,
                                   summary="Convertible: 4 transformer graphs" + (" + 2 codec graphs" if options.include_codec else "") + " (sampler stays in the host).",
                                   issues=issues, exporters=status, strategy={c.name: ["torch-legacy", "torch-dynamo"] for c in info.components}, estimates=est)

    # -- plan -----------------------------------------------------------------------------------------------------------------------
    def plan(self, source: ModelSource, options: ReverbOptions, config: JobConfig) -> ExportPlan:
        if not deps.have("reverb_tts"):
            raise UnsupportedModelError("reverb-tts is not installed (pip install reverb-tts)")
        wd = self._locate(source, options)
        cfg = _read_cfg(wd)
        d = _dims(cfg["llm_config"])
        shared = _Shared(wd, options)
        H, L, KV, HD, V = d["hidden"], d["layers"], d["kv"], d["hd"], d["vocab"]
        Pt, R, F = options.prefix_text_tokens, options.reference_frames, options.target_frames
        P = Pt + R
        rng = lambda tag, seed=0: np.random.default_rng(zlib.crc32(tag.encode()) + 1_000_003 * seed + config.seed)   # deterministic across runs
        f32 = np.float32
        tight, loose = Tolerance(atol=2e-4, rtol=1e-3, min_cosine=0.99999), Tolerance(atol=1e-5, rtol=1e-4, min_cosine=0.999999)

        # ---- core -----------------------------------------------------------------------------------------------------------------
        def core_in(S: int, Pp: int, tag: str, polish: bool = False) -> Arrays:
            r = rng(tag)
            bias = np.zeros((S, Pp + S), f32)
            if polish:
                bias[:, max(Pp - R, 0):Pp] = 1.0           # attention bias towards the reference frames (sampler's polish_bias)
            return {"x": (r.standard_normal((1, S, H)) * 0.5).astype(f32),
                    "past_k": (r.standard_normal((L, 1, KV, Pp, HD)) * 0.5).astype(f32),
                    "past_v": (r.standard_normal((L, 1, KV, Pp, HD)) * 0.5).astype(f32), "bias": bias}

        def core_prefix_state() -> Arrays:
            return core_in(P, 0, "prefix")

        def core_step(i: int) -> Arrays:           # the target pass after the prefix pass: only x and bias are fresh; past comes from the link
            r = rng("target-step", i)
            return {"x": (r.standard_normal((1, F, H)) * 0.5).astype(f32), "bias": np.zeros((F, P + F), f32)}

        core = ComponentExport(
            name="core", role="transformer", description="24-layer Qwen3 stack shared by prefix and target passes; explicit KV + attention bias",
            build=lambda: _mod("CoreGraph", shared.network), example_inputs=lambda: core_in(F, P, "core-example"),
            input_names=["x", "past_k", "past_v", "bias"], output_names=["hidden", "new_k", "new_v"],
            dynamic_axes={"x": {1: "S"}, "past_k": {3: "Pp"}, "past_v": {3: "Pp"}, "bias": {0: "S", 1: "T"}, "hidden": {1: "S"}, "new_k": {3: "S"}, "new_v": {3: "S"}},
            cases=[ReferenceCase("target-pass", lambda: core_in(F, P, "core-target"), "parity", "target pass over the cached prefix"),
                   ReferenceCase("prefix-pass", lambda: core_in(P, 0, "core-prefix"), "parity", "prefix pass: empty past (Pp = 0)"),
                   ReferenceCase("polish-pass", lambda: core_in(F, P, "core-polish", True), "parity", "polish pass: bias towards the reference frames"),
                   ReferenceCase("longer-target", lambda: core_in(F + 13, P + 25, "core-long"), "dynamic_shape", "other target / prefix lengths")],
            stateful=[StatefulCase(name="prefix-then-target", steps=2, initial=core_prefix_state, step_inputs=core_step,
                                   links=[StateLink("new_k", "past_k", "replace"), StateLink("new_v", "past_v", "replace")],
                                   description="the prefix pass's K/V become the target pass's past (what the host does between calls)")],
            stateful_component=True, tolerance=tight, output_kinds={}, exporters=["torch-legacy", "torch-dynamo"],
            quantize_exclude_ops=[])

        # ---- embeddings and heads -----------------------------------------------------------------------------------------------
        def ep_in(pt: int, r_: int, tag: str) -> Arrays:
            r = rng(tag)
            return {"text_ids": r.integers(0, V, (1, pt)).astype(np.int64), "ref_codes": r.integers(0, MASK, (1, CODEBOOKS, r_)).astype(np.int64)}

        def et_in(f: int, tag: str) -> Arrays:
            return {"tokens": rng(tag).integers(0, CODES, (1, CODEBOOKS, f)).astype(np.int64)}   # includes the mask token (1024)

        def hd_in(f: int, tag: str) -> Arrays:
            return {"hidden": (rng(tag).standard_normal((1, f, H)) * 1.0).astype(f32)}

        embed_prefix = ComponentExport(
            name="embed_prefix", role="embedding", description="factorized text embedding + summed reference-code embeddings",
            build=lambda: _mod("EmbedPrefix", shared.network), example_inputs=lambda: ep_in(Pt, R, "ep-example"),
            input_names=["text_ids", "ref_codes"], output_names=["x"],
            dynamic_axes={"text_ids": {1: "Pt"}, "ref_codes": {2: "R"}, "x": {1: "P"}},
            cases=[ReferenceCase("example", lambda: ep_in(Pt, R, "ep-a"), "parity"),
                   ReferenceCase("longer", lambda: ep_in(Pt + 31, R + 77, "ep-b"), "dynamic_shape")], tolerance=loose)
        embed_target = ComponentExport(
            name="embed_target", role="embedding", description="sum of the 8 codebook embeddings (mask token allowed)",
            build=lambda: _mod("EmbedTarget", shared.network), example_inputs=lambda: et_in(F, "et-example"),
            input_names=["tokens"], output_names=["x"], dynamic_axes={"tokens": {2: "F"}, "x": {1: "F"}},
            cases=[ReferenceCase("example", lambda: et_in(F, "et-a"), "parity"), ReferenceCase("longer", lambda: et_in(F + 40, "et-b"), "dynamic_shape")],
            tolerance=loose)
        heads = ComponentExport(
            name="heads", role="output heads", description="8 codebook heads + log-softmax, mask logit dropped",
            build=lambda: _mod("Heads", shared.network), example_inputs=lambda: hd_in(F, "hd-example"),
            input_names=["hidden"], output_names=["log_probs"], dynamic_axes={"hidden": {1: "F"}, "log_probs": {2: "F"}},
            cases=[ReferenceCase("example", lambda: hd_in(F, "hd-a"), "parity"), ReferenceCase("longer", lambda: hd_in(F + 40, "hd-b"), "dynamic_shape")],
            tolerance=Tolerance(atol=1e-4, rtol=1e-3, min_cosine=0.99999), output_kinds={"log_probs": "logits"})
        components = [core, embed_prefix, embed_target, heads]

        # ---- codec ------------------------------------------------------------------------------------------------------------------
        if options.include_codec:
            T = 20

            def dec_in(t: int, tag: str) -> Arrays:
                return {"codes": rng(tag).integers(0, MASK, (1, CODEBOOKS, t)).astype(np.int64)}

            def enc_in(seconds: float, tag: str) -> Arrays:
                from ...exporters.torch_support import to_numpy, to_tensor
                from .network import resample_24_to_16
                n = int(seconds * 24000) // HOP * HOP
                if options.reference_wav:
                    w = _load_wav24(options.reference_wav)[:n]
                    w = np.pad(w, (0, n - len(w))) if len(w) < n else w
                else:   # deterministic speech-like signal: harmonic stack with a slow envelope and a little noise
                    t = np.arange(n) / 24000.0
                    env = 0.5 * (1 + np.sin(2 * np.pi * 3.0 * t))
                    w = sum(np.sin(2 * np.pi * f0 * t) / k for k, f0 in enumerate((140, 280, 420, 560, 700), 1)) * env * 0.2 + \
                        rng(tag).standard_normal(n) * 0.003
                w24 = w.astype(f32)[None, None]
                w16 = to_numpy(resample_24_to_16(to_tensor(w24)))
                return {"wav24": w24, "wav16": w16}

            decoder = ComponentExport(
                name="codec_decoder", role="codec", description="Higgs Audio v2 decoder: codes [1,8,T] -> audio [1,1,T*960] at 24 kHz",
                build=lambda: _mod("CodecDecoder", shared.codec), example_inputs=lambda: dec_in(T, "dec-example"),
                input_names=["codes"], output_names=["audio"], dynamic_axes={"codes": {2: "T"}, "audio": {2: "N"}},
                cases=[ReferenceCase("example", lambda: dec_in(T, "dec-a"), "parity"), ReferenceCase("longer", lambda: dec_in(T + 31, "dec-b"), "dynamic_shape")],
                reference=lambda inp: {"audio": _official_decode(shared.codec, inp["codes"])},
                tolerance=Tolerance(atol=1e-5, rtol=1e-4, min_cosine=0.99999),
                output_kinds={"audio": "audio"}, provider_notes={"webgpu": WEBGPU_NOTE},
                known_issues=["Do not run on the WebGPU/Vulkan EP on AMD RADV GFX1013: silent output (see provider_notes)."])
            encoder = ComponentExport(
                name="codec_encoder", role="codec", description="Higgs Audio v2 encoder: wav24 [1,1,N] + wav16 [1,N16] -> codes [1,8,T] (caller resamples to 16 kHz)",
                build=lambda: _build_encoder(shared.codec), example_inputs=lambda: enc_in(1.4, "enc-example"),
                input_names=["wav24", "wav16"], output_names=["codes"],
                dynamic_axes={"wav24": {2: "N"}, "wav16": {1: "N16"}, "codes": {2: "T"}},
                cases=[ReferenceCase("example", lambda: enc_in(1.4, "enc-a"), "parity", "1.4 s"),
                       ReferenceCase("longer", lambda: enc_in(3.0, "enc-b"), "dynamic_shape", "3 s")],
                reference=lambda inp: {"codes": _official_encode(shared.codec, inp["wav24"])},
                tolerance=Tolerance(min_equal_fraction=0.99), output_kinds={"codes": "discrete"},
                known_issues=["wav16 must come from torchaudio's resample (97-98 % code match with SciPy resample_poly in the original study)."])
            components += [decoder, encoder]

        topology = [Edge("embed_prefix", "x", "core", "x"), Edge("embed_target", "x", "core", "x"), Edge("core", "hidden", "heads", "hidden"),
                    Edge("core", "new_k", "core", "past_k"), Edge("core", "new_v", "core", "past_v")]
        if options.include_codec:
            topology += [Edge("codec_encoder", "codes", "embed_prefix", "ref_codes"), Edge("(host sampler)", "codes", "codec_decoder", "codes")]

        assets = [AssetSpec("config", wd / "config.json", "config.json", description="Reverb checkpoint config (llm_config, quantization)")]
        tok = _tokenizer_dir(wd, source.path)
        for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json", "chat_template.jinja"):
            if tok and (tok / name).is_file():
                assets.append(AssetSpec("tokenizer", tok / name, f"tokenizer/{name}", required=name == "tokenizer.json"))
        voices = source.path / "voices"
        if voices.is_dir():
            assets += [AssetSpec("voice", v, f"voices/{v.name}", required=False, description="preset voice (codes + transcript)") for v in sorted(voices.glob("*.json"))]
        assets.append(AssetSpec("postprocessor", None, "sampler", required=True,
                                description="Masked-diffusion sampler is NOT exported: implement it in the host (reference reverb_tts/sampler.py; defaults "
                                            "steps=8, remask_ratio=0.25, t_shift=0.1, polish_bias=1.0, polish_guidance=0.5)"))
        plan = ExportPlan(components=components, topology=topology, assets=assets, quality_checks=[self._sampler_check(shared, options, d)],
                          runtime_requirements=["onnxruntime>=1.18", "host-side sampler (see assets/sampler)", "torchaudio-compatible 24->16 kHz resampler for codec_encoder"],
                          notes=["Prefix pass: embed_prefix -> core(past empty, bias zeros). Target passes: embed_target -> core(past = prefix K/V, bias 0 or polish) -> heads; "
                                 "discard core's new_k/new_v for target passes."])
        if options.include_codec:
            plan.component_licenses = {"codec_decoder": "llama3", "codec_encoder": "llama3"}
        return plan

    # -- quality check: the official sampler through the ONNX graphs ----------------------------------------------------------------------
    def _sampler_check(self, shared: _Shared, options: ReverbOptions, d: dict[str, int]) -> QualityCheck:
        def run(ctx: QualityContext) -> QualityResult:
            import torch
            from reverb_tts.sampler import Batch, Prompt, SamplerConfig, sample
            torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
            net = shared.network
            cfg = SamplerConfig(steps=options.quality_steps)
            r = np.random.default_rng(ctx.seed + 11)
            R, F = options.reference_frames, options.target_frames + 8
            ids = torch.from_numpy(r.integers(0, d["vocab"], options.prefix_text_tokens + 6)).long()
            ref = torch.from_numpy(r.integers(0, MASK, (CODEBOOKS, R))).long()
            p = Prompt(ids, ref, F, 7)
            with torch.inference_mode():
                want = sample(Batch(net, [p], cfg.polish_bias), [7], cfg)[0]
            got = sample(_OnnxBatch(ctx, p, cfg.polish_bias, d), [7], cfg)[0]
            per_cb = [round(float((want[c] == got[c]).float().mean()) * 100, 2) for c in range(CODEBOOKS)]
            agree = float((want == got).float().mean())
            exact = ctx.variant == "fp32" or ctx.variant.startswith("opt-")
            need = 1.0 if exact else options.quality_min_agreement_lowprec
            return QualityResult(agree >= need, {"code_agreement": agree, "per_codebook_percent": per_cb, "required": need, "frames": F, "steps": cfg.steps},
                                 f"official sampler through the ONNX graphs reproduces {agree * 100:.2f}% of the original codes (required {need * 100:.0f}%)")
        return QualityCheck("sampler-code-agreement", run, "official reverb_tts sampler driven through the ONNX graphs vs through PyTorch: codes must match")


class _OnnxBatch:
    """Drop-in for reverb_tts.sampler.Batch (batch size 1) backed by the exported graphs. Port of scripts/parity_onnx.py::OnnxBatch."""

    def __init__(self, ctx: QualityContext, p, polish_bias: float, d: dict[str, int]):
        import torch
        self.run = {n: ctx.runner(n).run for n in ("embed_prefix", "embed_target", "core", "heads")}
        self.n, self.device, self.frames, self.F = 1, torch.device("cpu"), [p.frames], p.frames
        self.target = torch.ones(1, self.F, dtype=torch.bool)
        x = self.run["embed_prefix"]({"text_ids": p.ids[None].numpy().astype(np.int64), "ref_codes": p.reference[None].numpy().astype(np.int64)})["x"]
        P, R = x.shape[1], p.reference.shape[1]
        empty = np.zeros((d["layers"], 1, d["kv"], 0, d["hd"]), np.float32)
        o = self.run["core"]({"x": x, "past_k": empty, "past_v": empty, "bias": np.zeros((P, P), np.float32)})
        self.k, self.v = o["new_k"], o["new_v"]
        self.bias = {}
        for pol in (False, True):
            b = np.zeros((self.F, P + self.F), np.float32)
            if pol:
                b[:, P - R:P] = polish_bias
            self.bias[pol] = b

    def log_probs(self, tokens, codebooks, polish: bool = False):
        import torch
        x = self.run["embed_target"]({"tokens": tokens.numpy().astype(np.int64)})["x"]
        h = self.run["core"]({"x": x, "past_k": self.k, "past_v": self.v, "bias": self.bias[polish]})["hidden"]
        lp = self.run["heads"]({"hidden": h})["log_probs"]
        return torch.from_numpy(lp)[:, codebooks.start:codebooks.stop]


# ------------------------------------------------------------------------------------------------------------------------ helpers
def _mod(cls: str, model):
    from . import network
    return getattr(network, cls)(model).eval()


def _load_wav24(path: str) -> np.ndarray:
    sf = deps.require("soundfile", "reading the reference wav", "reverb", package="soundfile")
    from ...exporters.torch_support import to_numpy, to_tensor
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(1)
    if sr != 24000:
        import torchaudio
        a = to_numpy(torchaudio.functional.resample(to_tensor(a), sr, 24000))
    return a


def _official_decode(codec, codes: np.ndarray) -> np.ndarray:
    import torch
    with torch.inference_mode():
        return codec.decode(torch.from_numpy(codes)).audio_values.float().numpy()


def _official_encode(codec, wav24: np.ndarray) -> np.ndarray:
    import torch
    with torch.inference_mode():
        return codec.encode(torch.from_numpy(wav24)).audio_codes.numpy()


def _build_encoder(codec):
    """The encoder wrapper's `pad_acoustic` flag depends on the codec version: pick the setting that reproduces the official codes
    (the original export script does the same probing)."""
    import torch
    from . import network
    from ...exporters.torch_support import to_tensor
    n = 24000 // HOP * HOP
    w24 = torch.sin(torch.arange(n).float() * 0.05)[None, None] * 0.1
    w16 = network.resample_24_to_16(w24)
    with torch.inference_mode():
        want = codec.encode(w24).audio_codes
        best, best_ok = None, -1.0
        for cand in (False, True):
            try:
                ok = (network.CodecEncoder(codec, cand).eval()(w24, w16) == want).float().mean().item()
            except RuntimeError:
                ok = -1.0
            if ok > best_ok:
                best, best_ok = cand, ok
    if best_ok < 0.99:
        raise UnsupportedModelError(f"the codec encoder wrapper reproduces only {best_ok * 100:.1f}% of the official codes with either padding mode")
    return network.CodecEncoder(codec, best).eval()
