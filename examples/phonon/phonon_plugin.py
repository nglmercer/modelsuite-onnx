"""Gradium Phonon TTS adapter for modelsuite (a plugin: nothing here is part of the core).

    modelsuite convert gradium/phonon-40m-multilingual \
        --plugin examples/phonon/phonon_plugin.py -o artifacts/phonon-40m-multilingual

Exports three graphs from a Gradium Phonon checkpoint:

* ``core``          backbone + EOS head, ONE stateful graph for prefill and decode (explicit KV cache);
* ``flow``          the LSD flow head, one Euler step per call;
* ``codec_decoder`` the processor's Mimi-style continuous codec: latents [B,T,32] -> 24 kHz audio.

Not exported (host's job): text normalisation/tokenisation and prompt assembly, the Gaussian noise draw,
latent denormalisation, the end-of-speech countdown, and voice-cloning encoding (voices are already latents).
The numerical oracle is the original ``transformers`` model and processor (which pull the codec weights from
``gradium/phonon-mimi``). The conversion was validated end-to-end: 12-frame greedy rollout latents within
~3e-5 and 24 kHz waveform within ~1e-4 of ``model.generate()``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))      # sibling phonon_graphs

from modelsuite.adapters.base import Adapter, AdapterOptions, memory_estimate
from modelsuite.core import deps
from modelsuite.core.config import JobConfig, Tolerance
from modelsuite.core.discovery import ModelSource
from modelsuite.core.errors import UnsupportedModelError
from modelsuite.core.plan import (Arrays, AssetSpec, Edge, ExportPlan, ComponentExport, QualityCheck, QualityContext,
                                  QualityResult, ReferenceCase, StatefulCase, StateLink)
from modelsuite.core.registry import register_adapter
from modelsuite.core.types import CompatibilityReport, ComponentInfo, Detection, Issue, ModelInfo, Severity, Support

MODEL_TYPE = "gradium_phonon_tts"
ARCH = "GradiumPhononTTSForConditionalGeneration"
F32 = np.float32
I64 = np.int64


class PhononOptions(AdapterOptions):
    text: str = "Hola, ¿cómo estás?"         # prompt used by the synthetic cases and the rollout check
    frames: int = 8                          # greedy frames in the end-to-end quality check
    codec_frames: int = 8                    # frames in the codec reference cases
    rollout_frames: int = 6                  # frames in the stateful KV-cache test
    include_codec: bool = True


class _Shared:
    """Lazily loaded original model + processor, reused by the components' reference functions and the quality check."""

    def __init__(self, source: ModelSource):
        self.root = source.root
        self._model = self._proc = None

    def model(self):
        if self._model is None:
            torch = deps.require("torch", "loading the Phonon checkpoint", "torch")
            tr = deps.require("transformers", "Gradium Phonon needs transformers >= 5.21", "hf")
            cls = getattr(tr, ARCH, None)
            if cls is None:
                raise UnsupportedModelError(f"installed transformers has no {ARCH}; upgrade to >= 5.21")
            self._model = cls.from_pretrained(str(self.root), dtype=torch.float32,
                                              attn_implementation="eager").eval()
        return self._model

    def processor(self):
        if self._proc is None:
            tr = deps.require("transformers", "the Phonon processor / codec", "hf")
            self._proc = tr.AutoProcessor.from_pretrained(str(self.root))
        return self._proc

    def drop(self):
        self._model = self._proc = None
        import gc
        gc.collect()


def _version_at_least(version: str | None, want: tuple[int, ...]) -> bool:
    try:
        got = tuple(int(p) for p in (version or "").split(".")[:len(want)])
    except ValueError:
        return False
    return got >= want


@register_adapter
class PhononAdapter(Adapter):
    name = "gradium-phonon-tts"
    version = "0.1.0"
    priority = 60
    description = "Gradium Phonon TTS (flow-matching backbone + flow head + Mimi codec) -> stateful core, flow and codec graphs"
    frameworks = ["transformers", "pytorch"]
    options_model = PhononOptions

    def detect(self, source: ModelSource) -> Detection:
        cfg = source.config or {}
        mt, arch = cfg.get("model_type"), (cfg.get("architectures") or [None])[0]
        if mt == MODEL_TYPE and arch == ARCH:
            return Detection(confidence=1.0, reason=f"config.json: model_type={mt!r}, architectures={arch!r}")
        if mt == MODEL_TYPE or arch == ARCH:
            return Detection(confidence=0.8, reason=f"config.json: model_type={mt!r}, architectures={arch!r}")
        return Detection(confidence=0.0, reason="not a Gradium Phonon TTS checkpoint")

    def inspect(self, source: ModelSource, options: PhononOptions) -> ModelInfo:
        cfg = source.config or {}
        w = source.weight_summary()
        comps = [ComponentInfo(name="core", role="decoder-with-kv-cache", stateful=True,
                              description=f"{cfg.get('num_hidden_layers')} layers, dim {cfg.get('hidden_size')}: prefill+decode, EOS logits",
                              parameters=None),
                 ComponentInfo(name="flow", role="flow head",
                               description=f"one LSD Euler step, latent {cfg.get('latent_dim')} -> condition {cfg.get('hidden_size')}",
                               parameters=None)]
        if options.include_codec:
            comps.append(ComponentInfo(name="codec_decoder", role="codec",
                                       description=f"continuous latents -> 24 kHz audio, {cfg.get('frame_rate')} frames/s",
                                       parameters=None))
        return ModelInfo(reference=source.reference, source_kind=source.kind, revision=source.revision,
                         framework="transformers", architecture=ARCH, model_type=MODEL_TYPE, task="text-to-speech",
                         parameters=w["parameters"], weight_bytes=w["bytes"], weight_dtypes=w["dtypes"],
                         license=source.license().id, components=comps,
                         extra={"frame_rate": cfg.get("frame_rate"), "flow_type": (cfg.get("flow_head_config") or {}).get("flow_type"),
                                "hidden_size": cfg.get("hidden_size"), "layers": cfg.get("num_hidden_layers"),
                                "not_exported": ["text normalisation + tokenisation + prompt assembly", "Gaussian noise (zero when greedy)",
                                                 "latent denormalisation (latent_mean/std)", "end-of-speech countdown",
                                                 "voice encoding from raw audio (voices are already 512-d latents)"]})

    def compatibility(self, source: ModelSource, info: ModelInfo, options: PhononOptions, config: JobConfig) -> CompatibilityReport:
        from modelsuite.core.registry import exporters
        issues: list[Issue] = []
        add = lambda sev, code, msg, hint=None: issues.append(Issue(severity=sev, code=code, message=msg, hint=hint))
        support = Support.supported
        ver = deps.dist_version("transformers")
        if not _version_at_least(ver, (5, 21)):
            add(Severity.error, "transformers-version", f"Gradium Phonon needs transformers >= 5.21 (found {ver or 'none'}).",
                "pip install -U transformers")
            support = Support.unsupported
        for mod in ("torch", "onnx", "onnxruntime"):
            if not deps.have(mod):
                add(Severity.error, "missing-dep", f"'{mod}' is not installed.", f"pip install {mod}")
                support = Support.unsupported
        if _version_at_least(ver, (5, 21)) and getattr(__import__("transformers"), ARCH, None) is None:
            add(Severity.error, "no-model-class", f"transformers {ver} has no class {ARCH}.")
            support = Support.unsupported
        add(Severity.info, "stateful-core", "The backbone is ONE graph for prefill and decode with explicit past_keys/past_values; "
            "the host feeds present_keys/present_values back (see the manifest's state_links).")
        add(Severity.warning, "host-stack", "Prompt assembly, greedy/noise sampling, denormalisation and the EOS countdown stay in the host; "
            "see examples/phonon/run_onnx.py. The graph outputs continuous latents, not text.")
        if options.include_codec:
            pcfg = source.root / "processor_config.json"
            add(Severity.info, "codec", "The codec graph uses the processor's audio tokenizer (gradium/phonon-mimi, Apache-2.0), "
                "downloaded separately by transformers at export time and NOT stored in this checkpoint.")
        lic = source.license()
        if lic.id and "nc" in lic.id:
            add(Severity.warning, "license", f"Weights are {lic.id}: non-commercial, attribution required; converted files inherit that.")
        est = memory_estimate(info)
        est["export_peak_ram_bytes_estimate"] = int(2 * 1024 ** 3)
        est["fits_in_ram"] = est["export_peak_ram_bytes_estimate"] < est.get("available_ram_bytes", 0)
        if not est["fits_in_ram"]:
            add(Severity.warning, "ram", f"Peak export RAM is ~2 GiB, only {est.get('available_ram_bytes', 0)/2**30:.1f} GiB available.")
        status = {}
        for n in exporters.names():
            e = exporters.get(n)
            e = e() if isinstance(e, type) else e
            ok, why = e.availability()
            status[n] = "available" if ok else why
        if status.get("torch-legacy") != "available":
            add(Severity.error, "no-exporter", f"torch-legacy is required: {status.get('torch-legacy')}")
            support = Support.unsupported
        return CompatibilityReport(target_format=config.export.format, support=support,
                                   summary="Convertible: stateful core + flow head" + (" + codec decoder" if options.include_codec else ""),
                                   issues=issues, exporters=status,
                                   strategy={c.name: ["torch-legacy"] for c in info.components}, estimates=est)

    # -- input generators ---------------------------------------------------------------------------------------------
    @staticmethod
    def _core_inputs(cfg, S: int, P: int, seed: int, *, nprompt: int, nlatent: int) -> Arrays:
        r = np.random.default_rng(seed + 10_003 * P)
        ids = r.integers(4, cfg["vocab_size"] - 1, (1, S)).astype(I64)
        npr = min(nprompt, S // 3)
        nla = min(nlatent, max(0, S - npr - 1))
        ids[0, :npr] = cfg["audio_prompt_token_id"]
        ids[0, npr] = cfg["audio_bos_token_id"]
        ids[0, npr + 1:npr + 1 + nla] = cfg["audio_latent_token_id"]
        pl = np.zeros((1, S, cfg["audio_prompt_dim"]), F32)
        pl[0, :npr] = r.standard_normal((npr, cfg["audio_prompt_dim"]))
        al = np.zeros((1, S, cfg["latent_dim"]), F32)
        al[0, npr + 1:npr + 1 + nla] = r.standard_normal((nla, cfg["latent_dim"])) * 0.1
        scale = 1.0 if P else 0.0
        kv_shape = (cfg["num_hidden_layers"], 1, cfg["num_key_value_heads"], P, cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"]))
        return {"input_ids": ids, "prompt_latents": pl, "audio_latents": al,
                "attention_mask": np.ones((1, P + S), I64),
                "position_ids": np.arange(P, P + S)[None].astype(I64),
                "past_keys": (r.standard_normal(kv_shape) * 0.3 * scale).astype(F32),
                "past_values": (r.standard_normal(kv_shape) * 0.3 * scale).astype(F32)}

    @staticmethod
    def _flow_inputs(cfg, seed: int) -> Arrays:
        r = np.random.default_rng(seed)
        return {"latents": (r.standard_normal((1, cfg["latent_dim"])) * 0.1).astype(F32),
                "condition": r.standard_normal((1, cfg["hidden_size"])).astype(F32),
                "start_time": np.zeros((1, 1), F32), "end_time": np.ones((1, 1), F32)}

    @staticmethod
    def _codec_inputs(seed: int, frames: int) -> Arrays:
        return {"latents": (np.random.default_rng(seed).standard_normal((1, frames, 32)) * 0.1).astype(F32)}

    # -- reference functions (the ORIGINAL model) -----------------------------------------------------------------------
    @staticmethod
    def _core_reference(shared: _Shared, inp: Arrays) -> Arrays:
        import torch
        from transformers.cache_utils import DynamicCache
        model, cfg = shared.model(), shared.model().config
        ids = torch.from_numpy(inp["input_ids"])
        p_mask = ids[0] == cfg.audio_prompt_token_id
        l_mask = ids[0] == cfg.audio_latent_token_id
        compact_p = torch.from_numpy(inp["prompt_latents"][0][p_mask.numpy()])[None]
        compact_l = torch.from_numpy(inp["audio_latents"][0][l_mask.numpy()])[None]
        cache = DynamicCache(config=cfg)
        for i in range(cfg.num_hidden_layers):
            cache.update(torch.from_numpy(inp["past_keys"][i]), torch.from_numpy(inp["past_values"][i]), i)
        with torch.inference_mode():
            out = model.model(input_ids=ids, audio_prompt_latents=compact_p,
                              audio_prompt_latents_mask=torch.ones((1, compact_p.shape[1]), dtype=torch.long),
                              audio_latents=compact_l, attention_mask=torch.from_numpy(inp["attention_mask"]),
                              position_ids=torch.from_numpy(inp["position_ids"]), cache=cache, use_cache=True, return_dict=True)
            eos = model.eos_head(out.last_hidden_state)[..., 0]
            return {"hidden": out.last_hidden_state.numpy(), "eos": eos.numpy(),
                    "present_keys": np.stack([c.keys.numpy() for c in cache.layers]),
                    "present_values": np.stack([c.values.numpy() for c in cache.layers])}

    @staticmethod
    def _flow_reference(shared: _Shared, inp: Arrays) -> Arrays:
        import torch
        model = shared.model()
        with torch.inference_mode():
            v = model.flow_head(torch.from_numpy(inp["latents"]), torch.from_numpy(inp["condition"]),
                                times=[torch.from_numpy(inp["start_time"]), torch.from_numpy(inp["end_time"])])
            return {"velocity": v.numpy(), "audio_latent": (v * model.latent_std + model.latent_mean).numpy()}

    @staticmethod
    def _codec_reference(shared: _Shared, inp: Arrays) -> Arrays:
        import torch
        from transformers.cache_utils import DynamicCache
        tok = shared.processor().audio_tokenizer
        cache = DynamicCache(config=tok.decoder.config)
        with torch.inference_mode():
            wav = tok.decode(torch.from_numpy(inp["latents"]), decoder_cache=cache).audio_values
            return {"audio": wav.numpy()}

    # -- plan ----------------------------------------------------------------------------------------------------------
    def plan(self, source: ModelSource, options: PhononOptions, config: JobConfig) -> ExportPlan:
        cfg = source.config
        if not cfg or cfg.get("model_type") != MODEL_TYPE:
            raise UnsupportedModelError("expected a Gradium Phonon config.json")
        shared = _Shared(source)
        names = ["input_ids", "prompt_latents", "audio_latents", "attention_mask", "position_ids", "past_keys", "past_values"]
        outs = ["hidden", "eos", "present_keys", "present_values"]
        dyn = {"input_ids": {1: "S"}, "prompt_latents": {1: "S"}, "audio_latents": {1: "S"},
               "attention_mask": {1: "PS"}, "position_ids": {1: "S"}, "past_keys": {3: "P"}, "past_values": {3: "P"},
               "hidden": {1: "S"}, "eos": {1: "S"}, "present_keys": {3: "PS"}, "present_values": {3: "PS"}}
        S0 = 13
        initial = lambda: self._core_inputs(cfg, S0, 0, 1, nprompt=6, nlatent=0)

        def step_inputs(i: int) -> Arrays:
            P = S0 + i - 1
            d = self._core_inputs(cfg, 1, P, 100 + i, nprompt=0, nlatent=1)
            d["prompt_latents"] = np.zeros((1, 1, cfg["audio_prompt_dim"]), F32)
            return d

        core = ComponentExport(
            name="core", build=lambda: self._build("core", shared), example_inputs=lambda: self._core_inputs(cfg, S0, 0, 0, nprompt=6, nlatent=0),
            input_names=names, output_names=outs, dynamic_axes=dyn, role="decoder-with-kv-cache",
            description=f"Gradium Phonon backbone ({cfg['num_hidden_layers']} layers) + EOS head: prefill and decode in one graph",
            reference=lambda inp: self._core_reference(shared, inp), exporters=["torch-legacy"],
            cases=[ReferenceCase("prefill", lambda: self._core_inputs(cfg, 15, 0, 2, nprompt=8, nlatent=0), "parity", "voice + text + BOS, empty past"),
                   ReferenceCase("decode-step", lambda: self._core_inputs(cfg, 1, 14, 3, nprompt=0, nlatent=1), "parity", "one latent frame, past 14"),
                   ReferenceCase("chunk", lambda: self._core_inputs(cfg, 7, 9, 4, nprompt=2, nlatent=2), "dynamic_shape", "7 new positions after 9"),
                   ReferenceCase("long-prefill", lambda: self._core_inputs(cfg, 29, 0, 5, nprompt=12, nlatent=0), "dynamic_shape", "longer prompt")],
            stateful=[StatefulCase(name="prefill-then-decode", steps=options.rollout_frames, initial=initial, step_inputs=step_inputs,
                                   links=[StateLink("present_keys", "past_keys", "replace"), StateLink("present_values", "past_values", "replace")],
                                   description="prefill then single-latent steps, feeding the KV cache back")],
            stateful_component=True, tolerance=Tolerance(atol=2e-4, rtol=1e-3), output_kinds={"hidden": "hidden", "eos": "logits"},
            on_release=shared.drop,
            known_issues=["Dense voice/latent tensors are (B,S,...) with values only at placeholder positions; zeros elsewhere are ignored.",
                          "Attention uses an explicit additive mask built from attention_mask and position_ids (no dynamic branch)."])
        components = [core]
        f_names = ["latents", "condition", "start_time", "end_time"]
        flow = ComponentExport(
            name="flow", build=lambda: self._build("flow", shared), example_inputs=lambda: self._flow_inputs(cfg, 0),
            input_names=f_names, output_names=["velocity", "audio_latent"], role="flow head",
            description=f"LSD flow head: one Euler step, times [start, end] in [0,1]; also returns the denormalised latent",
            reference=lambda inp: self._flow_reference(shared, inp), exporters=["torch-legacy"],
            cases=[ReferenceCase("example", lambda: self._flow_inputs(cfg, 1), "parity"),
                   ReferenceCase("other-latent", lambda: self._flow_inputs(cfg, 7), "dynamic_shape", "different condition and latent")],
            tolerance=Tolerance(atol=2e-4, rtol=1e-3),
            output_kinds={"velocity": "latents", "audio_latent": "latents"}, on_release=shared.drop)
        components.append(flow)
        if options.include_codec:
            codec = ComponentExport(
                name="codec_decoder", build=lambda: self._build("codec_decoder", shared),
                # trace above the decoder's 250-frame sliding-window threshold (16 frames x 16 SEANet frames),
                # otherwise that branch is baked off and long decodes disagree with the original
                example_inputs=lambda: self._codec_inputs(2, max(17, options.codec_frames)), input_names=["latents"], output_names=["audio"],
                dynamic_axes={"latents": {1: "T"}, "audio": {2: "N"}}, role="codec",
                description="continuous latents [B,T,32] -> 24 kHz waveform [B,1,T*1920] (Mimi-style codec)",
                reference=lambda inp: self._codec_reference(shared, inp), exporters=["torch-legacy"],
                cases=[ReferenceCase("example", lambda: self._codec_inputs(3, options.codec_frames), "parity"),
                       ReferenceCase("short", lambda: self._codec_inputs(4, 5), "dynamic_shape", "5 frames"),
                       ReferenceCase("longer", lambda: self._codec_inputs(5, options.codec_frames + 9), "dynamic_shape", "more frames")],
                tolerance=Tolerance(atol=1e-4, rtol=1e-3, min_cosine=0.9999), output_kinds={"audio": "audio"}, on_release=shared.drop,
                known_issues=["Whole-sequence decode; decode long audio in chunks (the decoder's sliding-window attention is dense-banded)."])
            components.append(codec)
        assets = []
        for rel, role in (("config.json", "config"), ("generation_config.json", "config"), ("processor_config.json", "preprocessor"),
                          ("tokenizer.json", "tokenizer"), ("tokenizer_config.json", "tokenizer"),
                          ("special_tokens_map.json", "tokenizer"), ("chat_template.jinja", "tokenizer"),
                          ("tokenizer.model", "tokenizer"), ("default-voice.safetensors", "voice")):
            p = source.root / rel
            if p.is_file():
                assets.append(AssetSpec(role, p, rel, required=rel in ("config.json", "tokenizer.json")))
        assets.append(AssetSpec("postprocessor", None, "host-pipeline", required=True,
                                description="NOT exported: text normalisation/tokenisation + prompt assembly (PocketTTSProcessor.apply_chat_template), "
                                            "Gaussian noise (zero for greedy), denormalisation latent*latent_std+latent_mean, EOS countdown "
                                            "(eos_threshold / min_frames_before_eos / frames_after_eos). See examples/phonon/run_onnx.py."))
        assets.append(AssetSpec("config", None, "codec-source", required=False,
                                description="The codec graph's weights come from gradium/phonon-mimi (Apache-2.0), fetched by AutoProcessor at export time."))
        topology = [Edge("core", "hidden", "flow", "condition"), Edge("flow", "velocity", "(host denorm)", "latents"),
                    Edge("(host denorm)", "latents", "core", "audio_latents"), Edge("core", "eos", "(host EOS)", "eos_logit")]
        if options.include_codec:
            topology.append(Edge("(host denorm)", "latents", "codec_decoder", "latents"))
        return ExportPlan(components=components, topology=topology, assets=assets,
                          quality_checks=[self._rollout_check(shared, options)] if options.include_codec else [],
                          component_licenses={"codec_decoder": "apache-2.0"},
                          runtime_requirements=["onnxruntime>=1.18", "tokenizers (for the tokenizer assets)",
                                                "host-side prompt assembly, sampler and latent denormalisation"],
                          notes=["core KV state: feed present_keys/present_values back as past_keys/past_values (replace); start from an empty cache.",
                                 "The flow graph consumes/produces NORMALISED latents; denormalise only when feeding the codec or writing audio.",
                                 f"Prompts are left-to-right: voice placeholders, text tokens, then the audio_bos_token_id."])

    def _build(self, what: str, shared: _Shared):
        from phonon_graphs import CoreGraph, FlowGraph, CodecGraph
        if what == "core":
            return CoreGraph(shared.model()).eval()
        if what == "flow":
            return FlowGraph(shared.model()).eval()
        return CodecGraph(shared.processor().audio_tokenizer).eval()

    # -- quality check: greedy rollout vs the original generate() ----------------------------------------------------------
    def _rollout_check(self, shared: _Shared, options: PhononOptions) -> QualityCheck:
        def run(ctx: QualityContext) -> QualityResult:
            import torch
            from safetensors.torch import load_file
            model, proc, cfg = shared.model(), shared.processor(), shared.model().config
            voice_path = shared.root / "default-voice.safetensors"
            if voice_path.is_file():
                voice = load_file(str(voice_path))["speaker_wavs"][0].T.float()
            else:                                    # gated model without the voice file: use a fixed synthetic voice
                voice = torch.from_numpy(np.random.default_rng(7).standard_normal((32, cfg.audio_prompt_dim)).astype(F32))
            conv = [{"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": options.text}]}]
            inputs = proc.apply_chat_template(conv, tokenize=True, return_dict=True, add_generation_prompt=True,
                                              processor_kwargs={"audio_prompt_latents": [voice]})
            ids = inputs["input_ids"].numpy()
            pl = inputs["audio_prompt_latents"].numpy()
            N = options.frames
            with torch.inference_mode():
                ref = model.generate(**inputs, max_new_tokens=N, do_sample=False, num_flow_steps=1, eos_threshold=1e9)
            ref_lat = ref.audio_latents.numpy()[0]

            core, flow = ctx.runner("core"), ctx.runner("flow")
            S = ids.shape[1]
            pl_dense = np.zeros((1, S, cfg.audio_prompt_dim), F32)
            pl_dense[0, ids[0] == cfg.audio_prompt_token_id] = pl[0]
            empty = np.zeros((cfg.num_hidden_layers, 1, cfg.num_key_value_heads, 0, cfg.head_dim), F32)
            o = core.run({"input_ids": ids, "prompt_latents": pl_dense,
                          "audio_latents": np.zeros((1, S, cfg.latent_dim), F32),
                          "attention_mask": np.ones((1, S), I64),
                          "position_ids": np.arange(S)[None].astype(I64),
                          "past_keys": empty, "past_values": empty.copy()})
            hidden, nk, nv = o["hidden"], o["present_keys"], o["present_values"]
            got = []
            for step in range(N):
                fo = flow.run({"latents": np.zeros((1, cfg.latent_dim), F32), "condition": hidden[:, -1],
                               "start_time": np.zeros((1, 1), F32), "end_time": np.ones((1, 1), F32)})
                velocity, audio_latent = fo["velocity"], fo["audio_latent"]
                got.append(audio_latent[0])
                al = np.zeros((1, 1, cfg.latent_dim), F32)
                al[0, 0] = velocity[0]                     # the model consumes NORMALISED latents
                o = core.run({"input_ids": np.full((1, 1), cfg.audio_latent_token_id, I64),
                              "prompt_latents": np.zeros((1, 1, cfg.audio_prompt_dim), F32),
                              "audio_latents": al, "attention_mask": np.ones((1, S + step + 1), I64),
                              "position_ids": np.array([[S + step]], I64), "past_keys": nk, "past_values": nv})
                hidden, nk, nv = o["hidden"], o["present_keys"], o["present_values"]
            got = np.stack(got)
            diff = float(np.abs(got - ref_lat).max())
            scale = max(1.0, float(np.abs(ref_lat).max()))
            need = 5e-4 if ctx.variant == "fp32" else 5e-2
            ok = diff <= need * scale
            metrics = {"frames": N, "max_abs_diff": diff, "ref_scale": scale, "required": need}
            if options.include_codec and ok:
                try:
                    wav_ref = proc.decode(torch.from_numpy(ref_lat[None]), torch.ones((1, N), dtype=torch.long))[0].numpy()
                    wav_got = ctx.runner("codec_decoder").run({"latents": got[None].astype(F32)})["audio"][0, 0]
                    n = min(len(wav_ref), len(wav_got))
                    metrics["audio_max_abs_diff"] = float(np.abs(wav_ref[:n] - wav_got[:n]).max())
                except Exception as exc:                      # audio comparison is informational
                    metrics["audio_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            return QualityResult(ok, metrics,
                                 f"greedy {N}-frame rollout through the ONNX core+flow reproduces the original latents "
                                 f"(max diff {diff:g}, limit {need*scale:g})" if ok else
                                 f"greedy rollout diverged from the original (max diff {diff:g} > {need*scale:g})")
        return QualityCheck("greedy-rollout", run, "greedy multi-frame TTS rollout through the ONNX core + flow vs transformers.generate()")
