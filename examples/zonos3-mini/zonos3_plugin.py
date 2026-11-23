"""ZONOS3-Mini adapter for modelsuite (a plugin: nothing here is part of the core).

    modelsuite convert Zyphra/ZONOS3-Mini --plugin examples/zonos3-mini/zonos3_plugin.py -o artifacts/zonos3-mini

Exports two graphs from Zyphra's release:

* ``lm``            the 800M-parameter hybrid Gated-DeltaNet / attention / latent-MoE language model, as ONE stateful graph that serves
                    prefill and decode (explicit KV cache, GDN matrices and conv history in and out);
* ``codec_decoder`` codes [1, 10, T] -> 44.1 kHz audio, with the multi-resolution iSTFT head written without complex tensors.

Everything else a TTS server needs (text normalisation, SuperBPE tokenizer, prompt assembly with the inter-codebook delay, sampling, delay
removal, speaker encoder, SSL teacher, aligner, edit LoRA) is *not* exported; see the README. The numerical oracle is Zyphra's own
plain-PyTorch reference (`zonos3.reference.lm`, `zonos3.codec`), imported from a clone via ZONOS3_SRC (see reference_env.py)."""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))      # sibling modules: reference_env, lm_graph, codec_graph, tiny_checkpoint

from modelsuite.adapters.base import Adapter, AdapterOptions, memory_estimate
from modelsuite.core import deps
from modelsuite.core.config import JobConfig, Tolerance
from modelsuite.core.discovery import ModelSource
from modelsuite.core.errors import UnsupportedModelError
from modelsuite.core.plan import (Arrays, AssetSpec, ComponentExport, Edge, ExportPlan, QualityCheck, QualityContext, QualityResult,
                                  ReferenceCase, StatefulCase, StateLink)
from modelsuite.core.registry import register_adapter
from modelsuite.core.types import CompatibilityReport, ComponentInfo, Detection, Issue, ModelInfo, Severity, Support

REPOSITORY = "Zyphra/ZONOS3-Mini"
GDN_HEADS, GDN_HEAD_DIM, GDN_KERNEL = 4, 128, 4


class Zonos3Options(AdapterOptions):
    source_dir: str | None = None          # <ZONOS3-Mini clone>/model/src (else $ZONOS3_SRC, else an installed `zonos3`)
    lm: str = "lm/zonos3_mini"             # LM directory inside the release (params.json + model.pth)
    include_codec: bool = True
    allow_pickle: bool = False             # the decoder-head checkpoint may need full pickle loading: off unless you trust the file
    text_rows: str | None = None           # optional path of a zonos-text-extra .pt (phoneme / start / pause rows) appended to the text table
    prefill_rows: int = 24                 # sizes of the synthetic cases
    codec_frames: int = 24
    rollout_steps: int = 6                 # greedy rollout of the quality check


def _lm_dir(root: Path, opts: Zonos3Options) -> Path | None:
    for d in (root / opts.lm, root):
        if (d / "params.json").is_file():
            return d
    return None


class _Shared:
    """The loaded reference LM / codec decoder, shared by every component and by the quality check."""

    def __init__(self, root: Path, lm_dir: Path, opts: Zonos3Options):
        self.root, self.lm_dir, self.opts = root, lm_dir, opts
        self._lm = self._codec = None

    def lm(self):
        if self._lm is None:
            import torch
            from zonos3.reference.lm import Zonos3LM
            self._lm = Zonos3LM.load(self.lm_dir, dtype=torch.float32).eval()
            if self.opts.text_rows:
                from zonos3.models.text_extra import TextExtraRows
                self._lm = self._lm.append_text_rows(TextExtraRows.load(self.opts.text_rows))
        return self._lm

    def drop_lm(self) -> None:
        """Free the 3.2 GB fp32 reference LM (reloaded lazily from the memory-mapped checkpoint when a check needs it again)."""
        self._lm = None
        import gc
        gc.collect()

    def drop_codec(self) -> None:
        self._codec = None
        import gc
        gc.collect()

    def codec(self):
        if self._codec is None:
            self._codec = load_codec(self.root, self.lm().config.audio_codec, self.opts)
        return self._codec


def load_codec(root: Path, audio_codec: str, opts: Zonos3Options):
    """The iSTFT-head decoder, built the way zonos3.tokenizer.matched_codec.load_model builds it (see docs in README), or a tiny test codec."""
    import torch
    from zonos3.codec.istft_head_a_20261002.model import ISTFTCodec
    tiny = root / "tiny_codec.pt"
    if tiny.is_file():
        blob = torch.load(tiny, map_location="cpu", weights_only=True)
        model = ISTFTCodec(blob["cfg"], blob["head"])
        model.load_state_dict(blob["state"], strict=True)
        return model.eval().requires_grad_(False)
    reg = json.loads((root / "registries" / "codec_registry.json").read_text())
    entry = next((e for e in reg.values() if e.get("codec") == audio_codec), None)
    if entry is None:
        raise UnsupportedModelError(f"codec '{audio_codec}' (params.json) is not in registries/codec_registry.json: {sorted(reg)}")
    dec = entry["decoder"]
    if dec.get("kind") != "istft_mr":
        raise UnsupportedModelError(f"decoder kind '{dec.get('kind')}' is not supported (only the multi-resolution iSTFT head, 'istft_mr')")
    campaign = root / dec["campaign"]
    cfg_file = next(iter(sorted(campaign.glob("config*.json"))), None)
    if cfg_file is None:
        raise UnsupportedModelError(f"no config*.json in {campaign}")
    cfg = json.loads(cfg_file.read_text())
    generator = torch.load(root / entry["checkpoint"], map_location="cpu", mmap=True, weights_only=True)
    if generator["step"] != entry["codec_step"]:
        raise UnsupportedModelError(f"generator checkpoint is step {generator['step']}, the registry expects {entry['codec_step']}")
    from codec_graph import build_empty_codec
    model = build_empty_codec(cfg["model"], cfg["head"])
    model.load_pretrained(generator["generator"])
    del generator
    ck = campaign / dec["checkpoint"]
    try:
        state = torch.load(ck, map_location="cpu", mmap=True, weights_only=True)
    except Exception as exc:
        if not opts.allow_pickle:
            raise UnsupportedModelError(f"{ck.name} cannot be loaded with weights_only=True ({type(exc).__name__}); loading it needs full pickle, "
                                        "which can execute code. Inspect the file, then pass --adapter-option allow_pickle=true") from exc
        state = torch.load(ck, map_location="cpu", mmap=True, weights_only=False)
    if dec.get("expect_step") is not None and int(state["step"]) != int(dec["expect_step"]):
        raise UnsupportedModelError(f"{ck.name} is step {state['step']}, the registry expects {dec['expect_step']}")
    if "trainable" in state:
        model.load_trainable_state_dict(state["trainable"])
    else:
        model.head.load_state_dict(state["head"], strict=True)
    for name in ("encoder", "to_internal", "semantic_encoder"):          # training-only parts the decoder never uses
        if hasattr(model, name):
            delattr(model, name)
    return model.eval().requires_grad_(False)


@register_adapter
class Zonos3Adapter(Adapter):
    name = "zonos3-mini"
    version = "0.1.0"
    priority = 60
    description = "ZONOS3-Mini TTS (hybrid GDN/attention MoE LM + iSTFT codec decoder) -> stateful LM graph + codec decoder graph"
    frameworks = ["pytorch", "zonos3"]
    options_model = Zonos3Options

    # -- detection / inspection ---------------------------------------------------------------------------------------------------
    def detect(self, source: ModelSource) -> Detection:
        if source.kind not in ("local-dir", "huggingface"):
            return Detection(confidence=0.0, reason="not a directory or Hub repository")
        if source.reference == REPOSITORY:
            return Detection(confidence=1.0, reason=f"the official repository {REPOSITORY}")
        d = _lm_dir(source.path, Zonos3Options())
        if d is not None:
            try:
                p = json.loads((d / "params.json").read_text()).get("model", {})
            except (OSError, json.JSONDecodeError):
                p = {}
            if str(p.get("attention_pattern", "")).startswith("gdn") and p.get("residual_type") == "full_attnres":
                return Detection(confidence=0.95, reason="params.json describes the ZONOS3 hybrid GDN/attention LM (full_attnres)")
        return Detection(confidence=0.0, reason="no ZONOS3 params.json (attention_pattern gdn*, residual_type full_attnres)")

    def weight_patterns(self, source: ModelSource, options: Zonos3Options) -> list[str] | None:
        pats = [f"{options.lm}/*", "registries/*", "README.md"]
        if options.include_codec:
            pats += ["autoencoder/generator.pt", "autoencoder/heads/full_mr/*"]
        return pats

    def _params(self, source: ModelSource, options: Zonos3Options) -> dict[str, Any]:
        d = _lm_dir(source.path, options)
        if d is None:
            raise UnsupportedModelError(f"no {options.lm}/params.json under {source.path}")
        return json.loads((d / "params.json").read_text())["model"]

    def inspect(self, source: ModelSource, options: Zonos3Options) -> ModelInfo:
        p = self._params(source, options)
        comps = [ComponentInfo(name="lm", role="language model (prefill + decode)", stateful=True,
                               description=f"{p['n_layers']} layers dim {p['dim']}: GDN/attention pattern {p['attention_pattern']}, latent MoE "
                                           f"{p['moe_n_experts']} experts top-{p['moe_router_topk']}, {p['n_codebooks']} codebooks x {p['codebook_size']}")]
        if options.include_codec:
            comps.append(ComponentInfo(name="codec_decoder", role="codec", description="codes -> 44.1 kHz audio (causal MR iSTFT head)"))
        return ModelInfo(reference=source.reference, source_kind=source.kind, revision=source.revision, framework="pytorch",
                         architecture="ZONOS3-Mini (hybrid GDN/attention MoE TTS)", model_type="zonos3", task="text-to-speech", parameters=None,
                         license=source.license().id, components=comps,
                         extra={"params": {k: p.get(k) for k in ("n_layers", "dim", "attention_pattern", "moe_n_experts", "n_codebooks", "codebook_size")},
                                "parameters_note": "about 800M total / 100M active (model card); the LM checkpoint is not safetensors, so no exact count is read",
                                "not_exported": ["text normalisation + SuperBPE tokenizer", "prompt assembly + delay pattern", "sampler", "speaker encoder",
                                                 "Wav2Vec2-BERT teacher", "aligner", "edit LoRA / accents / steering"]})

    def compatibility(self, source, info, options: Zonos3Options, config: JobConfig) -> CompatibilityReport:
        from modelsuite.core.registry import exporters
        import psutil
        issues: list[Issue] = []
        add = lambda sev, code, msg, hint=None: issues.append(Issue(severity=sev, code=code, message=msg, hint=hint))
        support = Support.supported
        try:
            from reference_env import ReferenceUnavailable, ensure_reference
            ensure_reference(options.source_dir, codec=options.include_codec)
        except Exception as exc:            # ReferenceUnavailable, or a missing optional package
            add(Severity.error, "reference-missing", str(exc), "git clone https://github.com/Zyphra/ZONOS3-Mini && export ZONOS3_SRC=$PWD/ZONOS3-Mini/model/src")
            support = Support.unsupported
        for mod in ("torch", "onnx", "onnxruntime"):
            if not deps.have(mod):
                add(Severity.error, f"{mod}-missing", f"'{mod}' is not installed.", f"pip install {mod}")
                support = Support.unsupported
        if options.include_codec and not deps.have("einops"):
            add(Severity.error, "einops-missing", "the codec decoder needs einops (and `pip install --no-deps descript-audio-codec==1.0.0`)")
            support = Support.unsupported
        add(Severity.info, "stateful-lm", "The LM is ONE graph for prefill and decode: GDN recurrence and MoE dispatch are ONNX Loops over rows; the host owns "
            "past_k/past_v, conv_state and gdn_state (the manifest's state_links say how to feed them back).")
        add(Severity.warning, "decode-speed", "Decode is sequential (a Loop per GDN/MoE layer). Measured on a 2-core CPU: ~0.08 s per frame vs the 0.067 s real-time "
            "budget at 15 Hz; Zyphra's ggml / CUDA / WebGPU runtimes are the real-time options.")
        add(Severity.warning, "not-exported", "Tokenizer, text normalisation, prompt assembly with the delay pattern, sampling, speaker encoder, SSL teacher, aligner "
            "and edit/steering features are NOT part of the graphs: the host must provide them (reference: Zyphra's zonos3.tts.*). Reference voices ship as "
            "precomputed codes/embeddings (web/voices, gguf/voices), so the speaker encoder is only needed for NEW reference audio.")
        add(Severity.info, "license", "Weights Apache-2.0 (model card), reference code MIT; third_party/antspeaker (speaker encoder) is not exported.")
        est = memory_estimate(info)
        est.update(export_peak_ram_bytes_estimate=int(5.5 * 2 ** 30), note="LM: 3.5 GB fp32 external data, peak RSS 5.4 GB measured while exporting (reference + traced graph)")
        est["fits_in_ram"] = est["export_peak_ram_bytes_estimate"] < psutil.virtual_memory().available
        est["available_ram_bytes"] = int(psutil.virtual_memory().available)
        if not est["fits_in_ram"]:
            add(Severity.warning, "ram", f"Export peaked at ~5.4 GiB; {psutil.virtual_memory().available / 2 ** 30:.1f} GiB is available now.")
        status = {}
        for n in exporters.names():
            e = exporters.get(n)
            e = e() if isinstance(e, type) else e
            ok, why = e.availability()
            status[n] = "available" if ok else why
        if status.get("torch-legacy") != "available":
            add(Severity.error, "no-exporter", f"torch-legacy is unavailable: {status.get('torch-legacy')} (scripted Loops need the legacy exporter)")
            support = Support.unsupported
        return CompatibilityReport(target_format=config.export.format, support=support,
                                   summary="Convertible: stateful LM graph" + (" + codec decoder graph" if options.include_codec else "") + " (the rest of the TTS stack stays in the host).",
                                   issues=issues, exporters=status, strategy={c.name: ["torch-legacy"] for c in info.components}, estimates=est)

    # -- plan ------------------------------------------------------------------------------------------------------------------------------
    def plan(self, source: ModelSource, options: Zonos3Options, config: JobConfig) -> ExportPlan:
        from reference_env import ReferenceUnavailable, ensure_reference
        try:
            ensure_reference(options.source_dir, codec=options.include_codec)
        except ReferenceUnavailable as exc:
            raise UnsupportedModelError(str(exc)) from exc
        root = source.path
        lm_dir = _lm_dir(root, options)
        if lm_dir is None:
            raise UnsupportedModelError(f"no {options.lm}/params.json under {root}")
        from zonos3.reference.lm import Zonos3LMConfig
        c = Zonos3LMConfig.from_params(json.loads((lm_dir / "params.json").read_text()))
        shared = _Shared(root, lm_dir, options)
        n_attn = len(c.attention_layers)
        n_gdn = c.n_layers - n_attn
        KV, HD, SP = c.n_kv_heads, c.head_dim, c.speaker_embedding_dim
        offsets = np.cumsum([0] + [n + 1 for n in c.condition_bucket_counts])           # start of each condition table in the concatenated table
        f32 = np.float32
        rng = lambda tag, seed=0: np.random.default_rng(zlib.crc32(tag.encode()) + 1_000_003 * seed + config.seed)

        def lm_inputs(T: int, P: int, tag: str, *, fresh: bool, conditioned: bool) -> Arrays:
            r = rng(tag)
            ids = np.concatenate([r.integers(0, c.codebook_size, (T, c.n_codebooks)), r.integers(0, c.text_vocab, (T, 1))], 1).astype(np.int64)
            ids[:, :c.n_codebooks] = np.where(r.random((T, 1)) < 0.4, c.audio_pad_id, ids[:, :c.n_codebooks])        # prompt-like rows: audio columns padded
            spk = np.zeros((T, SP), f32)
            flag = np.zeros(T, np.int64)
            cond = -np.ones(T, np.int64)
            if conditioned and T >= 1 + c.condition_count:
                spk[0] = r.standard_normal(SP).astype(f32)
                flag[0] = 1
                for k in range(c.condition_count):                                     # one row per table, ascending (what the reference expects)
                    cond[1 + k] = offsets[k] + int(r.integers(0, c.condition_bucket_counts[k] + 1))
            scale = 0.0 if fresh else 1.0
            return {"ids": ids, "speaker_vec": spk, "speaker_flag": flag, "cond_index": cond,
                    "past_k": (r.standard_normal((n_attn, P, KV, HD)) * 0.5 * scale).astype(f32),
                    "past_v": (r.standard_normal((n_attn, P, KV, HD)) * 0.5 * scale).astype(f32),
                    "conv_state": (r.standard_normal((n_gdn, 3 * GDN_HEADS * GDN_HEAD_DIM, GDN_KERNEL - 1)) * 0.5 * scale).astype(f32),
                    "gdn_state": (r.standard_normal((n_gdn, GDN_HEADS, GDN_HEAD_DIM, GDN_HEAD_DIM)) * 0.05 * scale).astype(f32)}

        def lm_reference(inp: Arrays) -> Arrays:
            """The ORIGINAL forward (Zyphra's reference), driven from the same explicit state."""
            import torch
            from zonos3.reference.lm import LMCache
            ref = shared.lm()
            cache = LMCache(position=inp["past_k"].shape[1])
            for a, layer in enumerate(c.attention_layers):
                cache.kv[layer] = (torch.from_numpy(inp["past_k"][a]), torch.from_numpy(inp["past_v"][a]))
            for g, layer in enumerate(l for l in range(c.n_layers) if c.uses_gdn(l)):
                cache.gdn_state[layer] = torch.from_numpy(inp["gdn_state"][g])
                cache.conv_state[layer] = torch.from_numpy(inp["conv_state"][g])
            P = cache.position
            rows = range(len(inp["ids"]))
            speaker = {P + t: torch.from_numpy(inp["speaker_vec"][t]) for t in rows if inp["speaker_flag"][t]}
            table_of = lambda idx: int(np.searchsorted(offsets, idx, side="right") - 1)
            cond = {P + t: int(inp["cond_index"][t] - offsets[table_of(inp["cond_index"][t])]) for t in rows if inp["cond_index"][t] >= 0}
            with torch.inference_mode():
                out = ref(torch.from_numpy(inp["ids"]), cache, speaker_rows=speaker or None, condition_rows=cond or None)
            return {"logits": out.logits.numpy(),
                    "new_k": np.stack([cache.kv[l][0].numpy() for l in c.attention_layers]), "new_v": np.stack([cache.kv[l][1].numpy() for l in c.attention_layers]),
                    "new_conv": np.stack([cache.conv_state[l].numpy() for l in range(c.n_layers) if c.uses_gdn(l)]),
                    "new_state": np.stack([cache.gdn_state[l].numpy() for l in range(c.n_layers) if c.uses_gdn(l)])}

        T0 = options.prefill_rows
        names = ["ids", "speaker_vec", "speaker_flag", "cond_index", "past_k", "past_v", "conv_state", "gdn_state"]
        dyn = {"ids": {0: "T"}, "speaker_vec": {0: "T"}, "speaker_flag": {0: "T"}, "cond_index": {0: "T"}, "past_k": {1: "P"}, "past_v": {1: "P"},
               "logits": {0: "T"}, "new_k": {1: "PT"}, "new_v": {1: "PT"}}

        def step_inputs(i: int) -> Arrays:
            r = rng("decode-row", i)
            ids = np.concatenate([r.integers(0, c.codebook_size, (1, c.n_codebooks)), np.full((1, 1), c.text_vocab)], 1).astype(np.int64)
            return {"ids": ids, "speaker_vec": np.zeros((1, SP), f32), "speaker_flag": np.zeros(1, np.int64), "cond_index": -np.ones(1, np.int64)}

        state_links = [StateLink("new_k", "past_k", "replace"), StateLink("new_v", "past_v", "replace"),
                       StateLink("new_conv", "conv_state", "replace"), StateLink("new_state", "gdn_state", "replace")]
        lm = ComponentExport(
            name="lm", role="language model", description="hybrid GDN/attention latent-MoE LM: prefill + decode in one graph, explicit state",
            build=lambda: _lm_module(shared), example_inputs=lambda: lm_inputs(5, 3, "lm-example", fresh=False, conditioned=True),
            input_names=names, output_names=["logits", "new_k", "new_v", "new_conv", "new_state"], dynamic_axes=dyn, reference=lm_reference,
            cases=[ReferenceCase("prefill", lambda: lm_inputs(T0, 0, "lm-prefill", fresh=True, conditioned=True), "parity", "prompt rows with a speaker row and the 3 condition rows, empty state"),
                   ReferenceCase("decode-step", lambda: lm_inputs(1, T0, "lm-decode", fresh=False, conditioned=False), "parity", "one frame continuing from a non-empty state"),
                   ReferenceCase("chunk", lambda: lm_inputs(7, 9, "lm-chunk", fresh=False, conditioned=False), "dynamic_shape", "a chunk of 7 rows after 9"),
                   ReferenceCase("long-prefill", lambda: lm_inputs(2 * T0 + 5, 0, "lm-long", fresh=True, conditioned=True), "dynamic_shape", "longer prompt")],
            stateful=[StatefulCase(name="prefill-then-decode", steps=6, links=state_links, step_inputs=step_inputs,
                                   initial=lambda: lm_inputs(T0 // 2, 0, "lm-seq", fresh=True, conditioned=True),
                                   description="prefill, then 5 single-frame steps feeding KV / GDN / conv state back (the host's decode loop)")],
            stateful_component=True, exporters=["torch-legacy"], on_release=shared.drop_lm,
            tolerance=Tolerance(atol=1e-4, rtol=2e-4, min_cosine=0.999999), output_kinds={"logits": "logits"},
            known_issues=["Prefill cost grows linearly with rows (Loop over rows for GDN and MoE); decode ~0.08 s/frame on a 2-core CPU."])
        components = [lm]
        if options.include_codec:
            def codec_in(frames: int, tag: str) -> Arrays:
                return {"codes": rng(tag).integers(0, 4096, (1, c.n_codebooks, frames)).astype(np.int64)}

            def codec_reference(inp: Arrays) -> Arrays:
                import torch
                with torch.inference_mode():
                    return {"audio": shared.codec()(torch.from_numpy(inp["codes"])).numpy()}
            F = options.codec_frames
            components.append(ComponentExport(
                name="codec_decoder", role="codec", description="codes [1,10,T] -> audio [1,1,2940*T] at 44.1 kHz (whole-sequence decode; chunk long audio)",
                build=lambda: _codec_module(shared), example_inputs=lambda: codec_in(F, "codec-example"), input_names=["codes"], output_names=["audio"],
                dynamic_axes={"codes": {2: "T"}, "audio": {2: "N"}}, reference=codec_reference,
                cases=[ReferenceCase("example", lambda: codec_in(F, "codec-a"), "parity"),
                       ReferenceCase("longer", lambda: codec_in(F + 17, "codec-b"), "dynamic_shape"),
                       ReferenceCase("short", lambda: codec_in(5, "codec-c"), "dynamic_shape")],
                tolerance=Tolerance(atol=1e-5, rtol=1e-4, min_cosine=0.99999), output_kinds={"audio": "audio"}, exporters=["torch-legacy"], on_release=shared.drop_codec,
                known_issues=["Attention at the 240 Hz stage is dense-banded: memory grows with T^2. Decode audio in chunks of a few seconds (overlap as Zyphra's stream does).",
                              "Non-streaming: equals ISTFTCodec.forward over the whole code sequence (Zyphra's streaming decoder reproduces that to 1e-5)."]))
        topology = [Edge("lm", s, "lm", d) for s, d in (("new_k", "past_k"), ("new_v", "past_v"), ("new_conv", "conv_state"), ("new_state", "gdn_state"))]
        topology.append(Edge("lm", "logits", "(host sampler)", "logits"))
        if options.include_codec:
            topology.append(Edge("(host sampler)", "codes", "codec_decoder", "codes"))
        assets = [AssetSpec("config", lm_dir / "params.json", "params.json", description="LM architecture (params.json)")]
        for rel, role in (("tokenizer/superbpe8192.json", "tokenizer"), ("text_rows/pause_tokens_s5000.json", "config")):
            if (lm_dir / rel).is_file():
                assets.append(AssetSpec(role, lm_dir / rel, rel, required=role == "tokenizer"))
        for rel in ("registries/codec_registry.json", "registries/silence_frame_opus_dual_3n_600k_sched600k_r4096_step600000.json"):
            if (root / rel).is_file():
                assets.append(AssetSpec("config", root / rel, rel, required=False, description="codec / silence-frame registry the prompt assembly reads"))
        assets.append(AssetSpec("postprocessor", None, "host-pipeline", required=True, description=(
            "NOT exported, implement in the host (reference: zonos3.tts.*): NeMo text normalisation + SuperBPE tokenisation (offset 4, bos 2, eos 3); prompt "
            f"assembly: speaker row + {c.condition_count} condition rows (bucket ids, count = unknown) + text rows + optional silence prefill, codebook j shifted by j "
            "frames (shear / shear_up); per-frame sampling (temperature 0.75, top_k 80, top_p 0.69, repetition penalty 2.4 over a 96-frame window, EOA ends); the "
            "condition-table offsets for cond_index are " + str(offsets.tolist()))))
        return ExportPlan(components=components, topology=topology, assets=assets, quality_checks=[self._rollout_check(shared, c, options, n_attn, n_gdn)],
                          runtime_requirements=["onnxruntime>=1.18 (Loop, TopK, Gather over external data)", "host-side tokenizer + prompt assembly + sampler"],
                          notes=["lm state: feed new_k/new_v/new_conv/new_state back as past_k/past_v/conv_state/gdn_state (all 'replace'); start from empty/zero state.",
                                 f"condition_index offsets {offsets.tolist()}; speaker rows carry a {SP}-dim embedding projected inside the graph."])

    # -- quality check ---------------------------------------------------------------------------------------------------------------------
    def _rollout_check(self, shared: _Shared, c, options: Zonos3Options, n_attn: int, n_gdn: int) -> QualityCheck:
        def run(ctx: QualityContext) -> QualityResult:
            import torch
            from zonos3.reference.lm import LMCache
            runner = ctx.runner("lm")
            r = np.random.default_rng(ctx.seed + 3)
            T0 = 8
            prompt = np.concatenate([np.full((T0, c.n_codebooks), c.audio_pad_id), r.integers(0, c.text_vocab, (T0, 1))], 1).astype(np.int64)
            f32 = np.float32
            state = {"past_k": np.zeros((n_attn, 0, c.n_kv_heads, c.head_dim), f32), "past_v": np.zeros((n_attn, 0, c.n_kv_heads, c.head_dim), f32),
                     "conv_state": np.zeros((n_gdn, 3 * GDN_HEADS * GDN_HEAD_DIM, GDN_KERNEL - 1), f32), "gdn_state": np.zeros((n_gdn, GDN_HEADS, GDN_HEAD_DIM, GDN_HEAD_DIM), f32)}
            ids, got = prompt, []
            for _ in range(options.rollout_steps):
                o = runner.run({"ids": ids, "speaker_vec": np.zeros((len(ids), c.speaker_embedding_dim), f32), "speaker_flag": np.zeros(len(ids), np.int64),
                                "cond_index": -np.ones(len(ids), np.int64), **state})
                state = {"past_k": o["new_k"], "past_v": o["new_v"], "conv_state": o["new_conv"], "gdn_state": o["new_state"]}
                frame = o["logits"][-1, :, : c.codebook_size + 1].argmax(-1)           # greedy over codes + EOA
                got.append(frame)
                ids = np.concatenate([frame, [c.text_vocab]])[None].astype(np.int64)
            ref, cache = shared.lm(), LMCache()
            ids_t, want = torch.from_numpy(prompt), []
            with torch.inference_mode():
                for _ in range(options.rollout_steps):
                    out = ref(ids_t, cache).logits
                    frame = out[-1, :, : c.codebook_size + 1].argmax(-1)
                    want.append(frame.numpy())
                    ids_t = torch.cat([frame, torch.tensor([c.text_vocab])])[None]
            agree = float(np.mean(np.stack(got) == np.stack(want)))
            exact = ctx.variant == "fp32" or ctx.variant.startswith("opt-")
            need = 1.0 if exact else 0.9
            return QualityResult(agree >= need, {"token_agreement": agree, "required": need, "steps": options.rollout_steps, "codebooks": c.n_codebooks},
                                 f"greedy {options.rollout_steps}-frame rollout through the ONNX LM reproduces {agree * 100:.1f}% of the reference tokens (required {need * 100:.0f}%)")
        return QualityCheck("greedy-rollout", run, "greedy frame rollout (prefill + decode steps, state fed back) vs Zyphra's reference LM")


def _lm_module(shared: _Shared):
    from lm_graph import LMGraph
    return LMGraph(shared.lm()).eval()


def _codec_module(shared: _Shared):
    from codec_graph import CodecDecoderGraph
    return CodecDecoderGraph(shared.codec()).eval()
