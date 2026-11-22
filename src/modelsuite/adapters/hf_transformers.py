"""Generic Hugging Face Transformers adapter.

What it converts (each path is exercised by tests against real tiny checkpoints):

* encoder-style and head models on text or images -- feature extraction, sequence / token classification, masked LM,
  question answering, image classification -- as ONE graph with dynamic batch/sequence axes;
* decoder-only causal LMs with a *uniform* KV cache (Llama / Mistral / Qwen / Phi-3 / Gemma-1 / Olmo / Granite families) as ONE
  graph that takes and returns the KV cache (prefill = empty past), so the same graph serves prefill and decode;
* any other causal LM as a no-cache logits graph (documented: decoding recomputes the sequence);
* everything else is *reported* (analyze) with the reason, and can be routed to the Optimum exporter (mode="optimum").

It deliberately does not try to be clever about architectures it cannot verify: exporters differ per model and
only parity runs prove a conversion."""
from __future__ import annotations

import inspect
from typing import Any, Literal

import numpy as np

from ..core import deps
from ..core.config import JobConfig, Tolerance
from ..core.discovery import ModelSource
from ..core.errors import UnsupportedModelError
from ..core.plan import (Arrays, AssetSpec, ComponentExport, ExportPlan, QualityCheck, QualityContext, QualityResult,
                         ReferenceCase, StatefulCase, StateLink)
from ..core.registry import register_adapter
from ..core.types import (CompatibilityReport, ComponentInfo, Detection, Issue, ModelInfo, Severity, Support)
from .base import Adapter, AdapterOptions, memory_estimate

#: model_type values whose decoder keeps one uniform (layers, batch, kv_heads, seq, head_dim) KV cache
UNIFORM_KV_MODEL_TYPES = {"llama", "mistral", "qwen2", "qwen3", "phi3", "gemma", "olmo", "olmo2", "granite", "smollm3"}

_TASKS = [("ForCausalLM", "text-generation"), ("LMHeadModel", "text-generation"), ("ForMaskedLM", "fill-mask"),
          ("ForSequenceClassification", "text-classification"), ("ForTokenClassification", "token-classification"),
          ("ForQuestionAnswering", "question-answering"), ("ForImageClassification", "image-classification"),
          ("ForMultipleChoice", "multiple-choice"), ("ForConditionalGeneration", "seq2seq"), ("ForSeq2SeqLM", "seq2seq"),
          ("ForSpeechSeq2Seq", "seq2seq-audio"), ("ForCTC", "audio-ctc"), ("ForAudioClassification", "audio-classification")]
_MODALITY = {"image-classification": "image", "audio-ctc": "audio", "audio-classification": "audio", "seq2seq-audio": "audio"}
_OUTPUTS = {"text-generation": ["logits"], "fill-mask": ["logits"], "text-classification": ["logits"],
            "token-classification": ["logits"], "question-answering": ["start_logits", "end_logits"],
            "image-classification": ["logits"], "feature-extraction": ["last_hidden_state"]}
#: our task name -> Optimum task (Optimum cannot infer the task from a local directory; causal LMs get the KV-cache variant)
_OPTIMUM_TASK = {"text-generation": "text-generation-with-past", "seq2seq": "text2text-generation-with-past",
                 "seq2seq-audio": "automatic-speech-recognition-with-past"}
_SEQ_OUT = {"text-generation", "fill-mask", "token-classification", "feature-extraction", "question-answering"}


class HFOptions(AdapterOptions):
    mode: Literal["auto", "plain", "kv-cache", "optimum"] = "auto"
    task: str | None = None                  # override the task inferred from the architecture name
    batch: int = 2
    seq_len: int = 7
    kv_past_len: int = 3
    greedy_tokens: int = 6                   # tokens decoded by the generation quality check


def _task_of(arch: str | None) -> str:
    for suffix, task in _TASKS:
        if arch and arch.endswith(suffix):
            return task
    return "feature-extraction"


def _transformers():
    return deps.require("transformers", "Hugging Face models", "hf")


def _arch_class(arch: str):
    return getattr(_transformers(), arch, None)


def _kv_dims(cfg: dict) -> tuple[int, int, int]:
    heads = cfg.get("num_attention_heads") or cfg.get("n_head")
    kv = cfg.get("num_key_value_heads") or heads
    hd = cfg.get("head_dim") or (cfg["hidden_size"] // heads)
    return cfg["num_hidden_layers"], int(kv), int(hd)


def _kv_blockers(cfg: dict) -> list[str]:
    why = []
    if cfg.get("model_type") not in UNIFORM_KV_MODEL_TYPES:
        why.append(f"model_type '{cfg.get('model_type')}' is not in the verified uniform-KV set {sorted(UNIFORM_KV_MODEL_TYPES)}")
    if cfg.get("layer_types") and len(set(cfg["layer_types"])) > 1:
        why.append("layers use different attention types (sliding/full): the KV cache is not uniform")
    if cfg.get("use_sliding_window") or (cfg.get("sliding_window") and cfg.get("use_sliding_window", True) and cfg.get("model_type") != "mistral"):
        why.append("sliding-window attention is configured")
    if cfg.get("is_encoder_decoder"):
        why.append("encoder-decoder model")
    return why


def _plan_mode(cfg: dict, task: str, opts: HFOptions) -> str:
    if opts.mode != "auto":
        return opts.mode
    if task == "text-generation":
        return "kv-cache" if not _kv_blockers(cfg) else "plain"
    return "plain"


@register_adapter
class HFTransformersAdapter(Adapter):
    name = "hf-transformers"
    description = "Hugging Face Transformers checkpoints (text/vision heads, causal LMs with uniform KV cache)"
    frameworks = ["transformers", "pytorch"]
    options_model = HFOptions
    priority = 5

    # -- detect / inspect ------------------------------------------------------------------------------------
    def detect(self, source: ModelSource) -> Detection:
        cfg = source.config
        if source.kind == "python-spec":
            return Detection(confidence=0.0, reason="python spec")
        if not cfg:
            return Detection(confidence=0.0, reason="no config.json")
        if cfg.get("model_type") or cfg.get("architectures"):
            has_w = any(p.suffix in (".safetensors", ".bin") for p in source.files()) or source.kind == "huggingface"
            return Detection(confidence=0.7 if has_w else 0.55,
                             reason=f"config.json with model_type={cfg.get('model_type')!r} architectures={cfg.get('architectures')}")
        return Detection(confidence=0.0, reason="config.json without model_type/architectures")

    def _arch(self, cfg: dict) -> str | None:
        a = cfg.get("architectures")
        return a[0] if a else None

    def inspect(self, source: ModelSource, options: HFOptions) -> ModelInfo:
        cfg = source.config
        arch = self._arch(cfg)
        task = options.task or _task_of(arch)
        w = source.weight_summary()
        info = ModelInfo(reference=source.reference, source_kind=source.kind, revision=source.revision,
                         framework="transformers", architecture=arch or cfg.get("model_type", "unknown"),
                         model_type=cfg.get("model_type"), task=task, parameters=w["parameters"], weight_bytes=w["bytes"],
                         weight_dtypes=w["dtypes"], license=source.license().id,
                         extra={"hidden_size": cfg.get("hidden_size"), "layers": cfg.get("num_hidden_layers"),
                                "vocab_size": cfg.get("vocab_size"), "transformers_version": cfg.get("transformers_version"),
                                "torch_dtype": cfg.get("torch_dtype") or cfg.get("dtype")})
        mode = _plan_mode(cfg, task, options)
        info.extra["planned_mode"] = mode
        info.components = [ComponentInfo(name="model", role="decoder-with-kv-cache" if mode == "kv-cache" else task,
                                         description=f"{arch} ({mode})", parameters=w["parameters"], stateful=mode == "kv-cache")]
        return info

    # -- compatibility -----------------------------------------------------------------------------------------------
    def compatibility(self, source: ModelSource, info: ModelInfo, options: HFOptions, config: JobConfig) -> CompatibilityReport:
        from ..core.registry import exporters
        cfg, arch, task = source.config, self._arch(source.config), info.task or "feature-extraction"
        issues: list[Issue] = []
        add = lambda sev, code, msg, hint=None: issues.append(Issue(severity=sev, code=code, message=msg, hint=hint))
        support = Support.supported
        mode = _plan_mode(cfg, task, options)

        if cfg.get("auto_map") and not config.export.trust_remote_code:
            add(Severity.error, "remote-code", "The checkpoint ships custom modeling code (auto_map) and trust_remote_code is off.",
                "Review the code, then set export.trust_remote_code=true.")
            support = Support.unsupported
        if deps.have("transformers"):
            from transformers import CONFIG_MAPPING
            if cfg.get("model_type") and cfg["model_type"] not in CONFIG_MAPPING and not cfg.get("auto_map"):
                add(Severity.error, "unknown-model-type", f"model_type '{cfg['model_type']}' is unknown to the installed transformers "
                    f"({deps.dist_version('transformers')}).", "Upgrade transformers or write an adapter.")
                support = Support.unsupported
            if arch and _arch_class(arch) is None and not cfg.get("auto_map"):
                add(Severity.error, "unknown-architecture", f"Class '{arch}' does not exist in the installed transformers.")
                support = Support.unsupported
        else:
            add(Severity.warning, "no-transformers", "transformers is not installed; analysis is based on config.json only.",
                "pip install 'modelsuite[hf]'")
            support = Support.unknown
        if cfg.get("quantization_config"):
            q = cfg["quantization_config"].get("quant_method", "unknown")
            add(Severity.error, "quantized-checkpoint", f"Checkpoint is already quantized ({q}); torch exporters cannot export packed weights.",
                "Convert from the original full-precision checkpoint, then quantize with `modelsuite optimize`.")
            support = Support.unsupported
        if task in ("seq2seq", "seq2seq-audio", "multiple-choice") or cfg.get("is_encoder_decoder"):
            add(Severity.warning, "unsupported-generic-task", f"Task '{task}' (encoder-decoder / multiple-choice) has no verified generic path.",
                "Use --adapter-option mode=optimum, or write an adapter that exports encoder and decoder as separate components.")
            if options.mode != "optimum":
                support = Support.unsupported
        if _MODALITY.get(task) == "audio":
            add(Severity.warning, "audio-modality", "Audio-input models are not covered by the generic adapter.",
                "Write a custom adapter (see the Reverb adapter for an audio example) or try mode=optimum.")
            if options.mode != "optimum":
                support = Support.unsupported
        if task == "text-generation":
            blockers = _kv_blockers(cfg)
            if mode == "kv-cache":
                add(Severity.info, "kv-cache", "Exported as a single prefill+decode graph with explicit KV-cache inputs/outputs.")
            elif mode == "plain":
                add(Severity.warning, "no-kv-cache", "No KV-cache graph: every decode step would recompute the whole sequence. "
                    + ("Reason: " + "; ".join(blockers) if blockers else ""), "Use mode=optimum for a cache-aware export.")
                support = Support.partial if support == Support.supported else support
        if cfg.get("num_local_experts") or cfg.get("num_experts") or cfg.get("n_routed_experts"):
            add(Severity.warning, "moe", "Mixture-of-experts routing is data-dependent; tracing may bake in one routing path.",
                "Rely on the parity cases; prefer an ONNX GenAI / Optimum export that supports MoE.")
            support = Support.partial if support == Support.supported else support
        rs = cfg.get("rope_scaling") or cfg.get("rope_parameters") or {}
        if isinstance(rs, dict) and (rs.get("rope_type") or rs.get("type")) in ("dynamic", "longrope", "yarn"):
            add(Severity.warning, "rope-scaling", f"rope scaling '{rs.get('rope_type') or rs.get('type')}' computes frequencies from the "
                "sequence length; the exported graph is only guaranteed for lengths covered by the parity cases.")
        if "bfloat16" in (info.weight_dtypes or []) or "BF16" in (info.weight_dtypes or []):
            add(Severity.info, "bf16-source", "Weights are bfloat16; the export casts to float32 (graph is fp32; use `optimize` for smaller variants).")

        est = memory_estimate(info)
        if est.get("fits_in_ram") is False:
            add(Severity.error, "ram", f"Estimated export peak RAM {est['export_peak_ram_bytes_estimate']/2**30:.1f} GiB exceeds the "
                f"{est['available_ram_bytes']/2**30:.1f} GiB currently available.", "Close other programs, add swap, or export on a larger machine.")
            support = Support.unsupported
        if est.get("needs_external_data"):
            add(Severity.info, "external-data", "fp32 weights exceed the protobuf limit; weights are stored in a .onnx.data sidecar.")

        statuses = {}
        for n in exporters.names():
            e = exporters.get(n)
            e = e() if isinstance(e, type) else e
            ok, why = e.availability()
            statuses[n] = "available" if ok else why
        chosen = ["optimum"] if mode == "optimum" else ["torch-legacy", "torch-dynamo"]
        if not any(statuses.get(c) == "available" for c in chosen):
            add(Severity.error, "no-exporter", f"None of the exporters for this path is usable: {', '.join(f'{c}: {statuses.get(c)}' for c in chosen)}")
            support = Support.unsupported
        summary = {Support.supported: f"Convertible with the generic path '{mode}'.", Support.partial: f"Convertible with limits (path '{mode}').",
                   Support.unsupported: "Not convertible with the generic adapter; see issues.",
                   Support.unknown: "Could not be fully analysed."}[support]
        return CompatibilityReport(target_format=config.export.format, support=support, summary=summary, issues=issues,
                                   exporters=statuses, strategy={"model": chosen}, estimates=est)

    # -- plan --------------------------------------------------------------------------------------------------------------
    def plan(self, source: ModelSource, options: HFOptions, config: JobConfig) -> ExportPlan:
        cfg = source.config
        arch = self._arch(cfg)
        if not arch:
            raise UnsupportedModelError("config.json has no 'architectures' entry; cannot choose a model class")
        task = options.task or _task_of(arch)
        mode = _plan_mode(cfg, task, options)
        if task in ("seq2seq", "seq2seq-audio", "multiple-choice", "audio-ctc", "audio-classification") and mode != "optimum":
            raise UnsupportedModelError(f"task '{task}' has no verified generic export path; use adapter option mode=optimum")
        if mode == "optimum":
            comp = self._optimum_component(source, cfg, arch, task, options, config)
            return ExportPlan(components=[comp], assets=self._assets(source),
                              notes=["Exported by the Optimum CLI (isolated interpreter); parity uses Optimum's input convention."])
        if mode == "kv-cache":
            comp = self._kv_component(source, cfg, arch, options, config)
            checks = [self._kv_consistency_check(), self._greedy_check(source, arch, options, config)]
        else:
            comp = self._plain_component(source, cfg, arch, task, options, config)
            checks = []
        return ExportPlan(components=[comp], assets=self._assets(source), quality_checks=checks,
                          runtime_requirements=[f"onnxruntime>=1.18", "tokenizers (for the tokenizer assets)"],
                          notes=[f"hf-transformers path: {mode}"])

    def _assets(self, source: ModelSource) -> list[AssetSpec]:
        names = {"config.json": "config", "generation_config.json": "config", "tokenizer.json": "tokenizer", "tokenizer_config.json": "tokenizer",
                 "special_tokens_map.json": "tokenizer", "vocab.json": "tokenizer", "merges.txt": "tokenizer", "vocab.txt": "tokenizer",
                 "tokenizer.model": "tokenizer", "preprocessor_config.json": "preprocessor", "added_tokens.json": "tokenizer"}
        out = []
        for f in source.files():
            if f.name in names and f.parent == source.root:
                out.append(AssetSpec(role=names[f.name], source=f, dest=f.name, required=names[f.name] != "config" or f.name == "config.json"))
        return out

    # -- component builders ------------------------------------------------------------------------------------------------------
    def _load(self, source: ModelSource, arch: str, config: JobConfig):
        torch = deps.require("torch", "loading the model", "torch")
        tr = _transformers()
        cls = _arch_class(arch)
        if cls is None:
            raise UnsupportedModelError(f"transformers has no class '{arch}'")
        kw: dict[str, Any] = {"trust_remote_code": config.export.trust_remote_code}
        params = inspect.signature(cls.from_pretrained).parameters
        for dtype_key in ("dtype", "torch_dtype"):
            try:
                model = cls.from_pretrained(str(source.root), **{dtype_key: torch.float32}, attn_implementation="eager", **kw)
                break
            except TypeError:
                continue
            except ValueError:
                model = cls.from_pretrained(str(source.root), **{dtype_key: torch.float32}, **kw)
                break
        else:
            model = cls.from_pretrained(str(source.root), **kw)
        return model.eval()

    def _forward_params(self, arch: str) -> set[str]:
        cls = _arch_class(arch)
        return set(inspect.signature(cls.forward).parameters) if cls else set()

    def _plain_component(self, source, cfg, arch, task, options: HFOptions, config: JobConfig) -> ComponentExport:
        outs = _OUTPUTS.get(task)
        if outs is None:
            raise UnsupportedModelError(f"no output convention for task '{task}'")
        rng = np.random.default_rng(config.seed)
        B, S = options.batch, options.seq_len
        image = _MODALITY.get(task) == "image"
        if image:
            size = cfg.get("image_size", 224)
            h, w = (size, size) if isinstance(size, int) else (size[0], size[1])
            ch = cfg.get("num_channels", 3)

            def make(b=B, hh=h, ww=w, seed=0):
                return {"pixel_values": np.random.default_rng(seed).standard_normal((b, ch, hh, ww)).astype(np.float32)}
            names = ["pixel_values"]
            dyn = {"pixel_values": {0: "batch"}, **{o: {0: "batch"} for o in outs}}
            cases = [ReferenceCase("example", lambda: make(), "parity"),
                     ReferenceCase("batch1", lambda: make(1, seed=3), "dynamic_shape", "batch of one")]
        else:
            vocab = cfg.get("vocab_size", 1000)
            fparams = self._forward_params(arch)
            use_tt = "token_type_ids" in fparams and (cfg.get("type_vocab_size") or 0) > 1
            names = ["input_ids", "attention_mask"] + (["token_type_ids"] if use_tt else [])

            def make(b=B, s=S, pad=1, seed=0):
                r = np.random.default_rng(seed)
                ids = r.integers(1, vocab, (b, s)).astype(np.int64)
                mask = np.ones((b, s), np.int64)
                if pad and b > 0 and s > 3:
                    mask[-1, s - min(pad, s - 2):] = 0     # right padding on the last row: forces the mask path
                d = {"input_ids": ids, "attention_mask": mask}
                if use_tt:
                    d["token_type_ids"] = np.zeros((b, s), np.int64)
                return d
            seq_out = task in _SEQ_OUT
            dyn = {n: {0: "batch", 1: "sequence"} for n in names}
            dyn.update({o: ({0: "batch", 1: "sequence"} if seq_out else {0: "batch"}) for o in outs})
            cases = [ReferenceCase("example", lambda: make(), "parity", "export example, last row right-padded"),
                     ReferenceCase("longer-batch1", lambda: make(1, 13, 0, 1), "dynamic_shape", "other batch and sequence length, no padding"),
                     ReferenceCase("wide-padded", lambda: make(3, 11, 4, 2), "dynamic_shape", "batch 3, heavy padding")]
        arch_ref = arch

        def build():
            torch = deps.require("torch")
            model = self._load(source, arch_ref, config)
            keys = names

            class Wrapper(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m

                def forward(self, *args):
                    out = self.m(**dict(zip(keys, args)), return_dict=True, **({"use_cache": False} if task == "text-generation" else {}))
                    return tuple(getattr(out, o) for o in outs)
            return Wrapper(model).eval()

        return ComponentExport(
            name="model", build=build, example_inputs=lambda: make(), input_names=names, output_names=outs, dynamic_axes=dyn,
            role=task, description=f"{arch} ({task}), single graph", cases=cases, tolerance=Tolerance(atol=2e-4, rtol=1e-3),
            output_kinds={o: "logits" for o in outs if "logits" in o})

    def _optimum_component(self, source, cfg, arch, task, options: HFOptions, config: JobConfig) -> ComponentExport:
        """Whole-model Optimum export. Heads / encoders keep our names (Optimum uses the same); causal LMs get Optimum's
        `position_ids` + per-layer `past_key_values.N.key/value` inputs, validated on prefill with an empty past."""
        opt_opts = {"source": str(source.root), "task": _OPTIMUM_TASK.get(task, task)}
        if task != "text-generation":
            comp = self._plain_component(source, cfg, arch, task, options, config)
            comp.exporters, comp.export_options = ["optimum"], opt_opts
            return comp
        L, KV, D = _kv_dims(cfg)
        vocab = cfg["vocab_size"]

        def make(b=2, s=options.seq_len, seed=0) -> Arrays:
            r = np.random.default_rng(seed)
            d: Arrays = {"input_ids": r.integers(1, vocab, (b, s)).astype(np.int64), "attention_mask": np.ones((b, s), np.int64),
                         "position_ids": np.broadcast_to(np.arange(s, dtype=np.int64), (b, s)).copy()}
            for i in range(L):
                d[f"past_key_values.{i}.key"] = np.zeros((b, KV, 0, D), np.float32)
                d[f"past_key_values.{i}.value"] = np.zeros((b, KV, 0, D), np.float32)
            return d
        holder: dict[str, Any] = {}

        def reference(inp: Arrays) -> Arrays:
            torch = deps.require("torch")
            if "m" not in holder:
                holder["m"] = self._load(source, arch, config)
            with torch.inference_mode():
                out = holder["m"](input_ids=torch.from_numpy(inp["input_ids"]), attention_mask=torch.from_numpy(inp["attention_mask"]),
                                  use_cache=False, return_dict=True)
            return {"logits": out.logits.numpy()}
        names = list(make().keys())
        return ComponentExport(
            name="model", build=lambda: None, example_inputs=lambda: make(), input_names=names, output_names=["logits"],
            role="decoder-with-kv-cache", description=f"{arch}: Optimum export (text-generation-with-past); validated on prefill with an empty past",
            exporters=["optimum"], export_options=opt_opts, reference=reference, tolerance=Tolerance(atol=5e-4, rtol=1e-3),
            cases=[ReferenceCase("prefill", lambda: make(), "parity", "empty past"),
                   ReferenceCase("prefill-longer", lambda: make(1, 13, 1), "dynamic_shape")], output_kinds={"logits": "logits"},
            known_issues=["Only prefill with an empty cache is compared to the original; decode steps are not covered for Optimum exports."])

    def _kv_component(self, source, cfg, arch, options: HFOptions, config: JobConfig) -> ComponentExport:
        L, KV, D = _kv_dims(cfg)
        vocab = cfg["vocab_size"]
        B, S, P0 = options.batch, options.seq_len, options.kv_past_len
        scale = 0.5

        def make(b=1, s=S, p=P0, seed=0) -> Arrays:
            r = np.random.default_rng(seed)
            return {"input_ids": r.integers(1, vocab, (b, s)).astype(np.int64),
                    "attention_mask": np.ones((b, p + s), np.int64),
                    "past_keys": (r.standard_normal((L, b, KV, p, D)) * scale).astype(np.float32),
                    "past_values": (r.standard_normal((L, b, KV, p, D)) * scale).astype(np.float32)}
        names = ["input_ids", "attention_mask", "past_keys", "past_values"]
        outs = ["logits", "present_keys", "present_values"]
        dyn = {"input_ids": {0: "batch", 1: "sequence"}, "attention_mask": {0: "batch", 1: "total_length"},
               "past_keys": {1: "batch", 3: "past_length"}, "past_values": {1: "batch", 3: "past_length"},
               "logits": {0: "batch", 1: "sequence"}, "present_keys": {1: "batch", 3: "total_length"},
               "present_values": {1: "batch", 3: "total_length"}}

        def build():
            torch = deps.require("torch")
            tr = _transformers()
            model = self._load(source, arch, config)
            cache_cls = tr.DynamicCache
            fwd = set(inspect.signature(model.forward).parameters)
            cache_arg = "past_key_values" if "past_key_values" in fwd else "cache"

            def new_cache():
                try:
                    return cache_cls(config=model.config)
                except TypeError:
                    return cache_cls()

            def layer_kv(c, i):
                if hasattr(c, "layers"):
                    return c.layers[i].keys, c.layers[i].values
                if hasattr(c, "key_cache"):
                    return c.key_cache[i], c.value_cache[i]
                return c[i][0], c[i][1]

            class Decoder(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m, self.L = m, m.config.num_hidden_layers

                def forward(self, input_ids, attention_mask, past_keys, past_values):
                    cache = new_cache()
                    for i in range(self.L):
                        cache.update(past_keys[i], past_values[i], i)
                    P = past_keys.shape[3]
                    pos = (P + torch.arange(input_ids.shape[1], device=input_ids.device))[None].expand(input_ids.shape[0], -1)
                    out = self.m(input_ids=input_ids, attention_mask=attention_mask, position_ids=pos, use_cache=True,
                                 return_dict=True, **{cache_arg: cache})
                    c = getattr(out, "past_key_values", None) or getattr(out, "cache", None) or cache
                    kv = [layer_kv(c, i) for i in range(self.L)]
                    return out.logits, torch.stack([k for k, _ in kv]), torch.stack([v for _, v in kv])
            return Decoder(model).eval()

        def initial():
            return make(1, S, 0, 5)

        def step_inputs(i: int) -> Arrays:
            r = np.random.default_rng(100 + i)
            total = S + i                   # tokens seen after this step
            return {"input_ids": r.integers(1, vocab, (1, 1)).astype(np.int64), "attention_mask": np.ones((1, total), np.int64)}

        def empty_state(i):
            return None
        stateful = StatefulCase(
            name="decode-8-steps", steps=8, initial=initial, step_inputs=step_inputs,
            links=[StateLink("present_keys", "past_keys", "replace"), StateLink("present_values", "past_values", "replace")],
            description="prefill (empty past) then 7 single-token steps feeding the cache back")
        cases = [ReferenceCase("prefill-with-past", lambda: make(1, S, P0, 0), "parity", "export example (non-empty past)"),
                 ReferenceCase("prefill-empty-past", lambda: make(1, 9, 0, 1), "dynamic_shape", "past length 0"),
                 ReferenceCase("decode-step", lambda: make(2, 1, 11, 2), "dynamic_shape", "one token, past 11, batch 2")]
        return ComponentExport(
            name="model", build=build, example_inputs=lambda: make(), input_names=names, output_names=outs, dynamic_axes=dyn,
            role="decoder-with-kv-cache", description=f"{arch}: prefill+decode in one graph (explicit KV cache)",
            cases=cases, stateful=[stateful], stateful_component=True, tolerance=Tolerance(atol=5e-4, rtol=1e-3),
            output_kinds={"logits": "logits"})

    # -- quality checks ------------------------------------------------------------------------------------------------------
    def _kv_consistency_check(self) -> QualityCheck:
        def run(ctx: QualityContext) -> QualityResult:
            runner = ctx.runner("model")
            sig = {t.name: t for t in getattr(runner, "signature", None).inputs}
            past = sig["past_keys"].shape
            L, KV, D = past[0], past[2], past[4]
            r = np.random.default_rng(ctx.seed)
            vocab = 50
            ids = r.integers(1, vocab, (1, 10)).astype(np.int64)
            z = lambda p: np.zeros((L, 1, KV, p, D), np.float32)
            full = runner.run({"input_ids": ids, "attention_mask": np.ones((1, 10), np.int64), "past_keys": z(0), "past_values": z(0)})["logits"]
            k, v = z(0), z(0)
            steps = []
            for t in range(10):
                o = runner.run({"input_ids": ids[:, t:t + 1], "attention_mask": np.ones((1, t + 1), np.int64), "past_keys": k, "past_values": v})
                k, v = o["present_keys"], o["present_values"]
                steps.append(o["logits"][:, 0])
            inc = np.stack(steps, 1)
            diff = float(np.abs(full - inc).max())
            scale = float(np.abs(full).max()) or 1.0
            ok = diff <= 5e-3 * max(1.0, scale)
            return QualityResult(ok, {"max_abs_diff": diff, "logit_scale": scale},
                                 "one-shot prefill and token-by-token decode through the cache agree" if ok else "cache path diverges from one-shot prefill")
        return QualityCheck("kv-cache-consistency", run, "full-sequence logits == incremental logits through the exported KV cache")

    def _greedy_check(self, source: ModelSource, arch: str, options: HFOptions, config: JobConfig) -> QualityCheck:
        def run(ctx: QualityContext) -> QualityResult:
            torch = deps.require("torch")
            model = self._load(source, arch, config)
            runner = ctx.runner("model")
            sig = {t.name: t for t in runner.signature.inputs}
            past = sig["past_keys"].shape
            L, KV, D = past[0], past[2], past[4]
            vocab = model.config.vocab_size
            prompt = np.random.default_rng(ctx.seed + 7).integers(1, vocab, (1, 6)).astype(np.int64)
            n = options.greedy_tokens
            with torch.inference_mode():
                ref = model.generate(torch.from_numpy(prompt), attention_mask=torch.ones(1, 6, dtype=torch.long), max_new_tokens=n,
                                     do_sample=False, min_new_tokens=n, pad_token_id=0)[0, 6:].numpy()
            k = np.zeros((L, 1, KV, 0, D), np.float32)
            v = k.copy()
            ids, got, total = prompt, [], 0
            for _ in range(n):
                total += ids.shape[1]
                o = runner.run({"input_ids": ids, "attention_mask": np.ones((1, total), np.int64), "past_keys": k, "past_values": v})
                k, v = o["present_keys"], o["present_values"]
                nxt = int(o["logits"][0, -1].argmax())
                got.append(nxt)
                ids = np.array([[nxt]], np.int64)
            match = float(np.mean(np.array(got) == ref))
            return QualityResult(match == 1.0, {"tokens": n, "agreement": match, "reference": ref.tolist(), "onnx": got},
                                 "greedy decode identical to the original model" if match == 1.0 else "greedy decode differs from the original model")
        return QualityCheck("greedy-generation", run, "greedy decoding through the ONNX KV cache vs transformers generate()", expensive=False)
