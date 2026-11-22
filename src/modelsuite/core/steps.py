"""Pipeline steps. Each takes a JobContext, mutates ctx and the manifest, and writes its report.

discover -> inspect -> analyze -> export -> optimize -> validate -> benchmark -> package
Heavy work (export, optimize, validate, benchmark) is skipped on resume when its results are already in the manifest
and the files on disk still match their checksums."""
from __future__ import annotations

import json
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from .. import __version__
from . import deps, licenses
from .config import JobConfig, Tolerance
from .context import JobContext
from .discovery import ModelSource, resolve
from .errors import ArtifactError, ConfigError, ExportError, ModelSuiteError, UnsupportedModelError
from .golden import GoldenStore
from .manifest import (Artifact, AssetRecord, ComponentRecord, Manifest, QuantizationRecord, Redistribution, SourceInfo, ToolVersion,
                       VariantRecord, collect_files, sha256_file, utcnow)
from .plan import Arrays, ComponentExport, ReferenceCase
from .registry import exporters as exporter_registry
from .registry import load_plugins
from .types import Issue, Severity, Status, Support

DEFAULT_TORCH_ORDER = ["torch-legacy", "torch-dynamo"]


# ============================================================================== discover / inspect / analyze
def discover(ctx: JobContext) -> None:
    from ..adapters.selection import select_adapter
    load_plugins(ctx.config.plugins)
    cfg = ctx.config
    ctx.say("discover", f"resolving {cfg.model}")
    ctx.source = resolve(cfg.model, cfg.revision, offline=cfg.offline)
    sel = select_adapter(ctx.source, cfg.adapter)
    ctx.adapter, ctx.detection, ctx.candidates = sel.adapter, sel.detection, sel.candidates
    ctx.adapter_options = ctx.adapter.parse_options(cfg.adapter_options)
    ctx.say("discover", f"adapter '{ctx.adapter.name}' (confidence {sel.detection.confidence:.2f}: {sel.detection.reason})")


def inspect_step(ctx: JobContext) -> None:
    ctx.info = ctx.adapter.inspect(ctx.source, ctx.adapter_options)
    if ctx.info.license is None:
        ctx.info.license = ctx.source.license().id
    ctx.save_report("inspect.json", {"model": ctx.info.model_dump(mode="json"), "adapter": {"name": ctx.adapter.name, "version": ctx.adapter.version,
                    "confidence": ctx.detection.confidence, "reason": ctx.detection.reason,
                    "candidates": [{"adapter": n, "confidence": d.confidence, "reason": d.reason} for n, d in ctx.candidates]},
                    "license": ctx.license().model_dump(mode="json")})


def analyze(ctx: JobContext, *, raise_on_unsupported: bool = True) -> None:
    ctx.compat = ctx.adapter.compatibility(ctx.source, ctx.info, ctx.adapter_options, ctx.config)
    lic = ctx.license()
    if lic.redistribution in ("no", "unknown", "conditional"):
        ctx.compat.issues.append(Issue(
            severity=Severity.warning if lic.redistribution != "no" else Severity.error, code="license",
            message=f"Model license: {lic.id or 'unknown'} ({lic.category}); redistribution of converted weights: {lic.redistribution}. {lic.notes}"))
    ctx.save_report("analyze.json", ctx.compat)
    if raise_on_unsupported and ctx.compat.support == Support.unsupported:
        errs = "; ".join(i.message for i in ctx.compat.errors) or ctx.compat.summary
        raise UnsupportedModelError(f"{ctx.info.architecture} cannot be converted: {errs} (full report: {ctx.artifact.reports_dir / 'analyze.json'})")


# ============================================================================== manifest bootstrap
def _tool(name: str) -> ToolVersion:
    return ToolVersion(name=name, version=deps.dist_version(name) or "?")


def new_manifest(ctx: JobContext) -> Manifest:
    lic = ctx.license()
    m = Manifest(
        suite_version=__version__,
        source=SourceInfo(reference=ctx.config.model, kind=ctx.source.kind, repo_id=ctx.source.repo_id, revision=ctx.source.revision,
                          revision_kind=ctx.source.revision_kind, framework=ctx.info.framework, architecture=ctx.info.architecture,
                          model_type=ctx.info.model_type, parameters=ctx.info.parameters, license=lic),
        adapter=ToolVersion(name=ctx.adapter.name, version=ctx.adapter.version), adapter_options=ctx.config.adapter_options,
        job_config=ctx.config.model_dump(mode="json"), job_fingerprint=ctx.config.fingerprint(), environment=deps.environment_snapshot())
    m.licenses = {"model": lic.model_dump(mode="json"),
                  "dependencies": licenses.dependency_licenses(["onnx", "onnxruntime", "torch", "transformers", "numpy", "onnxscript"])}
    m.redistribution = _redistribution(lic)
    m.notes.append(f"reproduce: modelsuite convert {ctx.config.model!r} --revision {ctx.source.revision} (see job_config in this manifest)")
    return m


def _redistribution(lic) -> Redistribution:
    notice = []
    if lic.category == "non-commercial":
        notice.append("Non-commercial use only; the converted files inherit this restriction.")
    if lic.notes:
        notice.append(lic.notes)
    if lic.redistribution == "allowed":
        return Redistribution(allowed=True, status="allowed", reason=f"license '{lic.id}' is permissive", required_notices=notice)
    if lic.redistribution == "no":
        return Redistribution(allowed=False, status="no", reason=f"license '{lic.id}' forbids redistributing derivatives", required_notices=notice)
    if lic.redistribution == "conditional":
        return Redistribution(allowed=None, status="conditional", reason=f"license '{lic.id}' allows redistribution only under conditions; read them", required_notices=notice)
    return Redistribution(allowed=None, status="unknown", reason="license not determined: do NOT redistribute the converted weights until it is confirmed", required_notices=notice)


def _apply_component_licenses(ctx: JobContext) -> None:
    """Record per-component licenses and make the redistribution verdict the most restrictive of all of them."""
    m, plan = ctx.manifest, ctx.plan
    if not plan.component_licenses:
        return
    infos = {c: licenses.classify(lid, "adapter") for c, lid in plan.component_licenses.items()}
    m.licenses["components"] = {c: i.model_dump(mode="json") for c, i in infos.items()}
    allowed, reasons, notices = [m.redistribution.allowed if m.redistribution else None], [], list(m.redistribution.required_notices if m.redistribution else [])
    statuses = [m.redistribution.status if m.redistribution else "unknown"]
    for c, i in infos.items():
        r = _redistribution(i)
        allowed.append(r.allowed)
        statuses.append(r.status)
        reasons.append(f"{c}: {r.reason}")
        notices += [f"{c}: {n}" for n in r.required_notices if n]
    verdict = False if False in allowed else (None if None in allowed else True)
    worst = next(st for st in ("no", "unknown", "conditional", "allowed") if st in statuses)       # most restrictive wins
    m.redistribution = Redistribution(allowed=verdict, status=worst, reason="; ".join([m.redistribution.reason if m.redistribution else "", *reasons]).strip("; "),
                                      required_notices=list(dict.fromkeys(notices)))


def load_manifest_or_new(ctx: JobContext) -> Manifest:
    if ctx.artifact.exists():
        m = ctx.artifact.read()
        if m.job_fingerprint != ctx.config.fingerprint() and not ctx.forced("all"):
            raise ConfigError(f"{ctx.artifact.root} holds a job with a different configuration (fingerprint {m.job_fingerprint} != "
                              f"{ctx.config.fingerprint()}). Use another --output or pass --force to redo it.")
        return m
    return new_manifest(ctx)


# ============================================================================== export
def _variant_files(artifact: Artifact, variant: str, components: list[str]) -> tuple[dict, dict]:
    files, sizes = {}, {}
    for c in components:
        g = artifact.graph_path(variant, c)
        entries = collect_files(artifact.root, g)
        files[c] = entries
        sizes[c] = sum(e.size for e in entries)
    return files, sizes


def _verify_component_files(artifact: Artifact, m: Manifest, variant: str, comp: str) -> bool:
    v = m.variants.get(variant)
    if not v or comp not in v.files:
        return False
    for fe in v.files[comp]:
        p = artifact.root / fe.path
        if not p.is_file() or p.stat().st_size != fe.size:
            return False
    return True


def _order(comp: ComponentExport, requested: str) -> list[str]:
    from ..exporters.base import order_exporters
    return order_exporters(comp, requested, DEFAULT_TORCH_ORDER)


def _smoke_parity(ctx: JobContext, comp: ComponentExport, path: Path) -> tuple[bool, dict[str, Any]]:
    """First parity case through ORT vs the original: used to choose between exporter strategies."""
    from ..exporters.torch_support import reference_runner
    from ..runtimes.ort import create_runner
    from ..validation.metrics import compare_arrays, summarize
    case = next((c for c in comp.cases if c.kind == "parity"), None)
    if case is None:
        return True, {"skipped": "no parity case"}
    inputs = case.inputs()
    ref = reference_runner(comp)(inputs)
    got = create_runner(path, "cpu", custom_op_libraries=ctx.plan.custom_op_libraries if ctx.plan else None).run(inputs)
    tol = case.tolerance or comp.tolerance or ctx.config.validation.tolerance
    comps = [compare_arrays(n, ref[n], got[n], tol, comp.output_kinds.get(n, "")) for n in comp.output_names if n in ref and n in got]
    s = summarize(comps)
    return bool(s["passed"] and comps), {"max_abs": s["max_abs"], "case": case.name, "reasons": [r for c in comps for r in c.reasons]}


def _export_component(ctx: JobContext, comp: ComponentExport):
    """Export one component; returns its record. Falls through the exporter list, gating on a parity smoke test when
    the exporter was left on 'auto' and there is more than one candidate."""
    from ..runtimes.onnx_utils import graph_stats, graph_signature
    out = ctx.artifact.graph_path("fp32", comp.name)
    out.parent.mkdir(parents=True, exist_ok=True)
    order = _order(comp, ctx.config.export.exporter)
    auto = ctx.config.export.exporter in ("auto", "") and len(order) > 1 and bool(comp.input_names)
    attempts: list[dict[str, Any]] = []
    chosen: Any = None
    kept_first: Any = None
    for name in order:
        try:
            exp = exporter_registry.get(name)
            exp = exp() if isinstance(exp, type) else exp
        except ModelSuiteError as exc:
            attempts.append({"exporter": name, "status": "unknown", "reason": str(exc)})
            continue
        ok, why = exp.availability()
        if not ok:
            attempts.append({"exporter": name, "status": "unavailable", "reason": why})
            continue
        sup, swhy = exp.supports(comp)
        if not sup:
            attempts.append({"exporter": name, "status": "unsupported", "reason": swhy})
            continue
        ctx.say("export", f"{comp.name}: exporting with {name}")
        try:
            res = exp.export(comp, out, ctx.config.export)
        except ExportError as exc:
            attempts.append({"exporter": name, "status": "failed", "reason": str(exc)[:600]})
            continue
        att = {"exporter": name, "strategy": res.strategy, "status": "exported", "seconds": res.seconds, "warnings": res.warnings}
        attempts.append(att)
        if auto:
            try:
                good, info = _smoke_parity(ctx, comp, out)
            except Exception as exc:
                good, info = False, {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
            att["smoke"] = info
            if not good:
                att["status"] = "parity-failed"
                ctx.say("export", f"{comp.name}: {name} output does not match the original ({info}); trying the next strategy")
                if kept_first is None:
                    kept_first = (res, out.with_name(f"{comp.name}.{name}.rejected.onnx"))
                    # keep the first result around in case no strategy passes
                    shutil.copy2(out, kept_first[1])
                    ext = out.with_name(out.name + ".data")
                    if ext.exists():
                        shutil.copy2(ext, kept_first[1].with_name(kept_first[1].name + ".data"))
                continue
        att["status"] = "ok"
        chosen = res
        break
    if chosen is None and kept_first is not None:
        res, backup = kept_first
        shutil.copy2(backup, out)
        if backup.with_name(backup.name + ".data").exists():
            shutil.copy2(backup.with_name(backup.name + ".data"), out.with_name(out.name + ".data"))
        chosen = res
        attempts.append({"exporter": res.exporter, "status": "kept", "reason": "no strategy passed the parity smoke test; the first export is kept so "
                         "validation can report exactly how it differs"})
    for f in out.parent.glob("*.rejected.onnx*"):
        f.unlink()
    if chosen is None:
        lines = "; ".join(f"{a['exporter']}: {a.get('status')} ({a.get('reason', '')[:160]})" for a in attempts)
        raise ExportError(f"no exporter produced a graph for component '{comp.name}': {lines}", attempts)
    sig = graph_signature(out)
    stats = graph_stats(out)
    rec = ComponentRecord(name=comp.name, role=comp.role, description=comp.description, signature=sig, stateful=comp.stateful_component,
                          state_links=[{"output": l.output, "input": l.input, "mode": l.mode, "axis": l.axis} for s in comp.stateful for l in s.links][:8],
                          exporter=chosen.exporter, strategy=chosen.strategy, opset=stats.get("opset"), known_issues=list(comp.known_issues),
                          provider_notes=dict(comp.provider_notes), export_attempts=attempts, quantize_exclude_ops=list(comp.quantize_exclude_ops),
                          output_kinds=dict(comp.output_kinds), tolerance=(comp.tolerance or ctx.config.validation.tolerance).model_dump())
    extras = [(n, p) for n, p in chosen.extra_components.items()]
    return rec, extras, chosen


def export(ctx: JobContext) -> None:
    cfg = ctx.config
    art = ctx.artifact
    m = ctx.manifest
    if cfg.export.fetch_weights and not ctx.source.weights_complete:
        ctx.say("export", f"downloading weights of {ctx.source.reference}")
    if cfg.export.fetch_weights:
        ctx.source.fetch_weights(ctx.adapter.weight_patterns(ctx.source, ctx.adapter_options))
    ctx.plan = ctx.adapter.plan(ctx.source, ctx.adapter_options, cfg)
    plan = ctx.plan
    _apply_component_licenses(ctx)
    m.runtime_requirements = ctx.adapter.runtime_requirements(plan)
    m.topology = [{"from": e.src, "output": e.output, "to": e.dst, "input": e.input} for e in plan.topology]
    m.notes.extend(n for n in plan.notes if n not in m.notes)
    golden = GoldenStore(art)
    cap = cfg.export.golden_max_mb * 1024 * 1024
    if "fp32" not in m.variants:
        m.variants["fp32"] = VariantRecord(name="fp32", precision="fp32")
    base = m.variants["fp32"]
    used_exporters: dict[str, str] = {}
    for comp in plan.components:
        if not ctx.forced("export") and _verify_component_files(art, m, "fp32", comp.name) and any(c.name == comp.name for c in m.components):
            ctx.say("export", f"{comp.name}: resumed (graph already exported and intact)")
            continue
        t = time.perf_counter()
        rec, extras, res = _export_component(ctx, comp)
        m.components = [c for c in m.components if c.name != comp.name] + [rec]
        for n, v in res.tool_versions.items():
            used_exporters[f"{rec.exporter}:{n}"] = v
        for n, p in extras:  # extra graphs a model-scope exporter produced
            from ..runtimes.onnx_utils import graph_signature, graph_stats
            ep = Path(p)
            m.components.append(ComponentRecord(name=n, role="extra", description=f"additional graph written by {rec.exporter}",
                                                signature=graph_signature(ep), exporter=rec.exporter, strategy=rec.strategy,
                                                opset=graph_stats(ep).get("opset")))
        # golden references from the ORIGINAL model while its module is loaded
        if cap > 0 and comp.cases and comp.input_names:
            from ..exporters.torch_support import reference_runner
            ref = reference_runner(comp)
            for case in comp.cases:
                try:
                    inputs = case.inputs()
                    outputs = ref(inputs)
                except Exception as exc:
                    ctx.say("export", f"{comp.name}/{case.name}: could not compute a reference ({type(exc).__name__}: {exc})")
                    continue
                est = sum(a.nbytes for a in inputs.values()) + sum(a.nbytes for a in outputs.values())
                if golden.bytes(comp.name) + est > cap:
                    ctx.say("export", f"{comp.name}/{case.name}: not bundled (would exceed golden_max_mb={cfg.export.golden_max_mb})")
                    continue
                golden.save(comp.name, case.name, case.kind, inputs, outputs)
            m.reference_cases[comp.name] = list(golden.cases(comp.name))
        comp.release()
        files, sizes = _variant_files(art, "fp32", [c.name for c in m.components if c.name == comp.name or c.name in dict(extras)])
        base.files.update(files)
        base.sizes_bytes.update(sizes)
        base.component_precision[comp.name] = "fp32"
        ctx.say("export", f"{comp.name}: done in {time.perf_counter() - t:.1f}s ({sum(sizes.values()) / 2 ** 20:.1f} MiB)")
        art.write(m)  # checkpoint: a crash after this point resumes at the next component
    if cfg.export.copy_assets:
        _copy_assets(ctx)
    if used_exporters:   # a fully resumed export exported nothing new: keep the versions recorded by the run that did the work
        merged = {e.name: e.version for e in m.exporters}
        merged.update(used_exporters)
        m.exporters = [ToolVersion(name=k, version=v) for k, v in sorted(merged.items())]
    m.default_variant = "fp32"
    art.write(m)
    ctx.save_report("export.json", {"components": [c.model_dump(mode="json") for c in m.components]})


def _copy_assets(ctx: JobContext) -> None:
    art, m, plan = ctx.artifact, ctx.manifest, ctx.plan
    art.assets_dir.mkdir(parents=True, exist_ok=True)
    seen = {a.path for a in m.assets}
    for a in plan.assets:
        if a.source is None:
            if f"assets/{a.dest}" not in seen:
                m.assets.append(AssetRecord(role=a.role, path=a.dest, required=a.required, description=a.description or "external (not bundled)",
                                            license_note=a.license_note))
            continue
        if not Path(a.source).is_file():
            if a.required:
                ctx.say("export", f"warning: required asset {a.source} is missing")
            continue
        dest = art.assets_dir / a.dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(a.source, dest)
        rel = dest.relative_to(art.root).as_posix()
        if rel not in seen:
            m.assets.append(AssetRecord(role=a.role, path=rel, required=a.required, description=a.description, sha256=sha256_file(dest),
                                        license_note=a.license_note))
    # files written by model-scope exporters next to the graphs
    for d in art.models_dir("fp32").glob("_*_files"):
        for f in d.iterdir():
            dest = art.assets_dir / f.name
            if not dest.exists():
                shutil.move(str(f), str(dest))
                m.assets.append(AssetRecord(role="config", path=dest.relative_to(art.root).as_posix(), sha256=sha256_file(dest)))
        shutil.rmtree(d, ignore_errors=True)


# ============================================================================== structural + optimize
def structural(ctx: JobContext, variant: str) -> dict[str, Any]:
    from ..validation.structural import check_graph
    m, art = ctx.manifest, ctx.artifact
    res = {}
    for c in m.variants[variant].files:
        r = check_graph(art.graph_path(variant, c))
        res[c] = r.model_dump(mode="json")
    v = m.variants[variant]
    v.validation = {**(v.validation or {}), "structural": res}
    return res


def variant_name(kind: str, value: str) -> str:
    return {"graph": f"opt-{value}"}.get(kind, value)


def planned_variants(cfg) -> list[tuple[str, str, str]]:
    """[(variant name, kind, value)] from the optimisation config."""
    out = []
    if cfg.graph_optimization != "none":
        out.append((f"opt-{cfg.graph_optimization}", "graph", cfg.graph_optimization))
    for p in cfg.precision:
        if p != "fp32":
            out.append((p, "precision", p))
    for q in cfg.quantization:
        if q != "none":
            out.append((q, "quantization", q))
    seen, uniq = set(), []
    for t in out:
        if t[0] not in seen:
            seen.add(t[0])
            uniq.append(t)
    return uniq


def optimize(ctx: JobContext, parent: str = "fp32") -> list[str]:
    from ..optimization import passes
    m, art, cfg = ctx.manifest, ctx.artifact, ctx.config.optimization
    golden = GoldenStore(art)
    created, report = [], {}
    for name, kind, value in planned_variants(cfg):
        if name in m.variants and not ctx.forced("optimize") and all(_verify_component_files(art, m, name, c) for c in m.variants[name].files) \
                and m.variants[name].files:
            ctx.say("optimize", f"{name}: resumed")
            continue
        ctx.say("optimize", f"building variant '{name}' from '{parent}'")
        v = VariantRecord(name=name, parent=parent, precision="fp32", graph_optimization="none")
        details: dict[str, Any] = {}
        for rec in m.components:
            src = art.graph_path(parent, rec.name)
            dst = art.graph_path(name, rec.name)
            if not src.is_file():
                continue
            try:
                if rec.name in cfg.exclude_components:
                    raise passes.OptimizationError("excluded by configuration")
                if kind == "graph":
                    r = passes.graph_optimize(src, dst, value)
                    v.graph_optimization = value
                elif kind == "precision" and value == "fp16":
                    r = passes.convert_fp16(src, dst, cfg.fp16_keep_io_types, cfg.fp16_op_block_list)
                    v.precision = "fp16"
                elif kind == "precision":
                    r = passes.cast_weights(src, dst, "float16" if value == "fp16-weights" else "bfloat16")
                    v.precision = value
                elif value == "int8-dynamic":
                    r = passes.quantize_int8_dynamic(src, dst, exclude_ops=rec.quantize_exclude_ops)
                    v.quantization = QuantizationRecord(method="int8-dynamic", details=r.details)
                elif value == "int8-static":
                    samples = passes.load_calibration(cfg.calibration_dir, cfg.calibration_samples) if cfg.calibration_dir \
                        else golden.all_inputs(rec.name)[:cfg.calibration_samples]
                    r = passes.quantize_int8_static(src, dst, samples, exclude_ops=rec.quantize_exclude_ops)
                    v.quantization = QuantizationRecord(method="int8-static", details={**r.details, "calibration": cfg.calibration_dir or "golden test inputs"})
                elif value == "int4":
                    r = passes.quantize_int4_weights(src, dst, cfg.int4_block_size, exclude_ops=rec.quantize_exclude_ops)
                    v.quantization = QuantizationRecord(method="int4", details=r.details)
                else:
                    raise ConfigError(f"unknown optimisation '{value}'")
                v.component_precision[rec.name] = value if kind != "graph" else f"fp32+{value}"
                v.notes.extend(f"{rec.name}: {n}" for n in r.notes)
                details[rec.name] = r.details
            except (passes.OptimizationError, ConfigError) as exc:
                # keep the parent's graph for this component and SAY SO; never pretend the variant is uniformly converted
                v.failed_components[rec.name] = str(exc)[:500]
                v.component_precision[rec.name] = f"{parent} (conversion failed)"
                for f in [dst, dst.with_name(dst.name + ".data")]:
                    if f.exists():
                        f.unlink()
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                if src.with_name(src.name + ".data").exists():
                    shutil.copy2(src.with_name(src.name + ".data"), dst.with_name(dst.name + ".data"))
                ctx.say("optimize", f"{name}/{rec.name}: {exc} -> kept {parent} graph for this component")
        if len(v.failed_components) == len(m.components):
            ctx.say("optimize", f"{name}: every component failed; variant not created")
            report[name] = {"status": "failed", "failed_components": v.failed_components}
            shutil.rmtree(art.models_dir(name), ignore_errors=True)
            continue
        v.files, v.sizes_bytes = _variant_files(art, name, [c.name for c in m.components if art.graph_path(name, c.name).is_file()])
        m.variants[name] = v
        structural(ctx, name)
        created.append(name)
        report[name] = {"status": "ok" if not v.failed_components else "partial", "failed_components": v.failed_components,
                        "size_bytes": v.total_bytes, "parent_size_bytes": m.variants[parent].total_bytes}
        art.write(m)
    ctx.save_report("optimize.json", report)
    return created


# ============================================================================== validate
def _same_inputs(a: Arrays, b: Arrays) -> bool:
    return a.keys() == b.keys() and all(a[k].shape == b[k].shape and np.array_equal(a[k], b[k]) for k in a)


def component_stub(rec: ComponentRecord, golden: GoldenStore, cfg: JobConfig) -> ComponentExport:
    """A ComponentExport rebuilt from the manifest + golden files, for validation without the original model."""
    cases = []
    for name, meta in golden.cases(rec.name).items():
        cases.append(ReferenceCase(name, (lambda n=name: golden.load(rec.name, n)[0]), meta.get("kind", "parity")))
    return ComponentExport(name=rec.name, build=lambda: None, example_inputs=lambda: {}, input_names=[t.name for t in rec.signature.inputs],
                           output_names=[t.name for t in rec.signature.outputs], cases=cases, output_kinds=dict(rec.output_kinds),
                           tolerance=Tolerance(**rec.tolerance) if rec.tolerance else None, provider_notes=dict(rec.provider_notes))


def _reference_fn(ctx: JobContext, comp: ComponentExport, golden: GoldenStore):
    from ..exporters.torch_support import reference_runner
    live = ctx.live and ctx.plan is not None and comp.name in [c.name for c in ctx.plan.components]
    live_comp = ctx.plan.component(comp.name) if live else None

    def fn(case: ReferenceCase, inputs: Arrays):
        if golden.has(comp.name, case.name) and (ctx.trust_golden or live_comp is None):
            gin, gout = golden.load(comp.name, case.name)
            if _same_inputs(gin, inputs):
                return gout
        if live_comp is not None:
            return reference_runner(live_comp)(inputs)
        return None
    return fn


def validate(ctx: JobContext, variants: list[str] | None = None, providers: list[str] | None = None, quality: bool | None = None) -> dict[str, Any]:
    from ..runtimes.ort import create_runner
    from ..validation import parity as P
    from ..validation import providers as PR
    from ..validation import stateful as S
    from ..core.plan import QualityContext, QualityResult
    m, art, vcfg = ctx.manifest, ctx.artifact, ctx.config.validation
    providers = providers or vcfg.providers
    quality = vcfg.run_quality_checks if quality is None else quality
    golden = GoldenStore(art)
    names = variants or list(m.variants)
    out: dict[str, Any] = {}
    custom_libs = ctx.plan.custom_op_libraries if ctx.plan else []
    for vn in names:
        v = m.variant(vn)
        if v.validation and v.validation.get("status") and not ctx.forced("validate") and variants is None:
            out[vn] = v.validation
            ctx.say("validate", f"{vn}: resumed (already validated: {v.validation['status']})")
            continue
        ctx.say("validate", f"validating variant '{vn}'")
        struct = structural(ctx, vn)
        comp_results: dict[str, Any] = {}
        failed = [c for c, r in struct.items() if r["status"] != "passed"]
        runners: dict[str, Any] = {}

        def runner_for(cn: str, vn=vn):
            if (vn, cn) not in runners:
                runners[(vn, cn)] = create_runner(art.graph_path(vn, cn), "cpu", threads=vcfg.threads, custom_op_libraries=custom_libs)
            return runners[(vn, cn)]
        for rec in m.components:
            if rec.name not in v.files:
                continue
            res: dict[str, Any] = {"structural": struct.get(rec.name)}
            comp = ctx.plan.component(rec.name) if (ctx.live and ctx.plan and rec.name in [c.name for c in ctx.plan.components]) else None
            if comp is None or not comp.cases:
                comp = component_stub(rec, golden, ctx.config)
            if struct.get(rec.name, {}).get("status") != "passed":
                res["parity"] = {"status": "skipped", "reason": "structural validation failed"}
                comp_results[rec.name] = res
                continue
            try:
                runner = runner_for(rec.name)
                reference = _reference_fn(ctx, comp, golden)
                prec = v.component_precision.get(rec.name, v.precision).split(" ")[0].split("+")[0]
                if v.quantization.method != "none":
                    prec = v.quantization.method
                rep = P.run_parity(comp, runner, reference, vcfg, precision=prec if prec != "fp32" else "fp32", signature=runner.signature)
                res["parity"] = rep.model_dump(mode="json")
                if vcfg.check_stateful and ctx.live and ctx.plan and rec.name in [c.name for c in ctx.plan.components]:
                    from ..exporters.torch_support import reference_runner
                    lc = ctx.plan.component(rec.name)
                    res["stateful"] = [S.run_stateful(lc, sc, runner, reference_runner(lc), vcfg, prec).model_dump(mode="json") for sc in lc.stateful]
                elif vcfg.check_stateful and rec.stateful:
                    res["stateful"] = [{"status": "skipped", "reason": "stateful cases need the original model (pass the model reference)"}]
                # provider compatibility
                prov_out = []
                first = golden.first_inputs(rec.name)
                if first is None and comp.cases:
                    first = (comp.cases[0].name, comp.cases[0].inputs())
                for p in providers:
                    if p.lower() in ("cpu", "cpuexecutionprovider") or first is None:
                        continue
                    pc = PR.check_provider(rec.name, art.graph_path(vn, rec.name), p, first[1], P.effective_tolerance(comp, None, vcfg, prec),
                                           known_issue=rec.provider_notes.get(p.lower()), threads=vcfg.threads, output_kinds=comp.output_kinds)
                    prov_out.append(pc.model_dump(mode="json"))
                if prov_out:
                    res["providers"] = prov_out
            except Exception as exc:
                res["parity"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc(limit=4)}
            comp_results[rec.name] = res
        # quality checks need every component of the variant
        qres: list[dict[str, Any]] = []
        if quality and ctx.live and ctx.plan and ctx.plan.quality_checks:
            qctx = QualityContext(variant=vn, runner=runner_for, seed=vcfg.seed)
            for qc in ctx.plan.quality_checks:
                ctx.say("validate", f"{vn}: quality check '{qc.name}'")
                try:
                    r = qc.run(qctx)
                except Exception as exc:
                    r = QualityResult(None, {}, f"{type(exc).__name__}: {exc}")
                qres.append({"name": qc.name, "description": qc.description, "passed": r.passed, "metrics": r.metrics, "message": r.message})
        v.validation = {**(v.validation or {}), "components": comp_results, "quality": qres, "validated_at": utcnow(),
                        "providers_requested": providers, "original_model_used": ctx.live, "status": _variant_status(v, comp_results, qres)}
        out[vn] = v.validation
        art.write(m)
    m.validation = {"status": _overall(m), "variants": {n: {"status": (m.variants[n].validation or {}).get("status", "not-run")} for n in m.variants},
                    "updated": utcnow()}
    ctx.save_report("validation.json", {n: m.variants[n].validation for n in m.variants})
    art.write(m)
    return out


def _variant_status(v: VariantRecord, comps: dict, quality: list) -> str:
    bad = False
    ran = False
    for r in comps.values():
        p = r.get("parity", {})
        if r["structural"] and r["structural"]["status"] != "passed":
            bad = True
        st = p.get("status")
        if st in ("failed", "error"):
            bad = True
        if st == "passed":
            ran = True
        for s in r.get("stateful", []):
            if s.get("status") in ("failed", "error"):
                bad = True
            if s.get("status") == "passed":
                ran = True
    if any(q["passed"] is False for q in quality):
        bad = True
    if any(q["passed"] for q in quality):
        ran = True
    if bad:
        return "failed"
    return "passed" if ran else "not-verified"


def _overall(m: Manifest) -> str:
    sts = [(v.validation or {}).get("status", "not-run") for v in m.variants.values()]
    if any(s == "failed" for s in sts):
        return "failed" if (m.variants.get(m.default_variant, VariantRecord(name="x")).validation or {}).get("status") == "failed" else "partial"
    if all(s == "passed" for s in sts):
        return "passed"
    return "partial" if any(s == "passed" for s in sts) else "not-verified"


# ============================================================================== benchmark
def benchmark(ctx: JobContext, variants: list[str] | None = None, providers: list[str] | None = None) -> dict[str, Any]:
    from ..benchmarking.compare import pipeline_latency_ms
    from ..benchmarking.runner import benchmark_graph
    from ..runtimes.onnx_utils import synth_inputs
    from ..optimization.compare import VariantMeasurement, recommend
    m, art, bcfg = ctx.manifest, ctx.artifact, ctx.config.benchmark
    providers = providers or bcfg.providers
    golden = GoldenStore(art)
    names = variants or list(m.variants)
    allres: dict[str, Any] = {}
    for vn in names:
        v = m.variant(vn)
        if v.benchmark and not ctx.forced("benchmark") and variants is None and v.benchmark.get("providers") == providers:
            allres[vn] = v.benchmark
            ctx.say("benchmark", f"{vn}: resumed")
            continue
        rows = []
        for rec in m.components:
            if rec.name not in v.files:
                continue
            first = golden.first_inputs(rec.name)
            if first:
                src, inputs = f"golden case '{first[0]}'", first[1]
            else:
                src, inputs = "synthetic (random) inputs: indices/masks may be unrepresentative", synth_inputs(rec.signature, bcfg.seed)
            for p in providers:
                ctx.say("benchmark", f"{vn}/{rec.name} on {p}")
                r = benchmark_graph(art.graph_path(vn, rec.name), p, inputs, bcfg, rec.name, vn, src)
                rows.append(r)
        by_provider = {}
        for p in providers:
            rs = [r for r in rows if r.provider == p.lower().replace("executionprovider", "") or r.provider == p]
            by_provider[p] = pipeline_latency_ms(rs)
        v.benchmark = {"providers": providers, "results": [r.model_dump(mode="json") for r in rows], "pipeline_latency_ms": by_provider,
                       "measured_at": utcnow(), "iterations": bcfg.iterations, "warmup": bcfg.warmup}
        allres[vn] = v.benchmark
        art.write(m)
    primary = providers[0]
    meas = []
    for n, v in m.variants.items():
        vs = v.validation or {}
        worst, cos = None, None
        for r in (vs.get("components") or {}).values():
            for c in (r.get("parity") or {}).get("cases", []):
                s = c.get("summary") or {}
                worst = max(worst or 0.0, s.get("max_abs") or 0.0)
                if s.get("min_cosine") is not None:
                    cos = min(cos if cos is not None else 1.0, s["min_cosine"])
        rss = [r["peak_rss_mb"] for r in (v.benchmark or {}).get("results", []) if r.get("peak_rss_mb")]
        meas.append(VariantMeasurement(variant=n, precision=v.precision, quantization=v.quantization.method, size_bytes=v.total_bytes,
                                       validation_passed=None if vs.get("status") in (None, "not-verified") else vs.get("status") == "passed",
                                       worst_max_abs=worst, min_cosine=cos, latency_ms=((v.benchmark or {}).get("pipeline_latency_ms") or {}).get(primary),
                                       peak_rss_mb=max(rss) if rss else None))
    rec_ = recommend(meas, baseline="fp32" if "fp32" in m.variants else next(iter(m.variants)), objective="balanced")
    m.benchmark = {"providers": providers, "recommendation": rec_.model_dump(mode="json"), "updated": utcnow()}
    ctx.save_report("benchmark.json", {"variants": {n: m.variants[n].benchmark for n in m.variants}, "recommendation": rec_.model_dump(mode="json")})
    art.write(m)
    return allres


# ============================================================================== package
def package(ctx: JobContext) -> Path:
    from .reports import render_text
    m, art = ctx.manifest, ctx.artifact
    m.environment = deps.environment_snapshot()
    (art.root / "environment.json").write_text(json.dumps(m.environment, indent=2) + "\n")
    bad = art.verify()
    if bad:
        raise ArtifactError("artifact integrity check failed after packaging: " + "; ".join(bad[:5]))
    art.write(m)
    text = render_text(m)
    ctx.artifact.reports_dir.mkdir(parents=True, exist_ok=True)
    (art.reports_dir / "summary.txt").write_text(text)
    ctx.save_report("summary.json", {"validation": m.validation, "benchmark": m.benchmark, "redistribution": m.redistribution})
    return art.manifest_path
