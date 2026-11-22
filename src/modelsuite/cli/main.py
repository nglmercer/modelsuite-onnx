"""modelsuite command line.

    modelsuite inspect   <model>                 what is this model? (architecture, size, components, license)
    modelsuite analyze   <model>                 can it be converted, with which exporters, with which caveats?
    modelsuite convert   <model> --format onnx   full resumable pipeline -> artifact directory
    modelsuite optimize  <artifact> --precision fp16
    modelsuite validate  [<model>] <artifact>    structural, parity, dynamic-shape, stateful, provider and quality checks
    modelsuite benchmark <artifact>
    modelsuite providers                         which execution providers actually work here
    modelsuite adapters                          registered adapters and exporters
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from .. import __version__
from ..core import registry
from ..core.config import BenchmarkConfig, ExportConfig, JobConfig, OptimizationConfig, ValidationConfig, load_config_file, merge
from ..core.errors import ModelSuiteError
from ..core.types import Support

EXIT_OK, EXIT_ERROR, EXIT_PROBLEM = 0, 1, 2


def _kv(items: list[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for it in items or []:
        if "=" not in it:
            raise ModelSuiteError(f"--adapter-option expects key=value, got '{it}'")
        k, v = it.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def _emit(step: str, msg: str) -> None:
    print(f"[{step}] {msg}", file=sys.stderr, flush=True)


def _quiet(step: str, msg: str) -> None:
    pass


def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--plugin", action="append", default=[], metavar="FILE|MODULE", help="load an adapter/exporter plugin (repeatable)")
    p.add_argument("--json", action="store_true", help="print machine-readable JSON instead of text")
    p.add_argument("-q", "--quiet", action="store_true", help="no progress on stderr")


def add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("model", help="local path, Hugging Face id (org/name) or python spec (file.py:factory)")
    p.add_argument("--revision", help="Hub revision / commit")
    p.add_argument("--adapter", help="force an adapter by name")
    p.add_argument("--adapter-option", action="append", metavar="KEY=VALUE", help="adapter option (JSON values), repeatable")
    p.add_argument("--offline", action="store_true", help="never touch the network")
    p.add_argument("--config", help="YAML/JSON job config; command-line flags override it")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="modelsuite", description="Convert, optimise, validate and benchmark ML models.")
    ap.add_argument("--version", action="version", version=f"modelsuite {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", help="architecture, size, components, license")
    add_model_args(p), add_common(p)
    p = sub.add_parser("analyze", help="compatibility and feasibility report")
    add_model_args(p), add_common(p)
    p.add_argument("--format", default="onnx", choices=["onnx"])
    p.add_argument("--trust-remote-code", action="store_true")

    p = sub.add_parser("convert", help="run the conversion pipeline")
    add_model_args(p), add_common(p)
    p.add_argument("--format", default="onnx", choices=["onnx"])
    p.add_argument("-o", "--output", help="artifact directory (default: artifacts/<model-name>)")
    p.add_argument("--exporter", help="auto | torch-legacy | torch-dynamo | optimum | ort-genai | <plugin>")
    p.add_argument("--opset", type=int)
    p.add_argument("--precision", action="append", choices=["fp16", "fp16-weights", "bf16-weights"], help="also build this precision variant (repeatable)")
    p.add_argument("--quantize", action="append", choices=["int8-dynamic", "int8-static", "int4"], help="also build this quantised variant (repeatable)")
    p.add_argument("--graph-opt", choices=["basic", "extended", "all"], help="also build an ORT-optimised graph variant")
    p.add_argument("--calibration-dir", help="directory of .npz calibration inputs for int8-static")
    p.add_argument("--providers", help="comma-separated providers for validation/benchmark (default cpu)")
    p.add_argument("--no-validate", action="store_true")
    p.add_argument("--benchmark", action="store_true", help="also benchmark every variant")
    p.add_argument("--iterations", type=int)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--no-golden", action="store_true", help="do not bundle reference inputs/outputs in the artifact")
    p.add_argument("--force", action="store_true", help="redo everything, even if the output holds a finished job")
    p.add_argument("--seed", type=int)

    p = sub.add_parser("optimize", help="add precision / quantisation / graph-optimised variants to an artifact")
    add_common(p)
    p.add_argument("artifact")
    p.add_argument("--precision", action="append", choices=["fp16", "fp16-weights", "bf16-weights"])
    p.add_argument("--quantize", action="append", choices=["int8-dynamic", "int8-static", "int4"])
    p.add_argument("--graph-opt", choices=["basic", "extended", "all"])
    p.add_argument("--calibration-dir")
    p.add_argument("--model", help="original model reference: enables live validation (stateful cases, quality checks)")
    p.add_argument("--no-validate", action="store_true")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--providers")
    p.add_argument("--force", action="store_true")
    p.add_argument("--offline", action="store_true")

    p = sub.add_parser("validate", help="validate an artifact against the original model (or its bundled golden references)")
    add_common(p)
    p.add_argument("targets", nargs="+", metavar="[model] artifact", help="`validate <model> <artifact>`, or `validate <artifact>` for golden-only mode")
    p.add_argument("--variant", action="append", help="only this variant (repeatable)")
    p.add_argument("--providers", help="comma-separated providers to compare against the CPU (default cpu)")
    p.add_argument("--no-quality", action="store_true", help="skip task-level quality checks")
    p.add_argument("--offline", action="store_true")

    p = sub.add_parser("benchmark", help="latency / throughput / memory per variant, component and provider")
    add_common(p)
    p.add_argument("artifact")
    p.add_argument("--variant", action="append")
    p.add_argument("--providers", help="comma-separated (default cpu)")
    p.add_argument("--iterations", type=int)
    p.add_argument("--warmup", type=int)
    p.add_argument("--threads", type=int)
    p.add_argument("--no-isolate", action="store_true", help="measure in this process (peak RSS then includes earlier work)")
    p.add_argument("--allow-partial-fallback", action="store_true", default=None)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("providers", help="execution providers: listed vs. actually working")
    add_common(p)
    p.add_argument("--no-probe", action="store_true", help="only list, do not run a probe graph on each provider")
    p = sub.add_parser("adapters", help="registered adapters (and exporters)")
    add_common(p)
    p = sub.add_parser("exporters", help="registered exporters and their capabilities")
    add_common(p)
    return ap


# ---------------------------------------------------------------------------------------------------------------------
def _config_from(args, **overrides) -> JobConfig:
    base: dict[str, Any] = {}
    if getattr(args, "config", None):
        base = load_config_file(args.config)
    base["model"] = args.model
    for key in ("revision", "adapter"):
        if getattr(args, key, None):
            base[key] = getattr(args, key)
    opts = _kv(getattr(args, "adapter_option", None))
    if opts:
        base["adapter_options"] = {**base.get("adapter_options", {}), **opts}
    if args.plugin:
        base["plugins"] = [*base.get("plugins", []), *args.plugin]
    if getattr(args, "offline", False):
        base["offline"] = True
    return JobConfig.model_validate(merge(base, overrides))


def _default_output(model: str) -> str:
    name = Path(model.split(":")[0]).stem if (":" in model and not model.startswith(("http", "/"))) else Path(model.rstrip("/")).name
    return str(Path("artifacts") / (name.replace("/", "__") or "model"))


def _print(obj: Any, args, text: str) -> None:
    print(json.dumps(obj, indent=2, default=str) if args.json else text)


def cmd_inspect(args) -> int:
    from ..core import pipeline
    from ..core.reports import render_info
    ctx = pipeline.inspect_only(_config_from(args), _quiet if args.quiet else _emit)
    lic = ctx.license()
    data = {"model": ctx.info.model_dump(mode="json"), "adapter": {"name": ctx.adapter.name, "version": ctx.adapter.version, "confidence": ctx.detection.confidence,
            "reason": ctx.detection.reason, "candidates": [{"adapter": n, "confidence": d.confidence, "reason": d.reason} for n, d in ctx.candidates]},
            "license": lic.model_dump(mode="json")}
    _print(data, args, render_info(ctx.info, ctx.adapter.name, ctx.detection.confidence) + f"\nLicense cat. {lic.category}; redistribution of converted weights: {lic.redistribution}")
    return EXIT_OK


def cmd_analyze(args) -> int:
    from ..core import pipeline
    from ..core.reports import render_compat, render_info
    cfg = _config_from(args, export={"trust_remote_code": True} if args.trust_remote_code else {})
    ctx = pipeline.analyze_only(cfg, _quiet if args.quiet else _emit)
    data = {"model": ctx.info.model_dump(mode="json"), "adapter": ctx.adapter.name, "compatibility": ctx.compat.model_dump(mode="json")}
    _print(data, args, render_info(ctx.info, ctx.adapter.name, ctx.detection.confidence) + "\n\n" + render_compat(ctx.compat))
    return EXIT_OK if ctx.compat.support in (Support.supported, Support.partial) else EXIT_PROBLEM


def _providers(arg: str | None) -> list[str] | None:
    return [p.strip() for p in arg.split(",") if p.strip()] if arg else None


def cmd_convert(args) -> int:
    from ..core import pipeline
    from ..core.manifest import Artifact
    from ..core.reports import render_text
    ov: dict[str, Any] = {"output": args.output or _default_output(args.model)}
    exp: dict[str, Any] = {}
    if args.exporter:
        exp["exporter"] = args.exporter
    if args.opset:
        exp["opset"] = args.opset
    if args.trust_remote_code:
        exp["trust_remote_code"] = True
    if args.no_golden:
        exp["golden_max_mb"] = 0
    if exp:
        ov["export"] = exp
    opt: dict[str, Any] = {}
    if args.precision:
        opt["precision"] = args.precision
    if args.quantize:
        opt["quantization"] = args.quantize
    if args.graph_opt:
        opt["graph_optimization"] = args.graph_opt
    if args.calibration_dir:
        opt["calibration_dir"] = args.calibration_dir
    if opt:
        ov["optimization"] = opt
    provs = _providers(args.providers)
    if provs:
        ov["validation"] = {"providers": provs}
        ov["benchmark"] = {"providers": provs}
    if args.iterations:
        ov.setdefault("benchmark", {})["iterations"] = args.iterations
    if args.seed is not None:
        ov["seed"] = args.seed
    cfg = _config_from(args, **ov)
    steps = [s for s in pipeline.STEP_ORDER if not (s == "validate" and args.no_validate) and not (s == "benchmark" and not args.benchmark)]
    cfg = cfg.model_copy(update={"steps": cfg.steps or steps})
    ctx = pipeline.convert(cfg, emit=_quiet if args.quiet else _emit, force={"all"} if args.force else set())
    m = ctx.artifact.read()
    _print({"artifact": str(ctx.artifact.root), "manifest": m.model_dump(mode="json")} if args.json else {}, args,
           render_text(m) + f"\nartifact: {ctx.artifact.root}\nmanifest: {ctx.artifact.manifest_path}")
    return EXIT_PROBLEM if (m.validation or {}).get("status") == "failed" else EXIT_OK


def cmd_optimize(args) -> int:
    from ..core import pipeline, steps
    from ..core.reports import render_text
    ov: dict[str, Any] = {"optimization": {k: v for k, v in {"precision": args.precision, "quantization": args.quantize, "graph_optimization": args.graph_opt,
                                                           "calibration_dir": args.calibration_dir}.items() if v}}
    if _providers(args.providers):
        ov["validation"] = {"providers": _providers(args.providers)}
        ov["benchmark"] = {"providers": _providers(args.providers)}
    if not ov["optimization"]:
        raise ModelSuiteError("nothing to do: pass --precision, --quantize and/or --graph-opt")
    from ..core.manifest import Artifact
    base = Artifact(args.artifact).read()
    cfg = JobConfig.model_validate(merge({**base.job_config, "output": args.artifact, "plugins": args.plugin or base.job_config.get("plugins", [])}, ov))
    cfg = cfg.model_copy(update={"optimization": OptimizationConfig.model_validate({**base.job_config.get("optimization", {}), **ov["optimization"]})})
    ctx = pipeline.open_artifact(args.artifact, cfg, _quiet if args.quiet else _emit, {"optimize"} if args.force else set(), model=args.model, offline=args.offline)
    created = steps.optimize(ctx)
    ctx.artifact.write(ctx.manifest)
    if not args.no_validate and created:
        steps.validate(ctx, variants=created)
    if args.benchmark and created:
        steps.benchmark(ctx, variants=["fp32", *created] if "fp32" in ctx.manifest.variants else created)
    steps.package(ctx)
    m = ctx.artifact.read()
    _print({"created": created, "variants": {n: v.model_dump(mode="json") for n, v in m.variants.items()}} if args.json else {}, args, render_text(m))
    bad = [n for n in created if (m.variants[n].validation or {}).get("status") == "failed"]
    return EXIT_PROBLEM if bad else EXIT_OK


def cmd_validate(args) -> int:
    from ..core import pipeline, steps
    from ..core.reports import render_text
    if len(args.targets) > 2:
        raise ModelSuiteError("validate takes [model] artifact")
    model, artifact = (args.targets[0], args.targets[1]) if len(args.targets) == 2 else (None, args.targets[0])
    ctx = pipeline.open_artifact(artifact, emit=_quiet if args.quiet else _emit, model=model, offline=args.offline,
                                 force={"validate"} if not args.variant else set())
    out = steps.validate(ctx, variants=args.variant, providers=_providers(args.providers), quality=not args.no_quality)
    steps.package(ctx)
    m = ctx.artifact.read()
    _print({"validation": out} if args.json else {}, args, render_text(m) + ("" if model else "\n(golden-only mode: stateful cases and quality checks need the original model; pass it as the first argument)"))
    return EXIT_PROBLEM if any(v.get("status") == "failed" for v in out.values()) else EXIT_OK


def cmd_benchmark(args) -> int:
    from ..core import pipeline, steps
    from ..core.reports import render_text
    ctx = pipeline.open_artifact(args.artifact, emit=_quiet if args.quiet else _emit, force={"benchmark"} if args.force else set())
    b = ctx.config.benchmark.model_dump()
    for k, v in (("iterations", args.iterations), ("warmup", args.warmup), ("threads", args.threads)):
        if v is not None:
            b[k] = v
    if args.no_isolate:
        b["isolate_process"] = False
    if args.allow_partial_fallback is not None:
        b["allow_partial_fallback"] = args.allow_partial_fallback
    ctx.config = ctx.config.model_copy(update={"benchmark": BenchmarkConfig.model_validate(b)})
    out = steps.benchmark(ctx, variants=args.variant, providers=_providers(args.providers))
    steps.package(ctx)
    m = ctx.artifact.read()
    lines = [render_text(m), "per-component results"]
    for vn, bm in out.items():
        for r in bm["results"]:
            lines.append(f"  {vn:<14} {r['component']:<14} {r['provider']:<6} " + (f"mean {r['mean_ms']:.2f} ms  p95 {r['p95_ms']:.2f} ms  first {r['first_inference_ms']:.1f} ms  "
                         f"load {r['load_seconds']:.2f}s  rss {r['peak_rss_mb']} MB" if r["status"] == "ok" else f"ERROR {r['error']}"))
    _print({"benchmark": out} if args.json else {}, args, "\n".join(lines))
    return EXIT_PROBLEM if any(r["status"] != "ok" for bm in out.values() for r in bm["results"]) else EXIT_OK


def cmd_providers(args) -> int:
    from ..runtimes.ort import probe_providers
    from ..core import deps
    if not deps.have("onnxruntime"):
        raise ModelSuiteError("onnxruntime is not installed (pip install 'modelsuite[onnx]')")
    st = probe_providers(deep=not args.no_probe)
    text = [f"onnxruntime {deps.dist_version('onnxruntime') or deps.dist_version('onnxruntime-gpu') or '?'}"]
    for p in st:
        text.append(f"  {'OK ' if p.usable else 'NO '} {p.short:<10} {p.name:<28} {'listed' if p.listed else ''}  {p.reason}")
    text.append("(a provider is 'OK' only if a probe graph really ran on it; install the matching onnxruntime build for the others)")
    _print([p.model_dump(mode="json") for p in st], args, "\n".join(text))
    return EXIT_OK


def cmd_adapters(args) -> int:
    from ..adapters.selection import ensure_builtin_adapters, instantiate
    ensure_builtin_adapters()
    rows = []
    for name, entry in registry.adapters.items():
        a = instantiate(entry)
        rows.append({"name": name, "version": a.version, "priority": a.priority, "frameworks": a.frameworks, "description": a.description,
                     "origin": registry.adapters.origin(name)})
    errs = registry.adapters.load_errors
    text = ["adapters"] + [f"  {r['name']:<18} v{r['version']:<7} {r['description']}  [{r['origin']}]" for r in rows]
    text += [f"  ! failed to load '{n}': {e}" for n, e in errs.items()]
    _print({"adapters": rows, "load_errors": errs}, args, "\n".join(text))
    return EXIT_OK


def cmd_exporters(args) -> int:
    from ..exporters.base import describe
    import modelsuite.exporters  # noqa: F401
    rows = []
    for name in registry.exporters.names():
        d = describe(name)
        rows.append({"name": name, **d})
    text = ["exporters"]
    for r in rows:
        c = r["capabilities"]
        text.append(f"  {r['name']:<14} {'available' if r['available'] else 'UNAVAILABLE':<12} {c['description']}")
        if not r["available"]:
            text.append(f"                 reason: {r['reason']}")
        for lim in c["limitations"]:
            text.append(f"                 limit: {lim}")
    _print(rows, args, "\n".join(text))
    return EXIT_OK


COMMANDS = {"inspect": cmd_inspect, "analyze": cmd_analyze, "convert": cmd_convert, "optimize": cmd_optimize, "validate": cmd_validate,
            "benchmark": cmd_benchmark, "providers": cmd_providers, "adapters": cmd_adapters, "exporters": cmd_exporters}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    import os
    if getattr(args, "quiet", False):
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        registry.load_plugins(args.plugin)
        return COMMANDS[args.command](args)
    except ModelSuiteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        print("interrupted (a re-run of the same command resumes the job)", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
