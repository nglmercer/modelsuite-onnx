"""High-level job orchestration used by the CLI and by library users."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from . import steps
from .config import JobConfig
from .context import Emit, JobContext, JobState, _silent
from .errors import ArtifactError, ConfigError
from .manifest import Artifact, Manifest

STEP_ORDER = ["discover", "inspect", "analyze", "export", "optimize", "validate", "benchmark", "package"]


def _run(state: JobState | None, name: str, fn: Callable[[], object]) -> object:
    t = time.perf_counter()
    try:
        out = fn()
    except Exception as exc:
        if state:
            state.mark(name, "failed", time.perf_counter() - t, f"{type(exc).__name__}: {str(exc)[:300]}")
        raise
    if state:
        state.mark(name, "done", time.perf_counter() - t)
    return out


def inspect_only(config: JobConfig, emit: Emit = _silent) -> JobContext:
    """discover + inspect + analyze, without writing an artifact manifest (reports go to a scratch dir if asked)."""
    ctx = JobContext(config=config, artifact=Artifact(Path(config.output)), emit=emit)
    steps.discover(ctx)
    steps.inspect_step(ctx)
    return ctx


def analyze_only(config: JobConfig, emit: Emit = _silent) -> JobContext:
    ctx = inspect_only(config, emit)
    steps.analyze(ctx, raise_on_unsupported=False)
    return ctx


def convert(config: JobConfig, *, emit: Emit = _silent, force: set[str] | None = None, until: str | None = None) -> JobContext:
    """Run the full conversion pipeline into config.output. Resumable: re-running the same command continues where it stopped."""
    wanted = config.steps or STEP_ORDER
    bad = [s for s in wanted if s not in STEP_ORDER]
    if bad:
        raise ConfigError(f"unknown step(s) {bad}; valid: {STEP_ORDER}")
    art = Artifact(config.output)
    ctx = JobContext(config=config, artifact=art, emit=emit, force=force or set(), trust_golden=True)
    state = JobState(art)
    if state.fingerprint and state.fingerprint != config.fingerprint() and not ctx.forced("all"):
        raise ConfigError(f"{art.root} contains a job with a different configuration; use another --output or --force")
    art.root.mkdir(parents=True, exist_ok=True)
    state.start(config.fingerprint(), config.model_dump(mode="json"))

    def active(step: str) -> bool:
        return step in wanted and (until is None or STEP_ORDER.index(step) <= STEP_ORDER.index(until))

    _run(state, "discover", lambda: steps.discover(ctx))
    _run(state, "inspect", lambda: steps.inspect_step(ctx))
    _run(state, "analyze", lambda: steps.analyze(ctx))
    ctx.manifest = steps.load_manifest_or_new(ctx)
    art.write(ctx.manifest)
    if active("export"):
        _run(state, "export", lambda: steps.export(ctx))
        steps.structural(ctx, "fp32")
        art.write(ctx.manifest)
    else:
        ctx.plan = ctx.adapter.plan(ctx.source, ctx.adapter_options, config)
    if active("optimize") and steps.planned_variants(config.optimization):
        _run(state, "optimize", lambda: steps.optimize(ctx))
    if active("validate"):
        _run(state, "validate", lambda: steps.validate(ctx))
    if active("benchmark"):
        _run(state, "benchmark", lambda: steps.benchmark(ctx))
    if active("package"):
        _run(state, "package", lambda: steps.package(ctx))
    return ctx


# --- operating on an existing artifact --------------------------------------------------------------------------
def open_artifact(path: str | Path, config: JobConfig | None = None, emit: Emit = _silent, force: set[str] | None = None,
                  model: str | None = None, revision: str | None = None, offline: bool = False) -> JobContext:
    """Context for an existing artifact. With `model`, the original is rebuilt through the recorded adapter ("live" mode:
    stateful cases, quality checks and uncached references work). Without it, validation uses the bundled golden references."""
    art = Artifact(path)
    m = art.read()
    cfg = config or JobConfig.model_validate({**m.job_config, "output": str(art.root)})
    cfg = cfg.model_copy(update={"output": str(art.root)})
    ctx = JobContext(config=cfg, artifact=art, emit=emit, force=force or set(), manifest=m, live=bool(model))
    if model:
        from ..adapters.selection import select_adapter
        from .discovery import resolve
        from .registry import load_plugins
        load_plugins(cfg.plugins)
        rev = revision or (m.source.revision if model == m.source.reference else None)
        ctx.source = resolve(model, rev, offline=offline)
        ctx.adapter = select_adapter(ctx.source, m.adapter.name).adapter
        ctx.adapter_options = ctx.adapter.parse_options(m.adapter_options)
        ctx.info = ctx.adapter.inspect(ctx.source, ctx.adapter_options)
        if ctx.source.revision and m.source.revision and ctx.source.revision != m.source.revision and ctx.source.revision_kind == m.source.revision_kind:
            emit("open", f"warning: model revision {ctx.source.revision} differs from the one converted ({m.source.revision}); parity may fail")
        if cfg.export.fetch_weights:
            ctx.source.fetch_weights(ctx.adapter.weight_patterns(ctx.source, ctx.adapter_options))
        ctx.plan = ctx.adapter.plan(ctx.source, ctx.adapter_options, cfg)
    return ctx
