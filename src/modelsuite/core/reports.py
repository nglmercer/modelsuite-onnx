"""Human-readable renderings of manifests and reports (the JSON files are the machine-readable truth)."""
from __future__ import annotations

from typing import Any

from .manifest import Manifest
from .types import CompatibilityReport, ModelInfo


def _mb(n: int | None) -> str:
    return "?" if n is None else f"{n / 2 ** 20:,.1f} MiB"


def _gb_or_mb(n: int | None) -> str:
    if n is None:
        return "?"
    return f"{n / 2 ** 30:.2f} GiB" if n >= 2 ** 30 else _mb(n)


def render_info(info: ModelInfo, adapter: str = "", confidence: float | None = None, license_: str | None = None) -> str:
    lines = [f"Model        {info.reference}", f"Source       {info.source_kind}" + (f" @ {info.revision}" if info.revision else ""),
             f"Framework    {info.framework}", f"Architecture {info.architecture}" + (f" ({info.model_type})" if info.model_type else ""),
             f"Task         {info.task or '-'}", f"Parameters   {info.parameters:,}" if info.parameters else "Parameters   unknown",
             f"Weights      {_gb_or_mb(info.weight_bytes)} {info.weight_dtypes or ''}", f"License      {info.license or 'unknown'}"]
    if adapter:
        lines.append(f"Adapter      {adapter}" + (f" (confidence {confidence:.2f})" if confidence is not None else ""))
    if info.components:
        lines.append("Components")
        for c in info.components:
            p = f"{c.parameters:,} params" if c.parameters else ""
            lines.append(f"  - {c.name:<16} {c.role:<24} {p}{'  [stateful]' if c.stateful else ''}")
    return "\n".join(lines)


def render_compat(c: CompatibilityReport) -> str:
    icon = {"supported": "OK ", "partial": "~  ", "unsupported": "NO ", "unknown": "?  "}[c.support.value]
    lines = [f"[{icon}] {c.support.value.upper()} for {c.target_format}: {c.summary}"]
    for i in c.issues:
        lines.append(f"  {i.severity.value:<7} {i.code}: {i.message}" + (f"\n          hint: {i.hint}" if i.hint else ""))
    if c.exporters:
        lines.append("Exporters    " + ", ".join(f"{k}={'ok' if v == 'available' else 'unavailable'}" for k, v in c.exporters.items()))
    if c.strategy:
        lines.append("Strategy     " + "; ".join(f"{k}: {' -> '.join(v)}" for k, v in c.strategy.items()))
    e = c.estimates
    if e.get("parameters"):
        lines.append(f"Estimates    fp32 {_gb_or_mb(e.get('fp32_bytes'))}, fp16 {_gb_or_mb(e.get('fp16_bytes'))}, int8 {_gb_or_mb(e.get('int8_bytes'))}, "
                     f"int4 {_gb_or_mb(e.get('int4_bytes'))}; export peak RAM ~{_gb_or_mb(e.get('export_peak_ram_bytes_estimate'))} "
                     f"(available {_gb_or_mb(e.get('available_ram_bytes'))})")
    return "\n".join(lines)


def render_text(m: Manifest) -> str:
    s = m.source
    lines = ["=" * 78, f"modelsuite artifact: {s.reference}  [{s.architecture}]", "=" * 78,
             f"source       {s.kind} {s.revision or ''} ({s.revision_kind})", f"adapter      {m.adapter.name} {m.adapter.version}",
             f"exporters    {', '.join(f'{e.name}={e.version}' for e in m.exporters) or '-'}",
             f"license      {(s.license.id if s.license else None) or 'unknown'}   redistribution of converted files: "
             + {"allowed": "allowed", "no": "NOT ALLOWED", "conditional": "CONDITIONAL (see notices)", "unknown": "UNKNOWN (do not redistribute)"}
             [m.redistribution.status if m.redistribution else "unknown"]]
    if m.redistribution:
        lines.append(f"             {m.redistribution.reason}")
        for n in m.redistribution.required_notices:
            lines.append(f"             - {n}")
    lines.append("")
    lines.append("components")
    for c in m.components:
        ins = ", ".join(f"{t.name}{t.shape}" for t in c.signature.inputs)
        outs = ", ".join(f"{t.name}{t.shape}" for t in c.signature.outputs)
        lines.append(f"  {c.name}: {c.exporter}/{c.strategy} opset {c.opset}{'  [stateful]' if c.stateful else ''}")
        lines.append(f"      in : {ins}")
        lines.append(f"      out: {outs}")
        for k in c.known_issues:
            lines.append(f"      ! {k}")
        for p, why in c.provider_notes.items():
            lines.append(f"      ! provider {p}: {why}")
    lines.append("")
    lines.append("variants")
    for n, v in m.variants.items():
        st = (v.validation or {}).get("status", "not validated")
        lines.append(f"  {n:<16} {_gb_or_mb(v.total_bytes):>12}  validation: {st}" + ("  (partial conversion)" if v.failed_components else ""))
        for c, why in v.failed_components.items():
            lines.append(f"      ! {c}: {why}")
        bm = (v.benchmark or {}).get("pipeline_latency_ms")
        if bm:
            lines.append("      latency (sum of components): " + ", ".join(f"{p}={ms} ms" for p, ms in bm.items() if ms is not None))
    rec = (m.benchmark or {}).get("recommendation")
    if rec:
        lines.append("")
        lines.append(f"recommendation ({rec['objective']}): {rec['recommended'] or 'none'}")
        for r in rec["reasons"]:
            lines.append(f"  - {r}")
        for w in rec["warnings"]:
            lines.append(f"  ! {w}")
    lines.append("")
    lines.append(f"overall validation: {(m.validation or {}).get('status', 'not-run')}")
    return "\n".join(lines) + "\n"
