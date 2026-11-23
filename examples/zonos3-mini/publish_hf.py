"""Stage (and optionally upload) a modelsuite artifact to the Hugging Face Hub.

    python publish_hf.py artifacts/zonos3-mini --repo melser/ZONOS3-Mini-onnx --variants fp32 fp16-weights --dry-run   # build + inspect the staging folder
    python publish_hf.py artifacts/zonos3-mini --repo melser/ZONOS3-Mini-onnx --variants fp32 fp16-weights             # ...and upload (needs `hf auth login` / HF_TOKEN)

Deliberately a script in the example, not a core command: the suite itself never uploads anything.

What it publishes: the chosen variants' graphs (+ external data), manifest.json, assets, validation/benchmark reports, SHA256SUMS and a generated model card.
What it does NOT publish: job/ state, reference/ (golden test inputs/outputs), environment-specific absolute paths.
Gate: the manifest's redistribution verdict. 'no' and 'unknown' abort; 'conditional' needs --accept-conditions after you have read the notices."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while block := fh.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def scrub(text: str, art: Path) -> str:
    """No local paths in published files: the artifact directory becomes '.', home directories '~', temp directories '<tmp>'."""
    text = text.replace(str(art), ".")
    text = re.sub(r"/home/[^/\"\s]+", "~", text)
    return re.sub(r"/tmp/[^\"\s,]+", "<tmp>", text)


def measured(m: dict, variants: list[str]) -> str:
    """A small table of the numbers behind 'validation passed', taken from the manifest (nothing is typed in by hand)."""
    v = m["variants"][variants[0]].get("validation") or {}
    lines = ["| graph | cases | max abs error vs reference | min cosine | argmax agreement |", "|---|---|---|---|---|"]
    for name, r in (v.get("components") or {}).items():
        cases = (r.get("parity") or {}).get("cases", [])
        if not cases:
            continue
        outs = [o for c in cases for o in c["outputs"]]
        cos = [o["cosine"] for o in outs if o.get("cosine") is not None]
        arg = [o["argmax_agreement"] for o in outs if o.get("argmax_agreement") is not None]
        lines.append(f"| `{name}` | {', '.join(c['name'] for c in cases)} | {max(c['summary']['max_abs'] for c in cases):.1e} | "
                     f"{min(cos):.12f} | {min(arg) * 100:.0f} % |" if cos and arg else
                     f"| `{name}` | {', '.join(c['name'] for c in cases)} | {max(c['summary']['max_abs'] for c in cases):.1e} | {min(cos):.12f} | n/a |" if cos else
                     f"| `{name}` | {', '.join(c['name'] for c in cases)} | {max(c['summary']['max_abs'] for c in cases):.1e} | n/a | n/a |")
    for name, r in (v.get("components") or {}).items():
        for st in r.get("stateful", []):
            lines.append(f"\nStateful `{name}` / `{st['case']}`: {st['status']}, {len(st['steps'])} steps, worst step error {st['worst_max_abs']:.1e}.")
    for q in v.get("quality", []):
        lines.append(f"\nQuality check `{q['name']}`: {'passed' if q['passed'] else 'FAILED'}: {q['message']}.")
    return "\n".join(lines)


def model_card(m: dict, repo: str, variants: list[str], upstream: str, files: list[tuple[str, int]]) -> str:
    lic = (m["source"].get("license") or {}).get("id") or "unknown"
    statuses = [(m["variants"][v].get("validation") or {}).get("status", "not-run") for v in variants]
    val = "passed" if statuses and all(x == "passed" for x in statuses) else "/".join(statuses)        # of the PUBLISHED variants only
    rows = []
    for v in variants:
        vr = m["variants"][v]
        st = (vr.get("validation") or {}).get("status", "not-run")
        rows.append(f"| `{v}` | {vr['precision']} | {sum(vr['sizes_bytes'].values()) / 2**30:.2f} GiB | {st} |")
    comps = []
    for c in m["components"]:
        ins = ", ".join(f"`{t['name']}` {t['shape']} {t['dtype']}" for t in c["signature"]["inputs"])
        outs = ", ".join(f"`{t['name']}` {t['shape']} {t['dtype']}" for t in c["signature"]["outputs"])
        comps.append(f"### `{c['name']}`{' (stateful)' if c.get('stateful') else ''}\n\n{c.get('description', '')}\n\n- inputs: {ins}\n- outputs: {outs}\n"
                     + "".join(f"- known issue: {k}\n" for k in c.get("known_issues", [])))
    notices = "".join(f"- {n}\n" for n in (m.get("redistribution") or {}).get("required_notices", []))
    table = measured(m, variants)
    return f"""---
license: {lic}
base_model: {upstream}
library_name: onnxruntime
pipeline_tag: text-to-speech
tags: [onnx, text-to-speech, tts, onnxruntime, modelsuite]
---

# {repo.split('/')[-1]}

ONNX conversion of [`{upstream}`](https://huggingface.co/{upstream}) made with [modelsuite](https://github.com/nglmercer/modelsuite-onnx)
(`examples/zonos3-mini`). **This repository contains the converted graphs only.** The tokenizer, text normalisation, prompt assembly with the
inter-codebook delay, sampler and speaker encoder are **not** included: the language model and codec decoder are the two tensor graphs below, and the
host application must provide the rest (reference: Zyphra's `zonos3` package).

Original model, weights and training: Zyphra, Apache-2.0. Please cite and credit the original authors.
{notices}
## Variants

| variant | precision | size | validation vs the original |
|---|---|---|---|
{chr(10).join(rows)}

Validation of the published variants: **{val}**. Every graph is compared with Zyphra's reference implementation on the same inputs
(prefill with a speaker row and condition rows, decode steps, different shapes, a multi-step stateful decode and a greedy rollout); full numbers in `reports/validation.json`.

{table}

Not validated and therefore not published: 16-bit / int8 / int4 variants. Only CPU was tested. No end-to-end audio was synthesised (the tokenizer, sampler
and speaker encoder are the host's job), so this validates the two tensor graphs, not a full TTS run.

## Graphs

{chr(10).join(comps)}
The language model is **one stateful graph for prefill and decode**: feed `new_k`/`new_v`/`new_conv`/`new_state` back as
`past_k`/`past_v`/`conv_state`/`gdn_state` (start from empty past and zero states). `cond_index` indexes the concatenation of the condition tables
(see `assets/host-pipeline` description in `manifest.json`).

## Use

```python
import onnxruntime as ort, numpy as np
lm = ort.InferenceSession("models/fp32/lm.onnx", providers=["CPUExecutionProvider"])
```

Verify the download against `SHA256SUMS`. Decode speed measured on a 2-core CPU: about 0.08 s per 15 Hz frame (real-time needs 0.067 s); only CPU was tested.

## Provenance

`manifest.json` records the source revision, exporter and library versions, per-graph signatures, checksums and the exact job configuration.
Converted with modelsuite {m.get('suite_version')} from `{m['source']['reference']}` @ `{m['source'].get('revision')}`.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("artifact")
    ap.add_argument("--repo", required=True, help="target repo id, e.g. melser/ZONOS3-Mini-onnx")
    ap.add_argument("--variants", nargs="+", default=["fp32", "fp16-weights"])
    ap.add_argument("--upstream", default="Zyphra/ZONOS3-Mini")
    ap.add_argument("--staging", help="staging directory (default: <artifact>/../<repo name>-hf)")
    ap.add_argument("--dry-run", action="store_true", help="build the staging folder, print what would be uploaded, upload nothing")
    ap.add_argument("--accept-conditions", action="store_true", help="the license is 'conditional': I have read the notices")
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()

    art = Path(a.artifact).resolve()
    m = json.loads((art / "manifest.json").read_text())
    red = m.get("redistribution") or {}
    status = red.get("status", "unknown")
    print(f"license: {(m['source'].get('license') or {}).get('id')}   redistribution verdict: {status}   ({red.get('reason', '')})")
    if status in ("no", "unknown"):
        print("refusing to publish: the license does not allow (or does not confirm) redistribution of converted weights", file=sys.stderr)
        return 2
    if status == "conditional" and not a.accept_conditions:
        print("the license is conditional; read the notices below and re-run with --accept-conditions\n  " + "\n  ".join(red.get("required_notices", [])), file=sys.stderr)
        return 2
    missing = [v for v in a.variants if v not in m["variants"]]
    if missing:
        print(f"variants not in the artifact: {missing} (has {list(m['variants'])})", file=sys.stderr)
        return 2
    bad = [v for v in a.variants if (m["variants"][v].get("validation") or {}).get("status") != "passed"]
    if bad:
        print(f"refusing to publish variants that did not pass validation: {bad}", file=sys.stderr)
        return 2
    problems = __import__("modelsuite.core.manifest", fromlist=["Artifact"]).Artifact(art).verify()
    if problems:
        print("artifact integrity check failed:\n  " + "\n  ".join(problems[:10]), file=sys.stderr)
        return 2

    stage = Path(a.staging) if a.staging else art.parent / (a.repo.split("/")[-1] + "-hf")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for v in a.variants:
        (stage / "models" / v).mkdir(parents=True)
        for files in m["variants"][v]["files"].values():
            for f in files:
                shutil.copy2(art / f["path"], stage / f["path"])
    if (art / "assets").is_dir():
        shutil.copytree(art / "assets", stage / "assets")
    (stage / "reports").mkdir()
    for r in ("validation.json", "benchmark.json", "summary.txt", "export.json", "analyze.json"):
        if (art / "reports" / r).is_file():
            shutil.copy2(art / "reports" / r, stage / "reports" / r)
    pub = dict(m)                                                       # publish only the chosen variants, and no local-path leakage
    pub["variants"] = {v: m["variants"][v] for v in a.variants}
    pub["default_variant"] = a.variants[0]
    pub["environment"] = {k: v for k, v in (m.get("environment") or {}).items() if k != "platform"}
    (stage / "manifest.json").write_text(scrub(json.dumps(pub, indent=2), art) + "\n")
    for r in (stage / "reports").glob("*"):
        r.write_text(scrub(r.read_text(), art))
    listing = sorted((p.relative_to(stage).as_posix(), p.stat().st_size) for p in stage.rglob("*") if p.is_file())
    (stage / "README.md").write_text(model_card(pub, a.repo, a.variants, a.upstream, listing))
    sums = [f"{sha256(p)}  {p.relative_to(stage).as_posix()}" for p in sorted(stage.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"]
    (stage / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    total = sum(s for _, s in listing)
    print(f"staged {len(listing) + 2} files, {total / 2**30:.2f} GiB in {stage}")
    for n, s in sorted(listing, key=lambda x: -x[1])[:8]:
        print(f"  {s / 2**20:10.1f} MiB  {n}")
    if a.dry_run:
        print("dry run: nothing uploaded")
        return 0
    from huggingface_hub import HfApi, get_token
    if not get_token():
        print("no Hugging Face token: run `hf auth login` (or set HF_TOKEN) and re-run; the staging folder is ready", file=sys.stderr)
        return 3
    api = HfApi()
    print("uploading as", api.whoami()["name"], "->", a.repo)
    api.create_repo(a.repo, repo_type="model", private=a.private, exist_ok=True)
    api.upload_folder(repo_id=a.repo, folder_path=str(stage), commit_message=f"ONNX conversion of {a.upstream} (modelsuite {m.get('suite_version')})")
    print(f"published: https://huggingface.co/{a.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
