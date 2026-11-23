"""Stage (and optionally upload) a modelsuite Phonon artifact to the Hugging Face Hub.

    python examples/phonon/publish_hf.py artifacts/phonon-40m-multilingual \
        --repo melser/phonon-40m-multilingual-onnx --dry-run
    python examples/phonon/publish_hf.py artifacts/phonon-40m-multilingual \
        --repo melser/phonon-40m-multilingual-onnx --accept-conditions

Deliberately an example script, not a core command: the suite itself never uploads anything.
Publishes the chosen variants' graphs, manifest, assets, reports, SHA256SUMS and a generated model card.
Does NOT publish job/ state, reference/ (golden data) or environment-specific absolute paths.
Gate: the manifest's redistribution verdict; 'conditional' needs --accept-conditions after reading the notices.
"""
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
    text = text.replace(str(art), ".")
    text = re.sub(r"/home/[^/\"\s]+", "~", text)
    return re.sub(r"/tmp/[^\"\s,]+", "<tmp>", text)


def measured(m: dict, variants: list[str], detailed: dict | None = None) -> str:
    # The manifest keeps a condensed validation summary; reports/validation.json has the live stateful and
    # quality-check results. Prefer the detailed one when publishing.
    v = ((detailed or {}).get(variants[0]) or m["variants"][variants[0]].get("validation") or {})
    lines = ["| graph | cases | max abs error vs original | min cosine |", "|---|---|---|---|"]
    for name, r in (v.get("components") or {}).items():
        cases = (r.get("parity") or {}).get("cases", [])
        if not cases:
            continue
        outs = [o for c in cases for o in c["outputs"]]
        cos = [o["cosine"] for o in outs if o.get("cosine") is not None]
        err = max(c["summary"]["max_abs"] for c in cases)
        lines.append(f"| `{name}` | {', '.join(c['name'] for c in cases)} | {err:.1e} | {min(cos):.12f} |" if cos else
                     f"| `{name}` | {', '.join(c['name'] for c in cases)} | {err:.1e} | n/a |")
    for name, r in (v.get("components") or {}).items():
        for st in r.get("stateful", []):
            if "steps" not in st:
                lines.append(f"\nStateful `{name}`: {st.get('status')} ({st.get('reason', 'no details in manifest')}).")
                continue
            lines.append(f"\nStateful `{name}` / `{st.get('case', st.get('name', 'stateful'))}`: {st['status']}, "
                         f"{len(st['steps'])} steps, worst step error {st.get('worst_max_abs', 0):.1e}.")
    for q in v.get("quality", []):
        lines.append(f"\nQuality check `{q['name']}`: {'passed' if q.get('passed') else 'FAILED'}: {q.get('message', '')} "
                     + json.dumps(q.get("metrics", {})) + ".")
    return "\n".join(lines)


def model_card(m: dict, repo: str, variants: list[str], upstream: str, detailed: dict | None = None) -> str:
    lic = (m["source"].get("license") or {}).get("id") or "unknown"
    statuses = [(m["variants"][v].get("validation") or {}).get("status", "not-run") for v in variants]
    val = "passed" if statuses and all(x == "passed" for x in statuses) else "/".join(statuses)
    rows = []
    for v in variants:
        vr = m["variants"][v]
        rows.append(f"| `{v}` | {vr['precision']} | {sum(vr['sizes_bytes'].values()) / 2**30:.2f} GiB | "
                    f"{(vr.get('validation') or {}).get('status', 'not-run')} |")
    comps = []
    for c in m["components"]:
        ins = ", ".join(f"`{t['name']}` {t['shape']} {t['dtype']}" for t in c["signature"]["inputs"])
        outs = ", ".join(f"`{t['name']}` {t['shape']} {t['dtype']}" for t in c["signature"]["outputs"])
        comps.append(f"### `{c['name']}`{' (stateful)' if c.get('stateful') else ''}\n\n{c.get('description', '')}\n\n"
                     f"- inputs: {ins}\n- outputs: {outs}\n"
                     + "".join(f"- known issue: {k}\n" for k in c.get("known_issues", [])))
    notices = "".join(f"- {n}\n" for n in (m.get("redistribution") or {}).get("required_notices", []))
    return f"""---
license: {lic}
base_model: {upstream}
library_name: onnxruntime
pipeline_tag: text-to-speech
tags: [onnx, onnxruntime, modelsuite, text-to-speech, tts, phonon, multilingual, es, en, fr, de, pt]
---

# {repo.split('/')[-1]}

ONNX conversion of [`{upstream}`](https://huggingface.co/{upstream}) made with
[modelsuite](https://github.com/nglmercer/modelsuite-onnx) (`examples/phonon`). **This repository contains the
converted graphs and the inference assets only** — no PyTorch weights.

Gradium Phonon is a flow-matching TTS: the `core` graph runs the backbone once for prefill and then once per
80 ms audio frame with an explicit KV cache, `flow` produces one latent per frame, and `codec_decoder` turns
latents into 24 kHz audio. Text tokenisation/prompt assembly, Gaussian noise (zero when greedy), latent
denormalisation (baked into `flow`) and the end-of-speech countdown stay in the host: see
[`examples/phonon/run_onnx.py`](https://github.com/nglmercer/modelsuite-onnx/blob/main/examples/phonon/run_onnx.py).

**Language/accent comes from the voice latents and the text** — the checkpoint has no language token. The
`default-voice.safetensors` in the upstream repo is not a Spanish voice; to speak Spanish pass a voice from
[`gradium/phonon-voices`](https://huggingface.co/gradium/phonon-voices), e.g. `--voice Iker` or `--voice Noa`.
The official runtime's `lang` only controls text normalisation (numbers, symbols, dates).

Credits: original model, tokenizer and voices by **Gradium** — CC BY-NC 4.0, non-commercial and attribution
required; the converted files inherit that. The codec graph's weights come from
[`gradium/phonon-mimi`](https://huggingface.co/gradium/phonon-mimi) (Apache-2.0).
{notices}
## Variants

| variant | precision | size | validation vs the original |
|---|---|---|---|
{chr(10).join(rows)}

Validation of the published variants: **{val}**. Full numbers in `reports/validation.json`.

{measured(m, variants, detailed)}

Only CPU was tested. The codec decoder is whole-sequence (chunk long audio); its graph is traced above the
decoder's 250-frame sliding-window threshold so shorter sequences also match. Only the fp32 variant is
published; fp16/int8 variants were not measured for this model.

## Graphs

{chr(10).join(comps)}

`core` KV state: feed `present_keys`/`present_values` back as `past_keys`/`past_values` (replace), starting
from an empty cache. Voice and latent embeddings are dense `(B,S,...)` tensors with values only at the
`audio_prompt_token_id` / `audio_latent_token_id` positions. `flow` returns both the normalised `velocity`
that the backbone consumes next step and the denormalised `audio_latent` for the codec/caller.

## Use

```sh
python -m pip install onnxruntime tokenizers safetensors
# Spanish voice + Spanish text normalisation (needs `pip install ptts`):
python examples/phonon/run_onnx.py . "Hola desde Phonon." -o speech.wav --voice Iker --lang es
```

Verify the download against `SHA256SUMS`.

## Provenance

`manifest.json` records the source revision, exporter and library versions, per-graph signatures, checksums
and the exact job configuration. Converted with modelsuite {m.get('suite_version')} from
`{m['source']['reference']}` @ `{m['source'].get('revision')}`.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("artifact")
    ap.add_argument("--repo", required=True, help="target repo id, e.g. melser/phonon-40m-multilingual-onnx")
    ap.add_argument("--variants", nargs="+", default=["fp32"])
    ap.add_argument("--upstream", default="gradium/phonon-40m-multilingual")
    ap.add_argument("--staging", help="staging directory (default: <artifact>/../<repo name>-hf)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--accept-conditions", action="store_true")
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()

    art = Path(a.artifact).resolve()
    m = json.loads((art / "manifest.json").read_text())
    red = m.get("redistribution") or {}
    status = red.get("status", "unknown")
    print(f"license: {(m['source'].get('license') or {}).get('id')}   redistribution verdict: {status}   ({red.get('reason', '')})")
    if status in ("no", "unknown"):
        print("refusing to publish: the license does not allow (or does not confirm) redistribution", file=sys.stderr)
        return 2
    if status == "conditional" and not a.accept_conditions:
        print("the license is conditional; read the notices and re-run with --accept-conditions:\n  "
              + "\n  ".join(red.get("required_notices", [])), file=sys.stderr)
        return 2
    missing = [v for v in a.variants if v not in m["variants"]]
    if missing:
        print(f"variants not in the artifact: {missing}", file=sys.stderr)
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
    for r in ("validation.json", "summary.txt", "export.json", "analyze.json"):
        if (art / "reports" / r).is_file():
            shutil.copy2(art / "reports" / r, stage / "reports" / r)
    pub = dict(m)
    pub["variants"] = {v: m["variants"][v] for v in a.variants}
    pub["default_variant"] = a.variants[0]
    pub["environment"] = {k: v for k, v in (m.get("environment") or {}).items() if k != "platform"}
    (stage / "manifest.json").write_text(scrub(json.dumps(pub, indent=2), art) + "\n")
    for r in (stage / "reports").glob("*"):
        r.write_text(scrub(r.read_text(), art))
    listing = sorted((p.relative_to(stage).as_posix(), p.stat().st_size) for p in stage.rglob("*") if p.is_file())
    report = art / "reports" / "validation.json"
    detailed = json.loads(report.read_text()) if report.is_file() else None
    (stage / "README.md").write_text(model_card(pub, a.repo, a.variants, a.upstream, detailed))
    sums = [f"{sha256(p)}  {p.relative_to(stage).as_posix()}" for p in sorted(stage.rglob("*"))
            if p.is_file() and p.name != "SHA256SUMS"]
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
        print("no Hugging Face token: run `hf auth login` (or set HF_TOKEN); the staging folder is ready", file=sys.stderr)
        return 3
    api = HfApi()
    print("uploading as", api.whoami()["name"], "->", a.repo)
    api.create_repo(a.repo, repo_type="model", private=a.private, exist_ok=True)
    api.upload_folder(repo_id=a.repo, folder_path=str(stage),
                      commit_message=f"ONNX conversion of {a.upstream} (modelsuite {m.get('suite_version')})")
    print(f"published: https://huggingface.co/{a.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
