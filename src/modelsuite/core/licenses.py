"""License detection and a deliberately conservative redistribution policy.

The suite never uploads anything, but every artifact it writes is derived from the source weights, so the
manifest carries the model's license and a verdict on whether the *derived* files may be redistributed. When
the license is unknown or restrictive the verdict is "no" / "unknown" and the report says so loudly."""
from __future__ import annotations

import importlib.metadata as md
import re
from pathlib import Path
from typing import Literal

from .types import Loose

Category = Literal["permissive", "weak-copyleft", "copyleft", "non-commercial", "restricted", "unknown"]
Redistribution = Literal["allowed", "conditional", "no", "unknown"]


class LicenseInfo(Loose):
    id: str | None                       # normalised identifier, or the raw text when unrecognised
    category: Category = "unknown"
    redistribution: Redistribution = "unknown"
    commercial_use: bool | None = None
    notes: str = ""
    source: str = ""                     # where the id came from: model-card | LICENSE file | config | user


# normalised id -> (category, redistribution, commercial use, note)
_TABLE: dict[str, tuple[Category, Redistribution, bool | None, str]] = {
    "apache-2.0": ("permissive", "allowed", True, "Keep the NOTICE/attribution and state changes."),
    "mit": ("permissive", "allowed", True, "Keep the copyright notice."),
    "bsd-3-clause": ("permissive", "allowed", True, "Keep the copyright notice."),
    "bsd-2-clause": ("permissive", "allowed", True, "Keep the copyright notice."),
    "isc": ("permissive", "allowed", True, ""),
    "unlicense": ("permissive", "allowed", True, ""),
    "cc0-1.0": ("permissive", "allowed", True, ""),
    "cc-by-4.0": ("permissive", "allowed", True, "Attribution required."),
    "cc-by-sa-4.0": ("weak-copyleft", "conditional", True, "Attribution and share-alike required."),
    "openrail": ("restricted", "conditional", True, "Use-based restrictions must be passed on."),
    "openrail++": ("restricted", "conditional", True, "Use-based restrictions must be passed on."),
    "creativeml-openrail-m": ("restricted", "conditional", True, "Use-based restrictions must be passed on."),
    "gpl-3.0": ("copyleft", "conditional", True, "Derived works must be GPL-3.0."),
    "agpl-3.0": ("copyleft", "conditional", True, "Derived works must be AGPL-3.0."),
    "lgpl-3.0": ("weak-copyleft", "conditional", True, ""),
    "cc-by-nc-4.0": ("non-commercial", "conditional", False, "Non-commercial only; attribution required; derived files inherit this."),
    "cc-by-nc-sa-4.0": ("non-commercial", "conditional", False, "Non-commercial only; share-alike; derived files inherit this."),
    "cc-by-nc-nd-4.0": ("non-commercial", "no", False, "NoDerivatives: converted weights are derivatives and must NOT be redistributed."),
    "cc-by-nd-4.0": ("restricted", "no", True, "NoDerivatives: converted weights are derivatives and must NOT be redistributed."),
    "llama2": ("restricted", "conditional", True, "Llama 2 Community License: attribution, AUP, 700M-MAU clause."),
    "llama3": ("restricted", "conditional", True, "Llama 3 Community License: attribution ('Built with Meta Llama 3'), AUP."),
    "llama3.1": ("restricted", "conditional", True, "Llama 3.1 Community License: attribution, AUP."),
    "gemma": ("restricted", "conditional", True, "Gemma Terms of Use: AUP flows down to derivatives."),
    "other": ("unknown", "unknown", None, "License is 'other': read the model's own license text."),
}

_ALIASES = {
    "apache 2.0": "apache-2.0", "apache-2": "apache-2.0", "apache license 2.0": "apache-2.0", "apache2": "apache-2.0",
    "mit license": "mit", "bsd": "bsd-3-clause", "cc-by-4": "cc-by-4.0", "cc by 4.0": "cc-by-4.0",
    "cc by-nc 4.0": "cc-by-nc-4.0", "cc-by-nc": "cc-by-nc-4.0", "cc by-nc-sa 4.0": "cc-by-nc-sa-4.0",
    "gpl-3": "gpl-3.0", "gplv3": "gpl-3.0", "agpl-3": "agpl-3.0", "llama-2": "llama2", "llama-3": "llama3",
    "llama3.2": "llama3.1", "llama3.3": "llama3.1", "gemma2": "gemma", "gemma3": "gemma",
}

_TEXT_SIGNATURES = [
    (r"apache license[\s,]*version 2\.0", "apache-2.0"),
    (r"permission is hereby granted, free of charge", "mit"),
    (r"creative commons attribution-noncommercial-noderivatives", "cc-by-nc-nd-4.0"),
    (r"creative commons attribution-noncommercial-sharealike", "cc-by-nc-sa-4.0"),
    (r"creative commons attribution-noncommercial", "cc-by-nc-4.0"),
    (r"creative commons attribution-sharealike", "cc-by-sa-4.0"),
    (r"creative commons attribution 4\.0", "cc-by-4.0"),
    (r"gnu affero general public license", "agpl-3.0"),
    (r"gnu general public license", "gpl-3.0"),
    (r"llama 3\.\d? ?community license|meta llama 3", "llama3"),
    (r"llama 2 community license", "llama2"),
    (r"gemma terms of use", "gemma"),
    (r"bsd 3-clause|redistribution and use in source and binary forms", "bsd-3-clause"),
]


def normalise(raw: str | None) -> str | None:
    if not raw:
        return None
    key = re.sub(r"\s+", " ", raw.strip().lower())
    return _ALIASES.get(key, key)


def classify(raw: str | None, source: str = "") -> LicenseInfo:
    lid = normalise(raw)
    if lid is None:
        return LicenseInfo(id=None, notes="No license found. Treat as all-rights-reserved until confirmed.", source=source)
    if lid in _TABLE:
        cat, redis, commercial, note = _TABLE[lid]
        return LicenseInfo(id=lid, category=cat, redistribution=redis, commercial_use=commercial, notes=note, source=source)
    return LicenseInfo(id=lid, notes="Unrecognised license identifier; read the original text before redistributing.", source=source)


def detect_from_text(text: str) -> str | None:
    low = text.lower()
    for pattern, lid in _TEXT_SIGNATURES:
        if re.search(pattern, low):
            return lid
    return None


def detect_model_license(model_dir: Path | None, model_card_license: str | None = None, config_license: str | None = None) -> LicenseInfo:
    """Resolve a model's license: model-card front matter wins, then config.json, then a LICENSE file."""
    if model_card_license and model_card_license.lower() != "other":
        return classify(model_card_license, "model-card")
    if config_license:
        return classify(config_license, "config")
    if model_dir is not None:
        for name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "LICENCE", "COPYING"):
            f = model_dir / name
            if f.is_file():
                try:
                    lid = detect_from_text(f.read_text(errors="ignore")[:20000])
                except OSError:
                    lid = None
                if lid:
                    return classify(lid, f"{name} file")
                return LicenseInfo(id="custom", notes=f"{name} present but not recognised; read it.", source=f"{name} file")
    if model_card_license:  # 'other' with no file to read
        return classify(model_card_license, "model-card")
    return classify(None)


def dependency_licenses(distributions: list[str]) -> dict[str, str | None]:
    """License string of each installed distribution (from its metadata), for the manifest."""
    out: dict[str, str | None] = {}
    for d in distributions:
        try:
            meta = md.metadata(d)
        except md.PackageNotFoundError:
            continue
        lic = meta.get("License-Expression") or meta.get("License")
        if not lic or len(lic) > 80:  # some projects dump the full license text into the field
            classifiers = [c.split("::")[-1].strip() for c in meta.get_all("Classifier") or [] if c.startswith("License ::")]
            lic = classifiers[0] if classifiers else (lic[:60] + "…" if lic else None)
        out[d] = lic
    return out
