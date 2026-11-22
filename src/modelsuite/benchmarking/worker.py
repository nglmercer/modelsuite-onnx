"""Subprocess entry point: reads a job from stdin, prints one JSON line. Importing this module has no side effects."""
from __future__ import annotations

import json
import sys

import numpy as np

from .runner import measure


def main() -> int:
    job = json.loads(sys.stdin.read())
    inputs = {k: v for k, v in np.load(job["inputs"]).items()}
    print(json.dumps(measure(job["path"], job["provider"], inputs, job["cfg"], job.get("component", ""), job.get("variant", ""))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
