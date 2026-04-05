"""
Manifest reading and writing for inputs/manifest.jsonl.

Schema (one JSON object per line)
----------------------------------
{
    "clip_id":    str,          # unique identifier
    "source_type": "video"|"frames",
    "path":       str,          # relative to repo root
    "fps":        float,
    "num_frames": int,
    "width":      int,
    "height":     int,
    "split":      "train"|"val"|"test"
}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from comic_baba.io.validators import validate_manifest_entry


def iter_manifest(manifest_path: str | Path) -> Iterator[dict]:
    """Yield validated manifest entries from a .jsonl file."""
    manifest_path = Path(manifest_path)
    with manifest_path.open() as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{manifest_path}:{lineno} — invalid JSON: {exc}") from exc
            validate_manifest_entry(entry, lineno=lineno)
            yield entry


def write_manifest(entries: list[dict], manifest_path: str | Path) -> None:
    """Write a list of manifest entries to a .jsonl file (overwrites)."""
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")
