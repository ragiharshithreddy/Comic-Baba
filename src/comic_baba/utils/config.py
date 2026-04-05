"""
YAML-based configuration loading and merging.

Config files live in `configs/`.  The resolved (merged) config is written
to `outputs/<run_id>/configs/resolved_config.yaml` for reproducibility.
"""

from __future__ import annotations

import copy
import datetime
import uuid
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return it as a dict."""
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Write *config* as YAML to *path* (creates parent dirs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        yaml.safe_dump(config, fh, default_flow_style=False, sort_keys=True)


def deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict that is *base* recursively updated with *override*."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def make_run_id(prefix: str = "") -> str:
    """Generate a unique run identifier: <prefix><YYYYMMDD_HHMMSS>_<short_uuid>."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:6]
    parts = [p for p in (prefix, ts, short_uuid) if p]
    return "_".join(parts)


def get_output_dirs(outputs_root: str | Path, run_id: str) -> dict[str, Path]:
    """Return a dict of standard output sub-directory paths for *run_id*."""
    from comic_baba.constants import (
        ARTIFACTS_DIR,
        CONFIGS_DIR,
        DEBUG_DIR,
        FRAMES_IN_DIR,
        FRAMES_OUT_DIR,
        METRICS_DIR,
        VIDEO_OUT_DIR,
    )

    run_dir = Path(outputs_root) / run_id
    artifacts = run_dir / ARTIFACTS_DIR
    return {
        "run": run_dir,
        "artifacts": artifacts,
        "frames_in": artifacts / FRAMES_IN_DIR,
        "frames_out": artifacts / FRAMES_OUT_DIR,
        "video_out": artifacts / VIDEO_OUT_DIR,
        "debug": artifacts / DEBUG_DIR,
        "metrics": run_dir / METRICS_DIR,
        "configs": run_dir / CONFIGS_DIR,
    }
