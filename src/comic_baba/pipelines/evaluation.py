"""
Stage 3 — Evaluation.

Computes temporal, identity, and quality metrics for each clip and writes
results to the metrics directory.

I/O contract
------------
Input:
    outputs/<run_id>/artifacts/frames_in/<clip_id>/frame_*.png
    outputs/<run_id>/artifacts/frames_out/<clip_id>/frame_*.png

Output:
    outputs/<run_id>/metrics/clip_metrics.jsonl  — one JSON line per clip
    outputs/<run_id>/metrics/summary.json        — aggregate statistics
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from comic_baba.eval.metrics_identity import compute_identity_drift
from comic_baba.eval.metrics_quality import compute_quality
from comic_baba.eval.metrics_temporal import compute_temporal_flicker
from comic_baba.io.frames import load_frames
from comic_baba.utils.config import get_output_dirs, load_config

logger = logging.getLogger(__name__)


def run_eval(
    config_path: str | Path = "configs/baseline.yaml",
    run_id: str | None = None,
) -> dict:
    """
    Execute the evaluation stage.

    Returns
    -------
    The summary dict (also written to metrics/summary.json).
    """
    config = load_config(config_path)

    if run_id is None:
        raise ValueError("run_id must be provided to run_eval.")

    dirs = get_output_dirs(config.get("paths", {}).get("outputs_root", "outputs"), run_id)
    dirs["metrics"].mkdir(parents=True, exist_ok=True)

    frames_out_root = dirs["frames_out"]

    if not frames_out_root.exists():
        raise FileNotFoundError(
            f"frames_out directory not found: {frames_out_root}. Run the inference stage first."
        )

    clip_dirs = sorted(p for p in frames_out_root.iterdir() if p.is_dir())
    if not clip_dirs:
        raise FileNotFoundError(f"No clip directories found in {frames_out_root}")

    all_clip_metrics: list[dict] = []

    with (dirs["metrics"] / "clip_metrics.jsonl").open("w") as out_fh:
        for clip_dir in clip_dirs:
            clip_id = clip_dir.name
            logger.info("  Evaluating clip %s …", clip_id)

            frames_out = load_frames(clip_dir)

            temporal = compute_temporal_flicker(frames_out)
            identity = compute_identity_drift(frames_out)
            quality = compute_quality(frames_out)  # no GT → psnr/ssim=None

            clip_result = {
                "clip_id": clip_id,
                **temporal,
                **identity,
                **quality,
            }
            out_fh.write(json.dumps(clip_result) + "\n")
            all_clip_metrics.append(clip_result)
            logger.info("    metrics: %s", clip_result)

    summary = _aggregate(all_clip_metrics)
    with (dirs["metrics"] / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)

    logger.info("Evaluation stage complete. run_id=%s", run_id)
    logger.info("Summary: %s", summary)
    return summary


def _aggregate(clip_metrics: list[dict]) -> dict:
    """Compute mean of all numeric fields across clips."""
    if not clip_metrics:
        return {}

    numeric_keys = [
        k
        for k, v in clip_metrics[0].items()
        if k != "clip_id" and isinstance(v, (int, float)) and v is not None
    ]
    summary: dict = {"num_clips": len(clip_metrics)}
    for key in numeric_keys:
        vals = [m[key] for m in clip_metrics if m.get(key) is not None]
        if vals:
            summary[f"{key}_mean"] = float(np.mean(vals))
    return summary
