"""
Stage 2 — Inference (interpolation + optional stabilisation).

Reads prepared frames from Stage 1, runs the configured interpolator,
optionally applies a stabiliser, and writes output frames.

I/O contract
------------
Input:
    outputs/<run_id>/artifacts/frames_in/<clip_id>/frame_*.png
    outputs/<run_id>/configs/resolved_config.yaml

Output:
    outputs/<run_id>/artifacts/frames_out/<clip_id>/frame_*.png
"""

from __future__ import annotations

import logging
from pathlib import Path

from comic_baba.io.frames import load_frames, save_frames
from comic_baba.models.interpolators import get_interpolator
from comic_baba.models.stabilizers import get_stabilizer
from comic_baba.utils.config import get_output_dirs, load_config
from comic_baba.utils.seed import set_seed

logger = logging.getLogger(__name__)


def run_infer(
    config_path: str | Path = "configs/baseline.yaml",
    run_id: str | None = None,
) -> None:
    """Execute the inference stage."""
    config = load_config(config_path)

    if run_id is None:
        # Try to infer run_id from a resolved config if present
        raise ValueError(
            "run_id must be provided to run_infer. Use the value returned by run_prepare."
        )

    set_seed(config.get("run", {}).get("seed", 42))
    dirs = get_output_dirs(config.get("paths", {}).get("outputs_root", "outputs"), run_id)

    # Build interpolator
    interp_cfg = config.get("interpolator", {})
    interp_name = interp_cfg.get("name", "blend")
    factor = interp_cfg.get("factor", 2)
    # Pass only constructor kwargs (not name or factor)
    interp_kwargs = {k: v for k, v in interp_cfg.items() if k not in ("name", "factor")}
    interpolator = get_interpolator(interp_name, **interp_kwargs)
    logger.info("Interpolator: %s  factor=%d", interpolator, factor)

    # Build stabilizer
    stab_cfg = config.get("stabilizer", {})
    stab_name = stab_cfg.get("name", "none")
    stab_kwargs = {k: v for k, v in stab_cfg.items() if k not in ("name",)}
    stabilizer = get_stabilizer(stab_name, **stab_kwargs)
    logger.info("Stabilizer: %s", stabilizer)

    frames_in_root = dirs["frames_in"]
    if not frames_in_root.exists():
        raise FileNotFoundError(
            f"frames_in directory not found: {frames_in_root}. Run the prepare stage first."
        )

    clip_dirs = sorted(p for p in frames_in_root.iterdir() if p.is_dir())
    if not clip_dirs:
        raise FileNotFoundError(f"No clip directories found in {frames_in_root}")

    for clip_dir in clip_dirs:
        clip_id = clip_dir.name
        logger.info("  Interpolating clip %s …", clip_id)
        frames = load_frames(clip_dir)
        interpolated = interpolator.interpolate(frames, factor)
        stabilised = stabilizer.stabilize(interpolated)
        out_dir = dirs["frames_out"] / clip_id
        save_frames(stabilised, out_dir)
        logger.info(
            "    %d → %d frames (factor=%d) saved to %s",
            len(frames),
            len(stabilised),
            factor,
            out_dir,
        )

    logger.info("Inference stage complete. run_id=%s", run_id)
