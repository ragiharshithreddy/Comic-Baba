"""
Stage 1 — Prepare Data.

Reads inputs/manifest.jsonl, validates each entry, decodes/copies frames
to the standardised output location, and writes resolved_config.yaml.

I/O contract
------------
Input:
    inputs/manifest.jsonl
    inputs/frames/<clip_id>/frame_*.png   (for source_type=frames)
    inputs/videos/<clip_id>.mp4            (for source_type=video)

Output:
    outputs/<run_id>/artifacts/frames_in/<clip_id>/frame_*.png
    outputs/<run_id>/configs/resolved_config.yaml
"""

from __future__ import annotations

import logging
from pathlib import Path

from comic_baba.io.frames import load_frames, resize_frame, save_frames
from comic_baba.io.manifest import iter_manifest
from comic_baba.io.video import decode_video_to_frames
from comic_baba.utils.config import get_output_dirs, load_config, make_run_id, save_config

logger = logging.getLogger(__name__)


def run_prepare(
    config_path: str | Path = "configs/baseline.yaml",
    run_id: str | None = None,
) -> str:
    """
    Execute the prepare stage.

    Returns
    -------
    The run_id used (useful when it was auto-generated).
    """
    config = load_config(config_path)
    run_id = run_id or make_run_id(config.get("run", {}).get("id_prefix", "run"))
    dirs = get_output_dirs(config.get("paths", {}).get("outputs_root", "outputs"), run_id)

    manifest_path = Path(config.get("paths", {}).get("manifest", "inputs/manifest.jsonl"))
    resize_wh: tuple[int, int] | None = None
    data_cfg = config.get("data", {})
    if data_cfg.get("resize"):
        resize_wh = (data_cfg["resize"]["width"], data_cfg["resize"]["height"])

    logger.info("run_id=%s | loading manifest from %s", run_id, manifest_path)
    entries = list(iter_manifest(manifest_path))
    logger.info("Found %d clips in manifest.", len(entries))

    for entry in entries:
        clip_id = entry["clip_id"]
        out_clip_dir = dirs["frames_in"] / clip_id
        out_clip_dir.mkdir(parents=True, exist_ok=True)

        src_path = Path(entry["path"])
        if not src_path.is_absolute():
            src_path = manifest_path.parent.parent / src_path

        logger.info("  Processing clip %s (source_type=%s)", clip_id, entry["source_type"])

        if entry["source_type"] == "frames":
            frames = load_frames(src_path)
        elif entry["source_type"] == "video":
            import tempfile

            with tempfile.TemporaryDirectory() as tmp_dir:
                decode_video_to_frames(src_path, tmp_dir)
                frames = load_frames(tmp_dir)
        else:
            raise ValueError(f"Unknown source_type: {entry['source_type']!r}")

        if resize_wh:
            frames = [resize_frame(f, resize_wh[0], resize_wh[1]) for f in frames]

        save_frames(frames, out_clip_dir)
        logger.info("    → saved %d frames to %s", len(frames), out_clip_dir)

    # Persist resolved config
    config["_run_id"] = run_id
    save_config(config, dirs["configs"] / "resolved_config.yaml")
    logger.info("Prepare stage complete. run_id=%s", run_id)
    return run_id
