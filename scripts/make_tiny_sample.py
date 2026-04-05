#!/usr/bin/env python
"""
scripts/make_tiny_sample.py
---------------------------
Generate a tiny synthetic comic clip for end-to-end smoke testing.

What this script creates
------------------------
- inputs/frames/tiny_clip/frame_000000.png … frame_000007.png
  (8 frames, 64×64, synthetic coloured rectangles)
- inputs/manifest.jsonl
  (one entry referencing the tiny_clip)

No real video or external data is needed.

Usage
-----
    python scripts/make_tiny_sample.py [--frames N] [--size WxH] [--out-dir OUTDIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def make_frame(idx: int, width: int, height: int) -> np.ndarray:
    """Return a synthetic H×W×3 uint8 frame for the given index."""
    rng = np.random.default_rng(seed=idx)
    # Gradient background that shifts slowly across frames
    base_hue = (idx * 30) % 256
    r = np.full((height, width), base_hue, dtype=np.uint8)
    g = np.full((height, width), (base_hue + 80) % 256, dtype=np.uint8)
    b = np.full((height, width), (base_hue + 160) % 256, dtype=np.uint8)
    frame = np.stack([r, g, b], axis=-1)

    # Draw a moving white rectangle (simulates a character moving across frames)
    x = int((idx / 8) * (width - width // 4))
    y = height // 4
    w = width // 4
    h = height // 2
    frame[y : y + h, x : x + w] = 255

    # Random noise (very small, simulates comic texture)
    noise = rng.integers(-5, 6, size=frame.shape, dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a tiny synthetic clip for testing.")
    parser.add_argument("--frames", type=int, default=8, help="Number of frames to generate.")
    parser.add_argument("--size", default="64x64", help="Frame size WxH (e.g. 64x64).")
    parser.add_argument("--out-dir", default=".", help="Repo root (where inputs/ will be created).")
    args = parser.parse_args()

    width, height = (int(x) for x in args.size.split("x"))
    repo_root = Path(args.out_dir).resolve()

    clip_id = "tiny_clip"
    frames_dir = repo_root / "inputs" / "frames" / clip_id
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.frames} frames ({width}×{height}) in {frames_dir} …")
    for i in range(args.frames):
        frame = make_frame(i, width, height)
        out_path = frames_dir / f"frame_{i:06d}.png"
        Image.fromarray(frame).save(out_path)
    print(f"  Saved {args.frames} frames.")

    manifest_path = repo_root / "inputs" / "manifest.jsonl"
    entry = {
        "clip_id": clip_id,
        "source_type": "frames",
        "path": f"inputs/frames/{clip_id}",
        "fps": 6,
        "num_frames": args.frames,
        "width": width,
        "height": height,
        "split": "test",
    }
    with manifest_path.open("w") as fh:
        fh.write(json.dumps(entry) + "\n")
    print(f"  Wrote manifest to {manifest_path}")
    print("Done.  Now run:")
    print("  python scripts/run_prepare.py")
    print("  python scripts/run_infer.py")
    print("  python scripts/run_eval.py")


if __name__ == "__main__":
    main()
