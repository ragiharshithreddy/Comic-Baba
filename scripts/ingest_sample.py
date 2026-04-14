#!/usr/bin/env python
"""
scripts/ingest_sample.py
------------------------
Ingest a sample comic clip for training.
"""

import json
from pathlib import Path
import numpy as np
from PIL import Image

def make_sample_frame(idx, width, height):
    rng = np.random.default_rng(seed=idx)
    frame = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    return frame

def main():
    repo_root = Path(__file__).parent.parent
    clip_id = "sample_training_clip"
    frames_dir = repo_root / "inputs" / "frames" / clip_id
    frames_dir.mkdir(parents=True, exist_ok=True)

    width, height = 256, 256
    num_frames = 10

    print(f"Generating {num_frames} frames for {clip_id}...")
    for i in range(num_frames):
        frame = make_sample_frame(i, width, height)
        Image.fromarray(frame).save(frames_dir / f"frame_{i:06d}.png")

    manifest_path = repo_root / "inputs" / "manifest.jsonl"
    entry = {
        "clip_id": clip_id,
        "source_type": "frames",
        "path": f"inputs/frames/{clip_id}",
        "fps": 10,
        "num_frames": num_frames,
        "width": width,
        "height": height,
        "split": "train",
    }

    with manifest_path.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")

    print(f"Added {clip_id} to {manifest_path}")

if __name__ == "__main__":
    main()
