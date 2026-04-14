import pytest
import numpy as np
import torch
from pathlib import Path
from comic_baba.models.interpolators.rife import RIFEInterpolator
from comic_baba.pipelines.train import run_train
from comic_baba.utils.config import save_config

def test_rife_interpolator_smoke():
    interpolator = RIFEInterpolator()
    frames = [np.zeros((64, 64, 3), dtype=np.uint8), np.ones((64, 64, 3), dtype=np.uint8) * 255]
    out = interpolator.interpolate(frames, factor=2)
    assert len(out) == 3
    assert out[0].shape == (64, 64, 3)
    assert out[1].shape == (64, 64, 3)
    assert out[2].shape == (64, 64, 3)

def test_training_pipeline_smoke(tmp_path):
    # Setup dummy manifest and frames
    manifest_path = tmp_path / "manifest.jsonl"
    frames_dir = tmp_path / "frames" / "test_clip"
    frames_dir.mkdir(parents=True)

    for i in range(3):
        frame = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        from PIL import Image
        Image.fromarray(frame).save(frames_dir / f"frame_{i:06d}.png")

    import json
    entry = {
        "clip_id": "test_clip",
        "source_type": "frames",
        "path": str(frames_dir),
        "fps": 10,
        "num_frames": 3,
        "width": 64,
        "height": 64,
        "split": "train",
    }
    with manifest_path.open("w") as fh:
        fh.write(json.dumps(entry) + "\n")

    config = {
        "run": {"seed": 42},
        "paths": {
            "manifest": str(manifest_path),
            "outputs_root": str(tmp_path / "outputs"),
        },
        "data": {
            "resize": {"width": 64, "height": 64}
        },
        "train": {
            "lr": 1e-4,
            "epochs": 1
        }
    }
    config_path = tmp_path / "config.yaml"
    save_config(config, config_path)

    run_train(config_path)

    assert (tmp_path / "outputs" / "checkpoints" / "latest.pt").exists()
