"""
tests/test_smoke_pipeline.py
----------------------------
End-to-end smoke test: generates a tiny synthetic clip, runs all three
pipeline stages, and asserts that the expected output files exist and that
the metrics JSON is valid.

This test must pass on CPU without any external model weights or GPU.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tiny_repo(tmp_path):
    """
    Set up a minimal repo-like directory with a tiny synthetic clip and a
    baseline config.  Returns a dict of key paths.
    """
    # Create a synthetic clip (4 frames, 32×32)
    clip_id = "smoke_clip"
    frames_dir = tmp_path / "inputs" / "frames" / clip_id
    frames_dir.mkdir(parents=True)

    for i in range(4):
        arr = np.full((32, 32, 3), i * 60, dtype=np.uint8)
        Image.fromarray(arr).save(frames_dir / f"frame_{i:06d}.png")

    # Write manifest
    manifest = tmp_path / "inputs" / "manifest.jsonl"
    entry = {
        "clip_id": clip_id,
        "source_type": "frames",
        "path": f"inputs/frames/{clip_id}",
        "fps": 6,
        "num_frames": 4,
        "width": 32,
        "height": 32,
        "split": "test",
    }
    manifest.write_text(json.dumps(entry) + "\n")

    # Write a minimal config
    import yaml

    config = {
        "run": {"id_prefix": "smoke", "seed": 0},
        "paths": {
            "manifest": str(manifest),
            "outputs_root": str(tmp_path / "outputs"),
        },
        "data": {"resize": None},
        "interpolator": {"name": "blend", "factor": 2},
        "stabilizer": {"name": "none"},
        "eval": {"temporal": True, "identity": True, "quality": True},
    }
    config_path = tmp_path / "configs" / "baseline.yaml"
    config_path.parent.mkdir()
    config_path.write_text(yaml.safe_dump(config))

    return {
        "root": tmp_path,
        "config": config_path,
        "frames_dir": frames_dir,
        "manifest": manifest,
        "outputs_root": tmp_path / "outputs",
        "clip_id": clip_id,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_prepare_stage(tiny_repo):
    """Stage 1 creates frames_in and resolved_config.yaml."""
    from comic_baba.pipelines.prepare_data import run_prepare

    run_id = run_prepare(config_path=tiny_repo["config"], run_id="smoke_run")
    assert run_id == "smoke_run"

    frames_in = tiny_repo["outputs_root"] / "smoke_run" / "artifacts" / "frames_in"
    clip_dir = frames_in / tiny_repo["clip_id"]
    assert clip_dir.exists(), f"frames_in clip dir not found: {clip_dir}"

    pngs = list(clip_dir.glob("frame_*.png"))
    assert len(pngs) == 4, f"Expected 4 frames, got {len(pngs)}"

    resolved_cfg = tiny_repo["outputs_root"] / "smoke_run" / "configs" / "resolved_config.yaml"
    assert resolved_cfg.exists(), "resolved_config.yaml not written"


def test_infer_stage(tiny_repo):
    """Stage 2 creates frames_out with factor×(N-1)+1 frames."""
    from comic_baba.pipelines.inference import run_infer
    from comic_baba.pipelines.prepare_data import run_prepare

    run_id = "smoke_run_infer"
    run_prepare(config_path=tiny_repo["config"], run_id=run_id)
    run_infer(config_path=tiny_repo["config"], run_id=run_id)

    frames_out = (
        tiny_repo["outputs_root"] / run_id / "artifacts" / "frames_out" / tiny_repo["clip_id"]
    )
    assert frames_out.exists(), f"frames_out dir not found: {frames_out}"

    pngs = list(frames_out.glob("frame_*.png"))
    # 4 input frames, factor=2 → 4 + (4-1)*1 = 7 output frames
    assert len(pngs) == 7, f"Expected 7 frames (factor=2, N=4), got {len(pngs)}"


def test_eval_stage(tiny_repo):
    """Stage 3 writes clip_metrics.jsonl and summary.json with expected keys."""
    from comic_baba.pipelines.evaluation import run_eval
    from comic_baba.pipelines.inference import run_infer
    from comic_baba.pipelines.prepare_data import run_prepare

    run_id = "smoke_run_eval"
    run_prepare(config_path=tiny_repo["config"], run_id=run_id)
    run_infer(config_path=tiny_repo["config"], run_id=run_id)
    summary = run_eval(config_path=tiny_repo["config"], run_id=run_id)

    metrics_dir = tiny_repo["outputs_root"] / run_id / "metrics"
    clip_jsonl = metrics_dir / "clip_metrics.jsonl"
    summary_json = metrics_dir / "summary.json"

    assert clip_jsonl.exists(), "clip_metrics.jsonl not written"
    assert summary_json.exists(), "summary.json not written"

    # Validate clip_metrics content
    lines = [json.loads(line) for line in clip_jsonl.read_text().strip().splitlines()]
    assert len(lines) == 1
    clip = lines[0]
    assert clip["clip_id"] == tiny_repo["clip_id"]
    assert "frame_diff_mean" in clip
    assert "identity_drift_mean" in clip
    assert "sharpness_mean" in clip

    # Validate summary
    assert isinstance(summary, dict)
    assert summary["num_clips"] == 1


def test_blend_interpolator_output_length():
    """BlendInterpolator output length follows the formula N + (N-1)*(factor-1)."""
    from comic_baba.models.interpolators.baseline_blend import BlendInterpolator

    interp = BlendInterpolator()
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(5)]
    for factor in (2, 3, 4):
        out = interp.interpolate(frames, factor)
        expected = len(frames) + (len(frames) - 1) * (factor - 1)
        assert len(out) == expected, f"factor={factor}: expected {expected}, got {len(out)}"


def test_blend_interpolator_values():
    """BlendInterpolator mid-frame is exactly the average of its neighbours."""
    from comic_baba.models.interpolators.baseline_blend import BlendInterpolator

    a = np.full((4, 4, 3), 0, dtype=np.uint8)
    b = np.full((4, 4, 3), 100, dtype=np.uint8)
    interp = BlendInterpolator()
    out = interp.interpolate([a, b], factor=2)
    assert len(out) == 3
    expected_mid = np.full((4, 4, 3), 50, dtype=np.uint8)
    np.testing.assert_array_equal(out[1], expected_mid)


def test_manifest_validation_missing_key():
    """validate_manifest_entry raises ValueError on missing required keys."""
    from comic_baba.io.validators import validate_manifest_entry

    with pytest.raises(ValueError, match="missing keys"):
        validate_manifest_entry({"clip_id": "x"})


def test_manifest_iter_valid(tmp_path):
    """iter_manifest yields correctly-shaped dicts for a valid manifest."""
    from comic_baba.io.manifest import iter_manifest

    entry = {
        "clip_id": "c1",
        "source_type": "frames",
        "path": "inputs/frames/c1",
        "fps": 6,
        "num_frames": 8,
        "width": 64,
        "height": 64,
        "split": "test",
    }
    mpath = tmp_path / "manifest.jsonl"
    mpath.write_text(json.dumps(entry) + "\n")
    entries = list(iter_manifest(mpath))
    assert len(entries) == 1
    assert entries[0]["clip_id"] == "c1"


def test_temporal_metrics_zero_diff():
    """All-identical frames should give frame_diff_mean == 0."""
    from comic_baba.eval.metrics_temporal import compute_temporal_flicker

    frames = [np.full((8, 8, 3), 128, dtype=np.uint8) for _ in range(5)]
    result = compute_temporal_flicker(frames)
    assert result["frame_diff_mean"] == pytest.approx(0.0)
    assert result["frame_diff_p95"] == pytest.approx(0.0)


def test_identity_metrics_zero_drift():
    """Identical frames should give identity_drift_mean == 0."""
    from comic_baba.eval.metrics_identity import compute_identity_drift

    frames = [np.full((8, 8, 3), 100, dtype=np.uint8) for _ in range(4)]
    result = compute_identity_drift(frames)
    assert result["identity_drift_mean"] == pytest.approx(0.0)
