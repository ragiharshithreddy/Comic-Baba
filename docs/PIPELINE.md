# Pipeline Architecture

## Overview

The Temporal Hallucinations pipeline consists of three sequential stages:

```
inputs/manifest.jsonl
        │
        ▼
┌───────────────────┐
│  Stage 1: Prepare │  validate + decode + standardise frames
└────────┬──────────┘
         │  outputs/<run_id>/artifacts/frames_in/
         ▼
┌────────────────────┐
│  Stage 2: Infer    │  interpolate + (optionally) stabilise
└────────┬───────────┘
         │  outputs/<run_id>/artifacts/frames_out/
         ▼
┌────────────────────┐
│  Stage 3: Evaluate │  temporal, identity, quality metrics
└────────────────────┘
         │
         ▼
outputs/<run_id>/metrics/clip_metrics.jsonl
outputs/<run_id>/metrics/summary.json
```

---

## Stage 1 — Prepare (`run_prepare`)

**Script:** `scripts/run_prepare.py`  
**Module:** `comic_baba.pipelines.prepare_data`

### Input
- `inputs/manifest.jsonl` — one JSON object per line (see `docs/DATA_FORMAT.md`)
- Raw clips referenced in the manifest:
  - `source_type=frames` → `inputs/frames/<clip_id>/frame_NNNNNN.png`
  - `source_type=video`  → `inputs/videos/<clip_id>.mp4`

### Processing
1. Parse and validate every manifest entry.
2. Decode videos to frames (if `source_type=video`).
3. Optionally resize every frame to `data.resize.{width,height}` from config.
4. Save standardised frames as `frame_NNNNNN.png` (zero-padded 6-digit index).

### Output
```
outputs/<run_id>/
  artifacts/
    frames_in/
      <clip_id>/
        frame_000000.png
        frame_000001.png
        …
  configs/
    resolved_config.yaml
```

---

## Stage 2 — Inference (`run_infer`)

**Script:** `scripts/run_infer.py`  
**Module:** `comic_baba.pipelines.inference`

### Input
- `outputs/<run_id>/artifacts/frames_in/<clip_id>/` (from Stage 1)
- Config: `interpolator.name`, `interpolator.factor`, `stabilizer.name`

### Processing
1. Load frames from `frames_in/<clip_id>/`.
2. Run interpolator: insert `(factor-1)` synthetic frames between each pair.
3. Run stabiliser (optional; default is no-op).
4. Save output frames.

### Output length formula
```
output_length = len(input_frames) + (len(input_frames) - 1) × (factor - 1)
```

### Output
```
outputs/<run_id>/
  artifacts/
    frames_out/
      <clip_id>/
        frame_000000.png
        …
```

### Extension points
- Add RIFE/FILM by sub-classing `BaseInterpolator` and registering in
  `comic_baba.models.interpolators.get_interpolator`.
- Add identity-lock or temporal-smoothing stabiliser by sub-classing
  `BaseStabilizer`.

---

## Stage 3 — Evaluate (`run_eval`)

**Script:** `scripts/run_eval.py`  
**Module:** `comic_baba.pipelines.evaluation`

### Input
- `outputs/<run_id>/artifacts/frames_out/<clip_id>/`
- `outputs/<run_id>/artifacts/frames_in/<clip_id>/` (for input-quality reference)

### Metrics computed

| Metric | Source | Description |
|--------|--------|-------------|
| `frame_diff_mean` | temporal | Mean abs pixel diff between consecutive frames |
| `frame_diff_p95` | temporal | 95th-percentile of the same |
| `warp_error_mean` | temporal | Optical-flow warping error (opencv required) |
| `identity_drift_mean` | identity | Mean L2 distance between consecutive embeddings |
| `identity_drift_p95` | identity | 95th-percentile identity drift |
| `psnr` | quality | Peak SNR vs GT (null if no GT) |
| `ssim` | quality | Structural similarity vs GT (null if no GT) |
| `sharpness_mean` | quality | Mean Laplacian variance (higher = sharper) |
| `contrast_mean` | quality | Mean RMS contrast |

### Output
```
outputs/<run_id>/
  metrics/
    clip_metrics.jsonl   ← one JSON line per clip
    summary.json         ← aggregate means across all clips
```

---

## Running the full pipeline

```bash
# Generate a synthetic test clip
python scripts/make_tiny_sample.py

# Run all three stages
python scripts/run_prepare.py --config configs/baseline.yaml  # prints RUN_ID=...
python scripts/run_infer.py  --config configs/baseline.yaml --run-id <RUN_ID>
python scripts/run_eval.py   --config configs/baseline.yaml --run-id <RUN_ID>

# Or with the CLI
comic-baba prepare --config configs/baseline.yaml
comic-baba infer   --config configs/baseline.yaml --run-id <RUN_ID>
comic-baba evaluate --config configs/baseline.yaml --run-id <RUN_ID>
```
