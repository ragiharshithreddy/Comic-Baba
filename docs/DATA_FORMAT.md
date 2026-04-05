# Data Format Specification

## Manifest (`inputs/manifest.jsonl`)

Every clip is described by one JSON object on a single line.

### Schema

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `clip_id` | string | ✓ | Unique identifier (no spaces, URL-safe) |
| `source_type` | `"video"` \| `"frames"` | ✓ | Whether the clip is stored as a video or frames |
| `path` | string | ✓ | Path relative to repo root |
| `fps` | number | ✓ | Frame rate of the *original* clip |
| `num_frames` | integer | ✓ | Number of frames in the clip |
| `width` | integer | ✓ | Original frame width (pixels) |
| `height` | integer | ✓ | Original frame height (pixels) |
| `split` | `"train"` \| `"val"` \| `"test"` | ✓ | Dataset split |

Optional keys (pass-through to pipeline):
| Key | Type | Description |
|-----|------|-------------|
| `tags` | list[string] | e.g. `["action", "dark_theme"]` |
| `source_url` | string | Original source URL |
| `checksum` | string | SHA-256 of the raw file |

### Example

```jsonl
{"clip_id": "fight_scene_01", "source_type": "frames", "path": "inputs/frames/fight_scene_01", "fps": 12, "num_frames": 48, "width": 512, "height": 512, "split": "train"}
{"clip_id": "run_clip_02", "source_type": "video", "path": "inputs/videos/run_clip_02.mp4", "fps": 24, "num_frames": 96, "width": 720, "height": 480, "split": "val"}
```

---

## Frame files (`inputs/frames/<clip_id>/`)

- Format: **PNG**, RGB (no alpha channel).
- Naming: `frame_NNNNNN.png` — zero-padded 6-digit index.
- Origin: frame 0 is the first frame of the clip.
- Frames are numbered sequentially without gaps.

### Example directory tree

```
inputs/frames/fight_scene_01/
  frame_000000.png
  frame_000001.png
  …
  frame_000047.png
```

---

## Video files (`inputs/videos/`)

- Format: MP4 (H.264 recommended).
- Frame rate should match the `fps` field in the manifest.
- Videos are decoded to frames in Stage 1 and are not required for Stage 2+.

---

## Output frame files (`outputs/<run_id>/artifacts/`)

### `frames_in/<clip_id>/`
Standardised copy of the input frames (possibly resized).
Same naming convention as input: `frame_NNNNNN.png`.

### `frames_out/<clip_id>/`
Interpolated (and optionally stabilised) output frames.
Naming: `frame_NNNNNN.png` starting from 0.

**Length formula:**
```
len(frames_out) = len(frames_in) + (len(frames_in) - 1) × (factor - 1)
```

---

## Metrics files (`outputs/<run_id>/metrics/`)

### `clip_metrics.jsonl`
One JSON object per line, one per clip.

```json
{
  "clip_id": "fight_scene_01",
  "frame_diff_mean": 0.032,
  "frame_diff_p95": 0.087,
  "warp_error_mean": null,
  "identity_drift_mean": 0.041,
  "identity_drift_p95": 0.089,
  "psnr": null,
  "ssim": null,
  "sharpness_mean": 142.3,
  "contrast_mean": 0.21
}
```

### `summary.json`
Aggregate statistics across all clips in the run.

```json
{
  "num_clips": 10,
  "frame_diff_mean_mean": 0.029,
  "frame_diff_p95_mean": 0.081,
  "identity_drift_mean_mean": 0.038,
  "sharpness_mean_mean": 137.1,
  "contrast_mean_mean": 0.19
}
```

---

## Resolved config (`outputs/<run_id>/configs/resolved_config.yaml`)

A verbatim copy of the config used for this run, with an added `_run_id` key.
This ensures every run is fully reproducible.
