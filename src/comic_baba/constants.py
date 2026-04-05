"""
Shared constants and prompt-string templates for the Comic-Baba pipeline.

Prompt strings are natural-language contracts that describe *exactly* what a
contributor must implement when adding a new module.  They serve as the
"specification" that teammates reference when writing code.

Usage
-----
>>> from comic_baba.constants import PROMPT_ADD_INTERPOLATOR
>>> print(PROMPT_ADD_INTERPOLATOR)
"""

# ---------------------------------------------------------------------------
# Frame naming convention
# ---------------------------------------------------------------------------
FRAME_PATTERN = "frame_{idx:06d}.png"  # e.g. frame_000042.png

# ---------------------------------------------------------------------------
# Output sub-directory names (relative to outputs/<run_id>/)
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = "artifacts"
METRICS_DIR = "metrics"
CONFIGS_DIR = "configs"

FRAMES_IN_DIR = "frames_in"  # decoded / standardised input frames
FRAMES_OUT_DIR = "frames_out"  # interpolated (+ optionally stabilised) frames
VIDEO_OUT_DIR = "video_out"  # exported MP4 (optional)
DEBUG_DIR = "debug"  # flow visualisations, embeddings, masks

# ---------------------------------------------------------------------------
# Manifest keys
# ---------------------------------------------------------------------------
MANIFEST_KEYS_REQUIRED = {"clip_id", "source_type", "path", "fps", "num_frames", "width", "height"}
MANIFEST_SOURCE_TYPES = {"video", "frames"}

# ---------------------------------------------------------------------------
# Prompt-string templates (natural-language contracts for contributors)
# ---------------------------------------------------------------------------

PROMPT_ADD_INTERPOLATOR = (
    "Implement a new interpolator by sub-classing "
    "`comic_baba.models.interpolators.base.BaseInterpolator`. "
    "The `interpolate(frames, factor)` method receives a list of H×W×3 uint8 numpy arrays "
    "and must return a new list of the same dtype with (factor-1) synthesised intermediate "
    "frames inserted between each consecutive pair of input frames.  "
    "Output frames must be saved as PNG files to "
    "`outputs/<run_id>/artifacts/frames_out/<clip_id>/frame_NNNNNN.png` "
    "(zero-padded 6-digit index).  "
    "The method must be deterministic given the same random seed.  "
    "Add a config key `interpolator.name` that maps to your class and document it in "
    "`configs/baseline.yaml`."
)

PROMPT_ADD_IDENTITY_METRIC = (
    "Implement an identity-drift metric in "
    "`comic_baba.eval.metrics_identity.compute_identity_drift`. "
    "Given a list of frames (H×W×3 uint8) and optional per-frame bounding boxes "
    "({'x1','y1','x2','y2'} dicts), extract an embedding vector for each frame "
    "(CLIP / DINO / face encoder — your choice), then compute "
    "mean and 95th-percentile L2 distance between consecutive embeddings. "
    "Return a dict with keys: `identity_drift_mean` (float), `identity_drift_p95` (float). "
    "Write results as a key in the per-clip JSON line in "
    "`outputs/<run_id>/metrics/clip_metrics.jsonl`."
)

PROMPT_ADD_TEMPORAL_METRIC = (
    "Implement a temporal-flicker metric in "
    "`comic_baba.eval.metrics_temporal.compute_temporal_flicker`. "
    "Given a list of frames (H×W×3 uint8), compute: "
    "(a) `frame_diff_mean` — mean absolute pixel difference between consecutive frames, "
    "normalised to [0,1]; "
    "(b) `frame_diff_p95` — 95th-percentile of the same distribution; "
    "(c) `warp_error_mean` (optional) — optical-flow warping error between consecutive frames. "
    "Return a dict.  If opencv is not installed, skip the warp_error field gracefully."
)

PROMPT_ADD_QUALITY_METRIC = (
    "Implement no-reference quality metrics in "
    "`comic_baba.eval.metrics_quality.compute_quality`. "
    "If ground-truth (GT) frames exist (same clip at higher FPS), compute PSNR and SSIM. "
    "If no GT exists, compute Laplacian variance (sharpness proxy) and log-RMS contrast. "
    "Return a dict with keys: `psnr` (float|None), `ssim` (float|None), "
    "`sharpness_mean` (float), `contrast_mean` (float)."
)

PROMPT_ADD_DATA_SOURCE = (
    "Write a data-ingestion script in `scripts/ingest_<source_name>.py`. "
    "The script must download or mount a dataset, extract frames at the configured FPS, "
    "resize to the configured resolution (width × height), "
    "save frames to `inputs/frames/<clip_id>/frame_NNNNNN.png`, "
    "and append one JSON line per clip to `inputs/manifest.jsonl` with keys: "
    "clip_id, source_type='frames', path, fps, num_frames, width, height, split. "
    "Include a checksum or version field so the dataset version is reproducible."
)

PROMPT_ADD_STABILIZER = (
    "Implement a stabiliser by sub-classing "
    "`comic_baba.models.stabilizers.base.BaseStabilizer`. "
    "The `stabilize(frames)` method receives a list of H×W×3 uint8 frames and returns "
    "a stabilised list of the same length and dtype. "
    "Typical approaches: temporal smoothing in feature space, optical-flow-guided warping, "
    "or identity-embedding-based frame selection.  "
    "The stabiliser is applied *after* interpolation, on frames_out. "
    "Add a config key `stabilizer.name` to enable/disable and document the I/O contract."
)
