# Task Assignments

Three teammates, three clear ownership areas.  
All tasks must be delivered as PRs (see `CONTRIBUTING.md`).

---

## Teammate A — Data & Ingestion

### Task A-1: Dataset ingestion script

**Branch:** `feature/data-ingestion`

**What to implement:**
Write `scripts/ingest_<source_name>.py` that downloads or mounts a real comic-clip dataset,
extracts frames at the configured FPS, resizes to the configured resolution,
saves frames to `inputs/frames/<clip_id>/frame_NNNNNN.png`,
and appends one manifest entry per clip to `inputs/manifest.jsonl`.

**Input:** dataset URL or local path  
**Output:** `inputs/manifest.jsonl` + `inputs/frames/<clip_id>/` directories

**Acceptance criteria:**
- [ ] At least 10 real comic clips ingested and registered in the manifest.
- [ ] All entries pass `comic_baba.io.validators.validate_manifest_entry`.
- [ ] Frames are correctly named (`frame_NNNNNN.png`).
- [ ] A `data/README.md` documents the dataset: source, license, version, checksum.
- [ ] `python scripts/run_prepare.py` succeeds on the real manifest.

**Prompt string (exact logic):**
> See `PROMPT_ADD_DATA_SOURCE` in `src/comic_baba/constants.py`.

---

### Task A-2: Configurable resizing and validation

**Branch:** `feature/data-transforms`

**What to implement:**
Extend `src/comic_baba/data/transforms.py` with any augmentations needed
for training (random flip, colour jitter).  Update `ComicClipDataset` to
accept a list of transforms.

**Acceptance criteria:**
- [ ] `ComicClipDataset` with transforms yields correct-shaped frames.
- [ ] Unit tests in `tests/test_transforms.py`.
- [ ] No changes to the manifest schema.

---

## Teammate B — Interpolation Model

### Task B-1: Real interpolator wrapper

**Branch:** `feature/interpolator-rife`  (or `feature/interpolator-film`)

**What to implement:**
Sub-class `BaseInterpolator` to wrap a pre-trained RIFE or FILM model.
Register the new class in `comic_baba.models.interpolators.get_interpolator`.
Add a config entry in `configs/rife_2x.yaml`.

**Input:** `list[np.ndarray]` (H×W×3 uint8), `factor: int`  
**Output:** `list[np.ndarray]` (same dtype, extended length)

**Acceptance criteria:**
- [ ] The class follows the `BaseInterpolator` contract.
- [ ] `pytest tests/test_interpolators.py` passes (including the new wrapper with a 4-frame test clip).
- [ ] A `docs/` entry or inline comment explains how to download the model weights.
- [ ] Smoke run on the tiny synthetic clip (`python scripts/run_infer.py --run-id smoke --config configs/rife_2x.yaml`) produces `frames_out/`.
- [ ] Inference time on a 10-frame 256×256 clip documented in `docs/EXPERIMENTS.md`.

**Prompt string (exact logic):**
> See `PROMPT_ADD_INTERPOLATOR` in `src/comic_baba/constants.py`.

---

### Task B-2: Colab/Kaggle training notebook

**Branch:** `feature/training-notebook`

**What to implement:**
A Jupyter notebook `notebooks/train_interpolator.ipynb` that fine-tunes the
chosen model on the ingested dataset.

**Acceptance criteria:**
- [ ] Notebook runs end-to-end on Colab/Kaggle (free tier GPU).
- [ ] Training config is a YAML file in `configs/`, not hardcoded in the notebook.
- [ ] Final checkpoint saved to a reproducible location (W&B artifact or Google Drive link in notebook).
- [ ] `docs/EXPERIMENTS.md` updated with the experiment entry.

---

## Teammate C — Evaluation & Integration

### Task C-1: Temporal flicker metric (complete implementation)

**Branch:** `feature/eval-temporal`

**What to implement:**
Replace the placeholder optical-flow metric in `metrics_temporal.py` with a
robust flow-based warping error (Farneback or RAFT when GPU available).

**Acceptance criteria:**
- [ ] `compute_temporal_flicker` returns `warp_error_mean` as a float (not `None`) when opencv is installed.
- [ ] All existing tests continue to pass.
- [ ] Metric value is logged in `clip_metrics.jsonl`.

**Prompt string:**
> See `PROMPT_ADD_TEMPORAL_METRIC` in `src/comic_baba/constants.py`.

---

### Task C-2: Identity-drift metric (real embedding)

**Branch:** `feature/eval-identity`

**What to implement:**
Replace the mean-colour placeholder in `metrics_identity.py` with a real
feature extractor (CLIP, DINO, or a fine-tuned face encoder).

**Acceptance criteria:**
- [ ] `compute_identity_drift` uses a proper deep embedding (documented).
- [ ] Works on CPU (model runs in < 10 s on a 10-frame 256×256 clip).
- [ ] `identity_drift_mean` measurably decreases when the identity-lock stabiliser is enabled.
- [ ] Unit test added in `tests/test_identity_metric.py`.

**Prompt string:**
> See `PROMPT_ADD_IDENTITY_METRIC` in `src/comic_baba/constants.py`.

---

### Task C-3: CI + integration ownership

**Branch:** `feature/ci-improvements`

**Responsibilities:**
- Keep `main` always green.
- Review and merge PRs from Teammates A and B.
- Ensure smoke test (`tests/test_smoke_pipeline.py`) passes on every PR.
- Update GitHub Actions workflow if new dependencies are added.

**Acceptance criteria:**
- [ ] All CI checks (lint + tests + smoke) pass on `main`.
- [ ] PR template checklist is enforced on every merged PR.

---

## Milestone schedule (suggested)

| Week | Milestone | Owner |
|------|-----------|-------|
| 1 | Baseline pipeline end-to-end on tiny clip | All (Teammate C integrates) |
| 2 | Real dataset ingested + RIFE/FILM wrapper | A + B |
| 3 | Real metrics + stabiliser | C |
| 4 | Training notebook + demo | B + A |
| Final | Report + comparison table | All |
