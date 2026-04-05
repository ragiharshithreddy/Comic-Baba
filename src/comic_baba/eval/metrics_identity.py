"""
Identity-drift metrics.

Measures how much the perceived "character identity" changes across frames.

CURRENT STATUS
--------------
Placeholder implementation that uses per-frame mean-colour vectors as a
proxy embedding.  This is intentionally simple and CPU-safe.

Metrics computed
----------------
identity_drift_mean : float
    Mean L2 distance between consecutive frame embeddings, normalised to [0, 1].
identity_drift_p95 : float
    95th-percentile of the same distribution.

EXTENSION POINT
---------------
Replace `_extract_embedding` with a real character/face encoder:
    - CLIP (openai/clip-vit-base-patch32) — style + content
    - DINOv2 (facebook/dinov2-base) — semantic content
    - ArcFace / InsightFace — face identity
    - Custom comic-character classifier fine-tuned on your dataset

See PROMPT_ADD_IDENTITY_METRIC in constants.py for the full contract.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_identity_drift(
    frames: list[np.ndarray],
    bboxes: list[dict] | None = None,
) -> dict:
    """
    Compute identity-drift metrics.

    Parameters
    ----------
    frames:
        List of H×W×3 uint8 numpy arrays.
    bboxes:
        Optional per-frame bounding boxes (dicts with keys x1,y1,x2,y2).
        When provided, embeddings are extracted from the crop rather than
        the full frame.

    Returns
    -------
    dict with keys: identity_drift_mean, identity_drift_p95.
    """
    if len(frames) < 2:
        return {"identity_drift_mean": 0.0, "identity_drift_p95": 0.0}

    embeddings = [_extract_embedding(f, bbox) for f, bbox in _zip_bboxes(frames, bboxes)]

    dists: list[float] = []
    for e1, e2 in zip(embeddings[:-1], embeddings[1:]):
        dist = float(np.linalg.norm(e1 - e2))
        dists.append(dist)

    return {
        "identity_drift_mean": float(np.mean(dists)),
        "identity_drift_p95": float(np.percentile(dists, 95)),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_embedding(frame: np.ndarray, bbox: dict | None = None) -> np.ndarray:
    """
    Extract a per-frame embedding vector.

    PLACEHOLDER: returns the normalised mean-colour vector of the frame (or
    crop).  Replace with a real encoder as described in PROMPT_ADD_IDENTITY_METRIC.
    """
    if bbox is not None:
        x1, y1 = int(bbox["x1"]), int(bbox["y1"])
        x2, y2 = int(bbox["x2"]), int(bbox["y2"])
        region = frame[y1:y2, x1:x2]
    else:
        region = frame

    mean_rgb = region.astype(np.float32).mean(axis=(0, 1))  # shape (3,)
    norm = np.linalg.norm(mean_rgb)
    if norm < 1e-8:
        return mean_rgb
    return mean_rgb / norm


def _zip_bboxes(frames, bboxes):
    if bboxes is None:
        return zip(frames, [None] * len(frames))
    return zip(frames, bboxes)
