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


_CLIP_MODEL = None
_CLIP_PROCESSOR = None

def _extract_embedding(frame: np.ndarray, bbox: dict | None = None) -> np.ndarray:
    """
    Extract a per-frame embedding vector using CLIP.
    """
    global _CLIP_MODEL, _CLIP_PROCESSOR
    import torch
    from transformers import CLIPProcessor, CLIPModel

    if _CLIP_MODEL is None:
        model_id = "openai/clip-vit-base-patch32"
        _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(model_id)
        _CLIP_MODEL = CLIPModel.from_pretrained(model_id)
        _CLIP_MODEL.eval()

    if bbox is not None:
        x1, y1 = int(bbox["x1"]), int(bbox["y1"])
        x2, y2 = int(bbox["x2"]), int(bbox["y2"])
        region = frame[y1:y2, x1:x2]
    else:
        region = frame

    inputs = _CLIP_PROCESSOR(images=region, return_tensors="pt")
    with torch.no_grad():
        outputs = _CLIP_MODEL.get_image_features(**inputs)

    # outputs is often a torch.Tensor if calling get_image_features directly
    # but some versions might return a wrapper.
    if not isinstance(outputs, torch.Tensor):
        embedding_tensor = outputs[0]
    else:
        embedding_tensor = outputs

    embedding = embedding_tensor.squeeze(0).cpu().numpy()
    norm = np.linalg.norm(embedding)
    if norm < 1e-8:
        return embedding
    return embedding / norm


def _zip_bboxes(frames, bboxes):
    if bboxes is None:
        return zip(frames, [None] * len(frames))
    return zip(frames, bboxes)
