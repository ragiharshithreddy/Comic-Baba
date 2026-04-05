"""
Temporal-flicker metrics.

Measures how much consecutive frames differ — a proxy for perceived flicker.

Metrics computed
----------------
frame_diff_mean : float
    Mean absolute pixel difference between consecutive frames, normalised
    to [0, 1].
frame_diff_p95 : float
    95th-percentile of per-pixel differences across all consecutive pairs.
warp_error_mean : float | None
    Mean warping error using dense optical flow (requires opencv).
    None if opencv is not installed.

All metrics are in [0, 1] unless stated otherwise.
Lower is better (less flicker).

EXTENSION POINT
---------------
Replace `warp_error_mean` with a flow-based metric that uses a stronger
optical flow estimator (RAFT, FlowNet, etc.) to get a more accurate
temporal consistency score.  See PROMPT_ADD_TEMPORAL_METRIC in constants.py.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_temporal_flicker(frames: list[np.ndarray]) -> dict:
    """
    Compute temporal-flicker metrics for a frame sequence.

    Parameters
    ----------
    frames:
        List of H×W×3 uint8 numpy arrays.

    Returns
    -------
    dict with keys: frame_diff_mean, frame_diff_p95, warp_error_mean.
    """
    if len(frames) < 2:
        return {
            "frame_diff_mean": 0.0,
            "frame_diff_p95": 0.0,
            "warp_error_mean": None,
        }

    diffs: list[np.ndarray] = []
    for a, b in zip(frames[:-1], frames[1:]):
        diff = np.abs(a.astype(np.float32) - b.astype(np.float32)) / 255.0
        diffs.append(diff)

    all_diffs = np.concatenate([d.ravel() for d in diffs])
    frame_diff_mean = float(np.mean(all_diffs))
    frame_diff_p95 = float(np.percentile(all_diffs, 95))

    warp_error_mean = _compute_warp_error(frames)

    return {
        "frame_diff_mean": frame_diff_mean,
        "frame_diff_p95": frame_diff_p95,
        "warp_error_mean": warp_error_mean,
    }


def _compute_warp_error(frames: list[np.ndarray]) -> float | None:
    """Compute mean optical-flow warping error (requires opencv)."""
    try:
        import cv2  # type: ignore[import]
    except ImportError:
        logger.debug("opencv not available; skipping warp_error_mean.")
        return None

    errors: list[float] = []
    for a, b in zip(frames[:-1], frames[1:]):
        gray_a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        gray_b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            gray_a,
            gray_b,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )
        h, w = gray_a.shape
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow[..., 0]).astype(np.float32)
        map_y = (grid_y + flow[..., 1]).astype(np.float32)
        warped = cv2.remap(
            a.astype(np.float32),
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        err = np.mean(np.abs(warped - b.astype(np.float32))) / 255.0
        errors.append(float(err))

    return float(np.mean(errors)) if errors else None
