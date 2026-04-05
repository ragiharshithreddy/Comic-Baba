"""
Image-quality metrics.

Metrics computed
----------------
psnr : float | None
    Peak signal-to-noise ratio (dB) vs. ground-truth frames.  None if no GT.
ssim : float | None
    Structural similarity index vs. ground-truth frames.  None if no GT.
sharpness_mean : float
    Mean Laplacian variance across frames (proxy for sharpness).
contrast_mean : float
    Mean RMS contrast across frames.

EXTENSION POINT
---------------
Add LPIPS (perceptual similarity) once torchmetrics or lpips package is
available.  See PROMPT_ADD_QUALITY_METRIC in constants.py for the contract.
"""

from __future__ import annotations

import numpy as np


def compute_quality(
    frames: list[np.ndarray],
    gt_frames: list[np.ndarray] | None = None,
) -> dict:
    """
    Compute image-quality metrics.

    Parameters
    ----------
    frames:
        Generated / interpolated frames (H×W×3 uint8).
    gt_frames:
        Optional ground-truth frames at the same indices.  Must have the
        same length as *frames* when provided.

    Returns
    -------
    dict with keys: psnr, ssim, sharpness_mean, contrast_mean.
    """
    if not frames:
        return {"psnr": None, "ssim": None, "sharpness_mean": 0.0, "contrast_mean": 0.0}

    sharpness_values = [_laplacian_variance(f) for f in frames]
    contrast_values = [_rms_contrast(f) for f in frames]

    result: dict = {
        "psnr": None,
        "ssim": None,
        "sharpness_mean": float(np.mean(sharpness_values)),
        "contrast_mean": float(np.mean(contrast_values)),
    }

    if gt_frames is not None:
        if len(gt_frames) != len(frames):
            raise ValueError(
                f"gt_frames length ({len(gt_frames)}) != frames length ({len(frames)})"
            )
        psnr_vals = [_psnr(f, g) for f, g in zip(frames, gt_frames)]
        ssim_vals = [_ssim(f, g) for f, g in zip(frames, gt_frames)]
        result["psnr"] = float(np.mean(psnr_vals))
        result["ssim"] = float(np.mean(ssim_vals))

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _laplacian_variance(frame: np.ndarray) -> float:
    """Return Laplacian variance of the greyscale frame (sharpness proxy)."""
    gray = np.mean(frame.astype(np.float32), axis=-1)
    # Simple discrete Laplacian via finite differences (no extra deps)
    lap = (
        gray[1:-1, 2:] + gray[1:-1, :-2] + gray[2:, 1:-1] + gray[:-2, 1:-1] - 4.0 * gray[1:-1, 1:-1]
    )
    return float(np.var(lap))


def _rms_contrast(frame: np.ndarray) -> float:
    """Return RMS contrast of the greyscale frame."""
    gray = np.mean(frame.astype(np.float32), axis=-1) / 255.0
    return float(np.sqrt(np.mean((gray - gray.mean()) ** 2)))


def _psnr(pred: np.ndarray, gt: np.ndarray, max_val: float = 255.0) -> float:
    """Peak signal-to-noise ratio (dB)."""
    mse = np.mean((pred.astype(np.float32) - gt.astype(np.float32)) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10.0 * np.log10(max_val**2 / mse))


def _ssim(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    k1: float = 0.01,
    k2: float = 0.03,
    max_val: float = 255.0,
) -> float:
    """Simplified global SSIM (no sliding window)."""
    p = pred.astype(np.float64)
    g = gt.astype(np.float64)
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    mu_p, mu_g = p.mean(), g.mean()
    sigma_p = p.std()
    sigma_g = g.std()
    sigma_pg = np.mean((p - mu_p) * (g - mu_g))
    num = (2 * mu_p * mu_g + c1) * (2 * sigma_pg + c2)
    den = (mu_p**2 + mu_g**2 + c1) * (sigma_p**2 + sigma_g**2 + c2)
    return float(num / den)
