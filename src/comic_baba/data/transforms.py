"""
Frame transforms (augmentations + normalisation).

All transforms operate on individual H×W×3 uint8 numpy arrays and return
the same type unless documented otherwise.

EXTENSION POINT
---------------
Add training-time augmentations (colour jitter, random crop, …) here.
Keep transforms deterministic when a seed is provided.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def center_crop(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Return a centre-cropped region of *frame*."""
    h, w = frame.shape[:2]
    top = max(0, (h - height) // 2)
    left = max(0, (w - width) // 2)
    return frame[top : top + height, left : left + width]


def resize(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Return *frame* resized to (width, height) using Lanczos resampling."""
    img = Image.fromarray(frame.astype(np.uint8))
    img = img.resize((width, height), Image.LANCZOS)
    return np.array(img)


def normalize_float(frame: np.ndarray) -> np.ndarray:
    """Convert uint8 [0,255] → float32 [0,1]."""
    return frame.astype(np.float32) / 255.0


def denormalize_uint8(frame: np.ndarray) -> np.ndarray:
    """Convert float32 [0,1] → uint8 [0,255] (clamped)."""
    return np.clip(frame * 255.0, 0, 255).astype(np.uint8)
