"""
Frame I/O utilities.

Handles reading / writing sequences of PNG frames that follow the naming
convention  frame_NNNNNN.png  (zero-padded, 6 digits).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from comic_baba.constants import FRAME_PATTERN


def load_frames(frames_dir: str | Path) -> list[np.ndarray]:
    """
    Load all PNG frames from *frames_dir* in sorted order.

    Returns
    -------
    list of H×W×3 uint8 numpy arrays.
    """
    frames_dir = Path(frames_dir)
    paths = sorted(frames_dir.glob("frame_*.png"))
    if not paths:
        raise FileNotFoundError(f"No frame_*.png files found in {frames_dir}")
    return [np.array(Image.open(p).convert("RGB")) for p in paths]


def save_frames(frames: list[np.ndarray], out_dir: str | Path) -> list[Path]:
    """
    Save a list of H×W×3 uint8 numpy arrays as PNGs.

    Returns
    -------
    list of saved file Paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for idx, frame in enumerate(frames):
        dest = out_dir / FRAME_PATTERN.format(idx=idx)
        Image.fromarray(frame.astype(np.uint8)).save(dest)
        saved.append(dest)
    return saved


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Return frame resized to (width, height) as uint8 RGB."""
    img = Image.fromarray(frame.astype(np.uint8))
    img = img.resize((width, height), Image.LANCZOS)
    return np.array(img)
