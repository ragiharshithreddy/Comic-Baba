"""
Visualisation utilities.

EXTENSION POINT
---------------
Add GIF / side-by-side video export, optical flow visualisation, and
per-frame metrics overlay here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def save_comparison_gif(
    frames_in: list[np.ndarray],
    frames_out: list[np.ndarray],
    out_path: str | Path,
    *,
    fps: float = 6.0,
    max_frames: int = 40,
) -> Path:
    """
    Save a side-by-side comparison GIF of input vs output frames.

    Both sequences are padded / truncated to the same display length
    (*max_frames* from each), placed left / right, and exported as a GIF.

    Parameters
    ----------
    frames_in, frames_out:
        H×W×3 uint8 arrays.
    out_path:
        Destination .gif file.
    fps:
        Frame rate of the output GIF.
    max_frames:
        Maximum frames sampled from each sequence for display.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _sample(seq, n):
        if len(seq) <= n:
            return list(seq)
        step = max(1, len(seq) // n)
        return seq[::step][:n]

    sample_in = _sample(frames_in, max_frames)
    sample_out = _sample(frames_out, max_frames)
    n = max(len(sample_in), len(sample_out))

    # Pad shorter sequence with last frame
    while len(sample_in) < n:
        sample_in.append(sample_in[-1])
    while len(sample_out) < n:
        sample_out.append(sample_out[-1])

    combined: list[Image.Image] = []
    for fi, fo in zip(sample_in, sample_out):
        h = max(fi.shape[0], fo.shape[0])
        w = fi.shape[1] + fo.shape[1] + 4
        canvas = np.ones((h, w, 3), dtype=np.uint8) * 128
        canvas[: fi.shape[0], : fi.shape[1]] = fi
        canvas[: fo.shape[0], fi.shape[1] + 4 :] = fo
        combined.append(Image.fromarray(canvas))

    duration_ms = int(1000 / max(fps, 1))
    combined[0].save(
        out_path,
        save_all=True,
        append_images=combined[1:],
        loop=0,
        duration=duration_ms,
    )
    return out_path
