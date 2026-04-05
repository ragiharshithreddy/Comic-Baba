"""
Video decoding utilities.

Decodes a video file to individual PNG frames using opencv-python-headless
when available, with a clear ImportError message otherwise.

EXTENSION POINT
---------------
Replace the opencv-based decoder with a faster/specialised loader
(e.g. decord, PyAV) without changing the public interface.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from comic_baba.io.frames import save_frames


def decode_video_to_frames(
    video_path: str | Path,
    out_dir: str | Path,
    *,
    max_frames: int | None = None,
) -> list[Path]:
    """
    Decode *video_path* to PNG frames saved in *out_dir*.

    Parameters
    ----------
    video_path:
        Path to the input video file.
    out_dir:
        Directory where frame_NNNNNN.png files will be written.
    max_frames:
        If set, decode at most this many frames.

    Returns
    -------
    List of saved frame Paths.
    """
    try:
        import cv2  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "opencv-python-headless is required for video decoding. "
            "Install it with: pip install opencv-python-headless"
        ) from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frames: list[np.ndarray] = []
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()

    if not frames:
        raise ValueError(f"No frames decoded from {video_path}")

    return save_frames(frames, out_dir)
