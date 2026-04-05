"""
BlendInterpolator — CPU-safe baseline interpolator.

Strategy
--------
For each consecutive pair of input frames (A, B), generate (factor-1)
intermediate frames by linear blending:

    I_t = round((1 - t) * A + t * B),  t ∈ {1/factor, 2/factor, …, (factor-1)/factor}

This is a trivial but fully functional baseline that:
- Requires no GPU.
- Requires no external model weights.
- Is deterministic.
- Serves as a correctness baseline for the evaluation suite.

EXTENSION POINT
---------------
Replace this class with a RIFE or FILM wrapper that honours the same
`interpolate(frames, factor) → frames` contract.  See PROMPT_ADD_INTERPOLATOR
in constants.py for the exact specification.
"""

from __future__ import annotations

import numpy as np

from comic_baba.models.interpolators.base import BaseInterpolator


class BlendInterpolator(BaseInterpolator):
    """
    Linear-blend frame interpolator (CPU-safe baseline).

    Parameters
    ----------
    duplicate_boundary:
        If True, duplicate the last frame so the output length is exactly
        ``len(frames) * factor``.  Default: False (standard behaviour).
    """

    def __init__(self, *, duplicate_boundary: bool = False) -> None:
        self.duplicate_boundary = duplicate_boundary

    def interpolate(self, frames: list[np.ndarray], factor: int) -> list[np.ndarray]:
        """Return a linearly blended up-sampled frame sequence."""
        if factor < 1:
            raise ValueError(f"factor must be >= 1, got {factor}")
        if factor == 1:
            return list(frames)
        if len(frames) < 2:
            # Nothing to interpolate; just return the single frame.
            return list(frames)

        out: list[np.ndarray] = []
        for i in range(len(frames) - 1):
            a = frames[i].astype(np.float32)
            b = frames[i + 1].astype(np.float32)
            out.append(frames[i])
            for step in range(1, factor):
                t = step / factor
                blended = np.round((1.0 - t) * a + t * b).clip(0, 255).astype(np.uint8)
                out.append(blended)
        # Append the last original frame.
        out.append(frames[-1])

        if self.duplicate_boundary:
            # Pad to exact factor × original length.
            target = len(frames) * factor
            while len(out) < target:
                out.append(out[-1].copy())

        return out
