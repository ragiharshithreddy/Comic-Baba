"""
TemporalSmoothingStabilizer — placeholder for temporal-smoothing stabilisation.

CURRENT STATUS: Placeholder — returns frames unchanged.

WHAT TO IMPLEMENT (see PROMPT_ADD_STABILIZER in constants.py)
-------------------------------------------------------------
Option A — Pixel-level temporal smoothing (simplest):
    Apply a 1-D Gaussian or box filter along the temporal axis for each
    pixel independently.  Fast but blurs motion.

Option B — Flow-guided warping (stronger):
    Compute optical flow between consecutive frames, then apply flow-guided
    weighted averaging to reduce jitter while preserving edges.

Option C — Feature-space smoothing (best for character coherence):
    Smooth latent/feature representations across time rather than pixel
    values, then decode back.

ACCEPTANCE CRITERIA (Teammate C)
---------------------------------
- Returns same-length list of H×W×3 uint8 arrays.
- frame_diff_p95 is measurably reduced vs. no smoothing on the test clip.
- Runtime on a 10-frame 256×256 clip < 2 s on CPU.
"""

from __future__ import annotations

import logging

import numpy as np

from comic_baba.models.stabilizers.base import BaseStabilizer

logger = logging.getLogger(__name__)


class TemporalSmoothingStabilizer(BaseStabilizer):
    """
    Temporal-smoothing stabiliser (placeholder implementation).

    Replace this class body with a real implementation.
    See the module docstring and PROMPT_ADD_STABILIZER for the contract.
    """

    def __init__(self, *, window_size: int = 3) -> None:
        self.window_size = window_size

    def stabilize(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        logger.warning(
            "TemporalSmoothingStabilizer is a placeholder — no smoothing applied. "
            "Implement the real logic as described in the module docstring."
        )
        return list(frames)
