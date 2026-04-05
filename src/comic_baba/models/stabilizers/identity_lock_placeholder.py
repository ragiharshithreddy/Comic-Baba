"""
IdentityLockStabilizer — placeholder for character-identity-based stabilisation.

CURRENT STATUS: Placeholder — returns frames unchanged.

WHAT TO IMPLEMENT (see PROMPT_ADD_STABILIZER in constants.py)
-------------------------------------------------------------
1. Extract per-frame character embeddings (CLIP / DINO / face encoder).
2. Detect frames whose embedding drifts beyond a threshold from the running
   mean.
3. Optionally replace or blend flagged frames with their neighbours to
   reduce identity discontinuities.

ACCEPTANCE CRITERIA (Teammate C)
---------------------------------
- Returns same-length list of H×W×3 uint8 arrays.
- identity_drift_mean is measurably reduced vs. no stabilisation on the
  test clip.
- Runtime on a 10-frame 256×256 clip < 5 s on CPU.
"""

from __future__ import annotations

import logging

import numpy as np

from comic_baba.models.stabilizers.base import BaseStabilizer

logger = logging.getLogger(__name__)


class IdentityLockStabilizer(BaseStabilizer):
    """
    Identity-lock stabiliser (placeholder implementation).

    Replace this class body with a real embedding-based stabiliser.
    See the module docstring and PROMPT_ADD_STABILIZER for the contract.
    """

    def stabilize(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        logger.warning(
            "IdentityLockStabilizer is a placeholder — no stabilisation applied. "
            "Implement the real logic as described in the module docstring."
        )
        return list(frames)
