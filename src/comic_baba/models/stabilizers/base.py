"""
Abstract base class for all frame stabilisers.

To add a new stabiliser:
1. Sub-class BaseStabilizer.
2. Implement `stabilize(frames)`.
3. Register in `comic_baba.models.stabilizers.__init__.get_stabilizer`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseStabilizer(ABC):
    """Base class for temporal stabilisation models."""

    @abstractmethod
    def stabilize(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """
        Apply temporal stabilisation to a frame sequence.

        Parameters
        ----------
        frames:
            List of H×W×3 uint8 numpy arrays (interpolated frames_out).

        Returns
        -------
        Stabilised list of H×W×3 uint8 numpy arrays, same length as input.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class PassthroughStabilizer(BaseStabilizer):
    """No-op stabilizer (returns frames unchanged)."""

    def stabilize(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        return list(frames)
