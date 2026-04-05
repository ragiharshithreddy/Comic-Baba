"""
Abstract base class for all frame interpolators.

To add a new interpolator:
1. Sub-class BaseInterpolator.
2. Implement `interpolate(frames, factor)`.
3. Register in `comic_baba.models.interpolators.__init__.get_interpolator`.
4. Add a config entry and document in configs/baseline.yaml.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseInterpolator(ABC):
    """Base class for frame interpolation models."""

    @abstractmethod
    def interpolate(self, frames: list[np.ndarray], factor: int) -> list[np.ndarray]:
        """
        Generate intermediate frames.

        Parameters
        ----------
        frames:
            Input sequence of H×W×3 uint8 numpy arrays.
        factor:
            Temporal up-sampling factor.  A factor of 2 inserts 1 new frame
            between each consecutive pair; factor 4 inserts 3.

        Returns
        -------
        New list of H×W×3 uint8 numpy arrays with length
        ``len(frames) + (len(frames) - 1) * (factor - 1)``.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
