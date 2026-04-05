"""Random seed utilities for reproducibility."""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Set random seeds for Python and NumPy (add torch if available)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore[import]

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
