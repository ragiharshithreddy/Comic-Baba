"""Stabilizer implementations."""

from comic_baba.models.stabilizers.base import BaseStabilizer
from comic_baba.models.stabilizers.identity_lock_placeholder import IdentityLockStabilizer
from comic_baba.models.stabilizers.temporal_smoothing_placeholder import (
    TemporalSmoothingStabilizer,
)

__all__ = ["BaseStabilizer", "IdentityLockStabilizer", "TemporalSmoothingStabilizer"]


def get_stabilizer(name: str, **kwargs) -> "BaseStabilizer":
    """
    Factory: return a stabilizer instance by name.

    Supported names
    ---------------
    "identity_lock"      — IdentityLockStabilizer (placeholder)
    "temporal_smoothing" — TemporalSmoothingStabilizer (placeholder)
    "none"               — PassthroughStabilizer (no-op)

    EXTENSION POINT
    ---------------
    Register production stabilizers here once implemented.
    See PROMPT_ADD_STABILIZER in constants.py.
    """
    from comic_baba.models.stabilizers.base import PassthroughStabilizer

    registry: dict[str, type] = {
        "none": PassthroughStabilizer,
        "identity_lock": IdentityLockStabilizer,
        "temporal_smoothing": TemporalSmoothingStabilizer,
    }
    if name not in registry:
        raise ValueError(f"Unknown stabilizer {name!r}. Available: {sorted(registry.keys())}.")
    return registry[name](**kwargs)
