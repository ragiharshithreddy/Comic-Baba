"""Interpolator implementations."""

from comic_baba.models.interpolators.base import BaseInterpolator
from comic_baba.models.interpolators.baseline_blend import BlendInterpolator

__all__ = ["BaseInterpolator", "BlendInterpolator"]


def get_interpolator(name: str, **kwargs) -> "BaseInterpolator":
    """
    Factory function: return an interpolator instance by name.

    Supported names
    ---------------
    "blend"   — BlendInterpolator (CPU-safe baseline, no external weights)

    EXTENSION POINT
    ---------------
    Register additional interpolators here:
        "rife"  — RIFEInterpolator (requires torch + RIFE weights)
        "film"  — FILMInterpolator (requires tensorflow + FILM weights)
    """
    registry: dict[str, type] = {
        "blend": BlendInterpolator,
    }
    if name not in registry:
        raise ValueError(
            f"Unknown interpolator {name!r}. Available: {sorted(registry.keys())}. "
            "See PROMPT_ADD_INTERPOLATOR in constants.py to add a new one."
        )
    return registry[name](**kwargs)
