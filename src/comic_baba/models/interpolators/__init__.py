"""Interpolator implementations."""

from comic_baba.models.interpolators.base import BaseInterpolator
from comic_baba.models.interpolators.baseline_blend import BlendInterpolator
from comic_baba.models.interpolators.rife import RIFEInterpolator

__all__ = ["BaseInterpolator", "BlendInterpolator", "RIFEInterpolator"]


def get_interpolator(name: str, **kwargs) -> "BaseInterpolator":
    """
    Factory function: return an interpolator instance by name.

    Supported names
    ---------------
    "blend"   — BlendInterpolator (CPU-safe baseline, no external weights)
    "rife"    — RIFEInterpolator (requires torch)

    EXTENSION POINT
    ---------------
    Register additional interpolators here:
        "rife"  — RIFEInterpolator (requires torch + RIFE weights)
        "film"  — FILMInterpolator (requires tensorflow + FILM weights)
    """
    registry: dict[str, type] = {
        "blend": BlendInterpolator,
        "rife": RIFEInterpolator,
    }
    if name not in registry:
        raise ValueError(
            f"Unknown interpolator {name!r}. Available: {sorted(registry.keys())}. "
            "See PROMPT_ADD_INTERPOLATOR in constants.py to add a new one."
        )
    return registry[name](**kwargs)
