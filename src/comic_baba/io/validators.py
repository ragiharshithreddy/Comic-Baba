"""
Manifest-entry validation.

All manifest validation logic lives here so it can be imported without
pulling in heavy dependencies.
"""

from __future__ import annotations

from comic_baba.constants import MANIFEST_KEYS_REQUIRED, MANIFEST_SOURCE_TYPES


def validate_manifest_entry(entry: dict, lineno: int = 0) -> None:
    """Raise ValueError if *entry* does not conform to the manifest schema."""
    loc = f"line {lineno}: " if lineno else ""

    missing = MANIFEST_KEYS_REQUIRED - set(entry.keys())
    if missing:
        raise ValueError(f"{loc}manifest entry missing keys: {sorted(missing)}")

    if entry["source_type"] not in MANIFEST_SOURCE_TYPES:
        raise ValueError(
            f"{loc}source_type must be one of {MANIFEST_SOURCE_TYPES}, got {entry['source_type']!r}"
        )

    for int_key in ("num_frames", "width", "height"):
        if not isinstance(entry[int_key], int) or entry[int_key] <= 0:
            raise ValueError(f"{loc}{int_key} must be a positive integer, got {entry[int_key]!r}")

    if not isinstance(entry["fps"], (int, float)) or entry["fps"] <= 0:
        raise ValueError(f"{loc}fps must be a positive number, got {entry['fps']!r}")

    if not isinstance(entry["clip_id"], str) or not entry["clip_id"]:
        raise ValueError(f"{loc}clip_id must be a non-empty string")
