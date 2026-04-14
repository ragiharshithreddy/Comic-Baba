"""
ComicClipDataset — iterable dataset over manifest entries.

Each item yielded is a dict:
    {
        "clip_id":  str,
        "frames":   list[np.ndarray],   # H×W×3 uint8
        "fps":      float,
        "metadata": dict,               # full manifest entry
    }

EXTENSION POINT
---------------
Add a PyTorch Dataset wrapper here when GPU training is needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from comic_baba.io.frames import load_frames, resize_frame
from comic_baba.io.manifest import iter_manifest
from comic_baba.io.video import decode_video_to_frames


class ComicClipDataset:
    """Iterate over comic clips described in a manifest file."""

    def __init__(
        self,
        manifest_path: str | Path,
        frames_root: str | Path | None = None,
        resize: tuple[int, int] | None = None,
        split: str | None = None,
        transforms: list | None = None,
    ) -> None:
        """
        Parameters
        ----------
        manifest_path:
            Path to `inputs/manifest.jsonl`.
        frames_root:
            Root directory used to resolve relative `path` entries for frames-type clips.
            Defaults to the repo root (parent of `inputs/`).
        resize:
            Optional (width, height) to resize every frame.
        split:
            If set, only yield clips whose `split` field matches.
        transforms:
            Optional list of transform functions to apply to each frame.
        """
        self.manifest_path = Path(manifest_path)
        self.frames_root = Path(frames_root) if frames_root else self.manifest_path.parent.parent
        self.resize = resize
        self.split = split
        self.transforms = transforms or []

    def __iter__(self) -> Iterator[dict]:
        for entry in iter_manifest(self.manifest_path):
            if self.split and entry.get("split") != self.split:
                continue
            frames = self._load_entry(entry)
            if self.resize:
                w, h = self.resize
                frames = [resize_frame(f, w, h) for f in frames]

            if self.transforms:
                applied_frames = []
                for f in frames:
                    for t in self.transforms:
                        f = t(f)
                    applied_frames.append(f)
                frames = applied_frames

            yield {
                "clip_id": entry["clip_id"],
                "frames": frames,
                "fps": entry["fps"],
                "metadata": entry,
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_entry(self, entry: dict) -> list[np.ndarray]:
        source_type = entry["source_type"]
        path = Path(entry["path"])
        if not path.is_absolute():
            path = self.frames_root / path

        if source_type == "frames":
            return load_frames(path)
        elif source_type == "video":
            import tempfile

            with tempfile.TemporaryDirectory() as tmp_dir:
                decode_video_to_frames(path, tmp_dir)
                return load_frames(tmp_dir)
        else:
            raise ValueError(f"Unknown source_type: {source_type!r}")
