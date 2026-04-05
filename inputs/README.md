# inputs/

This directory holds the manifest and raw data for the pipeline.

## Structure

```
inputs/
  manifest.jsonl        — one JSON object per clip (required)
  frames/               — decoded frame directories (gitignored for large datasets)
    <clip_id>/
      frame_000000.png
      …
  videos/               — optional raw video files (gitignored)
    <clip_id>.mp4
```

## Getting started

Generate a tiny synthetic sample:
```bash
python scripts/make_tiny_sample.py
```

This creates `inputs/frames/tiny_clip/` and `inputs/manifest.jsonl`.

## Data format

See [`../docs/DATA_FORMAT.md`](../docs/DATA_FORMAT.md) for the full manifest schema.

## Important

- **Do not commit large frame directories or video files.**
  Only `manifest.jsonl`, `.gitkeep` files, and `README.md` are tracked.
- Store large datasets on Kaggle Datasets, Google Drive, or your team's shared storage.
- Document your dataset in a `data/README.md` with source URL + checksum.
