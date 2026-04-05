#!/usr/bin/env python
"""
scripts/run_prepare.py
----------------------
Stage 1: validate manifest and extract / standardise frames.

Usage
-----
    python scripts/run_prepare.py [--config CONFIG] [--run-id RUN_ID]

Outputs
-------
    outputs/<run_id>/artifacts/frames_in/<clip_id>/frame_*.png
    outputs/<run_id>/configs/resolved_config.yaml

The run_id is printed to stdout on success (for piping to subsequent stages).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running the script directly from the repo root without `pip install -e .`.
# When the package is installed (e.g. in CI via `pip install -e .`), this is a no-op.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from comic_baba.pipelines.prepare_data import run_prepare
from comic_baba.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: prepare frames from manifest.")
    parser.add_argument("--config", default="configs/baseline.yaml", help="Config file path.")
    parser.add_argument(
        "--run-id", default=None, help="Run identifier (auto-generated if omitted)."
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    args = parser.parse_args()

    setup_logging(args.log_level)
    run_id = run_prepare(config_path=args.config, run_id=args.run_id)
    # Print run_id for shell scripting convenience.
    print(f"RUN_ID={run_id}")


if __name__ == "__main__":
    main()
