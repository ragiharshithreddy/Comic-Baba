#!/usr/bin/env python
"""
scripts/run_infer.py
--------------------
Stage 2: interpolate frames and optionally stabilise.

Usage
-----
    python scripts/run_infer.py --run-id RUN_ID [--config CONFIG]

The RUN_ID is the value printed by run_prepare.py (e.g. run_20240101_120000_abc123).

Outputs
-------
    outputs/<run_id>/artifacts/frames_out/<clip_id>/frame_*.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running the script directly from the repo root without `pip install -e .`.
# When the package is installed (e.g. in CI via `pip install -e .`), this is a no-op.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from comic_baba.pipelines.inference import run_infer
from comic_baba.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: interpolate and stabilise frames.")
    parser.add_argument("--config", default="configs/baseline.yaml", help="Config file path.")
    parser.add_argument("--run-id", required=True, help="Run identifier from run_prepare.py.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    args = parser.parse_args()

    setup_logging(args.log_level)
    run_infer(config_path=args.config, run_id=args.run_id)


if __name__ == "__main__":
    main()
