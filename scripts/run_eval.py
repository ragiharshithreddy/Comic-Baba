#!/usr/bin/env python
"""
scripts/run_eval.py
-------------------
Stage 3: compute evaluation metrics.

Usage
-----
    python scripts/run_eval.py --run-id RUN_ID [--config CONFIG]

Outputs
-------
    outputs/<run_id>/metrics/clip_metrics.jsonl
    outputs/<run_id>/metrics/summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running the script directly from the repo root without `pip install -e .`.
# When the package is installed (e.g. in CI via `pip install -e .`), this is a no-op.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from comic_baba.pipelines.evaluation import run_eval
from comic_baba.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3: compute evaluation metrics.")
    parser.add_argument("--config", default="configs/baseline.yaml", help="Config file path.")
    parser.add_argument("--run-id", required=True, help="Run identifier from run_prepare.py.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    args = parser.parse_args()

    setup_logging(args.log_level)
    summary = run_eval(config_path=args.config, run_id=args.run_id)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
