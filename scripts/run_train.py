#!/usr/bin/env python
"""
scripts/run_train.py
--------------------
Run the training pipeline.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from comic_baba.pipelines.train import run_train
from comic_baba.utils.logging import setup_logging

def main():
    parser = argparse.ArgumentParser(description="Train the interpolator.")
    parser.add_argument("--config", default="configs/baseline.yaml", help="Config file path.")
    parser.add_argument("--checkpoint", help="Path to checkpoint to resume from.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    args = parser.parse_args()

    setup_logging(args.log_level)
    run_train(config_path=args.config, checkpoint_path=args.checkpoint)

if __name__ == "__main__":
    main()
