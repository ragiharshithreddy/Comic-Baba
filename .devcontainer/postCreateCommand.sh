#!/bin/bash
# .devcontainer/postCreateCommand.sh
# Runs after the devcontainer is created.

set -euo pipefail

echo "=== Installing Comic-Baba package (editable + dev extras) ==="
pip install -e ".[dev]"

echo "=== Generating tiny sample clip ==="
python scripts/make_tiny_sample.py

echo "=== Running smoke test to verify environment ==="
pytest tests/ -v --tb=short -q

echo ""
echo "✅ Devcontainer ready! See README.md for quickstart."
