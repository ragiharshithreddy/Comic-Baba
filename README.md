# Comic-Baba — Temporal Hallucinations Pipeline

> Character-consistent frame interpolation for comic clips.

## What this project does

Given a short comic clip (image sequence), generate intermediate frames to:
1. **Increase FPS** (temporal hallucination of in-between frames)
2. **Keep character identity stable** across generated frames
3. **Reduce temporal flicker** (no jitter / style drift)

---

## Quickstart (CPU, no GPU required)

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Generate a tiny synthetic sample
python scripts/make_tiny_sample.py

# 3. Run the pipeline
python scripts/run_prepare.py --config configs/baseline.yaml
python scripts/run_infer.py   --config configs/baseline.yaml
python scripts/run_eval.py    --config configs/baseline.yaml

# 4. Inspect results
ls outputs/
```

All three scripts produce structured output under `outputs/<run_id>/`.

---

## Repository structure

```
Comic-Baba/
  configs/          — YAML experiment configs
  docs/             — pipeline docs, task assignments, data format spec
  inputs/           — manifest.jsonl + raw data (gitignored)
  outputs/          — generated artifacts + metrics (gitignored)
  scripts/          — CLI entry points for each pipeline stage
  src/comic_baba/   — Python package (all reusable logic lives here)
  tests/            — pytest suite (includes smoke test)
  .devcontainer/    — Codespaces / VS Code devcontainer
  .github/          — CI workflow + PR template
```

See [`docs/PIPELINE.md`](docs/PIPELINE.md) for a full stage-by-stage description.

---

## Team

| Role | Owner | Scope |
|------|-------|-------|
| Data & ingestion | Teammate A | `src/comic_baba/io/`, `scripts/run_prepare.py` |
| Interpolation model | Teammate B | `src/comic_baba/models/interpolators/`, `scripts/run_infer.py` |
| Evaluation & integration | Teammate C | `src/comic_baba/eval/`, `scripts/run_eval.py`, CI |

See [`docs/TASKS.md`](docs/TASKS.md) for per-task acceptance criteria.

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md).
