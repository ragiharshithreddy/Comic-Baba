# Contributing to Comic-Baba

Thank you for contributing!  This guide explains the collaboration workflow for the team.

---

## Branching strategy

- `main` is always runnable and protected.  No direct pushes.
- Create a feature branch for every task:
  ```
  git checkout -b feature/<topic>-<your-name>
  ```
- All changes go through Pull Requests.
- At least **1 approval** required before merge (2 for large changes).

### Branch naming examples
- `feature/data-ingestion-alice`
- `feature/interpolator-rife-bob`
- `feature/eval-identity-carol`
- `fix/manifest-validation-alice`

---

## Development setup

```bash
# 1. Clone (or open in Codespaces)
git clone https://github.com/ragiharshithreddy/Comic-Baba.git
cd Comic-Baba

# 2. Install in editable mode with dev dependencies
pip install -e ".[dev]"

# 3. Verify smoke test passes
python scripts/make_tiny_sample.py
pytest tests/ -v
```

---

## Before opening a PR

Run these checks locally:

```bash
# Lint
ruff check src/ scripts/ tests/

# Format check
ruff format --check src/ scripts/ tests/

# Tests
pytest tests/ -v
```

All three must pass.

---

## Pull Request checklist

Use the PR template (`.github/pull_request_template.md`).

Every PR must:
- [ ] Reference the issue or task it addresses (`Closes #N` or `Part of Task A-1`).
- [ ] Include a brief description of what changed and why.
- [ ] Pass all CI checks (lint + tests + smoke pipeline).
- [ ] Include a "how to test" section if the change is not obvious.
- [ ] Update `docs/` if behaviour or I/O contracts change.

---

## Code style

- Python 3.10+.
- Line length ≤ 100 characters.
- Use type hints for all public functions.
- No hardcoded paths — use the config system.
- Add docstrings to all public classes and functions.
- `ruff` is the linter + formatter.  Run `ruff format src/` to auto-format.

---

## Adding a new interpolator

1. Sub-class `BaseInterpolator` in `src/comic_baba/models/interpolators/`.
2. Register in `get_interpolator` in `src/comic_baba/models/interpolators/__init__.py`.
3. Add a config file in `configs/`.
4. Add tests in `tests/test_interpolators.py`.
5. Document in `docs/EXPERIMENTS.md`.

See `PROMPT_ADD_INTERPOLATOR` in `src/comic_baba/constants.py` for the exact spec.

---

## Adding a new metric

1. Implement in the appropriate `src/comic_baba/eval/metrics_*.py` file.
2. Wire it into `run_eval` in `src/comic_baba/pipelines/evaluation.py`.
3. Add tests.
4. Update `docs/DATA_FORMAT.md` with the new key in the metrics schema.

---

## Questions?

Open a GitHub Discussion or ping the team in Slack/Discord.
