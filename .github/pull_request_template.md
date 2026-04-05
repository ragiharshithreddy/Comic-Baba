## Summary

<!-- Describe what this PR does in 1-2 sentences. -->

## Related task / issue

<!-- Reference the task (e.g. "Task A-1 from docs/TASKS.md") or issue ("Closes #N"). -->

## How to test

<!-- Describe how a reviewer can verify your changes. Include commands if relevant. -->

```bash
# Example:
python scripts/make_tiny_sample.py
python scripts/run_prepare.py --config configs/baseline.yaml
```

## Checklist

- [ ] My code runs end-to-end on the tiny synthetic clip (CPU, no GPU required)
- [ ] I ran `ruff check src/ scripts/ tests/` — no errors
- [ ] I ran `pytest tests/ -v` — all tests pass
- [ ] I updated `docs/` if I changed an I/O contract or added a new feature
- [ ] The config file is updated / added if I changed pipeline behaviour
- [ ] My branch is up-to-date with `main`
- [ ] I added / updated tests for my changes

## Screenshots / sample output (optional)

<!-- Paste a snippet of terminal output or metrics JSON, if helpful. -->
