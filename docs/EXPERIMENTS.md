# Experiment Tracking

## Naming convention

Experiment IDs are auto-generated as:
```
<id_prefix>_<YYYYMMDD_HHMMSS>_<6-char UUID>
```

Override with `--run-id` for reproducibility or comparisons.

---

## Baseline experiment

| Setting | Value |
|---------|-------|
| Interpolator | `blend` (linear blend) |
| Factor | 2× |
| Stabilizer | `none` |
| Resolution | 256×256 |

Run:
```bash
python scripts/make_tiny_sample.py
python scripts/run_prepare.py --config configs/baseline.yaml
# → prints RUN_ID=run_YYYYMMDD_HHMMSS_xxxxxx
python scripts/run_infer.py   --run-id <RUN_ID>
python scripts/run_eval.py    --run-id <RUN_ID>
```

Expected output metrics (tiny synthetic clip):
- `frame_diff_mean`: ≈ 0.03–0.10 (depends on synthetic clip content)
- `identity_drift_mean`: ≈ 0.00–0.05

---

## Adding a new experiment

1. Copy `configs/baseline.yaml` to `configs/<exp_name>.yaml`.
2. Change the relevant fields (interpolator, factor, stabilizer, resize).
3. Run the pipeline with `--config configs/<exp_name>.yaml`.
4. Compare `summary.json` outputs.

---

## Planned experiments

| ID | Interpolator | Factor | Stabilizer | Notes |
|----|-------------|--------|------------|-------|
| baseline | blend | 2 | none | Linear blend; no GPU |
| rife_2x | rife | 2 | none | Add RIFE wrapper (Teammate B) |
| rife_4x | rife | 4 | none | Higher FPS target |
| rife_smooth | rife | 2 | temporal_smoothing | Add smoothing pass |
| rife_identity | rife | 2 | identity_lock | Real identity lock |

---

## Logging (optional but recommended)

For team-wide experiment visibility, use Weights & Biases:
```bash
pip install wandb
wandb login
# then add wandb.init(project="comic-baba") in the inference/eval scripts
```

All runs should at minimum log:
- resolved config
- clip_metrics.jsonl
- summary.json

---

## Comparing runs

```python
import json, glob, pandas as pd

summaries = []
for path in glob.glob("outputs/*/metrics/summary.json"):
    run_id = path.split("/")[1]
    s = json.load(open(path))
    s["run_id"] = run_id
    summaries.append(s)

df = pd.DataFrame(summaries).set_index("run_id")
print(df.sort_values("frame_diff_mean_mean"))
```
