# Comic-Baba — Temporal Hallucinations Pipeline

> Character-consistent frame interpolation for comic clips.

Comic-Baba is a research-and-engineering pipeline for generating temporally coherent in-between comic frames from sparse input frame sequences. The goal is to increase visual smoothness (effective FPS), preserve identity and style consistency, and reduce temporal artifacts such as flicker and jitter.

---

## What this project does

Given a short comic clip (image sequence), Comic-Baba generates intermediate frames to:
1. **Increase FPS** through temporal hallucination/interpolation
2. **Keep character identity stable** across generated frames
3. **Reduce temporal flicker** (jitter, style drift, inconsistent line rendering)

---

## Full problem statement

Comic-style motion synthesis is a difficult low-level and high-level vision problem. Unlike natural videos, comic frames often include:
- stylized edges and sparse textures,
- abrupt artistic transitions,
- exaggerated motion cues,
- non-photorealistic shading and line work,
- scene-to-scene style changes even within one sequence.

Traditional frame interpolation methods (optimized mostly for natural video) can fail in this regime by producing blurry line art, unstable facial details, identity drift across frames, and temporal inconsistencies that become obvious in playback.

**Problem to solve:**
Given an ordered sequence of comic frames \(I_1, I_2, ..., I_n\), generate one or more plausible intermediate frames between successive pairs such that:
1. **Temporal coherence** is maintained over the full sequence
2. **Character consistency** (face, clothing, color palette, silhouette) is preserved
3. **Style fidelity** (ink lines, panel aesthetics, shading language) is retained
4. **Perceptual quality** is high enough for downstream creative/production workflows
5. **Runtime and reproducibility constraints** are practical for iterative experimentation

Comic-Baba addresses this with a modular pipeline that separates:
- data preparation and manifest-driven ingestion,
- interpolation/inference,
- objective and perceptual evaluation,
- reproducible experiment configuration.

---

## State of the art (SOTA) context

Modern interpolation and temporal synthesis systems generally combine four ideas:

### 1) Flow-guided warping and occlusion reasoning
Many high-performing methods estimate motion fields (optical flow or learned motion representations), warp source frames, and synthesize missing regions under occlusion/disocclusion constraints.

### 2) Multi-scale feature fusion
SOTA pipelines often model motion and appearance at multiple scales so they can preserve both global structure and fine details (e.g., line boundaries, facial micro-features).

### 3) Perceptual + temporal objectives
Pixel losses alone are insufficient. Strong systems use perceptual terms, edge/structure losses, and temporal consistency terms to prevent frame-wise quality from causing sequence-wise instability.

### 4) Generative priors for hard regions
For challenging motion or heavy stylization, generative models (including diffusion-style priors in recent literature) are increasingly used to hallucinate plausible details while preserving style.

### Where Comic-Baba fits
Comic-Baba is designed as an experimentation framework around these SOTA principles for the **comic domain**, with explicit emphasis on:
- identity consistency,
- low flicker,
- reproducible stage-by-stage evaluation,
- practical CLI-first workflows.

> Note: "state of the art" evolves rapidly; this repository is structured to let teams swap model components and evaluate improvements without rewriting the full stack.

---

## Real-world use cases

1. **Digital comic motion enhancement**  
   Convert sparse frame assets into smoother animated clips for trailers, reels, and social previews.

2. **Webtoon and motion-comic production**  
   Increase perceived fluidity while preserving original illustration style.

3. **Localization pipelines**  
   Re-render transitions after text replacement while maintaining character/style continuity.

4. **Archival remastering**  
   Improve playback smoothness for legacy comic animations and storyboard sequences.

5. **Creator tooling and rapid prototyping**  
   Let artists test pacing and camera moves before committing to expensive manual in-betweening.

6. **Educational media and explainer content**  
   Build visually coherent animated sequences from limited illustrated assets.

7. **Game narrative and visual novel cutscenes**  
   Create lightweight animated transitions from static art with minimal asset overhead.

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
