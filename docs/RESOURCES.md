# Resources

## Models

### Frame interpolation

| Model | Paper | Code | Notes |
|-------|-------|------|-------|
| RIFE | [arXiv:2011.06294](https://arxiv.org/abs/2011.06294) | [hzwer/RIFE](https://github.com/hzwer/RIFE) | Fast, CPU-compatible for small clips |
| FILM | [arXiv:2202.04901](https://arxiv.org/abs/2202.04901) | [google-research/frame-interpolation](https://github.com/google-research/frame-interpolation) | High quality; TensorFlow |
| DAIN | [arXiv:1904.00831](https://arxiv.org/abs/1904.00831) | [baowenbo/DAIN](https://github.com/baowenbo/DAIN) | Depth-aware; requires GPU |
| EMA-VFI | [arXiv:2303.00440](https://arxiv.org/abs/2303.00440) | [MCG-NJU/EMA-VFI](https://github.com/MCG-NJU/EMA-VFI) | SOTA interpolation |

### Feature / identity embeddings

| Model | Repo | Notes |
|-------|------|-------|
| CLIP | [openai/CLIP](https://github.com/openai/CLIP) | Style + semantic; CPU-ok |
| DINOv2 | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | Dense features; good for characters |
| ArcFace | [InsightFace](https://github.com/deepinsight/insightface) | Face identity; works on close-ups |

### Optical flow

| Model | Notes |
|-------|-------|
| OpenCV Farneback | CPU; already integrated |
| RAFT | [princeton-vl/RAFT](https://github.com/princeton-vl/RAFT); GPU recommended |

---

## Datasets

### Comic / animation clips

| Dataset | URL | License | Notes |
|---------|-----|---------|-------|
| Anime Sketch Colorization Pair | Kaggle | CC-BY | Close to comic style |
| Comics in the Wild | [link](https://github.com/ybFang/ComicsInTheWild) | Research only | Panels, not clips |
| UCF101 (action clips) | [link](https://www.crcv.ucf.edu/data/UCF101.php) | Research | Useful for FPS baseline |
| Vimeo-90K | [link](http://toflow.csail.mit.edu/) | Research | Standard interpolation benchmark |

> **Note:** For comic-specific clips, collect from public domain comics (e.g. Project Gutenberg illustrations) or generate synthetic data using `scripts/make_tiny_sample.py`.

---

## Training infrastructure

### Recommended platforms

| Platform | Free GPU | Notes |
|----------|----------|-------|
| Google Colab | T4 (limited hours) | Good for quick iterations |
| Kaggle Notebooks | P100 / T4 (30 h/wk) | More stable; easy dataset hosting |
| Lightning AI | T4 | Good Codespaces-like experience |

### Experiment tracking

| Tool | URL | Free tier |
|------|-----|-----------|
| Weights & Biases | [wandb.ai](https://wandb.ai) | Yes (unlimited runs) |
| MLflow (local) | — | Yes |
| TensorBoard | — | Yes |

### Recommended W&B setup

```bash
pip install wandb
wandb login  # paste your API key
```

Then in your training script:
```python
import wandb
wandb.init(project="comic-baba", config=config_dict)
wandb.log({"loss": loss_val, "frame_diff_mean": metric})
wandb.save("outputs/<run_id>/metrics/summary.json")
```

---

## Key Python packages

| Package | Purpose | Install |
|---------|---------|---------|
| `Pillow` | Frame I/O | `pip install Pillow` |
| `numpy` | Array ops | `pip install numpy` |
| `pyyaml` | Config files | `pip install pyyaml` |
| `click` | CLI | `pip install click` |
| `opencv-python-headless` | Video decode + flow | `pip install opencv-python-headless` |
| `torch` + `torchvision` | GPU models | `pip install torch torchvision` |
| `transformers` | CLIP, DINO | `pip install transformers` |
| `wandb` | Experiment tracking | `pip install wandb` |
| `pytest` | Testing | `pip install pytest` |
| `ruff` | Linting | `pip install ruff` |

---

## References

- [Video Frame Interpolation Survey (2023)](https://arxiv.org/abs/2309.01146)
- [Temporal Consistency in Video Generation](https://arxiv.org/abs/2310.10647)
- [ComicFace: Face Detection in Comics](https://arxiv.org/abs/2101.03354)
