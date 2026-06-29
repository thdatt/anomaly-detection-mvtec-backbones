# Industrial Anomaly Detection — Backbone Comparison (MVTec-AD)

Code for my bachelor thesis. The idea is simple: take one fixed PatchCore-style
pipeline and **only swap the feature backbone**, then see how much the backbone
alone changes anomaly-detection results on MVTec-AD (all 15 categories).

Three backbones, increasing in strength:
**ResNet50** (supervised CNN) → **DINO ViT-S/8** (self-supervised) → **DINOv2 ViT-S/14** (self-supervised, higher resolution).

> Short version of what I found: in my experiments, stronger backbones consistently
> gave better detection, and most of the difference comes from a few hard,
> fine-grained categories (screw, capsule, pill). I'm reporting this as an
> observation under a fixed pipeline, not a general law.

---

## Results (full 15-category run)

| Backbone | Mean AUROC | Mean F1 |
|---|---:|---:|
| ResNet50 (layer2+3) | 0.9764 | 0.9155 |
| DINO ViT-S/8 | 0.9855 | 0.9472 |
| **DINOv2 ViT-S/14** | **0.9930** | **0.9615** |

Per-backbone, per-category numbers are in `results_resnet50.md`,
`results_dino_vit.md`, `results_dinov2.md` (these are the actual script outputs).

Where the backbone actually matters (F1 on hard categories):

| Category | ResNet50 | DINO | DINOv2 |
|---|---:|---:|---:|
| screw | 0.537 | 0.723 | **0.873** |
| capsule | 0.642 | 0.899 | 0.867 |
| pill | 0.908 | 0.940 | **0.986** |

Easy categories (bottle, hazelnut, leather, tile, metal_nut) are already near 1.0
AUROC for every backbone, so the backbone barely matters there.

---

## Quick start

```bash
# install (CUDA build of torch; change index-url if you're on CPU/other CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Download MVTec-AD (≈5 GB, not in this repo):
https://www.mvtec.com/company/research/datasets/mvtec-ad

Extract it like this (folder name `mvtec_anomaly_detection` next to the scripts):

```
mvtec_anomaly_detection/
  bottle/ cable/ ... zipper/
    train/good/*.png
    test/good/*.png
    test/<defect_type>/*.png
```

Then run a full 15-category evaluation for any backbone:

```bash
python evaluate_backbones.py resnet50     # -> results_resnet50.md
python evaluate_backbones.py dino_vit     # -> results_dino_vit.md
python evaluate_backbones.py dinov2        # -> results_dinov2.md
```

Look at how the threshold policy changes F1 for one category:

```bash
python tune_threshold.py dinov2 screw
```

---

## What's in here

| File | What it does |
|---|---|
| `anomaly_system.py` | Core pipeline: `Config`, the 3 backbone extractors, `MemoryBank` (k-center coreset), scoring, leakage-free threshold |
| `evaluate_backbones.py` | Runs the full 15-category evaluation for one backbone, writes `results_<backbone>.md` |
| `tune_threshold.py` | Threshold sweep (sigma / quantile) + the test-tuned oracle ceiling, per category |
| `Final_Code.ipynb` | Annotated walkthrough on a single category |
| `results_*.md` | Actual outputs for the three backbones |
| `make_figures.py` / `make_data_figures.py` | Regenerate the thesis figures (charts from the result numbers + heatmaps/ROC from the dataset) |
| `LEARNING_NOTES.md` | My own notes on the concepts (Vietnamese) |

---

## The pipeline (only the backbone changes)

```
image -> backbone -> L2-normalized patch features
                          |
                   k-center coreset memory bank (~10%)
                          |
                   nearest-neighbour distance per patch
                          |
                   Gaussian smoothing (sigma = 1.0)
                          |
                   image score = max patch score
                          |
                   threshold = mean + 2*std (from held-out good only)
                          |
                   OK / NG  (+ heatmap)
```

Two things I was careful about:

- **Leakage-free**: the threshold is set only on held-out *good* images (an 80/20
  split of the training set). Test labels are never used to pick the threshold.
- **Controlled**: preprocessing, memory bank, scoring and threshold are identical
  across all three runs — only the backbone differs (note: the official DINOv2 also
  runs at a higher input resolution, so resolution differs too; I mention this in the
  thesis).

---

## References

- PatchCore — https://arxiv.org/abs/2106.08265
- DINO — https://arxiv.org/abs/2104.14294
- DINOv2 — https://arxiv.org/abs/2304.07193
- MVTec-AD — https://www.mvtec.com/company/research/datasets/mvtec-ad
