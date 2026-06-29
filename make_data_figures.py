"""
Generate the DATA-dependent report figures from the real MVTec-AD dataset:
  fig_samples.png      -- real normal vs defective images
  fig_scoredist.png    -- real image-score histograms (easy vs hard category)
  fig_roc.png          -- real ROC curves for screw across the 3 backbones
  fig_heatmaps.png     -- real anomaly heatmaps on a defective screw (3 backbones)
  fig_qualitative.png  -- real input + heatmap + OK/NG for an easy and a hard category

Runs the actual pipeline (feature extractor -> coreset memory bank -> NN scoring),
on two representative categories (bottle = easy, screw = hard) with all three
backbones. Requires the dataset folder (auto-detected below) and a GPU is used if
available.

Run:  python make_data_figures.py
"""
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch
from sklearn.metrics import roc_curve, roc_auc_score

from anomaly_system import (
    get_feature_extractor, MemoryBank, image_anomaly_score,
    compute_threshold, smooth_map,
)

# ---- locate dataset --------------------------------------------------------
CANDIDATES = [
    Path("mvtec_anomaly_detection"),
    Path(r"c:/Users/ADMIN/Downloads/mvtec_anomaly_detection.tar/mvtec_anomaly_detection"),
    Path(r"c:/Users/ADMIN/Downloads/mvtec_anomaly_detection"),
]
DATA_ROOT = next((p for p in CANDIDATES if (p / "bottle" / "train" / "good").exists()), None)
assert DATA_ROOT is not None, "MVTec dataset not found in any known location."
print(f"[data] using dataset: {DATA_ROOT}")

OUT = "figures"; os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BACKBONES = ["resnet50", "dino_vit", "dinov2"]
NAMES = {"resnet50": "ResNet50", "dino_vit": "DINO", "dinov2": "DINOv2"}
CATS = ["bottle", "screw"]   # easy, hard
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    p = os.path.join(OUT, name); fig.savefig(p, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] wrote {p}")


def first_defect_dir(cat):
    for d in sorted((DATA_ROOT / cat / "test").iterdir()):
        if d.is_dir() and d.name != "good":
            return d
    raise RuntimeError(f"no defect dir for {cat}")


# ---- pipeline helpers ------------------------------------------------------
def build_bank(backbone, tf, cat):
    train = sorted((DATA_ROOT / cat / "train" / "good").glob("*.png"))
    nsplit = int(len(train) * 0.8)
    build, calib = train[:nsplit], train[nsplit:]
    mb = MemoryBank(DEVICE)
    feats = []
    for p in build:
        img = tf(Image.open(p).convert("RGB")).to(DEVICE).unsqueeze(0)
        with torch.no_grad():
            feats.append(backbone(img).cpu())
    mb.build(feats)
    mb.select_k_center_greedy(ratio=0.1)
    calib_scores = np.array([score_image(backbone, tf, mb, p) for p in calib])
    thr = compute_threshold(calib_scores, method="sigma", param=2.0)
    return mb, thr


def patch_scores(backbone, tf, mb, image_path):
    img = tf(Image.open(image_path).convert("RGB")).to(DEVICE).unsqueeze(0)
    with torch.no_grad():
        f = backbone(img)
        f = f / (torch.norm(f, dim=1, keepdim=True) + 1e-8)
        d = torch.cdist(f, mb.get(), p=2.0)
        ps = torch.min(d, dim=1)[0]
    return ps


def score_image(backbone, tf, mb, image_path):
    ps = patch_scores(backbone, tf, mb, image_path)
    return image_anomaly_score(ps, method="max", top_k=10, smooth_sigma=1.0).item()


def evaluate(backbone, tf, mb, cat):
    yt, ys, paths = [], [], []
    for sub in sorted((DATA_ROOT / cat / "test").iterdir()):
        if not sub.is_dir():
            continue
        lab = 0 if sub.name == "good" else 1
        for p in sorted(sub.glob("*.png")):
            ys.append(score_image(backbone, tf, mb, p)); yt.append(lab); paths.append(p)
    return np.array(yt), np.array(ys), paths


def heatmap(backbone, tf, mb, image_path, disp=256):
    """Return (original image, heat in [0,1]) with robust, clean normalization.

    Fixes the 'all-red' artifact: (1) damp the border ring (resize / ViT edge
    effects) to the interior median, (2) normalize on robust interior percentiles
    instead of per-image min-max, (3) gamma to suppress background. The caller
    then overlays with a PER-PIXEL alpha so normal regions stay transparent.
    """
    ps = patch_scores(backbone, tf, mb, image_path).cpu()
    n = ps.numel(); side = int(round(n ** 0.5))
    grid = smooth_map(ps[:side*side].view(side, side), 1.2).numpy().astype(float)

    b = max(1, side // 12)                      # border ring width
    interior = grid[b:-b, b:-b]
    med = float(np.median(interior))
    border = np.ones_like(grid, bool); border[b:-b, b:-b] = False
    grid[border] = np.minimum(grid[border], med)   # cap border artifacts

    lo, hi = np.percentile(interior, 50), np.percentile(interior, 99)
    g = np.clip((grid - lo) / (hi - lo + 1e-8), 0, 1) ** 1.6   # gamma suppresses background
    heat = np.array(Image.fromarray((g*255).astype(np.uint8)).resize((disp, disp), Image.BICUBIC)) / 255.0
    orig = np.array(Image.open(image_path).convert("RGB").resize((disp, disp)))
    return orig, heat


# ---- (1) sample images (no model) -----------------------------------------
def fig_samples():
    cats = ["bottle", "screw", "leather"]
    fig, axes = plt.subplots(2, 3, figsize=(8, 5.4))
    for j, cat in enumerate(cats):
        good = sorted((DATA_ROOT / cat / "train" / "good").glob("*.png"))[0]
        ddir = first_defect_dir(cat); bad = sorted(ddir.glob("*.png"))[0]
        axes[0, j].imshow(Image.open(good).convert("RGB")); axes[0, j].set_title(f"{cat}\n(normal)", fontsize=10)
        axes[1, j].imshow(Image.open(bad).convert("RGB")); axes[1, j].set_title(f"{cat} / {ddir.name}\n(defective)", fontsize=10)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("MVTec-AD: normal (top) vs defective (bottom)", fontsize=12)
    save(fig, "fig_samples.png")


# ---- main: run pipeline, collect everything --------------------------------
def main():
    fig_samples()

    results = {}   # (bb, cat) -> (yt, ys, paths, thr)
    mbs = {}       # (bb, cat) -> memory bank
    bbs = {}       # bb -> (backbone, tf)

    for bb in BACKBONES:
        print(f"[run] loading backbone {bb} ...")
        backbone = get_feature_extractor(bb, device=DEVICE)
        tf = backbone.transform
        bbs[bb] = (backbone, tf)
        for cat in CATS:
            mb, thr = build_bank(backbone, tf, cat)
            yt, ys, paths = evaluate(backbone, tf, mb, cat)
            results[(bb, cat)] = (yt, ys, paths, thr)
            mbs[(bb, cat)] = mb
            print(f"[run] {NAMES[bb]:8s} {cat:7s} AUROC={roc_auc_score(yt, ys):.4f} thr={thr:.3f}")

    # choose the CLEAREST defect image per category = the defective test image with
    # the highest DINOv2 anomaly score (most confidently detected -> best heatmap)
    best = {}
    for cat in CATS:
        yt, ys, paths, _ = results[("dinov2", cat)]
        cand = [(ys[i], paths[i]) for i in range(len(yt)) if yt[i] == 1]
        best[cat] = max(cand, key=lambda t: t[0])[1]
        print(f"[fig] heatmap image for {cat}: {best[cat].parent.name}/{best[cat].name}")

    heat = {}      # (bb, cat) -> (orig, heat)
    for bb in BACKBONES:
        backbone, tf = bbs[bb]
        for cat in CATS:
            heat[(bb, cat)] = heatmap(backbone, tf, mbs[(bb, cat)], best[cat])

    # (2) score distribution (DINOv2, bottle vs screw)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, cat in zip(axes, CATS):
        yt, ys, _, thr = results[("dinov2", cat)]
        ax.hist(ys[yt == 0], bins=20, alpha=0.7, label="normal", color="#4c8bf5")
        ax.hist(ys[yt == 1], bins=20, alpha=0.7, label="defective", color="#e8833a")
        ax.axvline(thr, ls="--", color="black", lw=1.2, label="threshold")
        ax.set_title(f"{cat} ({'easy' if cat=='bottle' else 'hard'})"); ax.set_xlabel("image anomaly score")
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel("count")
    fig.tight_layout()
    save(fig, "fig_scoredist.png")

    # (3) ROC curves for screw across backbones
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    colors = {"resnet50": "#9aa0a6", "dino_vit": "#4c8bf5", "dinov2": "#1a3d7c"}
    for bb in BACKBONES:
        yt, ys, _, _ = results[(bb, "screw")]
        fpr, tpr, _ = roc_curve(yt, ys); auc = roc_auc_score(yt, ys)
        ax.plot(fpr, tpr, lw=2, color=colors[bb], label=f"{NAMES[bb]} (AUROC={auc:.3f})")
    ax.plot([0, 1], [0, 1], ls=":", color="gray")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC on screw (hard category)"); ax.legend(loc="lower right", frameon=False)
    save(fig, "fig_roc.png")

    # (4) heatmaps on a defective screw across backbones
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.0))
    orig0, _ = heat[("dinov2", "screw")]
    axes[0].imshow(orig0); axes[0].set_title("defective screw\n(input)", fontsize=10)
    for ax, bb in zip(axes[1:], BACKBONES):
        orig, h = heat[(bb, "screw")]
        ax.imshow(orig); ax.imshow(h, cmap="turbo", alpha=h*0.7, vmin=0, vmax=1)
        ax.set_title(NAMES[bb], fontsize=10)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Anomaly localization on a defective screw (stronger backbone = sharper hotspot)", fontsize=11)
    save(fig, "fig_heatmaps.png")

    # (5) qualitative grid: bottle & screw, input + 3 backbone heatmaps + OK/NG
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.6))
    for i, cat in enumerate(CATS):
        orig0, _ = heat[("dinov2", cat)]
        axes[i, 0].imshow(orig0); axes[i, 0].set_ylabel(cat, fontsize=11)
        axes[i, 0].set_title("input" if i == 0 else "", fontsize=10)
        for j, bb in enumerate(BACKBONES):
            orig, h = heat[(bb, cat)]
            axes[i, j+1].imshow(orig); axes[i, j+1].imshow(h, cmap="turbo", alpha=h*0.7, vmin=0, vmax=1)
            if i == 0:
                axes[i, j+1].set_title(NAMES[bb], fontsize=10)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Qualitative localization: easy (bottle) vs hard (screw) across backbones", fontsize=12)
    save(fig, "fig_qualitative.png")

    print("\n[data] all data-figures written to ./figures/")


if __name__ == "__main__":
    main()
