"""
Threshold analysis tool: test various policies on held-out good scores.

Usage:
    python tune_threshold.py dinov2 screw
    python tune_threshold.py dino_vit screw 384   # optional ViT input resolution

Shows how precision, recall, F1 change across threshold policies.
The optional third argument overrides the ViT input resolution, which
reproduces the resolution sweep reported for the screw category (e.g.
DINO at 320/384). It is ignored for resnet50.

IMPORTANT: The threshold MUST be chosen from a policy applied to HELD-OUT GOOD images only.
The "ORACLE" row (best F1 on test labels) is shown only to show the ceiling — it requires
data leakage and is NOT usable. The gap between your chosen policy and the oracle reveals
how much headroom the feature representation leaves.
"""
import sys
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from pathlib import Path
from PIL import Image
from anomaly_system import (
    Config, get_feature_extractor, MemoryBank,
    image_anomaly_score, compute_threshold
)

# Parse arguments
BACKBONE = sys.argv[1] if len(sys.argv) > 1 else 'dinov2'
CATEGORY = sys.argv[2] if len(sys.argv) > 2 else 'screw'
IMG_SIZE = int(sys.argv[3]) if len(sys.argv) > 3 else None  # optional ViT resolution

# Setup
config = Config()
backbone = get_feature_extractor(BACKBONE, device=config.device, img_size=IMG_SIZE)
tf = backbone.transform
base_path = config.base_path


def extract_features(image_path):
    """Load image, extract features (on CPU)."""
    image = tf(Image.open(image_path).convert('RGB')).to(config.device).unsqueeze(0)
    with torch.no_grad():
        features = backbone(image)
    return features.cpu()


def score_image(image_path, memory_bank):
    """Compute anomaly score for one image."""
    image = tf(Image.open(image_path).convert('RGB')).to(config.device).unsqueeze(0)

    with torch.no_grad():
        features = backbone(image)
        features = features / (torch.norm(features, dim=1, keepdim=True) + 1e-8)
        distances = torch.cdist(features, memory_bank.get(), p=2.0)
        patch_scores = torch.min(distances, dim=1)[0]

    return image_anomaly_score(
        patch_scores,
        smooth_sigma=config.smooth_sigma
    ).item()


# Build memory bank + held-out split
train_dir = base_path / CATEGORY / 'train' / 'good'
all_train = sorted(train_dir.glob('*.png'))
n_split = int(len(all_train) * 0.8)
build_files = all_train[:n_split]
calib_files = all_train[n_split:] if n_split < len(all_train) else all_train[-3:]

mb = MemoryBank(config.device)
mb.build([extract_features(p) for p in build_files])
mb.select_k_center_greedy(ratio=config.coreset_ratio)

# Score calibration set (held-out good)
calib_scores = np.array([score_image(p, mb) for p in calib_files])

# Score test set
test_root = base_path / CATEGORY / 'test'
y_true, y_score = [], []
for subdir in sorted(test_root.iterdir()):
    if not subdir.is_dir():
        continue
    label = 0 if subdir.name == 'good' else 1
    for image_path in sorted(subdir.glob('*.png')):
        y_score.append(score_image(image_path, mb))
        y_true.append(label)

y_true = np.array(y_true)
y_score = np.array(y_score)
auc = roc_auc_score(y_true, y_score)
n_good = int((y_true == 0).sum())
n_def = int((y_true == 1).sum())


def format_row(name, thr):
    """Format metrics for a threshold."""
    y_pred = (y_score >= thr).astype(int)
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    return f"{name:28s} thr={thr:.4f}  P={p:.3f}  R={r:.3f}  F1={f:.4f}  FP={fp}/{n_good}"


# Print analysis
res_tag = f" @ {IMG_SIZE}px" if IMG_SIZE else ""
print(f"\n=== {BACKBONE}{res_tag} / {CATEGORY} ===  AUC={auc:.4f}  (good={n_good}, defect={n_def})")
print(f"calib good (held-out): mean={calib_scores.mean():.4f} std={calib_scores.std():.4f} "
      f"n={len(calib_scores)}\n")

print("-- POLICY: sigma (threshold = mean + k*std of held-out good) --")
for k in (1.0, 2.0, 3.0, 4.0):
    thr = compute_threshold(calib_scores, 'sigma', k)
    print("  " + format_row(f"sigma k={k}", thr))

print("\n-- POLICY: quantile (threshold = q-quantile of held-out good) --")
for q in (0.90, 0.95, 0.98, 0.99, 1.00):
    thr = compute_threshold(calib_scores, 'quantile', q)
    print("  " + format_row(f"quantile q={q}", thr))

# Oracle: best F1 on test (NOT usable, for reference only)
best_f1, best_thr = -1.0, None
for thr in np.unique(y_score):
    f = f1_score(y_true, (y_score >= thr).astype(int), zero_division=0)
    if f > best_f1:
        best_f1, best_thr = f, thr

print("\n-- CEILING (NOT usable: tuned on test labels) --")
print("  " + format_row("ORACLE best-F1 on test", best_thr))

print(f"\nINSTRUCTIONS:")
print(f"  1. Pick the highest-F1 row from sigma/quantile policies above")
print(f"  2. Set config.threshold_method and config.threshold_param to match")
print(f"  3. The gap to ORACLE shows remaining representation headroom\n")
