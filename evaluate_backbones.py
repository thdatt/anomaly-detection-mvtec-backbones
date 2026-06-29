"""
Evaluate anomaly detection across all 15 MVTec-AD categories.

Usage:
    python evaluate_backbones.py resnet50
    python evaluate_backbones.py dinov2 sigma 2.0

Outputs: results_<backbone>.md with per-category metrics and mean scores.
"""
import sys
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from pathlib import Path
from anomaly_system import (
    Config, get_feature_extractor, MemoryBank,
    image_anomaly_score, compute_threshold
)

# Parse arguments
BACKBONE = sys.argv[1] if len(sys.argv) > 1 else 'dinov2'
if len(sys.argv) > 2:
    THRESHOLD_METHOD = sys.argv[2]
else:
    THRESHOLD_METHOD = 'sigma'

if len(sys.argv) > 3:
    THRESHOLD_PARAM = float(sys.argv[3])
else:
    THRESHOLD_PARAM = 2.0

# Setup
config = Config()
config.threshold_method = THRESHOLD_METHOD
config.threshold_param = THRESHOLD_PARAM

backbone = get_feature_extractor(BACKBONE, device=config.device)
tf = backbone.transform
base_path = config.base_path
results_file = f"results_{BACKBONE}.md"


def extract_features(image_path):
    """Load image, extract and return features (on CPU)."""
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


def write_results(rows, total_categories, finished):
    """Write markdown results table."""
    status = "FINISHED" if finished else f"RUNNING... {len(rows)}/{total_categories} done"

    lines = [
        f"# Cross-Category Evaluation — backbone: {BACKBONE}",
        "",
        f"Status: {status}",
        f"Coreset ratio: {config.coreset_ratio} | Scoring: max over smoothed map "
        f"(smooth_sigma={config.smooth_sigma})",
        f"Threshold: method={config.threshold_method}, param={config.threshold_param} "
        f"(leakage-free: set on held-out good only)",
        "",
        "| Category | Build | Test good | Test def | Threshold | AUC-ROC | F1 | FP |",
        "|----------|------:|----------:|---------:|----------:|--------:|---:|---:|",
    ]

    for cat, n_build, n_good, n_def, thr, auc, f1, fp in rows:
        lines.append(
            f"| {cat} | {n_build} | {n_good} | {n_def} | {thr:.3f} | {auc:.4f} | {f1:.4f} | {fp}/{n_good} |"
        )

    if rows:
        aucs = [r[5] for r in rows]
        f1s = [r[6] for r in rows]
        lines.append(
            f"| **mean ({len(rows)})** | | | | | **{np.mean(aucs):.4f}** | **{np.mean(f1s):.4f}** | |"
        )

    Path(results_file).write_text("\n".join(lines) + "\n", encoding="utf-8")


# Main evaluation loop
categories = sorted([
    d.name for d in base_path.iterdir()
    if d.is_dir() and (d / 'train' / 'good').exists()
])

write_results([], len(categories), False)
print(f"[{BACKBONE}] Created {results_file}. Categories: {len(categories)}", flush=True)

from PIL import Image

rows = []
for cat in categories:
    # Split training: 80% memory bank, 20% threshold calibration (leakage-free)
    train_dir = base_path / cat / 'train' / 'good'
    all_train = sorted(train_dir.glob('*.png'))
    n_split = int(len(all_train) * 0.8)
    build_files = all_train[:n_split]
    calib_files = all_train[n_split:] if n_split < len(all_train) else all_train[-3:]

    # Build memory bank + select via k-center greedy
    mb = MemoryBank(config.device)
    mb.build([extract_features(p) for p in build_files])
    mb.select_k_center_greedy(ratio=config.coreset_ratio)

    # Calibrate threshold on held-out good scores only
    calib_scores = np.array([score_image(p, mb) for p in calib_files])
    threshold = compute_threshold(calib_scores, method=config.threshold_method, param=config.threshold_param)

    # Evaluate on test set
    test_root = base_path / cat / 'test'
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
    y_pred = (y_score >= threshold).astype(int)

    n_good = int((y_true == 0).sum())
    n_def = int((y_true == 1).sum())
    auc = roc_auc_score(y_true, y_score)
    f1 = f1_score(y_true, y_pred)
    fp = int(((y_pred == 1) & (y_true == 0)).sum())

    rows.append((cat, len(build_files), n_good, n_def, threshold, auc, f1, fp))
    print(f"[{BACKBONE}] {cat:12s} AUC={auc:.4f} F1={f1:.4f} FP={fp}/{n_good}", flush=True)
    write_results(rows, len(categories), len(rows) == len(categories))

    # Cleanup
    del mb
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"[{BACKBONE}] DONE -> {results_file}", flush=True)
