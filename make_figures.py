"""
Generate report figures into ./figures/ as PNGs.

Two kinds of figures:
  (A) DATA figures  -- built from the ACTUAL measured results of this project
                       (mean AUROC/F1 per backbone, per-category DINOv2, the screw
                       ablation, and the threshold sensitivity sweep).
  (B) DIAGRAMS      -- schematic illustrations (pipeline, backbones, coreset) drawn
                       with matplotlib; these are explanatory, not data.

Run:  python make_figures.py
Output: figures/*.png  (300 dpi, ready for includegraphics)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# Consistent palette
C_RESNET, C_DINO, C_DINOV2 = "#9aa0a6", "#4c8bf5", "#1a3d7c"
C_AUC, C_F1 = "#1a3d7c", "#e8833a"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote {path}")


# ---------------------------------------------------------------------------
# (A1) Main cross-backbone comparison: mean AUROC and mean F1
# ---------------------------------------------------------------------------
def fig_main_comparison():
    backbones = ["ResNet50", "DINO\nViT-S/8", "DINOv2\nViT-S/14"]
    auroc = [0.9764, 0.9855, 0.9930]
    f1 = [0.9155, 0.9472, 0.9615]
    x = np.arange(len(backbones)); w = 0.36
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    b1 = ax.bar(x - w/2, auroc, w, label="Mean AUROC", color=C_AUC)
    b2 = ax.bar(x + w/2, f1, w, label="Mean F1", color=C_F1)
    for b in list(b1) + list(b2):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.002,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(backbones)
    ax.set_ylim(0.88, 1.0); ax.set_ylabel("Score")
    ax.set_title("Monotonic improvement with backbone quality\n(mean over 15 MVTec-AD categories)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
    save(fig, "fig_main_comparison.png")


# ---------------------------------------------------------------------------
# (A2) Per-category DINOv2 results (AUROC and F1)
# ---------------------------------------------------------------------------
def fig_percategory_dinov2():
    cats = ["bottle","cable","capsule","carpet","grid","hazelnut","leather",
            "metal_nut","pill","screw","tile","toothbrush","transistor","wood","zipper"]
    auroc = [1.0000,0.9921,0.9677,1.0000,1.0000,1.0000,1.0000,1.0000,0.9921,
             0.9627,1.0000,0.9944,0.9950,0.9939,0.9974]
    f1 = [0.9921,0.9425,0.8673,0.9780,0.9500,1.0000,1.0000,0.9894,0.9857,
          0.8732,1.0000,0.9524,0.9524,0.9600,0.9793]
    x = np.arange(len(cats)); w = 0.4
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(x - w/2, auroc, w, label="AUROC", color=C_AUC)
    ax.bar(x + w/2, f1, w, label="F1", color=C_F1)
    ax.axhline(0.9930, ls="--", lw=1, color=C_AUC, alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(cats, rotation=45, ha="right")
    ax.set_ylim(0.80, 1.01); ax.set_ylabel("Score")
    ax.set_title("DINOv2 per-category results (threshold = mean + 2 x std)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    save(fig, "fig_percategory_dinov2.png")


# ---------------------------------------------------------------------------
# (A3) Hard categories: F1 across the three backbones
# ---------------------------------------------------------------------------
def fig_hard_categories():
    cats = ["screw", "capsule", "pill"]
    resnet = [0.537, 0.642, 0.908]
    dino   = [0.723, 0.899, 0.940]
    dinov2 = [0.873, 0.867, 0.986]
    x = np.arange(len(cats)); w = 0.26
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x - w, resnet, w, label="ResNet50", color=C_RESNET)
    ax.bar(x,     dino,   w, label="DINO",     color=C_DINO)
    ax.bar(x + w, dinov2, w, label="DINOv2",   color=C_DINOV2)
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylim(0.0, 1.12); ax.set_ylabel("F1 score")
    ax.set_title("Hard categories: F1 improves with stronger backbones")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
    save(fig, "fig_hard_categories.png")


# ---------------------------------------------------------------------------
# (A4) Screw ablation: resolution / backbone vs F1
# ---------------------------------------------------------------------------
def fig_screw_ablation():
    labels = ["DINO\n@224", "DINO\n@320", "DINO\n@384", "DINO @224\n+4-layer", "DINOv2\n@518"]
    f1 = [0.575, 0.608, 0.663, 0.592, 0.763]
    colors = [C_DINO, C_DINO, C_DINO, "#7aa7f0", C_DINOV2]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4.0))
    bars = ax.bar(x, f1, 0.6, color=colors)
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 0.85); ax.set_ylabel("F1 score (screw)")
    ax.set_title("Screw: higher resolution and DINOv2 close the gap")
    save(fig, "fig_screw_ablation.png")


# ---------------------------------------------------------------------------
# (A5) Threshold sensitivity: DINOv2 mean F1 vs sigma multiplier k
# ---------------------------------------------------------------------------
def fig_threshold_sensitivity():
    k = [1.0, 1.5, 2.0, 2.5, 3.0]
    meanf1 = [0.9575, 0.9573, 0.9615, 0.9473, 0.9364]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(k, meanf1, "-o", color=C_DINOV2, lw=2)
    best = int(np.argmax(meanf1))
    ax.scatter([k[best]], [meanf1[best]], s=120, facecolors="none",
               edgecolors=C_F1, linewidths=2, zorder=5)
    ax.annotate("best (k=2)", (k[best], meanf1[best]),
                textcoords="offset points", xytext=(10, -14), color=C_F1)
    ax.set_xlabel("threshold multiplier  k   (threshold = mean + k x std)")
    ax.set_ylabel("Mean F1 (15 categories)")
    ax.set_ylim(0.93, 0.965)
    ax.set_title("DINOv2 F1 is stable for k in [1, 2] and degrades by k=3")
    save(fig, "fig_threshold_sensitivity.png")


# ---------------------------------------------------------------------------
# (B1) Pipeline diagram (schematic)
# ---------------------------------------------------------------------------
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11, 2.6))
    ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 2.6)
    stages = ["Input\nimage", "Backbone\n(ResNet/DINO/\nDINOv2)", "L2 norm +\nmemory bank\n(coreset 10%)",
              "NN distance\n+ Gaussian\nsmoothing", "Max score\n+ threshold", "OK / NG\n+ heatmap"]
    n = len(stages); bw = 1.55; gap = (11 - n*bw) / (n+1)
    for i, s in enumerate(stages):
        xx = gap + i*(bw+gap)
        col = "#1a3d7c" if i == 1 else "#eef2fb"
        tc = "white" if i == 1 else "black"
        box = FancyBboxPatch((xx, 0.8), bw, 1.0, boxstyle="round,pad=0.02,rounding_size=0.08",
                             linewidth=1.2, edgecolor="#1a3d7c", facecolor=col)
        ax.add_patch(box)
        ax.text(xx+bw/2, 1.3, s, ha="center", va="center", fontsize=8.5, color=tc)
        if i < n-1:
            ax.add_patch(FancyArrowPatch((xx+bw, 1.3), (xx+bw+gap, 1.3),
                         arrowstyle="-|>", mutation_scale=14, color="#555"))
    ax.text(5.5, 2.35, "Only the backbone (Stage 2) changes across experiments",
            ha="center", fontsize=9, style="italic", color="#1a3d7c")
    save(fig, "fig_pipeline.png")


# ---------------------------------------------------------------------------
# (B2) Backbone comparison (schematic)
# ---------------------------------------------------------------------------
def fig_backbones():
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.axis("off"); ax.set_xlim(0, 9); ax.set_ylim(0, 3.4)
    data = [
        ("ResNet50", "Supervised CNN\n(ImageNet)", "224 x 224", "1536-d", C_RESNET),
        ("DINO ViT-S/8", "Self-supervised\n(ImageNet)", "224 x 224", "384-d", C_DINO),
        ("DINOv2 ViT-S/14", "Self-supervised\n(~142M images)", "518 x 518", "384-d", C_DINOV2),
    ]
    bw = 2.6; gap = (9 - 3*bw)/4
    for i, (name, kind, res, dim, col) in enumerate(data):
        xx = gap + i*(bw+gap)
        ax.add_patch(FancyBboxPatch((xx, 0.5), bw, 2.4, boxstyle="round,pad=0.03,rounding_size=0.1",
                     linewidth=1.5, edgecolor=col, facecolor="white"))
        ax.add_patch(FancyBboxPatch((xx, 2.35), bw, 0.55, boxstyle="round,pad=0.0,rounding_size=0.1",
                     linewidth=0, facecolor=col))
        ax.text(xx+bw/2, 2.62, name, ha="center", va="center", fontsize=10.5,
                color="white", fontweight="bold")
        ax.text(xx+bw/2, 1.85, kind, ha="center", va="center", fontsize=9)
        ax.text(xx+bw/2, 1.25, f"input: {res}", ha="center", va="center", fontsize=9)
        ax.text(xx+bw/2, 0.85, f"features: {dim}", ha="center", va="center", fontsize=9)
    ax.annotate("", xy=(9, 0.2), xytext=(0, 0.2),
                arrowprops=dict(arrowstyle="-|>", color="#888"))
    ax.text(4.5, 0.02, "increasing feature quality  -->", ha="center", fontsize=9, color="#888")
    save(fig, "fig_backbones.png")


# ---------------------------------------------------------------------------
# (B3) Coreset selection (illustrative scatter; real k-center on 2D points)
# ---------------------------------------------------------------------------
def fig_coreset():
    rng = np.random.default_rng(3)
    pts = np.vstack([rng.normal(m, 0.7, (120, 2)) for m in ([0,0],[4,1],[2,4],[5,4])])
    # k-center greedy on the 2D points
    m = max(1, int(len(pts) * 0.1))
    sel = [0]
    d = np.linalg.norm(pts - pts[0], axis=1)
    for _ in range(1, m):
        j = int(np.argmax(d)); sel.append(j)
        d = np.minimum(d, np.linalg.norm(pts - pts[j], axis=1))
    sel = np.array(sel)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.scatter(pts[:,0], pts[:,1], s=18, color="#c7cdd6", label="all normal features")
    ax.scatter(pts[sel,0], pts[sel,1], s=70, color="#e8833a", edgecolors="black",
               linewidths=0.5, label=f"coreset (10%)", zorder=5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("k-center-greedy keeps a diverse 10% that\ncovers the feature space")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    for sp in ax.spines.values():
        sp.set_visible(True)
    save(fig, "fig_coreset.png")


def fig_manual_auto():
    """Manual vs automated inspection comparison (schematic, no photos)."""
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.axis("off"); ax.set_xlim(0, 9); ax.set_ylim(0, 3.8)

    def panel(x, title, fc, ec, rows):
        ax.add_patch(FancyBboxPatch((x, 0.3), 3.7, 3.0, boxstyle="round,pad=0.03,rounding_size=0.1",
                     linewidth=1.6, edgecolor=ec, facecolor="white"))
        ax.add_patch(FancyBboxPatch((x, 2.75), 3.7, 0.55, boxstyle="round,pad=0.0,rounding_size=0.1",
                     linewidth=0, facecolor=fc))
        ax.text(x+1.85, 3.02, title, ha="center", va="center", fontsize=12, color="white", fontweight="bold")
        for i, (mark, txt) in enumerate(rows):
            yy = 2.4 - i*0.5
            ax.text(x+0.25, yy, mark, ha="left", va="center", fontsize=12,
                    color=("#cc3333" if mark == "✗" else "#2e8b57"))
            ax.text(x+0.7, yy, txt, ha="left", va="center", fontsize=10.5)

    panel(0.3, "Manual inspection", "#9aa0a6", "#9aa0a6", [
        ("✗", "Slow (a bottleneck)"),
        ("✗", "Costly to staff"),
        ("✗", "Inconsistent (fatigue)"),
        ("✗", "Does not scale"),
    ])
    panel(5.0, "Automated (vision)", "#1a3d7c", "#1a3d7c", [
        ("✓", "Fast, every unit"),
        ("✓", "Low marginal cost"),
        ("✓", "Consistent decisions"),
        ("✓", "24/7, scalable"),
    ])
    ax.add_patch(FancyArrowPatch((4.05, 1.8), (4.95, 1.8), arrowstyle="-|>", mutation_scale=20, color="#1a3d7c"))
    save(fig, "fig_manual_auto.png")


def fig_split():
    """Leakage-free split diagram (schematic)."""
    fig, ax = plt.subplots(figsize=(9, 4.0))
    ax.axis("off"); ax.set_xlim(0, 9); ax.set_ylim(0, 4.1)

    def box(x, y, w, h, text, fc, tc="black", fs=9.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                     linewidth=1.3, edgecolor="#1a3d7c", facecolor=fc))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs, color=tc)

    box(0.2, 2.4, 2.4, 0.9, "Normal training\nimages", "#eef2fb")
    box(3.4, 2.7, 2.2, 0.8, "Build set (80%)", "#1a3d7c", "white")
    box(3.4, 1.5, 2.2, 0.8, "Calibration (20%)", "#4c8bf5", "white")
    box(6.4, 2.7, 2.3, 0.8, "Memory bank", "#eef2fb")
    box(6.4, 1.5, 2.3, 0.8, "Threshold\n(mean + 2 x std)", "#fdf0e6")
    box(0.2, 0.3, 2.4, 0.9, "Test set\n(kept separate)", "#f3f4f6")
    ax.text(1.4, 0.05, "used only for the final metric", ha="center", fontsize=8, style="italic", color="#888")

    for (x0,y0),(x1,y1) in [((2.6,2.95),(3.4,3.1)), ((2.6,2.75),(3.4,1.9)),
                            ((5.6,3.1),(6.4,3.1)), ((5.6,1.9),(6.4,1.9))]:
        ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1), arrowstyle="-|>", mutation_scale=13, color="#555"))
    ax.text(4.5, 3.95, "Threshold is set only on held-out normal images (no test labels)",
            ha="center", fontsize=9, style="italic", color="#1a3d7c")
    save(fig, "fig_split.png")


if __name__ == "__main__":
    fig_main_comparison()
    fig_percategory_dinov2()
    fig_hard_categories()
    fig_screw_ablation()
    fig_threshold_sensitivity()
    fig_pipeline()
    fig_backbones()
    fig_coreset()
    fig_split()
    fig_manual_auto()
    print("\nAll figures written to ./figures/")
