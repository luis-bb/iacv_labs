"""
visualization.py – D2: All figures for the report.

- Keypoints per method (rich keypoints with scale & orientation)
- Correct/incorrect matches (green/red lines)
- Degradation curves: Matching Score vs k
- Pareto: time vs accuracy
- Tables from CSV aggregation
"""

import os
import csv
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd

import config as cfg


# ------------------------------------------------------------------
# Keypoint visualisation
# ------------------------------------------------------------------

def draw_keypoints_panel(images_and_kps, output_path, titles=None):
    """Draw rich keypoints for multiple methods side by side.

    Parameters
    ----------
    images_and_kps : list of (gray_image, keypoints, method_name)
    output_path    : str
    """
    n = len(images_and_kps)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (img, kps, name) in zip(axes, images_and_kps):
        if img.ndim == 2:
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_color = img.copy()

        out = cv2.drawKeypoints(img_color, kps, None,
                                 color=(0, 255, 0),
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        ax.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{name} ({len(kps)} kps)", fontsize=10)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved {output_path}")


# ------------------------------------------------------------------
# Match visualisation (green correct, red incorrect)
# ------------------------------------------------------------------

def draw_matches_colored(img1, kps1, img2, kps2,
                          correct, incorrect, output_path, title=""):
    """Draw correct matches in green and incorrect in red."""
    # Convert to color if needed
    if img1.ndim == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if img2.ndim == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    canvas = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:] = img2

    # Draw incorrect (red) first, then correct (green) on top
    for m in incorrect:
        pt1 = tuple(map(int, kps1[m.queryIdx].pt))
        pt2 = (int(kps2[m.trainIdx].pt[0]) + w1,
               int(kps2[m.trainIdx].pt[1]))
        cv2.line(canvas, pt1, pt2, (0, 0, 255), 1)

    for m in correct:
        pt1 = tuple(map(int, kps1[m.queryIdx].pt))
        pt2 = (int(kps2[m.trainIdx].pt[0]) + w1,
               int(kps2[m.trainIdx].pt[1]))
        cv2.line(canvas, pt1, pt2, (0, 255, 0), 1)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    ax.set_title(f"{title}  (green={len(correct)} correct, "
                 f"red={len(incorrect)} incorrect)", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved {output_path}")


# ------------------------------------------------------------------
# Degradation curve: Matching Score vs k
# ------------------------------------------------------------------

def plot_degradation(csv_path, output_path, metric="Matching_Score",
                     categories=("i", "v")):
    """Plot metric vs k (pair difficulty), one line per method."""
    df = pd.read_csv(csv_path)

    for cat in categories:
        sub = df[df["category"] == cat]
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        for method in sorted(sub["method"].unique()):
            msub = sub[sub["method"] == method]
            means = msub.groupby("k")[metric].mean()
            ax.plot(means.index, means.values, marker="o",
                    markersize=4, label=method)

        ax.set_xlabel("Pair difficulty k (1→k)")
        ax.set_ylabel(f"Mean {metric}")
        ax.set_title(f"Degradation ({cat} sequences)")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = output_path.replace(".png", f"_{cat}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"✓ Saved {out}")


# ------------------------------------------------------------------
# Pareto plot: time vs accuracy
# ------------------------------------------------------------------

def plot_pareto(csv_path, output_path):
    """Scatter of total_ms (x, log) vs Matching Score or HA@3 (y)."""
    df = pd.read_csv(csv_path)

    if "total_ms" not in df.columns:
        print("⚠ No timing data in CSV, skipping Pareto")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, title in [(axes[0], "Matching_Score", "Matching Score"),
                               (axes[1], "HA@3", "HA @ 3px")]:
        for method in sorted(df["method"].unique()):
            m = df[df["method"] == method]
            ax.scatter(m["total_ms"].mean(), m[metric].mean(),
                       s=60, label=method, zorder=5)
            ax.annotate(method, (m["total_ms"].mean(), m[metric].mean()),
                        fontsize=6, alpha=0.8)

        ax.set_xscale("log")
        ax.set_xlabel("Total time (ms, log)")
        ax.set_ylabel(f"Mean {title}")
        ax.set_title(f"Pareto: Time vs {title}")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"✓ Saved {output_path}")


# ------------------------------------------------------------------
# D1: Generate tables from CSV
# ------------------------------------------------------------------

def generate_tables(csv_path, output_dir):
    """Generate Tables A, B, C from the main results CSV."""
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    for cat in ["i", "v"]:
        sub = df[df["category"] == cat]
        if sub.empty:
            continue

        # Table A: Matching
        table_a = sub.groupby("method").agg({
            "n_features": "mean",
            "repeatability": "mean",
            "n_putatives": "mean",
            "PMR": "mean",
            "n_correct": "mean",
            "Precision": "mean",
            "Matching_Score": "mean",
            "n_correspondences": "mean",
            "Recall": "mean",
        }).round(4)
        table_a.to_csv(os.path.join(output_dir, f"Table_A_{cat}.csv"))

        # Table B: Cost
        cost_cols = ["detect_ms", "describe_ms", "match_ms", "total_ms",
                     "memory_bytes"]
        available = [c for c in cost_cols if c in sub.columns]
        if available:
            table_b = sub.groupby("method")[available].mean().round(2)
            table_b.to_csv(os.path.join(output_dir, f"Table_B_{cat}.csv"))

        # Table C: Homography
        ha_cols = ["mean_corner_err", "HA@1", "HA@3", "HA@5",
                   "n_inliers", "reproj_err_inliers"]
        available = [c for c in ha_cols if c in sub.columns]
        if available:
            table_c = sub.groupby("method")[available].mean().round(3)
            table_c.to_csv(os.path.join(output_dir, f"Table_C_{cat}.csv"))

    print(f"✓ Saved tables to {output_dir}")


# ------------------------------------------------------------------
# Main: generate all figures from results
# ------------------------------------------------------------------

def generate_all_figures():
    """Generate all D2 figures from existing CSV results."""
    csv_main = os.path.join(cfg.CSV_DIR, "main_results.csv")
    os.makedirs(cfg.FIGURES_DIR, exist_ok=True)

    if os.path.exists(csv_main):
        # Degradation curves
        plot_degradation(csv_main,
                         os.path.join(cfg.FIGURES_DIR, "degradation.png"))

        # Pareto
        plot_pareto(csv_main,
                    os.path.join(cfg.FIGURES_DIR, "pareto.png"))

        # Tables
        generate_tables(csv_main, os.path.join(cfg.RESULTS_DIR, "tables"))
    else:
        print(f"⚠ {csv_main} not found – run run_main.py first")


if __name__ == "__main__":
    generate_all_figures()
