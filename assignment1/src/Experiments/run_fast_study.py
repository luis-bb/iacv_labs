"""
run_fast_study.py – B5: FAST parameter study (Rosten & Drummond protocol).

Sweep: threshold × type × NMS on 42 settings.
Baselines: SIFT DoG and Harris.
Plots: repeatability vs #keypoints, detection time vs #keypoints.

Usage:
    cd src/
    python run_fast_study.py
"""

import os
import csv
import time
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
import data_loader as dl
from evaluation import compute_repeatability
from benchmark_timer import time_function


def run_fast_sweep():
    """Run FAST with all parameter combinations on all non-tuning pairs."""
    all_seqs = dl.get_sequence_dirs()
    test_seqs, _ = dl.split_tuning(all_seqs)

    if not test_seqs:
        print("No sequences found")
        return []

    results = []

    # FAST sweep
    for thresh in cfg.FAST_THRESHOLDS:
        for ftype in cfg.FAST_TYPES:
            type_name = {cv2.FastFeatureDetector_TYPE_9_16: "9_16",
                         cv2.FastFeatureDetector_TYPE_7_12: "7_12",
                         cv2.FastFeatureDetector_TYPE_5_8: "5_8"}[ftype]
            for nms in cfg.FAST_NMS_OPTIONS:
                label = f"FAST_{type_name}_t{thresh}_nms{int(nms)}"
                print(f"  {label}")

                for seq_name, seq_dir in test_seqs:
                    cat = dl.get_category(seq_name)
                    for k in cfg.TARGET_IMAGE_INDICES:
                        try:
                            img1, img2, H_gt = dl.load_pair(seq_dir, k)
                        except FileNotFoundError:
                            continue

                        fast = cv2.FastFeatureDetector_create(
                            threshold=thresh,
                            nonmaxSuppression=nms,
                            type=ftype)

                        # Time detection
                        def detect(img):
                            return fast.detect(img, None)

                        kps1, t1 = time_function(detect, img1, n_repeats=3)
                        kps2, t2 = time_function(detect, img2, n_repeats=3)
                        t_detect = (t1 + t2) / 2.0

                        # Megapixels
                        mpx = (img1.shape[0] * img1.shape[1]) / 1e6

                        # Repeatability (detection only)
                        rep, n_rep, n1v, n2v, n_feat = compute_repeatability(
                            kps1, kps2, H_gt, img1.shape, img2.shape)

                        results.append({
                            "detector": label,
                            "type": type_name,
                            "threshold": thresh,
                            "nms": int(nms),
                            "sequence": seq_name,
                            "category": cat,
                            "k": k,
                            "n_kps1": len(kps1),
                            "n_kps2": len(kps2),
                            "n_covis": n_feat,
                            "repeatability": rep,
                            "detect_ms_per_mpx": (t_detect * 1000) / mpx,
                        })

    # Harris baseline
    print("  Harris baselines...")
    for max_corners in [250, 500, 1000, 2000, 4000]:
        label = f"Harris_mc{max_corners}"
        for seq_name, seq_dir in test_seqs:
            cat = dl.get_category(seq_name)
            for k in cfg.TARGET_IMAGE_INDICES:
                try:
                    img1, img2, H_gt = dl.load_pair(seq_dir, k)
                except FileNotFoundError:
                    continue

                def detect_harris(img, mc=max_corners):
                    pts = cv2.goodFeaturesToTrack(
                        img, maxCorners=mc, qualityLevel=0.01,
                        minDistance=3, useHarrisDetector=True, k=0.04)
                    if pts is None:
                        return []
                    return [cv2.KeyPoint(float(p[0][0]), float(p[0][1]), 5)
                            for p in pts]

                kps1, t1 = time_function(detect_harris, img1, n_repeats=3)
                kps2, t2 = time_function(detect_harris, img2, n_repeats=3)

                mpx = (img1.shape[0] * img1.shape[1]) / 1e6
                rep, n_rep, n1v, n2v, n_feat = compute_repeatability(
                    kps1, kps2, H_gt, img1.shape, img2.shape)

                results.append({
                    "detector": label,
                    "type": "Harris",
                    "threshold": 0,
                    "nms": 1,
                    "sequence": seq_name,
                    "category": cat,
                    "k": k,
                    "n_kps1": len(kps1),
                    "n_kps2": len(kps2),
                    "n_covis": n_feat,
                    "repeatability": rep,
                    "detect_ms_per_mpx": ((t1 + t2) / 2 * 1000) / mpx,
                })

    # SIFT DoG baseline
    print("  SIFT DoG baselines...")
    for ct in [0.01, 0.02, 0.04, 0.08, 0.12]:
        label = f"DoG_ct{ct}"
        for seq_name, seq_dir in test_seqs:
            cat = dl.get_category(seq_name)
            for k in cfg.TARGET_IMAGE_INDICES:
                try:
                    img1, img2, H_gt = dl.load_pair(seq_dir, k)
                except FileNotFoundError:
                    continue

                def detect_dog(img, c=ct):
                    sift = cv2.xfeatures2d.SIFT_create(
                        nfeatures=0, contrastThreshold=c,
                        edgeThreshold=10, sigma=1.6)
                    return sift.detect(img, None)

                kps1, t1 = time_function(detect_dog, img1, n_repeats=3)
                kps2, t2 = time_function(detect_dog, img2, n_repeats=3)

                mpx = (img1.shape[0] * img1.shape[1]) / 1e6
                rep, n_rep, n1v, n2v, n_feat = compute_repeatability(
                    kps1, kps2, H_gt, img1.shape, img2.shape)

                results.append({
                    "detector": label,
                    "type": "DoG",
                    "threshold": ct,
                    "nms": 1,
                    "sequence": seq_name,
                    "category": cat,
                    "k": k,
                    "n_kps1": len(kps1),
                    "n_kps2": len(kps2),
                    "n_covis": n_feat,
                    "repeatability": rep,
                    "detect_ms_per_mpx": ((t1 + t2) / 2 * 1000) / mpx,
                })

    return results


def save_and_plot(results):
    os.makedirs(cfg.CSV_DIR, exist_ok=True)
    os.makedirs(cfg.FIGURES_DIR, exist_ok=True)

    # Save CSV
    csv_path = os.path.join(cfg.CSV_DIR, "fast_study.csv")
    if results:
        fields = sorted(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(results)
        print(f"✓ Saved {len(results)} rows to {csv_path}")

    # Aggregate for plotting: mean over sequences per detector setting
    from collections import defaultdict
    agg = defaultdict(lambda: {"kps": [], "rep": [], "time": []})
    for r in results:
        key = r["detector"]
        agg[key]["kps"].append(r["n_covis"])
        agg[key]["rep"].append(r["repeatability"])
        agg[key]["time"].append(r["detect_ms_per_mpx"])

    # Plot: repeatability vs #keypoints
    fig, ax = plt.subplots(figsize=(10, 6))
    for det, vals in sorted(agg.items()):
        mean_kps = np.mean(vals["kps"])
        mean_rep = np.mean(vals["rep"])
        color = "blue" if "FAST" in det else ("red" if "Harris" in det else "green")
        ax.scatter(mean_kps, mean_rep, label=det, alpha=0.7, c=color, s=30)
    ax.set_xscale("log")
    ax.set_xlabel("Mean #co-visible keypoints")
    ax.set_ylabel("Mean repeatability")
    ax.set_title("B5: Repeatability vs #Keypoints")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.FIGURES_DIR, "B5_rep_vs_kps.png"), dpi=150)
    plt.close(fig)
    print(f"✓ Saved B5 plot")


if __name__ == "__main__":
    np.random.seed(cfg.RANDOM_SEED)
    results = run_fast_sweep()
    save_and_plot(results)
