"""
run_synthetic.py – B6: Synthetic rotation and scale test (Rublee et al.).

Creates synthetic transformations of reference images and evaluates
all methods under controlled rotation, scale, and noise.

Usage:
    cd src/
    python run_synthetic.py
"""

import os
import csv
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
import data_loader as dl
from feature_extraction import extract
from matching import match_pipeline, classify_matches, estimate_homography
from evaluation import compute_repeatability, compute_matching_metrics, compute_homography_accuracy


def make_rotation_transform(img, angle_deg):
    """Rotate image by angle around centre, enlarging canvas.

    Returns (rotated_image, H_3x3_ground_truth).
    """
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2

    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

    # Compute new bounding box
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(np.ceil(h * sin_a + w * cos_a))
    new_h = int(np.ceil(h * cos_a + w * sin_a))

    # Adjust translation
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    rotated = cv2.warpAffine(img, M, (new_w, new_h),
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Build 3×3 homography
    H = np.eye(3)
    H[:2, :] = M
    return rotated, H


def make_scale_transform(img, scale):
    """Scale image by factor s. Returns (scaled_image, H_3x3)."""
    h, w = img.shape[:2]
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    H = np.diag([scale, scale, 1.0])
    return scaled, H


def add_noise(img, sigma=10):
    """Add Gaussian noise with given sigma."""
    noise = np.random.randn(*img.shape) * sigma
    noisy = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
    return noisy


def run_synthetic_test():
    """Run rotation and scale tests on reference images."""
    all_seqs = dl.get_sequence_dirs()
    test_seqs, _ = dl.split_tuning(all_seqs)

    # Use up to 30 sequences
    np.random.seed(cfg.RANDOM_SEED)
    if len(test_seqs) > 30:
        indices = np.random.choice(len(test_seqs), 30, replace=False)
        test_seqs = [test_seqs[i] for i in sorted(indices)]

    methods = list(cfg.METHOD_CONFIGS.keys())
    results = []

    print(f"Using {len(test_seqs)} sequences, {len(methods)} methods")

    for seq_name, seq_dir in test_seqs:
        # Load reference image (grayscale, 640 px)
        try:
            img1, _ = dl.load_image_gray(
                os.path.join(seq_dir, "1.ppm"),
                max_side=cfg.SYNTH_IMAGE_SIZE)
        except FileNotFoundError:
            continue

        # --- Rotation test ---
        for angle in cfg.SYNTH_ROTATIONS:
            img2, H_gt = make_rotation_transform(img1, angle)

            for method in methods:
                for noise in [False, True]:
                    test_img = add_noise(img2, cfg.SYNTH_NOISE_SIGMA) if noise else img2

                    try:
                        kps1, descs1 = extract(method, img1)
                        kps2, descs2 = extract(method, test_img)

                        if descs1 is None or descs2 is None:
                            continue

                        mcfg = cfg.METHOD_CONFIGS[method]
                        matches = match_pipeline(descs1, descs2, mcfg["norm"])
                        correct, incorrect, _ = classify_matches(
                            matches, kps1, kps2, H_gt)

                        rep, n_rep, _, _, n_feat = compute_repeatability(
                            kps1, kps2, H_gt, img1.shape, test_img.shape)

                        n_corr = n_rep
                        mm = compute_matching_metrics(
                            len(matches), len(correct), n_feat, n_corr)

                        H_est, mask = estimate_homography(matches, kps1, kps2)
                        ha = compute_homography_accuracy(H_est, H_gt, img1.shape)

                        results.append({
                            "method": method,
                            "sequence": seq_name,
                            "transform": "rotation",
                            "param": angle,
                            "noise": int(noise),
                            "repeatability": rep,
                            **mm,
                            **ha,
                        })
                    except Exception as e:
                        print(f"  ERR {method} rot={angle} noise={noise}: {e}")

        # --- Scale test ---
        for scale in cfg.SYNTH_SCALES:
            img2, H_gt = make_scale_transform(img1, scale)

            for method in methods:
                try:
                    kps1, descs1 = extract(method, img1)
                    kps2, descs2 = extract(method, img2)

                    if descs1 is None or descs2 is None:
                        continue

                    mcfg = cfg.METHOD_CONFIGS[method]
                    matches = match_pipeline(descs1, descs2, mcfg["norm"])
                    correct, incorrect, _ = classify_matches(
                        matches, kps1, kps2, H_gt)

                    rep, n_rep, _, _, n_feat = compute_repeatability(
                        kps1, kps2, H_gt, img1.shape, img2.shape)

                    n_corr = n_rep
                    mm = compute_matching_metrics(
                        len(matches), len(correct), n_feat, n_corr)

                    H_est, mask = estimate_homography(matches, kps1, kps2)
                    ha = compute_homography_accuracy(H_est, H_gt, img1.shape)

                    results.append({
                        "method": method,
                        "sequence": seq_name,
                        "transform": "scale",
                        "param": scale,
                        "noise": 0,
                        "repeatability": rep,
                        **mm,
                        **ha,
                    })
                except Exception as e:
                    print(f"  ERR {method} scale={scale}: {e}")

        print(f"  ✓ {seq_name}")

    return results


def save_and_plot(results):
    os.makedirs(cfg.CSV_DIR, exist_ok=True)
    os.makedirs(cfg.FIGURES_DIR, exist_ok=True)

    # Save CSV
    csv_path = os.path.join(cfg.CSV_DIR, "synthetic_results.csv")
    if results:
        fields = sorted(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(results)
        print(f"✓ Saved {len(results)} rows to {csv_path}")

    # --- Rotation plot ---
    from collections import defaultdict
    rot_data = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r["transform"] == "rotation" and r["noise"] == 0:
            rot_data[r["method"]][r["param"]].append(r["Matching_Score"])

    fig, ax = plt.subplots(figsize=(10, 6))
    for method in sorted(rot_data.keys()):
        angles = sorted(rot_data[method].keys())
        ms_vals = [np.mean(rot_data[method][a]) for a in angles]
        ax.plot(angles, ms_vals, marker="o", markersize=3, label=method)
    ax.set_xlabel("Rotation angle (°)")
    ax.set_ylabel("Mean Matching Score")
    ax.set_title("B6: Matching Score vs Rotation")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.FIGURES_DIR, "B6_rotation.png"), dpi=150)
    plt.close(fig)

    # --- Scale plot ---
    scale_data = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r["transform"] == "scale":
            scale_data[r["method"]][r["param"]].append(r["Matching_Score"])

    fig, ax = plt.subplots(figsize=(10, 6))
    for method in sorted(scale_data.keys()):
        scales = sorted(scale_data[method].keys())
        ms_vals = [np.mean(scale_data[method][s]) for s in scales]
        ax.plot(scales, ms_vals, marker="o", markersize=3, label=method)
    ax.set_xlabel("Scale factor")
    ax.set_ylabel("Mean Matching Score")
    ax.set_title("B6: Matching Score vs Scale")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.FIGURES_DIR, "B6_scale.png"), dpi=150)
    plt.close(fig)

    print("✓ Saved B6 plots")


if __name__ == "__main__":
    np.random.seed(cfg.RANDOM_SEED)
    results = run_synthetic_test()
    save_and_plot(results)
