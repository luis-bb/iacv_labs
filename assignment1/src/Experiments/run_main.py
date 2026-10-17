"""
run_main.py – Main benchmark runner (Phase B2).

Loops over all non-tuning HPatches pairs × all methods.
Produces a CSV with per-pair results ready for tables and figures.

Usage:
    cd src/
    python run_main.py
"""

import os
import sys
import csv
import time
import numpy as np

import config as cfg
import data_loader as dl
from feature_extraction import extract
from matching import match_pipeline, classify_matches, estimate_homography, inlier_stats
from evaluation import evaluate_pair
from benchmark_timer import time_pipeline


def run_identity_test():
    """B1.5 – Identity test: pair (1,1) with H = I.
    Repeatability must be 1.0 and precision ≈ 1.0 for every config.
    """
    print("\n" + "=" * 60)
    print("B1.5  IDENTITY TEST  (pair 1,1, H = I)")
    print("=" * 60)

    seqs = dl.get_sequence_dirs()
    if not seqs:
        print("⚠ No sequences found – skipping identity test")
        return True

    # Use first available sequence
    seq_name, seq_dir = seqs[0]
    img, _ = dl.load_image_gray(os.path.join(seq_dir, "1.ppm"))
    H_id = np.eye(3)

    all_pass = True
    for method in cfg.METHOD_CONFIGS:
        try:
            kps, descs = extract(method, img)
            if descs is None or len(kps) == 0:
                print(f"  {method:25s}  SKIP (no keypoints)")
                continue

            mcfg = cfg.METHOD_CONFIGS[method]
            matches = match_pipeline(descs, descs, mcfg["norm"])
            correct, incorrect, _ = classify_matches(matches, kps, kps, H_id)

            from evaluation import compute_repeatability
            rep, _, _, _, _ = compute_repeatability(kps, kps, H_id,
                                                     img.shape, img.shape)

            prec = len(correct) / len(matches) if matches else 0.0
            ok = rep > 0.99 and prec > 0.95
            status = "✓" if ok else "✗"
            if not ok:
                all_pass = False
            print(f"  {method:25s}  rep={rep:.3f}  prec={prec:.3f}  {status}")
        except Exception as e:
            print(f"  {method:25s}  ERROR: {e}")
            all_pass = False

    return all_pass


def run_main_benchmark(do_timing=True, methods=None):
    """B2 – Main run on HPatches.

    Returns the list of result dicts (also saved to CSV).
    """
    print("\n" + "=" * 60)
    print("B2  MAIN BENCHMARK")
    print("=" * 60)

    # Create output dirs
    os.makedirs(cfg.CSV_DIR, exist_ok=True)

    # Load sequences
    all_seqs = dl.get_sequence_dirs()
    test_seqs, tuning_seqs = dl.split_tuning(all_seqs)

    if not test_seqs:
        print("⚠ No test sequences found. Check HPATCHES_DIR in config.py")
        print(f"  Looking in: {os.path.abspath(cfg.HPATCHES_DIR)}")
        return []

    print(f"  Test sequences : {len(test_seqs)}")
    print(f"  Tuning excluded: {len(tuning_seqs)}")

    if methods is None:
        methods = list(cfg.METHOD_CONFIGS.keys())

    all_results = []
    csv_path = os.path.join(cfg.CSV_DIR, "main_results.csv")
    csv_fields = None

    total_pairs = len(test_seqs) * len(cfg.TARGET_IMAGE_INDICES)
    total_evals = total_pairs * len(methods)
    done = 0

    for method in methods:
        print(f"\n--- {method} ---")
        mcfg = cfg.METHOD_CONFIGS[method]
        norm = mcfg["norm"]

        for seq_name, seq_dir in test_seqs:
            cat = dl.get_category(seq_name)

            for k in cfg.TARGET_IMAGE_INDICES:
                done += 1
                try:
                    img1, img2, H_gt = dl.load_pair(seq_dir, k)
                except FileNotFoundError as e:
                    print(f"  SKIP {seq_name}/1→{k}: {e}")
                    continue

                # --- Extract features ---
                kps1, descs1 = extract(method, img1)
                kps2, descs2 = extract(method, img2)

                if descs1 is None or descs2 is None or \
                   len(kps1) == 0 or len(kps2) == 0:
                    print(f"  SKIP {seq_name}/1→{k}: no features")
                    continue

                # --- Match ---
                matches = match_pipeline(descs1, descs2, norm)

                # --- Classify correctness ---
                correct, incorrect, errors = classify_matches(
                    matches, kps1, kps2, H_gt)

                # --- Estimate homography ---
                H_est, mask = estimate_homography(matches, kps1, kps2)

                # --- Evaluate ---
                result = evaluate_pair(
                    kps1, descs1, kps2, descs2, H_gt,
                    img1.shape, img2.shape,
                    matches, correct, incorrect,
                    H_est, mask, method)

                # --- Timing (optional, expensive) ---
                if do_timing:
                    _, _, _, _, _, times = time_pipeline(
                        method, img1, img2, n_repeats=cfg.TIMING_REPEATS)
                    result.update(times)

                # --- Metadata ---
                result["method"] = method
                result["sequence"] = seq_name
                result["category"] = cat
                result["k"] = k

                all_results.append(result)

                if done % 50 == 0 or done == total_evals:
                    print(f"  [{done}/{total_evals}] {seq_name} k={k} "
                          f"MS={result['Matching_Score']:.3f} "
                          f"HA@3={result.get('HA@3', 0):.0f}")

    # --- Save CSV ---
    if all_results:
        csv_fields = sorted(all_results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n✓ Saved {len(all_results)} rows to {csv_path}")

    return all_results


def run_defaults_benchmark(methods=None):
    """Run a second pass with OpenCV default parameters (B1.3)."""
    print("\n" + "=" * 60)
    print("B1.3  DEFAULTS RUN")
    print("=" * 60)

    os.makedirs(cfg.CSV_DIR, exist_ok=True)
    all_seqs = dl.get_sequence_dirs()
    test_seqs, _ = dl.split_tuning(all_seqs)

    if methods is None:
        methods = ["SIFT", "KAZE", "ORB"]  # native pipelines only

    all_results = []

    for method in methods:
        print(f"\n--- {method} (defaults) ---")
        mcfg = cfg.METHOD_CONFIGS[method]
        norm = mcfg["norm"]

        for seq_name, seq_dir in test_seqs:
            cat = dl.get_category(seq_name)
            for k in cfg.TARGET_IMAGE_INDICES:
                try:
                    img1, img2, H_gt = dl.load_pair(seq_dir, k)
                except FileNotFoundError:
                    continue

                kps1, descs1 = extract(method, img1, use_defaults=True)
                kps2, descs2 = extract(method, img2, use_defaults=True)

                if descs1 is None or descs2 is None:
                    continue

                matches = match_pipeline(descs1, descs2, norm)
                correct, incorrect, _ = classify_matches(
                    matches, kps1, kps2, H_gt)
                H_est, mask = estimate_homography(matches, kps1, kps2)

                result = evaluate_pair(
                    kps1, descs1, kps2, descs2, H_gt,
                    img1.shape, img2.shape,
                    matches, correct, incorrect,
                    H_est, mask, method)

                result["method"] = method + "_defaults"
                result["sequence"] = seq_name
                result["category"] = cat
                result["k"] = k
                all_results.append(result)

    csv_path = os.path.join(cfg.CSV_DIR, "defaults_results.csv")
    if all_results:
        fields = sorted(all_results[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_results)
        print(f"\n✓ Saved {len(all_results)} rows to {csv_path}")

    return all_results


# ------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(cfg.RANDOM_SEED)

    # 1. Identity test (sanity check)
    ok = run_identity_test()
    if not ok:
        print("\n⚠ Identity test failed for some methods – review before continuing")

    # 2. Main benchmark
    results = run_main_benchmark(do_timing=True)

    # 3. Defaults run
    defaults = run_defaults_benchmark()

    print("\nDone. Results in:", os.path.abspath(cfg.CSV_DIR))
