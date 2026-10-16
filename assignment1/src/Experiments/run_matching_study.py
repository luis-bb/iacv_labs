"""
run_matching_study.py – B7: Matching strategy study (Lowe 2004).

1. Ratio sweep (precision–recall)
2. Lowe's claim verification
3. Strategy comparison: NN only, cross-check, ratio, ratio + one-to-one
4. Brute-force vs FLANN (approximate)
5. Wrong-norm check (ORB with L2)

Usage:
    cd src/
    python run_matching_study.py
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
from matching import (match_knn_ratio, enforce_one_to_one, match_nn_only,
                      match_cross_check, match_pipeline, classify_matches,
                      match_flann_sift, match_flann_binary)
from evaluation import compute_repeatability, compute_matching_metrics
from benchmark_timer import time_function


def ratio_sweep():
    """B7.1 – Sweep ratio from 0.5 to 1.0 for selected methods."""
    print("B7.1  Ratio sweep")
    all_seqs = dl.get_sequence_dirs()
    test_seqs, _ = dl.split_tuning(all_seqs)

    methods = ["SIFT", "ORB", "KAZE", "FAST+BRISK"]
    results = []

    for method in methods:
        mcfg = cfg.METHOD_CONFIGS[method]
        norm = mcfg["norm"]
        print(f"  {method}")

        for seq_name, seq_dir in test_seqs:
            cat = dl.get_category(seq_name)
            for k in cfg.TARGET_IMAGE_INDICES:
                try:
                    img1, img2, H_gt = dl.load_pair(seq_dir, k)
                except FileNotFoundError:
                    continue

                kps1, descs1 = extract(method, img1)
                kps2, descs2 = extract(method, img2)

                if descs1 is None or descs2 is None:
                    continue

                rep, n_rep, _, _, n_feat = compute_repeatability(
                    kps1, kps2, H_gt, img1.shape, img2.shape)

                for ratio in cfg.RATIO_SWEEP:
                    matches = match_knn_ratio(descs1, descs2, norm, ratio=ratio)
                    matches_1to1 = enforce_one_to_one(matches)
                    correct, incorrect, _ = classify_matches(
                        matches_1to1, kps1, kps2, H_gt)

                    n_put = len(matches_1to1)
                    n_corr = len(correct)
                    prec = n_corr / n_put if n_put > 0 else 0.0
                    recall = n_corr / n_rep if n_rep > 0 else 0.0

                    results.append({
                        "method": method,
                        "ratio": ratio,
                        "sequence": seq_name,
                        "category": cat,
                        "k": k,
                        "n_putatives": n_put,
                        "n_correct": n_corr,
                        "n_correspondences": n_rep,
                        "precision": prec,
                        "recall": recall,
                    })

    return results


def lowe_claim(results):
    """B7.2 – Check Lowe's claim: r=0.8 removes ~90% false, loses <5% correct."""
    print("\nB7.2  Lowe's claim (SIFT)")
    sift = [r for r in results if r["method"] == "SIFT"]

    # NN-only = ratio 1.0
    nn_data = {(r["sequence"], r["k"]): r for r in sift if r["ratio"] == 1.0}
    r08_data = {(r["sequence"], r["k"]): r for r in sift if r["ratio"] == 0.8}

    total_fp_nn, total_tp_nn = 0, 0
    total_fp_08, total_tp_08 = 0, 0

    for key in nn_data:
        if key not in r08_data:
            continue
        nn = nn_data[key]
        r08 = r08_data[key]

        fp_nn = nn["n_putatives"] - nn["n_correct"]
        tp_nn = nn["n_correct"]
        fp_08 = r08["n_putatives"] - r08["n_correct"]
        tp_08 = r08["n_correct"]

        total_fp_nn += fp_nn
        total_tp_nn += tp_nn
        total_fp_08 += fp_08
        total_tp_08 += tp_08

    if total_fp_nn > 0:
        fp_removed = 1 - total_fp_08 / total_fp_nn
        print(f"  False matches removed: {fp_removed * 100:.1f}% (paper: ~90%)")
    if total_tp_nn > 0:
        tp_lost = 1 - total_tp_08 / total_tp_nn
        print(f"  Correct matches lost:  {tp_lost * 100:.1f}% (paper: <5%)")


def strategy_comparison():
    """B7.3 – Compare NN only, cross-check, ratio, ratio+one-to-one."""
    print("\nB7.3  Strategy comparison (SIFT, ORB on v)")
    all_seqs = dl.get_sequence_dirs()
    test_seqs, _ = dl.split_tuning(all_seqs)
    v_seqs = [(n, p) for n, p in test_seqs if n.startswith("v_")]

    results = []
    for method in ["SIFT", "ORB"]:
        mcfg = cfg.METHOD_CONFIGS[method]
        norm = mcfg["norm"]

        strategies = {
            "NN_only": lambda d1, d2: match_nn_only(d1, d2, norm),
            "Cross_check": lambda d1, d2: match_cross_check(d1, d2, norm),
            "Ratio_only": lambda d1, d2: match_knn_ratio(d1, d2, norm, 0.8),
            "Ratio+1to1": lambda d1, d2: match_pipeline(d1, d2, norm, 0.8),
        }

        for seq_name, seq_dir in v_seqs:
            for k in cfg.TARGET_IMAGE_INDICES:
                try:
                    img1, img2, H_gt = dl.load_pair(seq_dir, k)
                except FileNotFoundError:
                    continue

                kps1, descs1 = extract(method, img1)
                kps2, descs2 = extract(method, img2)

                if descs1 is None or descs2 is None:
                    continue

                rep, n_rep, _, _, n_feat = compute_repeatability(
                    kps1, kps2, H_gt, img1.shape, img2.shape)

                for strat_name, strat_fn in strategies.items():
                    matches = strat_fn(descs1, descs2)
                    correct, _, _ = classify_matches(matches, kps1, kps2, H_gt)

                    n_put = len(matches)
                    n_corr = len(correct)

                    results.append({
                        "method": method,
                        "strategy": strat_name,
                        "sequence": seq_name,
                        "k": k,
                        "n_putatives": n_put,
                        "n_correct": n_corr,
                        "n_features": n_feat,
                        "n_correspondences": n_rep,
                        "PMR": n_put / n_feat if n_feat > 0 else 0,
                        "precision": n_corr / n_put if n_put > 0 else 0,
                        "recall": n_corr / n_rep if n_rep > 0 else 0,
                    })

    return results


def bf_vs_flann():
    """B7.4 – Brute-force vs FLANN (approximate)."""
    print("\nB7.4  BF vs FLANN")
    all_seqs = dl.get_sequence_dirs()
    test_seqs, _ = dl.split_tuning(all_seqs)

    results = []

    # Pick a few pairs for speed comparison
    pairs_to_test = []
    for seq_name, seq_dir in test_seqs[:5]:
        for k in [2, 5]:
            pairs_to_test.append((seq_name, seq_dir, k))

    for method, flann_fn in [("SIFT", match_flann_sift),
                              ("ORB", match_flann_binary)]:
        mcfg = cfg.METHOD_CONFIGS[method]
        norm = mcfg["norm"]

        for seq_name, seq_dir, k in pairs_to_test:
            try:
                img1, img2, H_gt = dl.load_pair(seq_dir, k)
            except FileNotFoundError:
                continue

            kps1, descs1 = extract(method, img1)
            kps2, descs2 = extract(method, img2)
            if descs1 is None or descs2 is None:
                continue

            # BF
            def run_bf():
                return match_pipeline(descs1, descs2, norm)
            bf_matches, t_bf = time_function(run_bf, n_repeats=5)
            bf_correct, _, _ = classify_matches(bf_matches, kps1, kps2, H_gt)

            # FLANN
            def run_flann():
                return flann_fn(descs1, descs2)
            fl_matches, t_fl = time_function(run_flann, n_repeats=5)
            fl_correct, _, _ = classify_matches(fl_matches, kps1, kps2, H_gt)

            results.append({
                "method": method,
                "sequence": seq_name,
                "k": k,
                "bf_matches": len(bf_matches),
                "bf_correct": len(bf_correct),
                "bf_time_ms": t_bf * 1000,
                "flann_matches": len(fl_matches),
                "flann_correct": len(fl_correct),
                "flann_time_ms": t_fl * 1000,
                "speedup": t_bf / t_fl if t_fl > 0 else 0,
            })

    return results


def wrong_norm_check():
    """B7.5 – ORB with NORM_L2 instead of Hamming."""
    print("\nB7.5  Wrong norm check (ORB + L2)")
    all_seqs = dl.get_sequence_dirs()
    test_seqs, _ = dl.split_tuning(all_seqs)

    results_hamming = []
    results_l2 = []

    for seq_name, seq_dir in test_seqs[:10]:
        for k in [2, 4]:
            try:
                img1, img2, H_gt = dl.load_pair(seq_dir, k)
            except FileNotFoundError:
                continue

            kps1, descs1 = extract("ORB", img1)
            kps2, descs2 = extract("ORB", img2)
            if descs1 is None or descs2 is None:
                continue

            # Correct norm
            m_ham = match_pipeline(descs1, descs2, cv2.NORM_HAMMING)
            c_ham, _, _ = classify_matches(m_ham, kps1, kps2, H_gt)

            # Wrong norm
            m_l2 = match_pipeline(descs1, descs2, cv2.NORM_L2)
            c_l2, _, _ = classify_matches(m_l2, kps1, kps2, H_gt)

            results_hamming.append(len(c_ham) / len(m_ham) if m_ham else 0)
            results_l2.append(len(c_l2) / len(m_l2) if m_l2 else 0)

    if results_hamming:
        print(f"  Hamming precision: {np.mean(results_hamming):.3f}")
        print(f"  L2 precision:      {np.mean(results_l2):.3f}")


def save_all(ratio_results, strategy_results, flann_results):
    os.makedirs(cfg.CSV_DIR, exist_ok=True)
    os.makedirs(cfg.FIGURES_DIR, exist_ok=True)

    for name, data in [("ratio_sweep", ratio_results),
                        ("strategy_comparison", strategy_results),
                        ("bf_vs_flann", flann_results)]:
        if data:
            path = os.path.join(cfg.CSV_DIR, f"B7_{name}.csv")
            fields = sorted(data[0].keys())
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(data)

    # Precision–Recall plot from ratio sweep
    if ratio_results:
        from collections import defaultdict
        agg = defaultdict(lambda: defaultdict(lambda: {"prec": [], "rec": []}))
        for r in ratio_results:
            agg[r["method"]][r["ratio"]]["prec"].append(r["precision"])
            agg[r["method"]][r["ratio"]]["rec"].append(r["recall"])

        fig, ax = plt.subplots(figsize=(8, 6))
        for method in sorted(agg.keys()):
            ratios = sorted(agg[method].keys())
            prec = [np.mean(agg[method][r]["prec"]) for r in ratios]
            rec = [np.mean(agg[method][r]["rec"]) for r in ratios]
            ax.plot(rec, prec, marker="o", label=method)
            for i, r in enumerate(ratios):
                ax.annotate(f"{r}", (rec[i], prec[i]), fontsize=6)

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("B7: Precision vs Recall (ratio sweep)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(cfg.FIGURES_DIR, "B7_prec_recall.png"), dpi=150)
        plt.close(fig)
        print("✓ Saved B7 plots")


if __name__ == "__main__":
    np.random.seed(cfg.RANDOM_SEED)

    ratio_results = ratio_sweep()
    lowe_claim(ratio_results)
    strategy_results = strategy_comparison()
    flann_results = bf_vs_flann()
    wrong_norm_check()
    save_all(ratio_results, strategy_results, flann_results)
