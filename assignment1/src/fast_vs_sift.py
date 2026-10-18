from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import detectors
import evaluation


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

FAST_THRESHOLDS = [5, 10, 15, 20, 30, 40, 60]
FAST_NONMAX_VALUES = [True, False]

MAX_FEATURES = 2000
REPEATABILITY_THRESHOLD_PX = 3.0

OUTPUT_DIR = Path("results/fast_parameter_sweep")


# ---------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------

def timed_detection(detector_fn, image):
    """Run detection and return keypoints and elapsed milliseconds."""
    start = time.perf_counter()
    keypoints = detector_fn(image)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return keypoints, elapsed_ms


def sift_detector_limited(image):
    """SIFT baseline with the same maximum feature count as FAST."""
    detector = cv2.SIFT_create(nfeatures=MAX_FEATURES)
    return detector.detect(image, None)


def make_fast_detector(threshold, nonmax_suppression):
    """Create a FAST detector function for one parameter configuration."""

    def detect(image):
        return detectors.fast_detector(
            image,
            threshold=threshold,
            nonmax_suppression=nonmax_suppression,
            max_features=MAX_FEATURES,
        )

    return detect


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate_pair(image1,image2,homography,detector_fn,method_name,pair_id):
    """Evaluate one detector on one image pair."""

    kps1, time1_ms = timed_detection(detector_fn, image1)
    kps2, time2_ms = timed_detection(detector_fn, image2)

    (repeatability,
        n_repeat,
        n1_visible,
        n2_visible,
        n_features,
    ) = evaluation.compute_repeatability(
        kps1,
        kps2,
        homography,
        image1.shape,
        image2.shape,
        threshold_px=REPEATABILITY_THRESHOLD_PX,
    )

    return {
        "pair_id": pair_id,
        "method": method_name,
        "n_keypoints_image1": len(kps1),
        "n_keypoints_image2": len(kps2),
        "n_covisible_image1": n1_visible,
        "n_covisible_image2": n2_visible,
        "n_features": n_features,
        "n_repeat": n_repeat,
        "repeatability": repeatability,
        "detection_time_image1_ms": time1_ms,
        "detection_time_image2_ms": time2_ms,
        "mean_detection_time_ms": (time1_ms + time2_ms) / 2.0,
    }


def run_fast_parameter_sweep(image_pairs, output_dir=OUTPUT_DIR):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configurations = []

    # SIFT baseline.
    configurations.append({
        "method": "SIFT",
        "detector_fn": sift_detector_limited,
    })

    # FAST parameter sweep.
    for threshold in FAST_THRESHOLDS:
        for nonmax in FAST_NONMAX_VALUES:
            configurations.append({
                "method": f"FAST_t{threshold}_nms{nonmax}",
                "detector_fn": make_fast_detector(threshold, nonmax),
                "threshold": threshold,
                "nonmax_suppression": nonmax,
            })

    records = []

    for pair_id, image1, image2, homography in image_pairs:
        if image1 is None or image2 is None:
            raise ValueError(f"Could not load images for pair {pair_id}")

        if homography is None:
            raise ValueError(f"Missing ground-truth homography for pair {pair_id}")

        for config in configurations:
            record = evaluate_pair(
                image1,
                image2,
                homography,
                config["detector_fn"],
                config["method"],
                pair_id,
            )

            record["threshold"] = config.get("threshold")
            record["nonmax_suppression"] = config.get(
                "nonmax_suppression"
            )

            records.append(record)

    results = pd.DataFrame(records)

    # Save per-pair results.
    csv_path = output_dir / "fast_parameter_sweep.csv"
    results.to_csv(csv_path, index=False)

    # Aggregate results across image pairs.
    summary = (
        results.groupby(
            ["method", "threshold", "nonmax_suppression"],
            dropna=False,
        )
        .agg(
            mean_keypoints_image1=("n_keypoints_image1", "mean"),
            mean_keypoints_image2=("n_keypoints_image2", "mean"),
            mean_covisible_image1=("n_covisible_image1", "mean"),
            mean_covisible_image2=("n_covisible_image2", "mean"),
            mean_repeatability=("repeatability", "mean"),
            mean_detection_time_ms=("mean_detection_time_ms", "mean"),
        )
        .reset_index()
    )

    summary_path = output_dir / "fast_parameter_sweep_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Plot 1: Repeatability vs number of detected keypoints (clean trade-off curves)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    fast_nms_true = summary[summary["nonmax_suppression"] == True].sort_values("threshold")
    fast_nms_false = summary[summary["nonmax_suppression"] == False].sort_values("threshold")
    sift_row = summary[summary["method"] == "SIFT"]

    if not fast_nms_true.empty:
        kps_true = (fast_nms_true["mean_keypoints_image1"] + fast_nms_true["mean_keypoints_image2"]) / 2.0
        ax.plot(
            kps_true,
            fast_nms_true["mean_repeatability"],
            color="#1f77b4",
            marker="o",
            markersize=7,
            linewidth=2.2,
            label="FAST (NMS = True)",
        )
        for x, y, t in zip(kps_true, fast_nms_true["mean_repeatability"], fast_nms_true["threshold"]):
            ax.annotate(
                f"t={int(t)}",
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color="#1f77b4",
                fontweight="bold",
            )

    if not fast_nms_false.empty:
        kps_false = (fast_nms_false["mean_keypoints_image1"] + fast_nms_false["mean_keypoints_image2"]) / 2.0
        ax.plot(
            kps_false,
            fast_nms_false["mean_repeatability"],
            color="#ff7f0e",
            linestyle="--",
            marker="s",
            markersize=7,
            linewidth=2.2,
            label="FAST (NMS = False)",
        )
        for x, y, t in zip(kps_false, fast_nms_false["mean_repeatability"], fast_nms_false["threshold"]):
            ax.annotate(
                f"t={int(t)}",
                (x, y),
                xytext=(5, -12),
                textcoords="offset points",
                fontsize=8,
                color="#d95f02",
                fontweight="bold",
            )

    if not sift_row.empty:
        sift_kps = (sift_row["mean_keypoints_image1"].iloc[0] + sift_row["mean_keypoints_image2"].iloc[0]) / 2.0
        sift_rep = sift_row["mean_repeatability"].iloc[0]
        ax.scatter(
            [sift_kps],
            [sift_rep],
            color="#d62728",
            marker="*",
            s=220,
            zorder=6,
            label=f"SIFT Baseline (rep={sift_rep:.3f})",
        )
        ax.axhline(sift_rep, color="#d62728", linestyle=":", linewidth=1.5, alpha=0.6)

    ax.set_xlabel("Mean Detected Keypoints per Image", fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean Repeatability (3 px)", fontsize=10, fontweight="bold")
    ax.set_title("FAST Parameter Study: Repeatability vs Keypoint Count", fontsize=11, fontweight="bold", pad=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)
    fig.tight_layout()

    fig.savefig(
        output_dir / "repeatability_vs_keypoints.png",
        dpi=200,
    )
    plt.close(fig)

    # Plot 2: Grouped bar chart comparing NMS On vs NMS Off per threshold
    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    thresholds = sorted([int(t) for t in FAST_THRESHOLDS])
    x_indices = np.arange(len(thresholds))
    width = 0.35

    nms_true_times = [
        fast_nms_true.loc[fast_nms_true["threshold"] == t, "mean_detection_time_ms"].values[0]
        if not fast_nms_true.loc[fast_nms_true["threshold"] == t].empty else 0.0
        for t in thresholds
    ]
    nms_false_times = [
        fast_nms_false.loc[fast_nms_false["threshold"] == t, "mean_detection_time_ms"].values[0]
        if not fast_nms_false.loc[fast_nms_false["threshold"] == t].empty else 0.0
        for t in thresholds
    ]

    ax.bar(
        x_indices - width / 2,
        nms_true_times,
        width,
        label="FAST (NMS On)",
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.8,
    )
    ax.bar(
        x_indices + width / 2,
        nms_false_times,
        width,
        label="FAST (NMS Off)",
        color="#ffbb78",
        edgecolor="black",
        linewidth=0.8,
    )

    if not sift_row.empty:
        sift_time = sift_row["mean_detection_time_ms"].iloc[0]
        ax.axhline(
            sift_time,
            color="#d62728",
            linestyle="--",
            linewidth=1.8,
            label=f"SIFT Baseline ({sift_time:.2f} ms)",
        )

    ax.set_xticks(x_indices)
    ax.set_xticklabels([f"t={t}" for t in thresholds], fontsize=10)
    ax.set_xlabel("FAST Intensity Threshold", fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean Detection Time per Image (ms)", fontsize=10, fontweight="bold")
    ax.set_title("FAST Detection Runtime vs Intensity Threshold & NMS", fontsize=11, fontweight="bold", pad=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)
    fig.tight_layout()

    fig.savefig(
        output_dir / "detection_runtime.png",
        dpi=200,
    )
    plt.close(fig)

    # Format markdown summary table for report
    md_summary = [
        "| Configuration | Threshold | NMS | Mean Keypoints | Repeatability | Runtime (ms) |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in summary.sort_values(["threshold", "nonmax_suppression"]).iterrows():
        t = row["threshold"]
        nms = "True" if row["nonmax_suppression"] is True else "False" if row["nonmax_suppression"] is False else "N/A"
        kps = (row["mean_keypoints_image1"] + row["mean_keypoints_image2"]) / 2.0
        rep = row["mean_repeatability"]
        runtime = row["mean_detection_time_ms"]
        method = row["method"]
        label = "SIFT" if method == "SIFT" else f"FAST"
        t_str = str(int(t)) if pd.notna(t) else "N/A"
        md_summary.append(f"| {label} | {t_str} | {nms} | {kps:.1f} | {rep:.3f} | {runtime:.2f} |")

    (output_dir / "fast_parameter_sweep_summary.md").write_text("\n".join(md_summary))

    print(f"Per-pair results: {csv_path}")
    print(f"Summary results:  {summary_path}")
    print(f"Plots saved in:   {output_dir.resolve()}")

    return results, summary