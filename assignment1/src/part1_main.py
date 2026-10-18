from pathlib import Path
import logging
import random

import cv2
import numpy as np
import pandas as pd
import torch

import detectors
import descriptors
import matching
import learned_models
import evaluation
import visualization

import fast_vs_sift # Experiments of 


# ============================================================
# Configuration
# ============================================================

DATASET_ROOT = Path("hpatches-sequences-release")
RESULTS_DIR = Path("results")
N_SEQUENCES = 2
RANDOM_SELECTION = False
SEED = 45
PAIRS = [(1, i) for i in range(2, 7)]

E = {
    1: {
        "name": "SIFT",
        "type": "classical",
        "detector": detectors.sift_detector,
        "descriptor": descriptors.compute_sift_descriptors,
        "norm": cv2.NORM_L2,
        "desc_dim": 128,
        "bytes_per_desc": 512,
        "device": "CPU",
    },
    2: {
        "name": "ORB",
        "type": "classical",
        "detector": detectors.orb_detector,
        "descriptor": descriptors.compute_orb_descriptors,
        "norm": cv2.NORM_HAMMING,
        "desc_dim": 256,
        "bytes_per_desc": 32,
        "device": "CPU",
    },
    3: {
        "name": "FAST_BRISK",
        "type": "classical",
        "detector": detectors.fast_detector,
        "descriptor": descriptors.compute_brisk_descriptors,
        "norm": cv2.NORM_HAMMING,
        "desc_dim": 512,
        "bytes_per_desc": 64,
        "device": "CPU",
    },
    4: {
        "name": "FAST_BRIEF",
        "type": "classical",
        "detector": detectors.fast_detector,
        "descriptor": descriptors.compute_brief_descriptors,
        "norm": cv2.NORM_HAMMING,
        "desc_dim": 256,
        "bytes_per_desc": 32,
        "device": "CPU",
    },
    5: {
        "name": "FAST_FREAK",
        "type": "classical",
        "detector": detectors.fast_detector,
        "descriptor": descriptors.compute_freak_descriptors,
        "norm": cv2.NORM_HAMMING,
        "desc_dim": 512,
        "bytes_per_desc": 64,
        "device": "CPU",
    },
    6: {
        "name": "KAZE",
        "type": "classical",
        "detector": detectors.kaze_detector,
        "descriptor": descriptors.compute_kaze_descriptors,
        "norm": cv2.NORM_L2,
        "desc_dim": 64,
        "bytes_per_desc": 256,
        "device": "CPU",
    },
    7: {
        "name": "SuperPoint_BF",
        "type": "learned",
        "matcher": "bf",
        "desc_dim": 256,
        "bytes_per_desc": 1024,
        "device": "CUDA" if torch.cuda.is_available() else "CPU",
    },
    8: {
        "name": "SuperPoint_LightGlue",
        "type": "learned",
        "matcher": "lightglue",
        "desc_dim": 256,
        "bytes_per_desc": 1024,
        "device": "CUDA" if torch.cuda.is_available() else "CPU",
    },
}
# ============================================================
# Timing
# ============================================================

def timed_call(function, *args, **kwargs):
    """Execute a function and return its result and runtime."""

    result, timing = evaluation.time_call(
        function,
        *args,
        repeats=1,
        **kwargs,
    )

    return result, timing["runtime_mean_s"]


# ============================================================
# Classical matching
# ============================================================

def classic_match(experiment, image1, image2):

    cfg = E[experiment]
    timings = {}

    detector = cfg["detector"]
    descriptor = cfg["descriptor"]

    # Detect keypoints in both images.
    kps1, timings["detection_time_image1_s"] = timed_call(
        detector, image1
    )

    kps2, timings["detection_time_image2_s"] = timed_call(
        detector, image2
    )

    if cfg['name'].startswith("FAST"):
        for kp in kps1:
            kp.size = 31.0
        for kp in kps2:
            kp.size = 31.0

    # Compute descriptors.
    (kps1, desc1), timings["descriptor_time_image1_s"] = timed_call(
        descriptor, image1, kps1
    )

    (kps2, desc2), timings["descriptor_time_image2_s"] = timed_call(
        descriptor, image2, kps2
    )

    # Match descriptors using A1 protocol: kNN(k=2) + ratio 0.8 + one-to-one
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        matches = []
        timings["matching_time_s"] = 0.0
    else:
        matches, timings["matching_time_s"] = timed_call(
            matching.match_pipeline,
            desc1,
            desc2,
            cfg["norm"],
            ratio=0.8,
        )

    # Measure descriptor memory.
    memory1 = evaluation.descriptor_memory(desc1)
    memory2 = evaluation.descriptor_memory(desc2)

    timings["descriptor_memory_image1_bytes"] = (
        memory1["descriptor_memory_bytes"]
    )
    timings["descriptor_memory_image2_bytes"] = (
        memory2["descriptor_memory_bytes"]
    )
    timings["descriptor_memory_bytes"] = (
        memory1["descriptor_memory_bytes"] + memory2["descriptor_memory_bytes"]
    ) / 2.0
    timings["descriptor_dim"] = cfg["desc_dim"]
    timings["descriptor_bytes_each"] = cfg["bytes_per_desc"]
    timings["device"] = cfg["device"]

    detect_s = timings["detection_time_image1_s"] + timings["detection_time_image2_s"]
    describe_s = timings["descriptor_time_image1_s"] + timings["descriptor_time_image2_s"]
    match_s = timings["matching_time_s"]
    timings["detect_ms"] = detect_s * 1000.0 / 2.0
    timings["describe_ms"] = describe_s * 1000.0 / 2.0
    timings["match_ms"] = match_s * 1000.0
    timings["total_ms"] = (detect_s + describe_s + match_s) * 1000.0

    return kps1, kps2, matches, timings


# ============================================================
# Learned matching
# ============================================================

def learned_match(experiment, image1, image2):
    cfg = E[experiment]
    timings = {}

    if cfg["matcher"] == "bf":
        # Extract SuperPoint keypoints and descriptors.
        (kps1, descs1), timings["detection_descriptor_time_image1_s"] = (
            timed_call(learned_models.extract_superpoint, image1)
        )

        (kps2, descs2), timings["detection_descriptor_time_image2_s"] = (
            timed_call(learned_models.extract_superpoint, image2)
        )

        # BFMatcher with kNN + ratio test 0.8 + one-to-one enforcement
        matches, timings["matching_time_s"] = timed_call(
            matching.match_pipeline,
            descs1,
            descs2,
            cv2.NORM_L2,
            ratio=0.8,
        )

    elif cfg["matcher"] == "lightglue":
        # Extract features and preserve native LightGlue dictionaries.
        (kps1, descs1, features1), timings[
            "detection_descriptor_time_image1_s"
        ] = timed_call(
            learned_models.extract_superpoint_features,
            image1,
            return_features=True,
        )

        (kps2, descs2, features2), timings[
            "detection_descriptor_time_image2_s"
        ] = timed_call(
            learned_models.extract_superpoint_features,
            image2,
            return_features=True,
        )

        (matches, scores), timings["matching_time_s"] = timed_call(
            learned_models.match_lightglue,
            features1,
            features2,
        )

        # Convert LightGlue index pairs to cv2.DMatch objects.
        matches = learned_models.lightglue_matches_to_cv(
            matches,
            scores,
        )

    else:
        raise ValueError(
            f"Unknown learned matcher: {cfg['matcher']}"
        )

    # Measure descriptor memory.
    memory1 = evaluation.descriptor_memory(descs1)
    memory2 = evaluation.descriptor_memory(descs2)

    timings["descriptor_memory_image1_bytes"] = (
        memory1["descriptor_memory_bytes"]
    )
    timings["descriptor_memory_image2_bytes"] = (
        memory2["descriptor_memory_bytes"]
    )
    timings["descriptor_memory_bytes"] = (
        memory1["descriptor_memory_bytes"] + memory2["descriptor_memory_bytes"]
    ) / 2.0
    timings["descriptor_dim"] = cfg["desc_dim"]
    timings["descriptor_bytes_each"] = cfg["bytes_per_desc"]
    timings["device"] = cfg["device"]

    extract_s = timings["detection_descriptor_time_image1_s"] + timings["detection_descriptor_time_image2_s"]
    match_s = timings["matching_time_s"]
    timings["detect_ms"] = extract_s * 1000.0 / 2.0
    timings["describe_ms"] = 0.0
    timings["match_ms"] = match_s * 1000.0
    timings["total_ms"] = (extract_s + match_s) * 1000.0

    return kps1, kps2, matches, timings


# ============================================================
# Evaluate one image pair
# ============================================================

def run_pair(experiment, image1, image2, homography):

    if experiment not in E:
        raise ValueError(f"Unknown experiment: {experiment}")

    experiment_config = E[experiment]

    if experiment_config["type"] == "classical":
        kps1, kps2, matches, timings = classic_match(
            experiment,
            image1,
            image2,
        )

    elif experiment_config["type"] == "learned":
        kps1, kps2, matches, timings = learned_match(
            experiment,
            image1,
            image2,
        )

    else:
        raise ValueError(
            f"Unknown experiment type: {experiment_config['type']}"
        )

    # Compute repeatability and visible keypoint statistics
    (
        repeatability,
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
    )

    # Count geometrically correct descriptor matches
    n_correct, errors, correct_mask = evaluation.count_correct_matches(
        kps1,
        kps2,
        matches,
        homography,
    )

    # Compute matching metrics
    if n_correct <= n_repeat:
        matching_metrics = evaluation.compute_matching_metrics(
            n_putatives=len(matches),
            n_correct=n_correct,
            n_features=n_features,
            n_correspondences=n_repeat,
        )
    else:
        matching_metrics = {
            "n_putatives": len(matches),
            "n_correct": n_correct,
            "n_features": n_features,
            "n_correspondences": n_repeat,
            "PMR": len(matches) / n_features if n_features else 0.0,
            "Precision": n_correct / len(matches) if matches else 0.0,
            "Matching_Score": n_correct / n_features if n_features else 0.0,
            "Recall": n_correct / n_repeat if n_repeat else 0.0,
        }

    # Estimate homography from putative matches using RANSAC.
    H_est, inlier_mask = matching.estimate_homography(
        matches,
        kps1,
        kps2,
    )

    # Evaluate homography accuracy against the ground-truth homography.
    homography_metrics = evaluation.homography_accuracy(
        H_est,
        homography,
        image1.shape,
    )

    # Compute RANSAC inlier statistics.
    inlier_metrics = evaluation.compute_inlier_metrics(
        kps1,
        kps2,
        matches,
        inlier_mask,
        H_est=H_est,
    )

    # Homography metric aliases
    ha_aliases = {
        "HA@1": homography_metrics.get("homography_correct_le_1px", False),
        "HA@3": homography_metrics.get("homography_correct_le_3px", False),
        "HA@5": homography_metrics.get("homography_correct_le_5px", False),
        "mean_corner_err": homography_metrics.get("mean_corner_error_px", float("nan")),
    }

    # Build results
    metrics = {
        "experiment": experiment,
        "experiment_name": experiment_config["name"],
        "repeatability": repeatability,
        "n_repeat": n_repeat,
        "n1_visible": n1_visible,
        "n2_visible": n2_visible,
        "n_features": n_features,
        "n_keypoints_image1": len(kps1),
        "n_keypoints_image2": len(kps2),
        "n_putative_matches": len(matches),
        **matching_metrics,
        **homography_metrics,
        **ha_aliases,
        **inlier_metrics,
        **timings,
    }
    return metrics


# ============================================================
# Dataset loading
# ============================================================

def load_sequence(sequence_name):

    directory = DATASET_ROOT / sequence_name

    images = {}
    homographies = {}

    for image_id in range(1, 7):

        image_path = directory / f"{image_id}.ppm"

        image = cv2.imread(str(image_path),cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(
                f"Could not load image: {image_path}"
            )

        images[image_id] = image

    for image_id in range(2, 7):

        homography_path = directory / f"H_1_{image_id}"

        if not homography_path.exists():
            raise FileNotFoundError(
                f"Could not load homography: {homography_path}"
            )

        homographies[image_id] = np.loadtxt(
            homography_path
        )

    return images, homographies


def list_sequences(n, random_selection=False, seed=None):

    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {DATASET_ROOT}"
        )

    all_sequences = sorted(path.name for path in DATASET_ROOT.iterdir() if path.is_dir())

    illumination = [sequence for sequence in all_sequences if sequence.startswith("i_")]

    viewpoint = [sequence for sequence in all_sequences if sequence.startswith("v_")]

    if random_selection:
        rng = random.Random(seed)

        rng.shuffle(illumination)
        rng.shuffle(viewpoint)

    selected_illumination = illumination[:n]
    selected_viewpoint = viewpoint[:n]

    return selected_illumination + selected_viewpoint


# ============================================================
# Run one experiment over one sequence
# ============================================================

def run_sequence(experiment, sequence, pairs):
    results = []
    images, homographies = load_sequence(sequence)

    for image_id1, image_id2 in pairs:

        try:
            image1 = images[image_id1]
            image2 = images[image_id2]

            if image_id1 == 1:
                homography = homographies[image_id2]
            else:
                homography = homographies[image_id2] @ np.linalg.inv(homographies[image_id1])


            metrics = run_pair(
                experiment,
                image1,
                image2,
                homography,
            )

            results.append({
                "experiment": experiment,
                "experiment_name": E[experiment]["name"],
                "sequence": sequence,
                "category": sequence[0],  # "i" or "v"
                "image1": image_id1,
                "image2": image_id2,
                "k": image_id2,
                "success": True,
                **metrics,
            })
        except Exception as exc:

            logging.exception(
                "%s failed on %s: image %s -> %s",
                E[experiment]["name"],
                sequence,
                image_id1,
                image_id2,
            )

            results.append({
                "experiment": experiment,
                "experiment_name": E[experiment]["name"],
                "sequence": sequence,
                "category": sequence[0],
                "image1": image_id1,
                "image2": image_id2,
                "k": image_id2,
                "success": False,
                "error": str(exc),
            })

    return results


def run_fast_parameter_investigation(sequences):
    """Run the FAST parameter sweep on the selected sequences."""

    image_pairs = []

    for sequence_name in sequences:
        images, homographies = load_sequence(sequence_name)

        for image_id in range(2, 7):
            image_pairs.append((
                f"{sequence_name}/1-{image_id}",
                images[1],
                images[image_id],
                homographies[image_id],
            ))

    logging.info(
        "Running FAST parameter investigation on %d image pairs",
        len(image_pairs),
    )

    fast_vs_sift.run_fast_parameter_sweep(image_pairs)

# ============================================================
# Save results
# ============================================================

def save_results(results):
    """Save pair-level results and aggregate successful evaluations."""

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.DataFrame(results)

    df.to_csv(
        RESULTS_DIR / "pair_results.csv",
        index=False,
    )

    if df.empty:
        logging.warning("No results to summarize.")
        return

    if "success" not in df.columns:
        logging.warning("Missing success column.")
        return

    successful = df.loc[df["success"]].copy()

    if successful.empty:
        logging.warning("No successful evaluations.")
        return

    # Calculate the success rate for each experiment.
    success_rate = (
        df.groupby("experiment")["success"]
        .mean()
        .rename("success_rate")
    )

    excluded = {
        "experiment",
        "image1",
        "image2",
        "success",
    }

    numeric_columns = [
        column
        for column in successful.select_dtypes(
            include="number"
        ).columns
        if column not in excluded
    ]

    summary = successful.groupby("experiment")[
        numeric_columns
    ].mean()

    summary = summary.join(success_rate)

    # Add readable experiment names.
    summary.insert(
        0,
        "experiment_name",
        [
            E[experiment]["name"]
            for experiment in summary.index
        ],
    )

    summary.to_csv(
        RESULTS_DIR / "summary.csv",
        index_label="experiment",
    )

    try:
        visualization.plot_summary(
            summary,
            output_dir=RESULTS_DIR,
        )
    except Exception:
        logging.exception("Could not generate summary plots.")

    logging.info(
        "Pair-level results saved to %s",
        RESULTS_DIR / "pair_results.csv",
    )

    logging.info(
        "Summary saved to %s",
        RESULTS_DIR / "summary.csv",
    )

def generate_keypoint_panels(sequences):
    """Save keypoint distribution panels for representative sequences across distinct detectors."""

    output_dir = RESULTS_DIR / "visualizations" / "keypoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    representative_sequences = {}

    for category, prefix in [("illumination", "i_"), ("viewpoint", "v_")]:
        sequence = next(
            (s for s in sequences if s.startswith(prefix)),
            None,
        )

        if sequence is not None:
            representative_sequences[category] = sequence

    for category, sequence in representative_sequences.items():
        images, _ = load_sequence(sequence)
        image = images[1]

        images_and_kps = [
            (image, detectors.sift_detector(image), "SIFT"),
            (image, detectors.orb_detector(image), "ORB"),
            (image, detectors.fast_detector(image), "FAST"),
            (image, detectors.kaze_detector(image), "KAZE"),
        ]

        # Include SuperPoint keypoints
        sp_kps, _ = learned_models.extract_superpoint(image)
        images_and_kps.append((image, sp_kps, "SuperPoint"))

        output_path = output_dir / f"{category}_keypoints.png"

        visualization.draw_keypoints_panel(
            images_and_kps,
            output_path,
        )

        logging.info(
            "Saved keypoint panel: %s",
            output_path,
        )

def generate_metric_plots(results):
    """Generate degradation and Pareto plots from benchmark results."""

    output_dir = RESULTS_DIR / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_rows = []

    for row in results:
        if not row.get("success", False):
            continue

        sequence = row["sequence"]

        # HPatches sequence type: illumination or viewpoint.
        if sequence.startswith("i_"):
            category = "i"
        elif sequence.startswith("v_"):
            category = "v"
        else:
            continue

        # The expected pairs are 1 -> k.
        plot_row = {
            "method": row["experiment_name"],
            "category": category,
            "k": row["image2"],
        }

        # Include the metrics available in the benchmark output.
        for column in (
            "Matching_Score",
            "HA@3",
            "total_ms",
        ):
            if column in row:
                plot_row[column] = row[column]

        plot_rows.append(plot_row)

    if not plot_rows:
        logging.warning("No successful results available for visualization.")
        return

    plot_csv = output_dir / "plot_results.csv"
    pd.DataFrame(plot_rows).to_csv(plot_csv, index=False)

    # Matching Score versus difficulty, separately for each category.
    try:
        paths = visualization.plot_degradation(
            csv_path=plot_csv,
            output_path=output_dir / "degradation.png",
            metric="Matching_Score",
            categories=("i", "v"),
        )
        logging.info("Saved degradation plots: %s", paths)
    except (ValueError, KeyError) as exc:
        logging.warning("Could not generate degradation plots: %s", exc)

    # Runtime versus Matching Score and HA@3, when those metrics exist.
    try:
        path = visualization.plot_pareto(
            csv_path=plot_csv,
            output_path=output_dir / "pareto.png",
        )
        logging.info("Saved Pareto plot: %s", path)
    except (ValueError, KeyError) as exc:
        logging.warning("Could not generate Pareto plot: %s", exc)

def generate_match_figures(sequences):
    """
    Generate correct/incorrect match visualizations for representative methods
    (SIFT, ORB, SuperPoint+LightGlue) on illumination and viewpoint sequences.
    Correct matches: forward transfer error <= 3 pixels (green); incorrect (red).
    """

    output_dir = RESULTS_DIR / "visualizations" / "matches"
    output_dir.mkdir(parents=True, exist_ok=True)

    illumination = next(
        (s for s in sequences if s.startswith("i_")),
        None,
    )
    viewpoint = next(
        (s for s in sequences if s.startswith("v_")),
        None,
    )

    selected = [
        ("illumination", illumination),
        ("viewpoint", viewpoint),
    ]

    methods_to_visualize = [
        (1, "SIFT"),
        (2, "ORB"),
        (8, "SuperPoint_LightGlue"),
    ]

    for category, sequence in selected:
        if sequence is None:
            logging.warning(
                "No %s sequence selected; skipping match visualization.",
                category,
            )
            continue

        images, homographies = load_sequence(sequence)
        image1 = images[1]
        image2 = images[2]
        H = homographies[2]

        for exp_id, method_name in methods_to_visualize:
            cfg = E[exp_id]
            if cfg["type"] == "classical":
                kps1, kps2, matches, _ = classic_match(
                    exp_id,
                    image1,
                    image2,
                )
            else:
                kps1, kps2, matches, _ = learned_match(
                    exp_id,
                    image1,
                    image2,
                )

            _, errors, correct_mask = evaluation.count_correct_matches(
                kps1,
                kps2,
                matches,
                H,
                threshold_px=3.0,
            )

            correct = [
                match
                for match, is_correct in zip(matches, correct_mask)
                if is_correct
            ]

            incorrect = [
                match
                for match, is_correct in zip(matches, correct_mask)
                if not is_correct
            ]

            precision = (len(correct) / len(matches) * 100.0) if matches else 0.0
            output_path = output_dir / f"{category}_{method_name}_matches.png"

            visualization.draw_matches_colored(
                image1,
                kps1,
                image2,
                kps2,
                correct,
                incorrect,
                output_path,
                title=f"{category.capitalize()} ({sequence}) 1→2: {method_name} (Prec: {precision:.1f}%)",
            )

            logging.info(
                "Saved %s %s match visualization: %s (%d correct, %d incorrect)",
                category,
                method_name,
                output_path,
                len(correct),
                len(incorrect),
            )

def df_to_markdown_table(df, title=None, float_formats=None):
    """Render a pandas DataFrame as a GitHub-flavored Markdown table with custom formatting."""
    df_copy = df.copy()
    if float_formats:
        for col, fmt in float_formats.items():
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].apply(
                    lambda v: fmt.format(v) if pd.notna(v) and isinstance(v, (int, float, np.number)) else (str(v) if pd.notna(v) else "-")
                )

    headers = [str(c) for c in df_copy.columns]
    lines = []
    if title:
        lines.append(f"### {title}\n")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df_copy.iterrows():
        row_str = [str(val) for val in row]
        lines.append("| " + " | ".join(row_str) + " |")
    return "\n".join(lines) + "\n"


def generate_report_tables(results):
    """Generate three final report tables, separately aggregated by category."""

    report_dir = RESULTS_DIR / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)

    if df.empty or "success" not in df.columns:
        logging.warning("No results available for report tables.")
        return

    df = df[df["success"] == True].copy()
    if df.empty:
        logging.warning("No successful results available for report tables.")
        return

    if "category" not in df.columns:
        df["category"] = df["sequence"].str[0]

    all_markdown = [
        "# Benchmark Report Tables\n",
        "Aggregated quantitative evaluation on HPatches dataset across Illumination (i_*) and Viewpoint (v_*) sequences.\n",
    ]

    # --- TABLE 1: Matching Quality ---
    t1_cols = {
        "n_features": "#features",
        "repeatability": "Repeat.",
        "n_putatives": "#putative",
        "PMR": "PMR",
        "n_correct": "#correct",
        "Precision": "Precision",
        "Matching_Score": "Matching score",
        "n_correspondences": "#corresp.",
        "Recall": "Recall",
    }
    t1_formats = {
        "#features": "{:.1f}",
        "Repeat.": "{:.3f}",
        "#putative": "{:.1f}",
        "PMR": "{:.3f}",
        "#correct": "{:.1f}",
        "Precision": "{:.3f}",
        "Matching score": "{:.3f}",
        "#corresp.": "{:.1f}",
        "Recall": "{:.3f}",
    }

    t1_frames = []
    for cat_code, cat_name in [("i", "Illumination"), ("v", "Viewpoint")]:
        sub = df[df["category"] == cat_code]
        if sub.empty:
            continue
        agg = sub.groupby(["experiment", "experiment_name"], as_index=False)[list(t1_cols.keys())].mean()
        agg = agg.rename(columns={"experiment_name": "Method", **t1_cols}).drop(columns=["experiment"])
        agg.insert(0, "Category", cat_name)
        t1_frames.append(agg)

    if t1_frames:
        t1_df = pd.concat(t1_frames, ignore_index=True)
        t1_df.to_csv(report_dir / "table1_matching_quality.csv", index=False)
        t1_md = df_to_markdown_table(t1_df, "Table 1: Matching Quality", t1_formats)
        (report_dir / "table1_matching_quality.md").write_text(t1_md)
        all_markdown.append(t1_md)

    # --- TABLE 2: Runtime and Cost ---
    df["_mem_kb"] = df["descriptor_memory_bytes"].fillna(0) / 1024.0

    t2_cols = {
        "detect_ms": "Detect (ms)",
        "describe_ms": "Describe (ms)",
        "match_ms": "Match (ms)",
        "total_ms": "Total (ms)",
        "descriptor_dim": "Descriptor dim",
        "descriptor_bytes_each": "Bytes/desc.",
        "_mem_kb": "Memory (KB)",
        "device": "Device",
    }
    t2_formats = {
        "Detect (ms)": "{:.2f}",
        "Describe (ms)": "{:.2f}",
        "Match (ms)": "{:.2f}",
        "Total (ms)": "{:.2f}",
        "Descriptor dim": "{:.0f}",
        "Bytes/desc.": "{:.0f}",
        "Memory (KB)": "{:.1f}",
    }

    t2_frames = []
    for cat_code, cat_name in [("i", "Illumination"), ("v", "Viewpoint")]:
        sub = df[df["category"] == cat_code]
        if sub.empty:
            continue
        agg = sub.groupby(["experiment", "experiment_name"], as_index=False).agg({
            "detect_ms": "mean",
            "describe_ms": "mean",
            "match_ms": "mean",
            "total_ms": "mean",
            "descriptor_dim": "first",
            "descriptor_bytes_each": "first",
            "_mem_kb": "mean",
            "device": "first",
        })
        agg = agg.rename(columns={"experiment_name": "Method", **t2_cols}).drop(columns=["experiment"])
        agg.insert(0, "Category", cat_name)
        t2_frames.append(agg)

    if t2_frames:
        t2_df = pd.concat(t2_frames, ignore_index=True)
        t2_df.to_csv(report_dir / "table2_runtime_cost.csv", index=False)
        t2_md = df_to_markdown_table(t2_df, "Table 2: Runtime and Computational Cost", t2_formats)
        (report_dir / "table2_runtime_cost.md").write_text(t2_md)
        all_markdown.append(t2_md)

    # --- TABLE 3: Homography Quality ---
    df["_ha1_pct"] = df["homography_correct_le_1px"].astype(float) * 100.0
    df["_ha3_pct"] = df["homography_correct_le_3px"].astype(float) * 100.0
    df["_ha5_pct"] = df["homography_correct_le_5px"].astype(float) * 100.0

    t3_cols = {
        "_ha1_pct": "HA@1px (%)",
        "_ha3_pct": "HA@3px (%)",
        "_ha5_pct": "HA@5px (%)",
        "n_ransac_inliers": "RANSAC inliers",
        "mean_inlier_reprojection_error_px": "Mean inlier error (px)",
    }
    t3_formats = {
        "HA@1px (%)": "{:.1f}",
        "HA@3px (%)": "{:.1f}",
        "HA@5px (%)": "{:.1f}",
        "RANSAC inliers": "{:.1f}",
        "Mean inlier error (px)": "{:.2f}",
    }

    t3_frames = []
    for cat_code, cat_name in [("i", "Illumination"), ("v", "Viewpoint")]:
        sub = df[df["category"] == cat_code]
        if sub.empty:
            continue
        agg = sub.groupby(["experiment", "experiment_name"], as_index=False)[list(t3_cols.keys())].mean()
        agg = agg.rename(columns={"experiment_name": "Method", **t3_cols}).drop(columns=["experiment"])
        agg.insert(0, "Category", cat_name)
        t3_frames.append(agg)

    if t3_frames:
        t3_df = pd.concat(t3_frames, ignore_index=True)
        t3_df.to_csv(report_dir / "table3_homography_quality.csv", index=False)
        t3_md = df_to_markdown_table(t3_df, "Table 3: Homography Estimation Quality", t3_formats)
        (report_dir / "table3_homography_quality.md").write_text(t3_md)
        all_markdown.append(t3_md)

    (report_dir / "all_report_tables.md").write_text("\n".join(all_markdown))
    logging.info("Report tables saved in %s (both CSV and Markdown)", report_dir)


def main():

    # Select sequences once for both experiments.
    sequences = list_sequences(
        n=N_SEQUENCES,
        random_selection=RANDOM_SELECTION,
        seed=SEED,
    )

    logging.info("Selected %d sequences: %s", len(sequences), sequences)

    # Run the main experiments.
    results = []

    for experiment in E:
        logging.info(
            "Running experiment %s: %s",
            experiment,
            E[experiment]["name"],
        )

        for sequence in sequences:
            sequence_results = run_sequence(
                experiment,
                sequence,
                PAIRS,
            )
            results.extend(sequence_results)

    save_results(results)
    generate_report_tables(results)
    generate_keypoint_panels(sequences)
    generate_match_figures(sequences)
    generate_metric_plots(results)
    run_fast_parameter_investigation(sequences)


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    main()