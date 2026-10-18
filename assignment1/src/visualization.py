"""
visualization.py — Visualization and reporting for the HPatches benchmark.

Responsibilities
----------------
1. Visualize keypoints and descriptor matches.
2. Plot degradation curves and time-accuracy comparisons.
3. Aggregate benchmark results into summary tables.
4. Generate all report artifacts.
5. Provide standalone tests using real PPM images when available.

Expected main results columns
-----------------------------
method, category, k, n_features, repeatability, n_putatives,
PMR, n_correct, Precision, Matching_Score, n_correspondences,
Recall

Optional columns include
------------------------
detect_ms, describe_ms, match_ms, total_ms, memory_bytes,
mean_corner_err, HA@1, HA@3, HA@5, n_inliers,
reproj_err_inliers
"""

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

MATCHING_COLUMNS = [
    "n_features",
    "repeatability",
    "n_putatives",
    "PMR",
    "n_correct",
    "Precision",
    "Matching_Score",
    "n_correspondences",
    "Recall",
]

COST_COLUMNS = [
    "detect_ms",
    "describe_ms",
    "match_ms",
    "total_ms",
    "memory_bytes",
]

HOMOGRAPHY_COLUMNS = [
    "mean_corner_err",
    "HA@1",
    "HA@3",
    "HA@5",
    "n_inliers",
    "reproj_err_inliers",
]

REQUIRED_RESULTS_COLUMNS = {
    "method",
    "category",
    "k",
}

METHOD_STYLES = {
    "SIFT": {
        "color": "#1f77b4",       # Deep blue
        "linestyle": "-",
        "marker": "o",
        "label": "SIFT",
    },
    "KAZE": {
        "color": "#17becf",       # Cyan
        "linestyle": "-",
        "marker": "^",
        "label": "KAZE",
    },
    "ORB": {
        "color": "#ff7f0e",       # Orange
        "linestyle": "--",
        "marker": "s",
        "label": "ORB",
    },
    "FAST_BRIEF": {
        "color": "#2ca02c",       # Green
        "linestyle": "--",
        "marker": "D",
        "label": "FAST+BRIEF",
    },
    "FAST_BRISK": {
        "color": "#8c564b",       # Brown
        "linestyle": "--",
        "marker": "v",
        "label": "FAST+BRISK",
    },
    "FAST_FREAK": {
        "color": "#e377c2",       # Pink
        "linestyle": "--",
        "marker": "<",
        "label": "FAST+FREAK",
    },
    "SuperPoint_BF": {
        "color": "#9467bd",       # Purple
        "linestyle": "-.",
        "marker": "p",
        "label": "SuperPoint+BF",
    },
    "SuperPoint_LightGlue": {
        "color": "#d62728",       # Crimson red
        "linestyle": "-",
        "marker": "*",
        "label": "SuperPoint+LightGlue",
    },
}

def get_method_style(method_name):
    """Retrieve consistent, distinct styling for a feature matching method."""
    if method_name in METHOD_STYLES:
        return METHOD_STYLES[method_name]
    norm = str(method_name).replace(" ", "_").replace("+", "_")
    for key, style in METHOD_STYLES.items():
        if key.lower() == norm.lower():
            return style
    return {
        "color": "#7f7f7f",
        "linestyle": "-",
        "marker": "o",
        "label": str(method_name),
    }


# ------------------------------------------------------------------
# General helpers
# ------------------------------------------------------------------

def ensure_parent_directory(path):
    """Create the parent directory of a file path."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def load_results_csv(csv_path):
    """
    Load benchmark results and validate their essential columns.

    Returns
    -------
    pd.DataFrame
    """
    csv_path = Path(csv_path)

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Results CSV does not exist: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    missing = REQUIRED_RESULTS_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"Results CSV is missing required columns: "
            f"{sorted(missing)}"
        )

    return df


def save_figure(fig, output_path, dpi=150):
    """Save and close a Matplotlib figure."""
    output_path = Path(output_path)
    ensure_parent_directory(output_path)

    try:
        fig.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
        )
    finally:
        plt.close(fig)

    return output_path


def to_color_image(image):
    """Convert a grayscale or BGR image to a three-channel BGR image."""
    image = np.asarray(image)

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.ndim == 3 and image.shape[2] == 3:
        return image.copy()

    raise ValueError(
        "Expected a grayscale or three-channel BGR image."
    )


# ------------------------------------------------------------------
# 1. Keypoint visualization
# ------------------------------------------------------------------

def draw_keypoints_panel(
    images_and_kps,
    output_path,
    titles=None,
):
    """
    Draw keypoints for multiple methods side by side.

    Parameters
    ----------
    images_and_kps : list
        Each element is (image, keypoints, method_name).

    output_path : str or Path
        Output image path.

    titles : list[str], optional
        Custom panel titles. When omitted, method names are used.

    Returns
    -------
    Path
        Path of the saved figure.
    """
    if not images_and_kps:
        raise ValueError("images_and_kps cannot be empty.")

    if titles is not None and len(titles) != len(images_and_kps):
        raise ValueError(
            "titles must have the same length as images_and_kps."
        )

    n_panels = len(images_and_kps)

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(5 * n_panels, 5),
        squeeze=False,
    )

    try:
        for i, (image, keypoints, method_name) in enumerate(
            images_and_kps
        ):
            image_bgr = to_color_image(image)

            rendered = cv2.drawKeypoints(
                image_bgr,
                keypoints,
                None,
                color=(0, 255, 0),
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )

            ax = axes[0, i]
            ax.imshow(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB))

            title = (
                titles[i]
                if titles is not None
                else method_name
            )

            ax.set_title(
                f"{title} ({len(keypoints)} keypoints)",
                fontsize=10,
            )
            ax.axis("off")

        fig.tight_layout()

    except Exception:
        plt.close(fig)
        raise

    return save_figure(fig, output_path)


# ------------------------------------------------------------------
# 2. Match visualization
# ------------------------------------------------------------------

def create_match_canvas(img1, img2):
    """
    Place two images side by side on a common canvas.

    Returns
    -------
    canvas : np.ndarray
    offset_x : int
        Horizontal offset of image 2.
    """
    img1 = to_color_image(img1)
    img2 = to_color_image(img2)

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    canvas = np.zeros(
        (max(h1, h2), w1 + w2, 3),
        dtype=np.uint8,
    )

    canvas[:h1, :w1] = img1
    canvas[:h2, w1:w1 + w2] = img2

    return canvas, w1


def draw_match_lines(
    canvas,
    img1_width,
    keypoints1,
    keypoints2,
    matches,
    color,
    thickness=1,
):
    """Draw a collection of descriptor matches on a shared canvas."""
    for match in matches:
        if not (
            0 <= match.queryIdx < len(keypoints1)
            and 0 <= match.trainIdx < len(keypoints2)
        ):
            raise IndexError(
                "A match references a keypoint outside the supplied lists."
            )

        x1, y1 = keypoints1[match.queryIdx].pt
        x2, y2 = keypoints2[match.trainIdx].pt

        point1 = (int(round(x1)), int(round(y1)))
        point2 = (
            int(round(x2)) + img1_width,
            int(round(y2)),
        )

        cv2.line(
            canvas,
            point1,
            point2,
            color,
            thickness,
        )

    return canvas


def draw_matches_colored(
    img1,
    kps1,
    img2,
    kps2,
    correct,
    incorrect,
    output_path,
    title="",
):
    """
    Visualize correct matches in green and incorrect matches in red.

    OpenCV uses BGR internally:
        Green = (0, 255, 0)
        Red   = (0, 0, 255)
    """
    canvas, offset_x = create_match_canvas(img1, img2)

    # Draw incorrect matches first so correct matches remain visible.
    draw_match_lines(
        canvas,
        offset_x,
        kps1,
        kps2,
        incorrect,
        color=(0, 0, 255),
    )

    draw_match_lines(
        canvas,
        offset_x,
        kps1,
        kps2,
        correct,
        color=(0, 255, 0),
    )

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

    ax.set_title(
        f"{title} "
        f"(green={len(correct)} correct, "
        f"red={len(incorrect)} incorrect)",
        fontsize=10,
    )
    ax.axis("off")
    fig.tight_layout()

    return save_figure(fig, output_path)


# ------------------------------------------------------------------
# 3. Degradation curves
# ------------------------------------------------------------------

def aggregate_degradation_data(
    df,
    metric="Matching_Score",
    category="i",
):
    """
    Calculate mean metric values by method and difficulty.

    Parameters
    ----------
    df : pd.DataFrame
        Benchmark results.

    metric : str
        Metric to aggregate.

    category : str
        HPatches category, e.g. 'i' or 'v'.

    Returns
    -------
    pd.DataFrame
        Columns: method, k, metric.
    """
    required = {"method", "category", "k", metric}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Cannot aggregate degradation data; "
            f"missing columns: {sorted(missing)}"
        )

    subset = df.loc[
        df["category"] == category,
        ["method", "k", metric],
    ]

    return (
        subset.groupby(
            ["method", "k"],
            as_index=False,
        )[metric]
        .mean()
        .sort_values(["method", "k"])
    )


def plot_degradation(
    csv_path,
    output_path,
    metric="Matching_Score",
    categories=("i", "v"),
):
    """
    Plot metric versus pair difficulty for each category.

    Produces one figure per category, plus a combined side-by-side figure.

    Returns
    -------
    list[Path]
        Paths of generated figures.
    """
    df = load_results_csv(csv_path)

    if metric not in df.columns:
        raise ValueError(
            f"Metric '{metric}' is missing from {csv_path}."
        )

    output_path = Path(output_path)

    if output_path.suffix.lower() != ".png":
        raise ValueError("output_path must have a .png extension.")

    output_paths = []
    category_data = {}

    for category in categories:
        aggregated = aggregate_degradation_data(
            df,
            metric=metric,
            category=category,
        )

        if aggregated.empty:
            continue

        category_data[category] = aggregated

        fig, ax = plt.subplots(figsize=(8.5, 5.5))

        try:
            for method, method_data in aggregated.groupby("method"):
                style = get_method_style(method)
                ax.plot(
                    method_data["k"],
                    method_data[metric],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    markersize=7,
                    linewidth=2.2,
                    label=style["label"],
                )

            cat_title = (
                "Illumination Variation (i_*)"
                if category == "i"
                else "Viewpoint Variation (v_*)"
                if category == "v"
                else f"Category: {category}"
            )

            ax.set_title(
                f"Degradation Analysis: Matching Score vs Difficulty\n{cat_title}",
                fontsize=11,
                fontweight="bold",
                pad=10,
            )
            ax.set_xlabel("Pair difficulty k (Image 1 → Image k)", fontsize=10, fontweight="bold")
            ax.set_ylabel(f"Mean {metric.replace('_', ' ')}", fontsize=10, fontweight="bold")
            ax.set_xticks([2, 3, 4, 5, 6])
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                frameon=True,
                facecolor="white",
                edgecolor="#cccccc",
                fontsize=8.5,
                borderaxespad=0.,
            )
            fig.tight_layout()

        except Exception:
            plt.close(fig)
            raise

        category_path = output_path.with_name(
            f"{output_path.stem}_{category}"
            f"{output_path.suffix}"
        )

        output_paths.append(
            save_figure(fig, category_path, dpi=200)
        )

    # Combined side-by-side figure (Illumination vs Viewpoint)
    if "i" in category_data and "v" in category_data:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)

        try:
            configs = [
                ("i", "Illumination Robustness (i_*)"),
                ("v", "Viewpoint Robustness (v_*)"),
            ]
            for idx, (cat, title) in enumerate(configs):
                ax = axes[idx]
                agg = category_data[cat]

                for method, method_data in agg.groupby("method"):
                    style = get_method_style(method)
                    ax.plot(
                        method_data["k"],
                        method_data[metric],
                        color=style["color"],
                        linestyle=style["linestyle"],
                        marker=style["marker"],
                        markersize=7,
                        linewidth=2.2,
                        label=style["label"],
                    )

                ax.set_title(title, fontsize=11, fontweight="bold")
                ax.set_xlabel("Pair difficulty k (Image 1 → Image k)", fontsize=10, fontweight="bold")
                if idx == 0:
                    ax.set_ylabel(f"Mean {metric.replace('_', ' ')}", fontsize=10, fontweight="bold")
                ax.set_xticks([2, 3, 4, 5, 6])
                ax.set_ylim(-0.02, 1.02)
                ax.grid(True, linestyle="--", alpha=0.5)

            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                bbox_to_anchor=(0.99, 0.85),
                loc="upper left",
                frameon=True,
                facecolor="white",
                edgecolor="#cccccc",
                fontsize=9,
            )
            fig.subplots_adjust(right=0.83, wspace=0.15)

            combined_path = output_path.with_name(
                f"{output_path.stem}_combined{output_path.suffix}"
            )
            output_paths.append(
                save_figure(fig, combined_path, dpi=200)
            )

        except Exception:
            plt.close(fig)

    return output_paths


# ------------------------------------------------------------------
# 4. Pareto plots
# ------------------------------------------------------------------

def aggregate_pareto_data(
    df,
    accuracy_metric,
):
    """
    Aggregate runtime and accuracy by method.

    Returns a DataFrame containing the mean runtime and mean accuracy
    for each method.
    """
    required = {"method", "total_ms", accuracy_metric}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Cannot aggregate Pareto data; "
            f"missing columns: {sorted(missing)}"
        )

    return (
        df.groupby("method", as_index=False)
        .agg(
            mean_time_ms=("total_ms", "mean"),
            mean_accuracy=(accuracy_metric, "mean"),
        )
    )


def plot_pareto(csv_path, output_path):
    """
    Compare runtime against Matching Score and HA@3.

    A metric is plotted only if its column exists.

    Returns
    -------
    Path or None
        Saved figure path, or None if no supported metric is available.
    """
    df = load_results_csv(csv_path)

    if "total_ms" not in df.columns:
        return None

    metrics = [
        ("Matching_Score", "Matching Score"),
        ("HA@3", "Homography Accuracy HA@3px (%)"),
    ]

    metrics = [
        (column, title)
        for column, title in metrics
        if column in df.columns
    ]

    if not metrics:
        return None

    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(7.5 * len(metrics), 5.5),
        squeeze=False,
    )

    # Staggered annotation offsets to prevent text collision
    offsets = {
        "FAST_BRIEF": (-25, 12),
        "FAST_BRISK": (10, -16),
        "FAST_FREAK": (-30, -18),
        "ORB": (10, 8),
        "SIFT": (10, -12),
        "KAZE": (10, 8),
        "SuperPoint_BF": (-45, 12),
        "SuperPoint_LightGlue": (10, -14),
    }

    try:
        for i, (metric, title) in enumerate(metrics):
            ax = axes[0, i]

            aggregated = aggregate_pareto_data(
                df,
                accuracy_metric=metric,
            )

            for row in aggregated.itertuples(index=False):
                runtime = row.mean_time_ms
                accuracy = row.mean_accuracy

                # If HA@3 is stored as 0..1 ratio, convert to %
                if metric == "HA@3" and accuracy <= 1.0:
                    accuracy = accuracy * 100.0

                if (
                    not np.isfinite(runtime)
                    or runtime <= 0
                    or not np.isfinite(accuracy)
                ):
                    continue

                style = get_method_style(row.method)

                ax.scatter(
                    runtime,
                    accuracy,
                    color=style["color"],
                    marker=style["marker"],
                    s=120,
                    edgecolors="black",
                    linewidths=1.2,
                    zorder=5,
                    label=style["label"],
                )

                norm_name = str(row.method).replace(" ", "_").replace("+", "_")
                offset = offsets.get(norm_name, (10, 5))

                ax.annotate(
                    style["label"],
                    (runtime, accuracy),
                    xytext=offset,
                    textcoords="offset points",
                    fontsize=8,
                    fontweight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.25",
                        fc="white",
                        ec=style["color"],
                        alpha=0.9,
                        lw=1.2,
                    ),
                    zorder=6,
                )

            ax.set_xscale("log")
            ax.set_xlabel("Mean Total Time per Pair (ms, log scale)", fontsize=10, fontweight="bold")
            ax.set_ylabel(f"Mean {title}", fontsize=10, fontweight="bold")
            ax.set_title(f"Efficiency vs Accuracy: Time vs {title}", fontsize=11, fontweight="bold", pad=10)
            ax.grid(True, which="both", linestyle="--", alpha=0.4)

        fig.tight_layout()

    except Exception:
        plt.close(fig)
        raise

    return save_figure(fig, output_path, dpi=200)


# ------------------------------------------------------------------
# 5. Summary tables
# ------------------------------------------------------------------

def aggregate_summary_table(df, columns):
    """
    Aggregate available numeric metrics by method.

    Returns an empty DataFrame if no requested columns are available.
    """
    available = [
        column
        for column in columns
        if column in df.columns
    ]

    if not available:
        return pd.DataFrame()

    return (
        df.groupby("method")[available]
        .mean(numeric_only=True)
        .round(4)
        .sort_index()
    )


def generate_tables(csv_path, output_dir):
    """
    Generate matching, computational cost, and homography tables.

    Creates up to three CSV files per category.

    Returns
    -------
    list[Path]
        Paths of generated tables.
    """
    df = load_results_csv(csv_path)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table_definitions = [
        ("A", MATCHING_COLUMNS, 4),
        ("B", COST_COLUMNS, 2),
        ("C", HOMOGRAPHY_COLUMNS, 3),
    ]

    output_paths = []

    for category in ("i", "v"):
        subset = df.loc[df["category"] == category]

        if subset.empty:
            continue

        for table_name, columns, decimals in table_definitions:
            table = aggregate_summary_table(
                subset,
                columns,
            )

            if table.empty:
                continue

            table = table.round(decimals)

            output_path = (
                output_dir / f"Table_{table_name}_{category}.csv"
            )

            table.to_csv(output_path)
            output_paths.append(output_path)

    return output_paths


# ------------------------------------------------------------------
# 6. Pipeline orchestration
# ------------------------------------------------------------------

def generate_all_figures(
    csv_path=None,
    figures_dir=None,
    tables_dir=None,
):
    """
    Generate report figures and tables from benchmark results.

    Paths default to the values in config.py.

    Returns
    -------
    dict[str, list[Path] | Path | None]
        Generated artifacts grouped by type.
    """
    if csv_path is None:
        csv_path = Path(cfg.CSV_DIR) / "main_results.csv"

    if figures_dir is None:
        figures_dir = Path(cfg.FIGURES_DIR)

    if tables_dir is None:
        tables_dir = Path(cfg.RESULTS_DIR) / "tables"

    csv_path = Path(csv_path)
    figures_dir = Path(figures_dir)

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Main results CSV not found: {csv_path}"
        )

    figures_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "degradation": plot_degradation(
            csv_path,
            figures_dir / "degradation.png",
        ),
        "pareto": plot_pareto(
            csv_path,
            figures_dir / "pareto.png",
        ),
        "tables": generate_tables(
            csv_path,
            tables_dir,
        ),
    }

    return artifacts


# ------------------------------------------------------------------
# 7. Standalone tests
# ------------------------------------------------------------------

def _create_test_results_csv(output_path):
    """Create a small synthetic benchmark dataset for testing."""
    rows = []

    methods = {
        "SIFT": {
            "score": 0.70,
            "repeatability": 0.75,
            "time": 25.0,
            "ha3": 0.80,
        },
        "ORB": {
            "score": 0.55,
            "repeatability": 0.60,
            "time": 8.0,
            "ha3": 0.65,
        },
    }

    for category in ("i", "v"):
        for k in range(1, 6):
            for method, values in methods.items():
                rows.append({
                    "method": method,
                    "category": category,
                    "k": k,
                    "n_features": 500,
                    "repeatability": (
                        values["repeatability"] / k
                    ),
                    "n_putatives": 100,
                    "PMR": 0.20,
                    "n_correct": int(
                        100 * values["score"] / k
                    ),
                    "Precision": values["score"] / k,
                    "Matching_Score": values["score"] / k,
                    "n_correspondences": 100,
                    "Recall": values["score"] / k,
                    "detect_ms": values["time"] * 0.3,
                    "describe_ms": values["time"] * 0.3,
                    "match_ms": values["time"] * 0.4,
                    "total_ms": values["time"],
                    "memory_bytes": 4096,
                    "mean_corner_err": 2.0 * k,
                    "HA@1": 0.4 / k,
                    "HA@3": values["ha3"] / k,
                    "HA@5": 0.9 / k,
                    "n_inliers": 50,
                    "reproj_err_inliers": 1.5,
                })

    pd.DataFrame(rows).to_csv(output_path, index=False)


def _load_test_images(image_dir):
    """
    Load two PPM images, falling back to synthetic images if necessary.

    Returns
    -------
    image1, image2, image_names
    """
    image_dir = Path(image_dir)

    image_paths = (
        sorted(image_dir.glob("*.ppm"))
        if image_dir.is_dir()
        else []
    )

    if len(image_paths) >= 2:
        image1 = cv2.imread(
            str(image_paths[0]),
            cv2.IMREAD_GRAYSCALE,
        )
        image2 = cv2.imread(
            str(image_paths[1]),
            cv2.IMREAD_GRAYSCALE,
        )

        if image1 is not None and image2 is not None:
            return (
                image1,
                image2,
                (image_paths[0].name, image_paths[1].name),
            )

    # Synthetic images keep the test runnable without a dataset.
    image1 = np.zeros((300, 400), dtype=np.uint8)

    cv2.rectangle(
        image1,
        (50, 50),
        (200, 200),
        255,
        3,
    )
    cv2.circle(
        image1,
        (280, 130),
        40,
        180,
        3,
    )
    cv2.putText(
        image1,
        "TEST",
        (70, 270),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        255,
        2,
    )

    image2 = image1.copy()

    cv2.line(
        image2,
        (10, 10),
        (100, 100),
        120,
        2,
    )

    return image1, image2, ("synthetic_1", "synthetic_2")


def plot_summary(summary=None, output_dir=None):
    """
    Generate summary figures and tables from this benchmark's CSV format.

    Expects pair_results.csv and, optionally, a summary DataFrame
    indexed by experiment ID.
    """
    if output_dir is None:
        output_dir = cfg.RESULTS_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_csv = output_dir / "pair_results.csv"

    if not pair_csv.is_file():
        raise FileNotFoundError(
            f"Pair-level results CSV not found: {pair_csv}"
        )

    df = pd.read_csv(pair_csv)

    required = {"experiment", "experiment_name"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Pair results CSV is missing required columns: "
            f"{sorted(missing)}"
        )

    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "figures": [],
        "tables": [],
    }

    # --------------------------------------------------------------
    # Figure 1: Matching Score by experiment
    # --------------------------------------------------------------
    if "Matching_Score" in df.columns:
        grouped = (
            df.groupby("experiment_name")["Matching_Score"]
            .mean()
            .sort_values(ascending=False)
        )

        fig, ax = plt.subplots(figsize=(10, 5))

        try:
            grouped.plot(kind="bar", ax=ax)

            ax.set_title("Mean Matching Score by Method")
            ax.set_xlabel("Method")
            ax.set_ylabel("Matching Score")
            ax.tick_params(axis="x", labelrotation=30)
            ax.grid(axis="y", alpha=0.3)

            fig.tight_layout()

        except Exception:
            plt.close(fig)
            raise

        path = save_figure(
            fig,
            figures_dir / "matching_score_by_method.png",
        )
        artifacts["figures"].append(path)

    # --------------------------------------------------------------
    # Figure 2: Repeatability by experiment
    # --------------------------------------------------------------
    if "repeatability" in df.columns:
        grouped = (
            df.groupby("experiment_name")["repeatability"]
            .mean()
            .sort_values(ascending=False)
        )

        fig, ax = plt.subplots(figsize=(10, 5))

        try:
            grouped.plot(kind="bar", ax=ax)

            ax.set_title("Mean Repeatability by Method")
            ax.set_xlabel("Method")
            ax.set_ylabel("Repeatability")
            ax.tick_params(axis="x", labelrotation=30)
            ax.grid(axis="y", alpha=0.3)

            fig.tight_layout()

        except Exception:
            plt.close(fig)
            raise

        path = save_figure(
            fig,
            figures_dir / "repeatability_by_method.png",
        )
        artifacts["figures"].append(path)

    # --------------------------------------------------------------
    # Figure 3: Matching Score vs matching time
    # --------------------------------------------------------------
    if (
        "Matching_Score" in df.columns
        and "matching_time_s" in df.columns
    ):
        grouped = (
            df.groupby("experiment_name", as_index=False)
            .agg(
                matching_score=("Matching_Score", "mean"),
                matching_time_s=("matching_time_s", "mean"),
            )
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        try:
            for row in grouped.itertuples(index=False):
                if not (
                    np.isfinite(row.matching_score)
                    and np.isfinite(row.matching_time_s)
                ):
                    continue

                ax.scatter(
                    row.matching_time_s * 1000,
                    row.matching_score,
                    s=60,
                )

                ax.annotate(
                    row.experiment_name,
                    (
                        row.matching_time_s * 1000,
                        row.matching_score,
                    ),
                    fontsize=8,
                )

            ax.set_title("Matching Score vs Matching Time")
            ax.set_xlabel("Mean Matching Time (ms)")
            ax.set_ylabel("Mean Matching Score")
            ax.grid(alpha=0.3)

            fig.tight_layout()

        except Exception:
            plt.close(fig)
            raise

        path = save_figure(
            fig,
            figures_dir / "matching_score_vs_time.png",
        )
        artifacts["figures"].append(path)

    # --------------------------------------------------------------
    # Table 1: Aggregated benchmark metrics
    # --------------------------------------------------------------
    metrics = [
        "repeatability",
        "n_features",
        "n_putatives",
        "n_correct",
        "n_correspondences",
        "PMR",
        "Precision",
        "Matching_Score",
        "Recall",
        "matching_time_s",
        "success",
    ]

    available = [column for column in metrics if column in df.columns]

    if available:
        table = (
            df.groupby(
                ["experiment", "experiment_name"],
                as_index=False,
            )[available]
            .mean(numeric_only=True)
            .sort_values("experiment")
        )

        table_path = tables_dir / "benchmark_summary.csv"
        table.to_csv(table_path, index=False)
        artifacts["tables"].append(table_path)

    # --------------------------------------------------------------
    # Table 2: Results by sequence
    # --------------------------------------------------------------
    if "sequence" in df.columns:
        sequence_metrics = [
            column
            for column in (
                "repeatability",
                "Matching_Score",
                "Precision",
                "Recall",
                "matching_time_s",
            )
            if column in df.columns
        ]

        if sequence_metrics:
            table = (
                df.groupby(
                    ["experiment_name", "sequence"],
                    as_index=False,
                )[sequence_metrics]
                .mean(numeric_only=True)
            )

            table_path = tables_dir / "benchmark_by_sequence.csv"
            table.to_csv(table_path, index=False)
            artifacts["tables"].append(table_path)

    return artifacts

def run_visualization_tests(
    output_dir="visualization_test_output",
    image_dir=None,
):
    """
    Exercise the visualization and reporting functions.

    Uses PPM images when available, otherwise synthetic images.
    All generated files are written to output_dir.
    """
    import tempfile

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if image_dir is None:
        image_dir = (
            Path(__file__).resolve().parent.parent
            / "data"
            / "images"
        )

    print("Testing visualization.py")
    print(f"Output directory: {output_dir.resolve()}")

    # --------------------------------------------------------------
    # Test 1: Load images
    # --------------------------------------------------------------
    image1, image2, image_names = _load_test_images(image_dir)

    print(f"Image 1: {image_names[0]}, shape={image1.shape}")
    print(f"Image 2: {image_names[1]}, shape={image2.shape}")

    # --------------------------------------------------------------
    # Test 2: Detect keypoints
    # --------------------------------------------------------------
    sift = cv2.SIFT_create()

    keypoints1 = sift.detect(image1, None)
    keypoints2 = sift.detect(image2, None)

    print(
        f"Detected keypoints: "
        f"image 1={len(keypoints1)}, "
        f"image 2={len(keypoints2)}"
    )

    keypoint_path = output_dir / "test_keypoints.png"

    draw_keypoints_panel(
        [
            (image1, keypoints1, "SIFT image 1"),
            (image2, keypoints2, "SIFT image 2"),
        ],
        keypoint_path,
    )

    # --------------------------------------------------------------
    # Test 3: Draw synthetic correct/incorrect matches
    # --------------------------------------------------------------
    n_matches = min(
        10,
        len(keypoints1),
        len(keypoints2),
    )

    matches = [
        cv2.DMatch(i, i, 0.0)
        for i in range(n_matches)
    ]

    correct = matches[: n_matches // 2]
    incorrect = matches[n_matches // 2 :]

    matches_path = output_dir / "test_matches.png"

    draw_matches_colored(
        image1,
        keypoints1,
        image2,
        keypoints2,
        correct,
        incorrect,
        matches_path,
        title="Visualization test",
    )

    # --------------------------------------------------------------
    # Test 4: Create synthetic results CSV
    # --------------------------------------------------------------
    results_path = output_dir / "test_results.csv"

    _create_test_results_csv(results_path)

    df = load_results_csv(results_path)

    # Test aggregation independently of plotting.
    degradation = aggregate_degradation_data(
        df,
        metric="Matching_Score",
        category="i",
    )

    assert not degradation.empty
    print(f"Degradation aggregation: {len(degradation)} rows")

    pareto = aggregate_pareto_data(
        df,
        accuracy_metric="Matching_Score",
    )

    assert not pareto.empty
    print(f"Pareto aggregation: {len(pareto)} methods")

    # --------------------------------------------------------------
    # Test 5: Generate degradation plots
    # --------------------------------------------------------------
    degradation_paths = plot_degradation(
        results_path,
        output_dir / "degradation.png",
    )

    print(f"Generated degradation plots: {len(degradation_paths)}")

    # --------------------------------------------------------------
    # Test 6: Generate Pareto plot
    # --------------------------------------------------------------
    pareto_path = plot_pareto(
        results_path,
        output_dir / "pareto.png",
    )

    assert pareto_path is not None

    # --------------------------------------------------------------
    # Test 7: Generate summary tables
    # --------------------------------------------------------------
    table_paths = generate_tables(
        results_path,
        output_dir / "tables",
    )

    assert table_paths
    print(f"Generated summary tables: {len(table_paths)}")

    # --------------------------------------------------------------
    # Test 8: Test complete orchestration
    # --------------------------------------------------------------
    pipeline_dir = output_dir / "pipeline"

    artifacts = generate_all_figures(
        csv_path=results_path,
        figures_dir=pipeline_dir / "figures",
        tables_dir=pipeline_dir / "tables",
    )

    assert artifacts["degradation"]
    assert artifacts["pareto"] is not None
    assert artifacts["tables"]

    # --------------------------------------------------------------
    # Verify generated files
    # --------------------------------------------------------------
    expected_files = [
        keypoint_path,
        matches_path,
        *degradation_paths,
        pareto_path,
        *table_paths,
    ]

    missing = [
        path
        for path in expected_files
        if not path.is_file()
    ]

    if missing:
        raise AssertionError(
            f"Some expected output files were not created: {missing}"
        )

    print("\nAll visualization tests passed.")
    print(f"Generated outputs: {output_dir.resolve()}")


if __name__ == "__main__":
    run_visualization_tests()
