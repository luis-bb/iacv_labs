"""
evaluation.py — Quantitative metrics for the HPatches benchmark.

Implemented metrics
-------------------
1. Co-visible keypoints
2. Repeatability (one-to-one assignment, geometric threshold)
3. Feature count: min(N1_visible, N2_visible)
4. Maximum repeatable correspondences
5. PMR, precision, matching score, recall, and consistency checks
6. Homography corner accuracy at configurable thresholds
7. RANSAC inlier statistics and inlier reprojection error
8. Runtime and descriptor-memory helpers

Conventions
-----------
- Keypoint coordinates are (x, y) in the coordinate system of the images
  passed to this module.
- H maps image 1 coordinates to image 2 coordinates.
- Correctness uses forward transfer error <= threshold_px.
- Co-visibility means the projected keypoint centre falls inside the other
  image; it does not model occlusion or the full keypoint support region.
- Recall uses the maximum number of repeatable pairs as its denominator.
- PMR = putative_matches / min(N1_visible, N2_visible).
- Matching score = correct_matches / min(N1_visible, N2_visible).
"""

from time import perf_counter

import numpy as np
from scipy.optimize import linear_sum_assignment
try:
    import config as cfg
except ImportError:
    from src import config as cfg

DEFAULT_THRESHOLD_PX = getattr(cfg, "CORRECT_MATCH_THRESHOLD_PX", 3.0)
threshold_px = DEFAULT_THRESHOLD_PX


# ------------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------------

def _keypoint_points(keypoints):
    """Return keypoint coordinates as an (N, 2) float64 array."""
    if not keypoints:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray([kp.pt for kp in keypoints], dtype=np.float64).reshape(-1, 2)


def _project_points(points, H, eps=1e-12):
    """Project (N, 2) points with H; invalid projections become NaN."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float64)

    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3) or not np.all(np.isfinite(H)):
        raise ValueError("H must be a finite 3x3 homography.")

    homogeneous = np.column_stack((points, np.ones(len(points))))
    projected = (H @ homogeneous.T).T
    denominator = projected[:, 2]

    result = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(projected).all(axis=1) & (np.abs(denominator) > eps)
    result[valid] = projected[valid, :2] / denominator[valid, None]
    result[~np.isfinite(result).all(axis=1)] = np.nan
    return result


def _inside_image(points, shape):
    """Boolean mask for finite points whose centres lie inside image bounds."""
    height, width = shape[:2]
    finite = np.isfinite(points).all(axis=1)
    return (
        finite
        & (points[:, 0] >= 0)
        & (points[:, 0] < width)
        & (points[:, 1] >= 0)
        & (points[:, 1] < height)
    )


# ------------------------------------------------------------------
# 1. Co-visible keypoints
# ------------------------------------------------------------------

def get_covisible_indices(kps1, kps2, H, shape1, shape2):
    """
    Return indices of keypoints whose projected centres lie inside the
    opposite image.

    Returns
    -------
    vis1_idx, vis2_idx : np.ndarray[int]
        Indices of keypoints in image 1 and image 2, respectively.
        Each image is handled independently, including when the other has
        no keypoints.
    """
    pts1 = _keypoint_points(kps1)
    pts2 = _keypoint_points(kps2)

    try:
        H = np.asarray(H, dtype=np.float64)
        if H.shape != (3, 3) or not np.all(np.isfinite(H)):
            return np.empty(0, dtype=int), np.empty(0, dtype=int)
        H_inv = np.linalg.inv(H)
    except (np.linalg.LinAlgError, ValueError):
        return np.empty(0, dtype=int), np.empty(0, dtype=int)

    projected_1_to_2 = _project_points(pts1, H)
    projected_2_to_1 = _project_points(pts2, H_inv)

    vis1 = np.flatnonzero(_inside_image(projected_1_to_2, shape2))
    vis2 = np.flatnonzero(_inside_image(projected_2_to_1, shape1))
    return vis1, vis2


# ------------------------------------------------------------------
# 2. Repeatability (one-to-one assignment)
# ------------------------------------------------------------------

def compute_repeatability(kps1, kps2, H, shape1, shape2, threshold_px=None):

    if threshold_px is None:
        threshold_px = getattr(cfg, "CORRECT_MATCH_THRESHOLD_PX", DEFAULT_THRESHOLD_PX)
    if threshold_px <= 0:
        raise ValueError("threshold_px must be positive.")

    vis1, vis2 = get_covisible_indices(kps1, kps2, H, shape1, shape2)
    n1_vis, n2_vis = len(vis1), len(vis2)
    n_features = min(n1_vis, n2_vis)

    if n1_vis == 0 or n2_vis == 0:
        return 0.0, 0, n1_vis, n2_vis, n_features

    pts1 = _keypoint_points(kps1)[vis1]
    pts2 = _keypoint_points(kps2)[vis2]
    projected = _project_points(pts1, H)

    # Invalid projections are not expected after co-visibility filtering,
    # but handle them defensively.
    dist = np.linalg.norm(projected[:, None, :] - pts2[None, :, :], axis=2)
    dist[~np.isfinite(dist)] = np.inf
    valid = dist <= threshold_px

    # Every valid edge costs at most threshold_px. A penalty greater than
    # min(n1_vis, n2_vis) * threshold_px makes gaining one extra valid pair
    # preferable to any possible change in the total valid-edge distance.
    max_valid_pairs = min(n1_vis, n2_vis)
    penalty = (max_valid_pairs + 1) * (threshold_px + 1.0)
    cost = np.where(valid, dist, penalty)

    rows, cols = linear_sum_assignment(cost)
    n_repeat = int(np.count_nonzero(valid[rows, cols]))
    repeatability = n_repeat / n_features if n_features else 0.0

    return repeatability, n_repeat, n1_vis, n2_vis, n_features


# ------------------------------------------------------------------
# 3. Geometric correctness of descriptor matches
# ------------------------------------------------------------------

def count_correct_matches(
    keypoints1, keypoints2, matches, H, threshold_px=None
):
    """
    Count geometrically correct putative matches using ground-truth H.

    Parameters
    ----------
    keypoints1, keypoints2
        Keypoint lists used to produce `matches`.
    matches
        OpenCV DMatch objects, e.g. after KNN ratio filtering and any
        one-to-one filtering required by the chosen protocol.
    H
        Ground-truth homography mapping image 1 to image 2.
    threshold_px
        A match is correct when forward transfer error <= threshold_px.

    Returns
    -------
    n_correct : int
    errors_px : np.ndarray
        Forward transfer errors for each input match, in match order.
    correct_mask : np.ndarray[bool]
    """
    if threshold_px is None:
        threshold_px = getattr(cfg, "CORRECT_MATCH_THRESHOLD_PX", DEFAULT_THRESHOLD_PX)
    if threshold_px <= 0:
        raise ValueError("threshold_px must be positive.")

    if not matches:
        return 0, np.empty(0, dtype=float), np.empty(0, dtype=bool)

    pts1 = np.asarray(
        [keypoints1[m.queryIdx].pt for m in matches], dtype=np.float64
    )
    pts2 = np.asarray(
        [keypoints2[m.trainIdx].pt for m in matches], dtype=np.float64
    )
    projected = _project_points(pts1, H)
    errors = np.linalg.norm(projected - pts2, axis=1)
    correct = np.isfinite(errors) & (errors <= threshold_px)
    return int(correct.sum()), errors, correct


# ------------------------------------------------------------------
# 4. PMR, precision, matching score, recall + sanity checks
# ------------------------------------------------------------------

def compute_matching_metrics(
    n_putatives, n_correct, n_features, n_correspondences
):
    """
    Calculate PMR, precision, matching score and recall.

    Definitions
    -----------
    PMR = n_putatives / n_features
    Precision = n_correct / n_putatives
    Matching_Score = n_correct / n_features
    Recall = n_correct / n_correspondences

    Here n_correspondences is the maximum number of repeatable pairs
    computed by compute_repeatability().

    Zero-denominator metrics are returned as 0.0. Counts must follow the
    same one-to-one matching protocol and geometric threshold.
    """
    counts = {
        "n_putatives": n_putatives,
        "n_correct": n_correct,
        "n_features": n_features,
        "n_correspondences": n_correspondences,
    }
    for name, value in counts.items():
        if not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer.")

    if n_correct > n_putatives:
        raise ValueError("n_correct cannot exceed n_putatives.")
    if n_correct > n_correspondences:
        raise ValueError(
            "n_correct exceeds n_correspondences. Check one-to-one matching, "
            "co-visibility, and whether both use the same geometric threshold."
        )

    pmr = n_putatives / n_features if n_features else 0.0
    precision = n_correct / n_putatives if n_putatives else 0.0
    matching_score = n_correct / n_features if n_features else 0.0
    recall = n_correct / n_correspondences if n_correspondences else 0.0

    if not np.isclose(matching_score, pmr * precision, atol=1e-9, rtol=0.0):
        raise AssertionError(
            "Metric consistency check failed: Matching_Score != PMR * Precision."
        )
    if recall > 1.0 + 1e-9:
        raise AssertionError(f"Recall exceeds 1: {recall}")

    return {
        **counts,
        "PMR": pmr,
        "Precision": precision,
        "Matching_Score": matching_score,
        "Recall": recall,
    }


# ------------------------------------------------------------------
# 5. Homography accuracy
# ------------------------------------------------------------------

def homography_corner_errors(H_est, H_gt, image_shape):
    """
    Compare estimated and ground-truth homographies using forward transfer
    errors at the four image corners.

    H_est and H_gt must map image 1 coordinates to image 2 coordinates.
    image_shape is the shape of image 1.

    Returns the four Euclidean corner transfer errors in pixels.
    """
    if H_est is None or H_gt is None:
        return np.full(4, np.nan, dtype=float)

    height, width = image_shape[:2]
    corners = np.array(
        [
            [0.0, 0.0],
            [width - 1.0, 0.0],
            [width - 1.0, height - 1.0],
            [0.0, height - 1.0],
        ],
        dtype=np.float64,
    )

    try:
        est = _project_points(corners, H_est)
        gt = _project_points(corners, H_gt)
    except (ValueError, np.linalg.LinAlgError):
        return np.full(4, np.nan, dtype=float)

    return np.linalg.norm(est - gt, axis=1)


def homography_accuracy(H_est, H_gt, image_shape, thresholds=(1, 3, 5)):
    """
    Summarize homography corner transfer error.

    The per-pair corner error is the mean Euclidean transfer error over the
    four image-1 corners. Each threshold result is a Boolean indicating
    whether this pair's mean corner error is <= that threshold. Aggregate
    these Booleans across pairs to report benchmark accuracy percentages.
    """
    errors = homography_corner_errors(H_est, H_gt, image_shape)
    finite = np.isfinite(errors)
    mean_error = float(errors.mean()) if finite.all() else float("nan")

    result = {
        "corner_errors_px": errors,
        "mean_corner_error_px": mean_error,
        "max_corner_error_px": (
            float(errors.max()) if finite.all() else float("nan")
        ),
    }
    for threshold in thresholds:
        result[f"homography_correct_le_{threshold}px"] = (
            bool(mean_error <= threshold) if np.isfinite(mean_error) else False
        )
    return result


# ------------------------------------------------------------------
# 6. RANSAC inlier statistics
# ------------------------------------------------------------------

def compute_inlier_metrics(
    keypoints1, keypoints2, matches, inlier_mask, H_est=None
):
    """
    Calculate inlier count/ratio and mean forward reprojection error.

    `inlier_mask` is the mask returned by cv2.findHomography for the exact
    match list passed to that function.
    """
    n_matches = len(matches)
    if inlier_mask is None:
        mask = np.zeros(n_matches, dtype=bool)
    else:
        mask = np.asarray(inlier_mask).reshape(-1).astype(bool)
        if len(mask) != n_matches:
            raise ValueError("inlier_mask length must equal len(matches).")

    n_inliers = int(mask.sum())
    result = {
        "n_ransac_matches": n_matches,
        "n_ransac_inliers": n_inliers,
        "ransac_inlier_ratio": n_inliers / n_matches if n_matches else 0.0,
        "mean_inlier_reprojection_error_px": float("nan"),
    }

    if H_est is not None and n_inliers:
        inlier_matches = [m for m, keep in zip(matches, mask) if keep]
        _, errors, _ = count_correct_matches(
            keypoints1, keypoints2, inlier_matches, H_est, threshold_px=float("inf")
        )
        finite = np.isfinite(errors)
        if finite.any():
            result["mean_inlier_reprojection_error_px"] = float(errors[finite].mean())

    return result


# ------------------------------------------------------------------
# 7. Runtime and descriptor-memory helpers
# ------------------------------------------------------------------

def time_call(function, *args, repeats=None, **kwargs):
    """
    Time a callable repeatedly and return (last_result, timing_dict).

    `function` should perform only the stage being timed, so detection,
    description, and matching can be measured independently.
    """
    if repeats is None:
        repeats = getattr(cfg, "TIMING_REPEATS", 5)
    if repeats < 1:
        raise ValueError("repeats must be at least 1.")

    durations = []
    result = None
    for _ in range(repeats):
        start = perf_counter()
        result = function(*args, **kwargs)
        durations.append(perf_counter() - start)

    return result, {
        "runtime_mean_s": float(np.mean(durations)),
        "runtime_std_s": float(np.std(durations)),
        "runtime_median_s": float(np.median(durations)),
        "runtime_repeats": repeats,
    }


def descriptor_memory(descriptors):
    """
    Estimate descriptor-array storage in bytes.

    This is the NumPy array's payload size; it excludes keypoint objects,
    Python/OpenCV overhead, and temporary matching allocations.
    """
    if descriptors is None:
        return {
            "descriptor_count": 0,
            "descriptor_dim": 0,
            "descriptor_dtype": None,
            "descriptor_bytes_each": 0,
            "descriptor_memory_bytes": 0,
        }

    descriptors = np.asarray(descriptors)
    if descriptors.ndim != 2:
        raise ValueError("descriptors must be a 2D array.")

    n_descriptors, dimension = descriptors.shape
    bytes_each = int(dimension * descriptors.dtype.itemsize)
    return {
        "descriptor_count": int(n_descriptors),
        "descriptor_dim": int(dimension),
        "descriptor_dtype": str(descriptors.dtype),
        "descriptor_bytes_each": bytes_each,
        "descriptor_memory_bytes": int(descriptors.nbytes),
    }


# ------------------------------------------------------------------
# 8. Standalone testing
# ------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path
    import cv2

    threshold_px = getattr(cfg, "CORRECT_MATCH_THRESHOLD_PX", DEFAULT_THRESHOLD_PX)

    data = Path("hpatches-sequences-release/i_ajuntament/")

    image_path1 = data / "1.ppm"
    image_path2 = data / "2.ppm" 

    print("=" * 70)
    print("EVALUATION MODULE TEST")
    print("=" * 70)
    print(f"Image 1: {image_path1}")
    print(f"Image 2: {image_path2}")

    # --------------------------------------------------------------
    # Load images
    # --------------------------------------------------------------
    image1 = cv2.imread(str(image_path1), cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(str(image_path2), cv2.IMREAD_GRAYSCALE)

    if image1 is None or image2 is None:
        raise RuntimeError("Failed to load one or both images.")

    print(f"Image 1 shape: {image1.shape}")
    print(f"Image 2 shape: {image2.shape}")

    # --------------------------------------------------------------
    # Detect keypoints and compute descriptors
    # --------------------------------------------------------------
    sift = cv2.SIFT_create()

    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    print("\n--- Feature Detection ---")
    print(f"Keypoints in image 1: {len(keypoints1)}")
    print(f"Keypoints in image 2: {len(keypoints2)}")

    if descriptors1 is None or descriptors2 is None:
        raise RuntimeError("No descriptors were detected in one or both images.")

    # Test descriptor-memory helpers.
    print("\n--- Descriptor Memory ---")
    print("Image 1:", descriptor_memory(descriptors1))
    print("Image 2:", descriptor_memory(descriptors2))

    # --------------------------------------------------------------
    # Match descriptors
    # --------------------------------------------------------------
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    def match_descriptors():
        knn_matches = matcher.knnMatch(
            descriptors1, descriptors2, k=2
        )

        return [
            m
            for pair in knn_matches
            if len(pair) == 2
            for m, n in [pair]
            if m.distance < 0.75 * n.distance
        ]

    matches, timing = time_call(match_descriptors)

    # Enforce one-to-one matching.
    matches.sort(key=lambda m: m.distance)

    unique_matches = []
    used_query = set()
    used_train = set()

    for match in matches:
        if (
            match.queryIdx not in used_query
            and match.trainIdx not in used_train
        ):
            unique_matches.append(match)
            used_query.add(match.queryIdx)
            used_train.add(match.trainIdx)

    matches = unique_matches

    print("\n--- Descriptor Matching ---")
    print(f"Ratio-filtered, one-to-one matches: {len(matches)}")
    print(f"Matching runtime: {timing['runtime_mean_s']:.6f} s")

    if len(matches) < 4:
        raise RuntimeError(
            "At least four matches are required to estimate a homography."
        )

    # --------------------------------------------------------------
    # Estimate homography using RANSAC
    # --------------------------------------------------------------
    points1 = np.float32(
        [keypoints1[m.queryIdx].pt for m in matches]
    ).reshape(-1, 1, 2)

    points2 = np.float32(
        [keypoints2[m.trainIdx].pt for m in matches]
    ).reshape(-1, 1, 2)

    H_est, inlier_mask = cv2.findHomography(
        points1,
        points2,
        cv2.RANSAC,
        threshold_px,
    )

    if H_est is None or inlier_mask is None:
        raise RuntimeError("RANSAC failed to estimate a homography.")

    inlier_mask = inlier_mask.ravel().astype(bool)

    print("\n--- Estimated Homography ---")
    print(H_est)

    # --------------------------------------------------------------
    # Test co-visibility and repeatability
    # --------------------------------------------------------------
    vis1, vis2 = get_covisible_indices(
        keypoints1,
        keypoints2,
        H_est,
        image1.shape,
        image2.shape,
    )

    (
        repeatability,
        n_repeat,
        n1_visible,
        n2_visible,
        n_features,
    ) = compute_repeatability(
        keypoints1,
        keypoints2,
        H_est,
        image1.shape,
        image2.shape,
        threshold_px,
    )

    print("\n--- Co-Visibility ---")
    print(f"Visible keypoints in image 1: {len(vis1)}")
    print(f"Visible keypoints in image 2: {len(vis2)}")

    print("\n--- Repeatability ---")
    print(f"Visible keypoints in image 1: {n1_visible}")
    print(f"Visible keypoints in image 2: {n2_visible}")
    print(f"Feature count: {n_features}")
    print(f"Repeatable correspondences: {n_repeat}")
    print(f"Repeatability: {repeatability:.4f}")

    # --------------------------------------------------------------
    # Evaluate geometrically correct descriptor matches
    # --------------------------------------------------------------
    visible_matches = [
        m for m in matches
        if m.queryIdx in set(vis1)
        and m.trainIdx in set(vis2)
    ]

    n_correct, errors, correct_mask = count_correct_matches(
        keypoints1,
        keypoints2,
        visible_matches,
        H_est,
        threshold_px,
    )

    n_putatives = len(visible_matches)

    print("\n--- Geometric Match Evaluation ---")
    print(f"Putative matches: {n_putatives}")
    print(f"Correct matches: {n_correct}")

    if len(errors):
        finite_errors = errors[np.isfinite(errors)]

        if len(finite_errors):
            print(
                f"Mean forward transfer error: "
                f"{finite_errors.mean():.4f} px"
            )

    # The matching metrics include a consistency check requiring
    # n_correct <= n_correspondences. With an estimated homography,
    # this is not guaranteed, so report any inconsistency explicitly.
    if n_correct <= n_repeat:
        matching_metrics = compute_matching_metrics(
            n_putatives=n_putatives,
            n_correct=n_correct,
            n_features=n_features,
            n_correspondences=n_repeat,
        )

        print("\n--- Matching Metrics ---")
        for name, value in matching_metrics.items():
            print(f"{name}: {value}")
    else:
        print(
            "\nMatching metrics skipped: the number of correct matches "
            "exceeds the repeatability-based correspondence count. "
            "Check the estimated homography and matching protocol."
        )

    # --------------------------------------------------------------
    # Test RANSAC inlier statistics
    # --------------------------------------------------------------
    inlier_metrics = compute_inlier_metrics(
        keypoints1,
        keypoints2,
        matches,
        inlier_mask,
        H_est,
    )

    print("\n--- RANSAC Inlier Metrics ---")
    for name, value in inlier_metrics.items():
        print(f"{name}: {value}")

    # --------------------------------------------------------------
    # Test homography accuracy helper
    # --------------------------------------------------------------
    # No ground-truth homography is available here, so compare H_est
    # against itself to verify the function's expected behaviour.
    accuracy = homography_accuracy(
        H_est,
        H_est,
        image1.shape,
    )

    print("\n--- Homography Accuracy (Sanity Check) ---")
    for name, value in accuracy.items():
        print(f"{name}: {value}")

    print("\n" + "=" * 70)
    print("EVALUATION TEST COMPLETED")
    print("=" * 70)