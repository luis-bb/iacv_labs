"""
evaluation.py – All metrics for the HPatches benchmark (plan A2).

Metrics
-------
1. Co-visible keypoints
2. Repeatability (one-to-one, 3 px)
3. #features = min(N1_vis, N2_vis)
4. #correspondences = repeat_matches (max possible correct matches)
5. PMR, Precision, Matching Score, Recall  + sanity check
6. Homography accuracy (corner error ≤ 1, 3, 5 px)
7. Inlier stats
8. Memory footprint
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

import config as cfg


# ------------------------------------------------------------------
# 1. Co-visible keypoints
# ------------------------------------------------------------------

def get_covisible_indices(kps1, kps2, H, shape1, shape2):
    """Project keypoints through H and its inverse; keep those landing
    inside the other image.

    Returns (vis1_idx, vis2_idx) – integer index arrays.
    """
    h1, w1 = shape1[:2]
    h2, w2 = shape2[:2]

    pts1 = np.array([kp.pt for kp in kps1]) if len(kps1) else np.zeros((0, 2))
    pts2 = np.array([kp.pt for kp in kps2]) if len(kps2) else np.zeros((0, 2))

    if len(pts1) == 0 or len(pts2) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.array([], dtype=int), np.array([], dtype=int)

    # pts1 → img2
    ones1 = np.ones((len(pts1), 1))
    proj1 = (H @ np.hstack([pts1, ones1]).T).T
    proj1 = proj1[:, :2] / proj1[:, 2:3]
    mask1 = (proj1[:, 0] >= 0) & (proj1[:, 0] < w2) & \
            (proj1[:, 1] >= 0) & (proj1[:, 1] < h2)

    # pts2 → img1
    ones2 = np.ones((len(pts2), 1))
    proj2 = (H_inv @ np.hstack([pts2, ones2]).T).T
    proj2 = proj2[:, :2] / proj2[:, 2:3]
    mask2 = (proj2[:, 0] >= 0) & (proj2[:, 0] < w1) & \
            (proj2[:, 1] >= 0) & (proj2[:, 1] < h1)

    return np.where(mask1)[0], np.where(mask2)[0]


# ------------------------------------------------------------------
# 2. Repeatability (one-to-one via Hungarian)
# ------------------------------------------------------------------

def compute_repeatability(kps1, kps2, H, shape1, shape2,
                          threshold_px=None):
    """Repeatability with one-to-one assignment (linear_sum_assignment).

    Returns
    -------
    repeatability   : float
    n_repeat        : int   – number of repeated pairs (= correspondences)
    n1_vis, n2_vis  : int
    n_features      : int   – min(n1_vis, n2_vis)
    """
    if threshold_px is None:
        threshold_px = cfg.CORRECT_MATCH_THRESHOLD_PX

    vis1, vis2 = get_covisible_indices(kps1, kps2, H, shape1, shape2)
    n1_vis, n2_vis = len(vis1), len(vis2)

    if n1_vis == 0 or n2_vis == 0:
        return 0.0, 0, n1_vis, n2_vis, 0

    pts1 = np.array([kps1[i].pt for i in vis1])
    pts2 = np.array([kps2[i].pt for i in vis2])

    # Project pts1 → img2 through H
    ones = np.ones((len(pts1), 1))
    proj = (H @ np.hstack([pts1, ones]).T).T
    proj = proj[:, :2] / proj[:, 2:3]

    # Distance matrix: (n1_vis, n2_vis)
    dist = np.linalg.norm(proj[:, np.newaxis, :] - pts2[np.newaxis, :, :],
                          axis=2)

    # Hungarian (minimise total distance)
    row_ind, col_ind = linear_sum_assignment(dist)

    # Count valid (< threshold)
    n_repeat = int(sum(1 for r, c in zip(row_ind, col_ind)
                       if dist[r, c] < threshold_px))

    n_features = min(n1_vis, n2_vis)
    repeatability = n_repeat / n_features if n_features > 0 else 0.0

    return repeatability, n_repeat, n1_vis, n2_vis, n_features


# ------------------------------------------------------------------
# 5. PMR, Precision, Matching Score, Recall + sanity check
# ------------------------------------------------------------------

def compute_matching_metrics(n_putatives, n_correct, n_features,
                             n_correspondences):
    """Compute all four metrics and run the mandatory sanity check.

    Parameters
    ----------
    n_putatives      : int – number of putative matches (after one-to-one)
    n_correct        : int – number of correct matches (< 3 px via GT H)
    n_features       : int – min(N1_vis, N2_vis)
    n_correspondences: int – repeatability count (max possible correct)

    Returns dict with PMR, Precision, Matching_Score, Recall.
    """
    pmr = n_putatives / n_features if n_features > 0 else 0.0
    precision = n_correct / n_putatives if n_putatives > 0 else 0.0
    ms = n_correct / n_features if n_features > 0 else 0.0
    recall = n_correct / n_correspondences if n_correspondences > 0 else 0.0

    # Sanity checks (A2.6)
    assert abs(ms - pmr * precision) < 1e-9, \
        f"Sanity fail: MS={ms:.6f} != PMR*Prec={pmr*precision:.6f}"
    assert recall <= 1.0 + 1e-9, f"Recall > 1: {recall}"

    return {
        "n_putatives": n_putatives,
        "n_correct": n_correct,
        "n_features": n_features,
        "n_correspondences": n_correspondences,
        "PMR": pmr,
        "Precision": precision,
        "Matching_Score": ms,
        "Recall": recall,
    }


# ------------------------------------------------------------------
# 7. Homography accuracy (corner error)
# ------------------------------------------------------------------

def compute_homography_accuracy(H_est, H_gt, img_shape):
    """Mean corner error and accuracy at 1, 3, 5 px thresholds.

    If H_est is None (RANSAC failed), error = ∞ and accuracy = 0.
    """
    if H_est is None:
        return {
            "mean_corner_err": float('inf'),
            "HA@1": 0.0, "HA@3": 0.0, "HA@5": 0.0,
        }

    h, w = img_shape[:2]
    corners = np.array([
        [0,     0,     1],
        [w - 1, 0,     1],
        [w - 1, h - 1, 1],
        [0,     h - 1, 1],
    ], dtype=float).T  # (3, 4)

    proj_gt = H_gt @ corners
    proj_gt = proj_gt[:2] / proj_gt[2:3]

    proj_est = H_est @ corners
    proj_est = proj_est[:2] / proj_est[2:3]

    errs = np.linalg.norm(proj_gt - proj_est, axis=0)  # (4,)
    mean_err = float(np.mean(errs))

    return {
        "mean_corner_err": mean_err,
        "HA@1": float(mean_err <= 1.0) * 100,
        "HA@3": float(mean_err <= 3.0) * 100,
        "HA@5": float(mean_err <= 5.0) * 100,
    }


# ------------------------------------------------------------------
# 9. Memory footprint
# ------------------------------------------------------------------

def compute_memory(n_kps, method_name):
    """N_kps × bytes_per_descriptor."""
    mcfg = cfg.METHOD_CONFIGS.get(method_name, {})
    bpd = mcfg.get("bytes_per_desc", 64)
    return n_kps * bpd


# ------------------------------------------------------------------
# Full evaluation for one pair
# ------------------------------------------------------------------

def evaluate_pair(kps1, descs1, kps2, descs2, H_gt, shape1, shape2,
                  matches, correct_matches, incorrect_matches,
                  H_est, ransac_mask, method_name):
    """Run all metrics for a single image pair.

    Returns a flat dict ready for CSV export.
    """
    # Repeatability and co-visibility
    rep, n_repeat, n1v, n2v, n_feat = compute_repeatability(
        kps1, kps2, H_gt, shape1, shape2)

    n_correspondences = n_repeat  # A2.4

    # Matching metrics
    n_putatives = len(matches)
    n_correct = len(correct_matches)
    mm = compute_matching_metrics(n_putatives, n_correct,
                                  n_feat, n_correspondences)

    # Homography accuracy
    ha = compute_homography_accuracy(H_est, H_gt, shape1)

    # Inlier stats
    from matching import inlier_stats
    n_inliers, reproj_err = inlier_stats(matches, kps1, kps2,
                                          H_est, ransac_mask)

    # Memory
    mem = compute_memory(len(kps1), method_name)

    result = {
        "n_kps1": len(kps1),
        "n_kps2": len(kps2),
        "n1_vis": n1v,
        "n2_vis": n2v,
        "n_features": n_feat,
        "repeatability": rep,
        "n_correspondences": n_correspondences,
    }
    result.update(mm)
    result.update(ha)
    result.update({
        "n_inliers": n_inliers,
        "reproj_err_inliers": reproj_err,
        "memory_bytes": mem,
    })
    return result
