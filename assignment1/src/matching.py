"""
matching.py – Matching pipeline: kNN + ratio test + one-to-one.

Implements the A1 protocol:
  - kNN (k=2) + ratio test 0.8
  - one-to-one enforcement (keep best match per target keypoint)
  - Correct-match verification via ground-truth homography
"""

import cv2
import numpy as np

import config as cfg


# ------------------------------------------------------------------
# Core matching
# ------------------------------------------------------------------

def match_knn_ratio(desc1, desc2, norm_type, ratio=None, k=2):
    """kNN matching with Lowe's ratio test.

    Returns list of cv2.DMatch (only the best match per query that
    passes the ratio test).
    """
    if ratio is None:
        ratio = cfg.RATIO_TEST_THRESHOLD

    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return []

    bf = cv2.BFMatcher(norm_type)
    raw = bf.knnMatch(desc1, desc2, k=k)

    good = []
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def enforce_one_to_one(matches):
    """Remove duplicate target assignments, keeping the shortest distance.

    After ratio test, multiple query keypoints may match the same target.
    Keep only the best (lowest distance) for each trainIdx.
    """
    best = {}
    for m in matches:
        tid = m.trainIdx
        if tid not in best or m.distance < best[tid].distance:
            best[tid] = m
    return list(best.values())


def match_pipeline(desc1, desc2, norm_type, ratio=None):
    """Full A1 matching: kNN → ratio test → one-to-one."""
    matches = match_knn_ratio(desc1, desc2, norm_type, ratio)
    matches = enforce_one_to_one(matches)
    return matches


# ------------------------------------------------------------------
# Match correctness via ground-truth homography
# ------------------------------------------------------------------

def classify_matches(matches, kps1, kps2, H_gt, threshold_px=None):
    """Classify putative matches as correct or incorrect.

    A match (kp1, kp2) is correct if ‖H·kp1 − kp2‖ < threshold.

    Returns
    -------
    correct   : list of cv2.DMatch
    incorrect : list of cv2.DMatch
    errors    : np.ndarray of reprojection errors for each match
    """
    if threshold_px is None:
        threshold_px = cfg.CORRECT_MATCH_THRESHOLD_PX

    if len(matches) == 0:
        return [], [], np.array([])

    pts1 = np.array([kps1[m.queryIdx].pt for m in matches])
    pts2 = np.array([kps2[m.trainIdx].pt for m in matches])

    # Project pts1 through H_gt
    ones = np.ones((pts1.shape[0], 1))
    pts1_h = np.hstack([pts1, ones])
    proj = (H_gt @ pts1_h.T).T
    proj = proj[:, :2] / proj[:, 2:3]

    errors = np.linalg.norm(proj - pts2, axis=1)

    correct = [m for m, e in zip(matches, errors) if e < threshold_px]
    incorrect = [m for m, e in zip(matches, errors) if e >= threshold_px]

    return correct, incorrect, errors


# ------------------------------------------------------------------
# Homography estimation
# ------------------------------------------------------------------

def estimate_homography(matches, kps1, kps2, threshold=None, max_iters=None,
                        confidence=None, seed=None):
    """Estimate homography with RANSAC from putative matches.

    Returns
    -------
    H      : 3×3 np.ndarray or None
    mask   : inlier mask or None
    """
    if len(matches) < 4:
        return None, None

    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    if seed is not None:
        cv2.setRNGSeed(int(seed))
    H, mask = cv2.findHomography(
        pts1, pts2,
        method=cfg.RANSAC_METHOD,
        ransacReprojThreshold=cfg.RANSAC_REPROJ_THRESHOLD if threshold is None else threshold,
        maxIters=cfg.RANSAC_MAX_ITERS if max_iters is None else max_iters,
        confidence=cfg.RANSAC_CONFIDENCE if confidence is None else confidence,
    )
    return H, mask


def inlier_stats(matches, kps1, kps2, H_est, mask):
    """Compute inlier count and mean reprojection error on inliers."""
    if H_est is None or mask is None:
        return 0, float('inf')

    mask_flat = mask.ravel().astype(bool)
    n_inliers = int(mask_flat.sum())

    if n_inliers == 0:
        return 0, float('inf')

    inlier_matches = [m for m, ok in zip(matches, mask_flat) if ok]
    pts1 = np.float32([kps1[m.queryIdx].pt for m in inlier_matches])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in inlier_matches])

    ones = np.ones((len(pts1), 1))
    proj = (H_est @ np.hstack([pts1, ones]).T).T
    proj = proj[:, :2] / proj[:, 2:3]

    errors = np.linalg.norm(proj - pts2, axis=1)
    return n_inliers, float(np.mean(errors))


# ------------------------------------------------------------------
# B7 – Extra matching strategies
# ------------------------------------------------------------------

def match_nn_only(desc1, desc2, norm_type):
    """Nearest-neighbour only (no ratio test, no cross-check)."""
    if desc1 is None or desc2 is None:
        return []
    bf = cv2.BFMatcher(norm_type)
    return sorted(bf.match(desc1, desc2), key=lambda m: m.distance)


def match_cross_check(desc1, desc2, norm_type):
    """Cross-check matching (mutual best)."""
    if desc1 is None or desc2 is None:
        return []
    bf = cv2.BFMatcher(norm_type, crossCheck=True)
    return sorted(bf.match(desc1, desc2), key=lambda m: m.distance)


def match_ratio_only(desc1, desc2, norm_type, ratio=None):
    """Ratio test only (no one-to-one)."""
    return match_knn_ratio(desc1, desc2, norm_type, ratio)


def match_flann_sift(desc1, desc2, ratio=None):
    """FLANN with KD-tree for float descriptors (SIFT, KAZE)."""
    if ratio is None:
        ratio = cfg.RATIO_TEST_THRESHOLD
    if desc1 is None or desc2 is None:
        return []

    index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    desc1 = np.float32(desc1)
    desc2 = np.float32(desc2)
    raw = flann.knnMatch(desc1, desc2, k=2)

    good = []
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return enforce_one_to_one(good)


def match_flann_binary(desc1, desc2, ratio=None):
    """FLANN with LSH for binary descriptors (ORB, BRIEF, etc.)."""
    if ratio is None:
        ratio = cfg.RATIO_TEST_THRESHOLD
    if desc1 is None or desc2 is None:
        return []

    index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                        table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    raw = flann.knnMatch(desc1, desc2, k=2)

    good = []
    for pair in raw:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return enforce_one_to_one(good)


def match_mutual_ratio(desc1, desc2, norm_type, ratio=None):
    """Part 2: ratio test in BOTH directions, then mutual consistency.

    Part 1 match_pipeline keeps its original one-to-one filtering protocol.
    """
    ratio = cfg.RATIO_TEST_THRESHOLD if ratio is None else ratio
    if not 0 < ratio < 1:
        raise ValueError('ratio must be between zero and one')
    if desc1 is None or desc2 is None or min(len(desc1), len(desc2)) < 2:
        return []
    forward = match_knn_ratio(desc1, desc2, norm_type, ratio)
    reverse = match_knn_ratio(desc2, desc1, norm_type, ratio)
    backward = {(m.trainIdx, m.queryIdx) for m in reverse}
    return sorted((m for m in forward if (m.queryIdx, m.trainIdx) in backward),
                  key=lambda m: m.distance)
