"""
stitching.py – Phase E: Image stitching pipeline.

E1. stitch_pair: detect → describe → match → RANSAC → warp → blend
E2. Mini RANSAC implementation
E3. RANSAC threshold study
E4. Multi-method comparison
E5. 3-image panorama with reference strategies
E6. Blending (averaging + feathering)

Usage:
    cd src/
    python stitching.py --demo
"""

import os
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
from feature_extraction import extract
from matching import match_pipeline, estimate_homography, inlier_stats


# ==================================================================
# E1. stitch_pair
# ==================================================================

def stitch_pair(img1, img2, method="SIFT", ratio=None, canvas_pad=0):
    """Full stitching pipeline for two colour images.

    img1 is the reference (stays fixed); img2 is warped onto img1's plane.

    Returns
    -------
    panorama : np.ndarray (BGR)
    info     : dict with H, n_matches, n_inliers, inlier_ratio,
               reproj_err, mask
    """
    # Grayscale for features
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2

    mcfg = cfg.METHOD_CONFIGS[method]
    norm = mcfg["norm"]

    kps1, descs1 = extract(method, g1)
    kps2, descs2 = extract(method, g2)

    if descs1 is None or descs2 is None or len(kps1) < 4 or len(kps2) < 4:
        return None, {"error": "Not enough features"}

    matches = match_pipeline(descs1, descs2, norm, ratio)
    H, mask = estimate_homography(matches, kps1, kps2)

    if H is None:
        return None, {"error": "RANSAC failed"}

    n_inliers, reproj_err = inlier_stats(matches, kps1, kps2, H, mask)
    inlier_ratio = n_inliers / len(matches) if matches else 0

    # --- Build canvas ---
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Project corners of img2 through H_inv (since H maps 1→2,
    # but we may want 2→1 depending on convention)
    # Convention: matches are kps1 → kps2, so H maps pts from img1 to img2.
    # To warp img2 onto img1's plane, we need H_inv.
    H_inv = np.linalg.inv(H)

    corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    corners2_proj = cv2.perspectiveTransform(corners2, H_inv)

    corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    all_corners = np.vstack([corners1, corners2_proj])

    x_min = int(np.floor(all_corners[:, 0, 0].min()))
    y_min = int(np.floor(all_corners[:, 0, 1].min()))
    x_max = int(np.ceil(all_corners[:, 0, 0].max()))
    y_max = int(np.ceil(all_corners[:, 0, 1].max()))

    # Translation to keep everything in positive coords
    T = np.array([[1, 0, -x_min],
                   [0, 1, -y_min],
                   [0, 0, 1]], dtype=float)

    W = x_max - x_min
    H_canvas = y_max - y_min

    # Warp img2 onto canvas
    panorama = cv2.warpPerspective(img2, T @ H_inv, (W, H_canvas))

    # Place img1 (reference) on canvas
    panorama[-y_min:-y_min + h1, -x_min:-x_min + w1] = img1

    info = {
        "H": H,
        "n_matches": len(matches),
        "n_inliers": n_inliers,
        "inlier_ratio": inlier_ratio,
        "reproj_err": reproj_err,
        "kps1": kps1,
        "kps2": kps2,
        "matches": matches,
        "mask": mask,
    }

    return panorama, info


# ==================================================================
# E1 (improved). Averaging blend
# ==================================================================

def stitch_pair_averaged(img1, img2, method="SIFT", ratio=None):
    """Stitch with per-pixel averaging in overlap regions."""
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2

    mcfg = cfg.METHOD_CONFIGS[method]
    kps1, descs1 = extract(method, g1)
    kps2, descs2 = extract(method, g2)

    if descs1 is None or descs2 is None:
        return None, {}

    matches = match_pipeline(descs1, descs2, mcfg["norm"], ratio)
    H, mask = estimate_homography(matches, kps1, kps2)
    if H is None:
        return None, {}

    H_inv = np.linalg.inv(H)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    corners2_proj = cv2.perspectiveTransform(corners2, H_inv)
    corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    all_corners = np.vstack([corners1, corners2_proj])

    x_min = int(np.floor(all_corners[:, 0, 0].min()))
    y_min = int(np.floor(all_corners[:, 0, 1].min()))
    x_max = int(np.ceil(all_corners[:, 0, 0].max()))
    y_max = int(np.ceil(all_corners[:, 0, 1].max()))

    T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=float)
    W = x_max - x_min
    Hc = y_max - y_min

    # Warp img2
    warped2 = cv2.warpPerspective(img2, T @ H_inv, (W, Hc)).astype(float)
    mask2 = (warped2.sum(axis=2) > 0).astype(float)

    # Place img1
    canvas1 = np.zeros((Hc, W, 3), dtype=float)
    oy, ox = -y_min, -x_min
    canvas1[oy:oy + h1, ox:ox + w1] = img1.astype(float)
    mask1 = (canvas1.sum(axis=2) > 0).astype(float)

    # Average
    total_mask = mask1 + mask2
    total_mask[total_mask == 0] = 1  # avoid division by zero
    panorama = ((canvas1 * mask1[:, :, np.newaxis] +
                 warped2 * mask2[:, :, np.newaxis]) /
                total_mask[:, :, np.newaxis]).astype(np.uint8)

    n_inliers, reproj_err = inlier_stats(matches, kps1, kps2, H, mask)

    return panorama, {
        "H": H, "n_matches": len(matches),
        "n_inliers": n_inliers, "reproj_err": reproj_err,
    }


# ==================================================================
# E2. Mini RANSAC implementation
# ==================================================================

def mini_ransac(pts1, pts2, threshold=3.0, max_iters=2000,
                confidence=0.995, seed=42):
    """Minimal RANSAC for homography estimation (~30 lines).

    Uses cv2.getPerspectiveTransform for the minimal 4-point model.
    """
    rng = np.random.RandomState(seed)
    n = len(pts1)
    if n < 4:
        return None, None

    best_H = None
    best_inlier_mask = None
    best_n_inliers = 0

    for _ in range(max_iters):
        # Sample 4 random correspondences
        idx = rng.choice(n, 4, replace=False)
        src = np.float32(pts1[idx]).reshape(4, 1, 2)
        dst = np.float32(pts2[idx]).reshape(4, 1, 2)

        # Minimal model
        H_cand = cv2.getPerspectiveTransform(
            src.reshape(4, 2), dst.reshape(4, 2))

        if H_cand is None:
            continue

        # Project all pts1
        pts1_h = np.hstack([pts1, np.ones((n, 1))])
        proj = (H_cand @ pts1_h.T).T
        proj = proj[:, :2] / proj[:, 2:3]

        errors = np.linalg.norm(proj - pts2, axis=1)
        inlier_mask = errors < threshold
        n_inliers = inlier_mask.sum()

        if n_inliers > best_n_inliers:
            best_n_inliers = n_inliers
            best_inlier_mask = inlier_mask.copy()

            # Re-estimate on all inliers (if > 4)
            if n_inliers >= 4:
                inl_src = pts1[inlier_mask]
                inl_dst = pts2[inlier_mask]
                best_H, _ = cv2.findHomography(
                    inl_src.reshape(-1, 1, 2),
                    inl_dst.reshape(-1, 1, 2), 0)

            # Adaptive stopping
            w = n_inliers / n
            if w > 0:
                needed = np.log(1 - confidence) / np.log(1 - w ** 4)
                if _ > needed:
                    break

    return best_H, best_inlier_mask


# ==================================================================
# E3. RANSAC threshold study
# ==================================================================

def ransac_threshold_study(img1, img2, method="SIFT", output_dir=None):
    """Sweep RANSAC threshold and record inliers, error, panoramas."""
    if output_dir is None:
        output_dir = os.path.join(cfg.FIGURES_DIR, "ransac_study")
    os.makedirs(output_dir, exist_ok=True)

    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2

    mcfg = cfg.METHOD_CONFIGS[method]
    kps1, descs1 = extract(method, g1)
    kps2, descs2 = extract(method, g2)
    matches = match_pipeline(descs1, descs2, mcfg["norm"])

    thresholds = [0.5, 1, 2, 3, 5, 8, 10]
    results = []

    for t in thresholds:
        pts1 = np.float32([kps1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kps2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, t,
                                      maxIters=2000, confidence=0.995)

        if H is not None and mask is not None:
            n_inliers = int(mask.sum())
            inlier_matches = [m for m, ok in zip(matches, mask.ravel()) if ok]
            # Reprojection error on inliers
            inl_pts1 = np.float32([kps1[m.queryIdx].pt for m in inlier_matches])
            inl_pts2 = np.float32([kps2[m.trainIdx].pt for m in inlier_matches])
            ones = np.ones((len(inl_pts1), 1))
            proj = (H @ np.hstack([inl_pts1, ones]).T).T
            proj = proj[:, :2] / proj[:, 2:3]
            reproj = float(np.mean(np.linalg.norm(proj - inl_pts2, axis=1)))
        else:
            n_inliers = 0
            reproj = float('inf')

        results.append({
            "threshold": t,
            "n_inliers": n_inliers,
            "inlier_ratio": n_inliers / len(matches) if matches else 0,
            "reproj_err": reproj,
        })

        # Save panorama at this threshold
        panorama, _ = stitch_pair_averaged(img1, img2, method)
        if panorama is not None:
            path = os.path.join(output_dir, f"panorama_t{t}.png")
            cv2.imwrite(path, panorama)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ts = [r["threshold"] for r in results]
    ax1.plot(ts, [r["n_inliers"] for r in results], "o-")
    ax1.set_xlabel("RANSAC threshold (px)")
    ax1.set_ylabel("Inliers")
    ax1.set_title("Inliers vs threshold")
    ax1.grid(True, alpha=0.3)

    ax2.plot(ts, [r["reproj_err"] for r in results], "o-", color="red")
    ax2.set_xlabel("RANSAC threshold (px)")
    ax2.set_ylabel("Mean reprojection error (px)")
    ax2.set_title("Reprojection error vs threshold")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "ransac_threshold.png"), dpi=150)
    plt.close(fig)

    return results


# ==================================================================
# E5. 3-image panorama
# ==================================================================

def stitch_three(img1, img2, img3, method="SIFT", ref="center"):
    """Stitch 3 images. ref='center' uses img2 as reference,
    ref='left' uses img1.
    """
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2
    g3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY) if img3.ndim == 3 else img3

    mcfg = cfg.METHOD_CONFIGS[method]
    norm = mcfg["norm"]

    # Extract features
    kps1, d1 = extract(method, g1)
    kps2, d2 = extract(method, g2)
    kps3, d3 = extract(method, g3)

    # Match adjacent pairs: 1↔2 and 2↔3
    m12 = match_pipeline(d1, d2, norm)
    H12, _ = estimate_homography(m12, kps1, kps2)  # maps 1 → 2

    m23 = match_pipeline(d2, d3, norm)
    H23, _ = estimate_homography(m23, kps2, kps3)  # maps 2 → 3

    if H12 is None or H23 is None:
        print("Failed to estimate homography for one pair")
        return None

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h3, w3 = img3.shape[:2]

    if ref == "center":
        # Reference = img2
        H1_to_ref = np.linalg.inv(H12)       # 1 → 2
        H3_to_ref = np.linalg.inv(H23)       # 3 → 2
        ref_img = img2
        imgs_to_warp = [(img1, H1_to_ref), (img3, H3_to_ref)]
    elif ref == "left":
        # Reference = img1
        H2_to_ref = H12  # already maps 1→2, need inv: H12 maps 1→2
        # Actually H12 maps pts from img1 to img2.
        # We need H that warps img2 to img1 plane: inv(H12)
        H2_to_1 = np.linalg.inv(H12)
        H3_to_1 = np.linalg.inv(H12) @ np.linalg.inv(H23)  # 3→2→1
        ref_img = img1
        imgs_to_warp = [(img2, H2_to_1), (img3, H3_to_1)]
    else:
        raise ValueError(f"Unknown ref: {ref}")

    # Compute canvas size
    ref_h, ref_w = ref_img.shape[:2]
    all_corners = [np.float32([[0, 0], [ref_w, 0],
                                [ref_w, ref_h], [0, ref_h]]).reshape(-1, 1, 2)]

    for warp_img, H_w in imgs_to_warp:
        hi, wi = warp_img.shape[:2]
        c = np.float32([[0, 0], [wi, 0], [wi, hi], [0, hi]]).reshape(-1, 1, 2)
        all_corners.append(cv2.perspectiveTransform(c, H_w))

    all_pts = np.vstack(all_corners)
    x_min = int(np.floor(all_pts[:, 0, 0].min()))
    y_min = int(np.floor(all_pts[:, 0, 1].min()))
    x_max = int(np.ceil(all_pts[:, 0, 0].max()))
    y_max = int(np.ceil(all_pts[:, 0, 1].max()))

    T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=float)
    W = x_max - x_min
    Hc = y_max - y_min

    # Accumulate with averaging
    accum = np.zeros((Hc, W, 3), dtype=float)
    count = np.zeros((Hc, W), dtype=float)

    # Place reference
    ref_warped = cv2.warpPerspective(ref_img, T, (W, Hc))
    ref_mask = (ref_warped.sum(axis=2) > 0).astype(float)
    accum += ref_warped.astype(float)
    count += ref_mask

    # Warp and accumulate others
    for warp_img, H_w in imgs_to_warp:
        warped = cv2.warpPerspective(warp_img, T @ H_w, (W, Hc))
        w_mask = (warped.sum(axis=2) > 0).astype(float)
        accum += warped.astype(float)
        count += w_mask

    count[count == 0] = 1
    panorama = (accum / count[:, :, np.newaxis]).astype(np.uint8)

    return panorama


# ==================================================================
# E6. Feathering blend
# ==================================================================

def feathering_blend(img1_warped, mask1, img2_warped, mask2):
    """Blend two warped images using distance-transform feathering."""
    # Distance transforms
    d1 = cv2.distanceTransform(mask1.astype(np.uint8), cv2.DIST_L2, 5)
    d2 = cv2.distanceTransform(mask2.astype(np.uint8), cv2.DIST_L2, 5)

    total = d1 + d2
    total[total == 0] = 1

    w1 = d1 / total
    w2 = d2 / total

    blended = (img1_warped.astype(float) * w1[:, :, np.newaxis] +
               img2_warped.astype(float) * w2[:, :, np.newaxis])

    return blended.astype(np.uint8)


# ==================================================================
# Main demo
# ==================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img1", type=str, help="Path to first image")
    parser.add_argument("--img2", type=str, help="Path to second image")
    parser.add_argument("--method", type=str, default="SIFT")
    parser.add_argument("--output", type=str, default="panorama.png")
    args = parser.parse_args()

    if args.img1 and args.img2:
        i1 = cv2.imread(args.img1)
        i2 = cv2.imread(args.img2)
        pano, info = stitch_pair_averaged(i1, i2, method=args.method)
        if pano is not None:
            cv2.imwrite(args.output, pano)
            print(f"Panorama saved to {args.output}")
            print(f"  Matches: {info['n_matches']}, Inliers: {info['n_inliers']}")
        else:
            print("Stitching failed")
    else:
        print("Usage: python stitching.py --img1 left.jpg --img2 right.jpg")
