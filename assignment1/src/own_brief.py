"""
own_brief.py – Custom BRIEF descriptor (Calonder et al. 2012).

Sampling strategies
-------------------
G I   : both points uniform over the patch
G II  : both points isotropic Gaussian around the centre, σ² = S²/25
G III : first point Gaussian σ² = S²/25, second Gaussian around the
        first with σ² = S²/100

Preprocessing: Gaussian blur 9×9, σ = 2 (paper setting).
Patch size S = 48, default 256 tests (32 bytes).
"""

import numpy as np
import cv2


# ------------------------------------------------------------------
# Pair generation (deterministic, saved to disk, reused)
# ------------------------------------------------------------------

def generate_pairs(strategy, n_bits=256, patch_size=48, seed=42):
    """Generate (n_bits, 4) array of (x1, y1, x2, y2) offsets.

    Each offset is in [-S/2, S/2) relative to patch centre.
    """
    rng = np.random.RandomState(seed)
    half = patch_size // 2
    sigma_main = patch_size / 5.0         # σ² = S²/25  → σ = S/5
    sigma_fine = patch_size / 10.0        # σ² = S²/100 → σ = S/10

    if strategy == "GI":
        # Both uniform
        pairs = rng.uniform(-half, half, size=(n_bits, 4))
    elif strategy == "GII":
        # Both isotropic Gaussian, clipped to patch
        pairs = rng.normal(0, sigma_main, size=(n_bits, 4))
    elif strategy == "GIII":
        # First point: Gaussian σ_main; second: Gaussian around first, σ_fine
        xy1 = rng.normal(0, sigma_main, size=(n_bits, 2))
        xy2 = xy1 + rng.normal(0, sigma_fine, size=(n_bits, 2))
        pairs = np.hstack([xy1, xy2])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Clip to patch and round to int
    pairs = np.clip(pairs, -half, half - 1).round().astype(np.int32)
    return pairs


def save_pairs(pairs, path):
    np.save(path, pairs)


def load_pairs(path):
    return np.load(path)


# ------------------------------------------------------------------
# Descriptor computation
# ------------------------------------------------------------------

def compute_own_brief(gray, keypoints, pairs, patch_size=48, smooth=True):
    """Compute own BRIEF descriptors for a list of cv2.KeyPoint.

    Parameters
    ----------
    gray       : (H, W) uint8 image (already grayscale)
    keypoints  : list of cv2.KeyPoint
    pairs      : (n_bits, 4) int32 array of offsets
    patch_size : int
    smooth     : bool – apply Gaussian blur (paper default: True)

    Returns
    -------
    kept_kps   : list of cv2.KeyPoint (border-safe)
    descriptors: (N, n_bits//8) uint8 array, or empty
    """
    if smooth:
        gray = cv2.GaussianBlur(gray, (9, 9), 2)

    h, w = gray.shape[:2]
    half = patch_size // 2
    n_bits = pairs.shape[0]
    n_bytes = n_bits // 8

    # Vectorise: collect all valid keypoints first
    kept_kps = []
    centres = []
    for kp in keypoints:
        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
        if x < half or y < half or x >= w - half or y >= h - half:
            continue
        kept_kps.append(kp)
        centres.append((x, y))

    if len(centres) == 0:
        return [], np.empty((0, n_bytes), dtype=np.uint8)

    centres = np.array(centres, dtype=np.int32)  # (N, 2)

    # Pixel coordinates for all keypoints × all pairs
    # p coords: centre + (pairs[:, 0], pairs[:, 1])
    # q coords: centre + (pairs[:, 2], pairs[:, 3])
    px = centres[:, 0:1] + pairs[np.newaxis, :, 0]   # (N, n_bits)
    py = centres[:, 1:2] + pairs[np.newaxis, :, 1]
    qx = centres[:, 0:1] + pairs[np.newaxis, :, 2]
    qy = centres[:, 1:2] + pairs[np.newaxis, :, 3]

    # Clip to image bounds (safety, though border check above should suffice)
    px = np.clip(px, 0, w - 1)
    py = np.clip(py, 0, h - 1)
    qx = np.clip(qx, 0, w - 1)
    qy = np.clip(qy, 0, h - 1)

    # Intensity lookups
    p_vals = gray[py, px]  # (N, n_bits)
    q_vals = gray[qy, qx]

    # Bit rule: bit = 1 if I(p) < I(q)
    bits = (p_vals < q_vals).astype(np.uint8)  # (N, n_bits)

    # Pack into bytes
    descriptors = np.packbits(bits, axis=1)  # (N, ceil(n_bits/8))
    descriptors = descriptors[:, :n_bytes]   # exact trim

    return kept_kps, descriptors


# ------------------------------------------------------------------
# Convenience: detect FAST + compute own BRIEF
# ------------------------------------------------------------------

def detect_and_compute_own_brief(gray, strategy="GII", n_bits=256,
                                  patch_size=48, smooth=True,
                                  fast_threshold=20, seed=42,
                                  max_keypoints=None,
                                  fast_kp_size=31.0):
    """Full pipeline: FAST detection → own BRIEF description.

    Returns (keypoints, descriptors).
    """
    # FAST detection
    fast = cv2.FastFeatureDetector_create(
        threshold=fast_threshold,
        nonmaxSuppression=True,
        type=cv2.FastFeatureDetector_TYPE_9_16,
    )
    kps = fast.detect(gray, None)

    # Set kp.size for consistency
    for kp in kps:
        kp.size = fast_kp_size

    # Generate (or reuse) pairs
    pairs = generate_pairs(strategy, n_bits=n_bits,
                           patch_size=patch_size, seed=seed)

    # Compute descriptor
    kps, descs = compute_own_brief(gray, kps, pairs,
                                    patch_size=patch_size, smooth=smooth)

    # Budget
    if max_keypoints is not None and len(kps) > max_keypoints:
        # Sort by response (FAST score)
        indices = np.argsort([-kp.response for kp in kps])[:max_keypoints]
        kps = [kps[i] for i in indices]
        descs = descs[indices]

    return kps, descs
