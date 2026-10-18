import os
import glob
import cv2
import numpy as np
from pathlib import Path
import config as cfg


# ------------------------------------------------------------------
# Image helpers
# ------------------------------------------------------------------

def _read_image(path, flags):
    """Read an image from a path, including paths with Unicode characters."""
    image_data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(image_data, flags)


def load_image_gray(path, max_side=None):
    """Load an image as grayscale, optionally cap its longest side.

    Returns
    -------
    gray : np.ndarray  (H, W) uint8
    scale : float      resize factor applied (1.0 if none)
    """
    img = _read_image(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    if max_side is None:
        max_side = cfg.MAX_RESOLUTION_SIDE

    h, w = img.shape[:2]
    longest = max(h, w)
    if longest > max_side:
        scale = max_side / longest
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0

    return img, scale


def load_image_color(path, max_side=None):
    """Same as above but returns BGR for stitching / visualisation."""
    img = _read_image(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    if max_side is None:
        max_side = cfg.MAX_RESOLUTION_SIDE

    h, w = img.shape[:2]
    longest = max(h, w)
    if longest > max_side:
        scale = max_side / longest
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0

    return img, scale


def correct_homography(H, scale1, scale2):
    """Correct a ground-truth homography when images have been resized.

    If img1 was scaled by s1 and img2 by s2, the corrected homography is:
        H' = S2 · H · S1⁻¹
    where S = diag(s, s, 1).
    """
    S1_inv = np.diag([1.0 / scale1, 1.0 / scale1, 1.0])
    S2 = np.diag([scale2, scale2, 1.0])
    return S2 @ H @ S1_inv


# ------------------------------------------------------------------
# HPatches loading
# ------------------------------------------------------------------

def load_homography(path):
    """Load a 3×3 homography from a text file (space-separated)."""
    return np.loadtxt(str(path)).reshape(3, 3)


def get_sequence_dirs(hpatches_dir=None):
    """Return sorted lists of (name, path) for i_* and v_* sequences."""
    if hpatches_dir is None:
        hpatches_dir = cfg.HPATCHES_DIR

    all_dirs = sorted(glob.glob(os.path.join(hpatches_dir, "i_*"))) + \
               sorted(glob.glob(os.path.join(hpatches_dir, "v_*")))

    sequences = []
    for d in all_dirs:
        name = os.path.basename(d)
        # Only include directories that actually have 1.ppm
        if os.path.isfile(os.path.join(d, "1.ppm")):
            sequences.append((name, d))
    return sequences


def split_tuning(sequences, tuning_names=None):
    """Split sequences into (test, tuning) based on names."""
    if tuning_names is None:
        tuning_names = cfg.TUNING_SEQUENCES
    test = [(n, p) for n, p in sequences if n not in tuning_names]
    tuning = [(n, p) for n, p in sequences if n in tuning_names]
    return test, tuning


def get_category(seq_name):
    """Return 'i' or 'v' from a sequence name like 'i_ajuntament'."""
    return seq_name[0]


def load_pair(seq_dir, k, max_side=None):
    """Load a (1, k) pair from a sequence directory.

    Returns
    -------
    img1, img2 : np.ndarray  grayscale
    H          : np.ndarray  3×3 (corrected for any resize)
    """
    img1, s1 = load_image_gray(os.path.join(seq_dir, "1.ppm"), max_side)
    img2, s2 = load_image_gray(os.path.join(seq_dir, f"{k}.ppm"), max_side)
    H_raw = load_homography(os.path.join(seq_dir, f"H_1_{k}"))
    H = correct_homography(H_raw, s1, s2)
    return img1, img2, H


def load_pair_color(seq_dir, k, max_side=None):
    """Load a (1, k) pair in colour."""
    img1, s1 = load_image_color(os.path.join(seq_dir, "1.ppm"), max_side)
    img2, s2 = load_image_color(os.path.join(seq_dir, f"{k}.ppm"), max_side)
    H_raw = load_homography(os.path.join(seq_dir, f"H_1_{k}"))
    H = correct_homography(H_raw, s1, s2)
    return img1, img2, H


def iter_pairs(sequences, max_side=None):
    """Yield (seq_name, category, k, img1, img2, H) for all pairs."""
    for seq_name, seq_dir in sequences:
        cat = get_category(seq_name)
        for k in cfg.TARGET_IMAGE_INDICES:
            img1, img2, H = load_pair(seq_dir, k, max_side)
            yield seq_name, cat, k, img1, img2, H


def load_stitching_image(path, max_side=None):
    """Colour input plus exact resize metadata; max_side=0 disables resizing.

    Part 1 loaders retain their existing return signatures.
    """
    max_side = cfg.PART2_MAX_SIDE if max_side is None else max_side
    if max_side < 0:
        raise ValueError('max_side cannot be negative')
    path = Path(path)
    image = _read_image(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Could not load {path}')
    h, w = image.shape[:2]
    if max_side and max(h, w) > max_side:
        scale = max_side / max(h, w)
        image = cv2.resize(image, (max(1, round(w*scale)), max(1, round(h*scale))),
                           interpolation=cv2.INTER_AREA)
    return image, {'path': str(path), 'original_width': w, 'original_height': h,
                   'processed_width': image.shape[1], 'processed_height': image.shape[0],
                   'scale_x': image.shape[1]/w, 'scale_y': image.shape[0]/h}
