"""
feature_extraction.py – Unified interface for all detector/descriptor methods.

Usage:
    kps, descs = extract("SIFT", gray_image)
    kps, descs = extract("FAST+BRISK", gray_image)
"""

import cv2
import numpy as np

import config as cfg
from own_brief import detect_and_compute_own_brief


# ------------------------------------------------------------------
# Detector factories
# ------------------------------------------------------------------

def _make_sift(params=None):
    p = params or cfg.DETECTOR_PARAMS["SIFT"]
    return cv2.xfeatures2d.SIFT_create(
        nfeatures=0,  # no cap here, we do it ourselves
        nOctaveLayers=p["nOctaveLayers"],
        contrastThreshold=p["contrastThreshold"],
        edgeThreshold=p["edgeThreshold"],
        sigma=p["sigma"],
    )

def _make_kaze(params=None):
    p = params or cfg.DETECTOR_PARAMS["KAZE"]
    return cv2.xfeatures2d.KAZE_create(
        threshold=p["threshold"],
        nOctaves=p["nOctaves"],
        nOctaveLayers=p["nOctaveLayers"],
    )

def _make_orb(params=None):
    p = params or cfg.DETECTOR_PARAMS["ORB"]
    return cv2.ORB_create(
        nfeatures=p["nfeatures"],
        scaleFactor=p["scaleFactor"],
        nlevels=p["nlevels"],
        edgeThreshold=p["edgeThreshold"],
        firstLevel=0,
        WTA_K=p["WTA_K"],
        scoreType=p["scoreType"],
        patchSize=p["patchSize"],
        fastThreshold=p["fastThreshold"],
    )

def _make_fast(params=None):
    p = params or cfg.DETECTOR_PARAMS["FAST"]
    return cv2.FastFeatureDetector_create(
        threshold=p["threshold"],
        nonmaxSuppression=p["nonmaxSuppression"],
        type=p.get("type", cv2.FastFeatureDetector_TYPE_9_16),
    )


# ------------------------------------------------------------------
# Descriptor factories
# ------------------------------------------------------------------

def _make_brief():
    return cv2.xfeatures2d.BriefDescriptorExtractor_create(bytes=32)

def _make_brisk_desc():
    return cv2.xfeatures2d.BRISK_create()

def _make_freak():
    return cv2.xfeatures2d.FREAK_create()


# ------------------------------------------------------------------
# Budget: keep top-N keypoints by response
# ------------------------------------------------------------------

def _apply_budget(kps, descs, max_kps):
    if max_kps is None or len(kps) <= max_kps:
        return kps, descs
    # Sort by response descending
    indices = np.argsort([-kp.response for kp in kps])[:max_kps]
    kps = [kps[i] for i in indices]
    if descs is not None:
        descs = descs[indices]
    return kps, descs


def _apply_budget_kps_only(kps, max_kps):
    """Budget for keypoints only (before compute)."""
    if max_kps is None or len(kps) <= max_kps:
        return kps
    indices = np.argsort([-kp.response for kp in kps])[:max_kps]
    return [kps[i] for i in indices]


# ------------------------------------------------------------------
# Unified extraction
# ------------------------------------------------------------------

def extract(method_name, gray, max_keypoints=None, use_defaults=False):
    """Detect + describe with the given method.

    Parameters
    ----------
    method_name  : str – key in cfg.METHOD_CONFIGS
    gray         : (H, W) uint8 grayscale image
    max_keypoints: int or None – keypoint budget (default: cfg.MAX_KEYPOINTS)
    use_defaults : bool – use OpenCV default params instead of calibrated

    Returns
    -------
    kps  : list of cv2.KeyPoint
    descs: np.ndarray – descriptor matrix
    """
    if max_keypoints is None:
        max_keypoints = cfg.MAX_KEYPOINTS

    mcfg = cfg.METHOD_CONFIGS[method_name]
    det_name = mcfg["detector"]
    desc_name = mcfg["descriptor"]

    params_dict = cfg.DETECTOR_DEFAULTS if use_defaults else cfg.DETECTOR_PARAMS

    # --- Native pipelines (detectAndCompute) ---
    if method_name == "SIFT":
        obj = _make_sift(params_dict.get("SIFT"))
        kps, descs = obj.detectAndCompute(gray, None)
        kps, descs = _apply_budget(kps, descs, max_keypoints)
        return kps, descs

    if method_name == "KAZE":
        obj = _make_kaze(params_dict.get("KAZE"))
        kps, descs = obj.detectAndCompute(gray, None)
        kps, descs = _apply_budget(kps, descs, max_keypoints)
        return kps, descs

    if method_name == "ORB":
        obj = _make_orb(params_dict.get("ORB"))
        kps, descs = obj.detectAndCompute(gray, None)
        # ORB already limits via nfeatures, but apply budget anyway
        kps, descs = _apply_budget(kps, descs, max_keypoints)
        return kps, descs

    # --- FAST + X pipelines ---
    if det_name == "FAST":
        # Own BRIEF variants
        if desc_name.startswith("OwnBRIEF"):
            strategy = desc_name.split("_")[1]  # GI, GII, GIII
            kps, descs = detect_and_compute_own_brief(
                gray, strategy=strategy, n_bits=256,
                fast_threshold=params_dict["FAST"]["threshold"],
                max_keypoints=max_keypoints,
                fast_kp_size=cfg.FAST_KP_SIZE,
            )
            return kps, descs

        # OpenCV FAST + OpenCV descriptor
        fast = _make_fast(params_dict.get("FAST"))
        kps = fast.detect(gray, None)

        # Set kp.size for BRISK/FREAK/BRIEF (B2 step 3 of plan)
        for kp in kps:
            kp.size = cfg.FAST_KP_SIZE

        # Pre-budget before compute (avoid unnecessary work)
        kps = _apply_budget_kps_only(kps, max_keypoints)

        # Compute descriptor
        if desc_name == "BRIEF":
            descriptor = _make_brief()
        elif desc_name == "BRISK":
            descriptor = _make_brisk_desc()
        elif desc_name == "FREAK":
            descriptor = _make_freak()
        else:
            raise ValueError(f"Unknown descriptor: {desc_name}")

        kps, descs = descriptor.compute(gray, kps)
        # B1 counting rule: count #features AFTER description
        # (OpenCV silently drops keypoints too close to border)
        kps, descs = _apply_budget(kps, descs, max_keypoints)
        return kps, descs

    raise ValueError(f"Unknown method: {method_name}")


# ------------------------------------------------------------------
# Separate detect / describe for timing
# ------------------------------------------------------------------

def detect_only(method_name, gray, max_keypoints=None, use_defaults=False):
    """Run only detection (for timing separation)."""
    if max_keypoints is None:
        max_keypoints = cfg.MAX_KEYPOINTS

    params_dict = cfg.DETECTOR_DEFAULTS if use_defaults else cfg.DETECTOR_PARAMS
    mcfg = cfg.METHOD_CONFIGS[method_name]
    det_name = mcfg["detector"]

    if det_name == "SIFT":
        obj = _make_sift(params_dict.get("SIFT"))
        kps = obj.detect(gray, None)
    elif det_name == "KAZE":
        obj = _make_kaze(params_dict.get("KAZE"))
        kps = obj.detect(gray, None)
    elif det_name == "ORB":
        obj = _make_orb(params_dict.get("ORB"))
        kps = obj.detect(gray, None)
    elif det_name == "FAST":
        fast = _make_fast(params_dict.get("FAST"))
        kps = fast.detect(gray, None)
        for kp in kps:
            kp.size = cfg.FAST_KP_SIZE
    else:
        raise ValueError(f"Unknown detector: {det_name}")

    kps = _apply_budget_kps_only(kps, max_keypoints)
    return kps


def describe_only(method_name, gray, kps):
    """Run only description on pre-detected keypoints (for timing)."""
    mcfg = cfg.METHOD_CONFIGS[method_name]
    desc_name = mcfg["descriptor"]
    det_name = mcfg["detector"]

    if desc_name == "SIFT":
        obj = _make_sift()
        kps, descs = obj.compute(gray, kps)
    elif desc_name == "KAZE":
        obj = _make_kaze()
        kps, descs = obj.compute(gray, kps)
    elif desc_name == "ORB":
        obj = _make_orb()
        kps, descs = obj.compute(gray, kps)
    elif desc_name == "BRIEF":
        obj = _make_brief()
        kps, descs = obj.compute(gray, kps)
    elif desc_name == "BRISK":
        obj = _make_brisk_desc()
        kps, descs = obj.compute(gray, kps)
    elif desc_name == "FREAK":
        obj = _make_freak()
        kps, descs = obj.compute(gray, kps)
    elif desc_name.startswith("OwnBRIEF"):
        strategy = desc_name.split("_")[1]
        from own_brief import generate_pairs, compute_own_brief
        pairs = generate_pairs(strategy, n_bits=256, patch_size=48, seed=42)
        kps, descs = compute_own_brief(gray, kps, pairs)
    else:
        raise ValueError(f"Unknown descriptor: {desc_name}")

    return kps, descs
