"""
config.py – Full evaluation protocol for Assignment #1
All thresholds, budgets and method definitions in one place.
"""

import cv2
import numpy as np

# =====================================================================
# A1. General protocol
# =====================================================================

HPATCHES_DIR = "hpatches-sequences-release"  # full-image sequences

REFERENCE_IMAGE_INDEX = 1
TARGET_IMAGE_INDICES = [2, 3, 4, 5, 6]

# Pre-processing
MAX_RESOLUTION_SIDE = 1024  # longest side cap

# Keypoint budget (top-N by response)
MAX_KEYPOINTS = 2000

# Matching
KNN_K = 2
RATIO_TEST_THRESHOLD = 0.8
CORRECT_MATCH_THRESHOLD_PX = 3.0

# RANSAC
RANSAC_METHOD = cv2.RANSAC
RANSAC_REPROJ_THRESHOLD = 3.0
RANSAC_MAX_ITERS = 2000
RANSAC_CONFIDENCE = 0.995
RANDOM_SEED = 42

# Timing
TIMING_REPEATS = 5  # median of >= 5 runs (after 1 warm-up)

# =====================================================================
# B1. Tuning set – 6 sequences excluded from final tables
# =====================================================================

_rng = np.random.RandomState(RANDOM_SEED)
# We'll pick 3 i_* and 3 v_* at dataset-load time; store names here
# once we know the full list.  For now, use fixed names that exist in
# the standard HPatches-sequences release.
TUNING_SEQUENCES = {
    "i_dc", "i_fruits", "i_fog",
    "v_adam", "v_bark", "v_there",
}

# =====================================================================
# B1. Per-detector calibrated thresholds (≈ 2000 kps on 1024 px)
# These are starting values; re-calibrate on the tuning set if needed.
# =====================================================================

DETECTOR_PARAMS = {
    "SIFT":  {"contrastThreshold": 0.02, "edgeThreshold": 10, "sigma": 1.6, "nOctaveLayers": 3},
    "KAZE":  {"threshold": 0.0008, "nOctaves": 4, "nOctaveLayers": 4},
    "ORB":   {"nfeatures": 2000, "scaleFactor": 1.2, "nlevels": 8,
              "edgeThreshold": 31, "patchSize": 31, "fastThreshold": 20,
              "scoreType": cv2.ORB_HARRIS_SCORE, "WTA_K": 2},
    "FAST":  {"threshold": 20, "nonmaxSuppression": True,
              "type": cv2.FastFeatureDetector_TYPE_9_16},
}

# Default params (OpenCV defaults, for the "defaults run")
DETECTOR_DEFAULTS = {
    "SIFT":  {"contrastThreshold": 0.04, "edgeThreshold": 10, "sigma": 1.6, "nOctaveLayers": 3},
    "KAZE":  {"threshold": 0.001, "nOctaves": 4, "nOctaveLayers": 4},
    "ORB":   {"nfeatures": 500, "scaleFactor": 1.2, "nlevels": 8,
              "edgeThreshold": 31, "patchSize": 31, "fastThreshold": 20,
              "scoreType": cv2.ORB_HARRIS_SCORE, "WTA_K": 2},
    "FAST":  {"threshold": 10, "nonmaxSuppression": True,
              "type": cv2.FastFeatureDetector_TYPE_9_16},
}

# kp.size to set on FAST keypoints before describing (match ORB patch)
FAST_KP_SIZE = 31.0

# =====================================================================
# Method configurations table (plan §B)
# =====================================================================

METHOD_CONFIGS = {
    # --- native pipelines ---
    "SIFT": {
        "detector": "SIFT",
        "descriptor": "SIFT",
        "norm": cv2.NORM_L2,
        "bytes_per_desc": 512,
        "desc_dim": 128,
        "is_binary": False,
    },
    "KAZE": {
        "detector": "KAZE",
        "descriptor": "KAZE",
        "norm": cv2.NORM_L2,
        "bytes_per_desc": 256,
        "desc_dim": 64,
        "is_binary": False,
    },
    "ORB": {
        "detector": "ORB",
        "descriptor": "ORB",
        "norm": cv2.NORM_HAMMING,
        "bytes_per_desc": 32,
        "desc_dim": 256,
        "is_binary": True,
    },
    # --- FAST + X ---
    "FAST+BRIEF": {
        "detector": "FAST",
        "descriptor": "BRIEF",
        "norm": cv2.NORM_HAMMING,
        "bytes_per_desc": 32,
        "desc_dim": 256,
        "is_binary": True,
    },
    "FAST+BRISK": {
        "detector": "FAST",
        "descriptor": "BRISK",
        "norm": cv2.NORM_HAMMING,
        "bytes_per_desc": 64,
        "desc_dim": 512,
        "is_binary": True,
    },
    "FAST+FREAK": {
        "detector": "FAST",
        "descriptor": "FREAK",
        "norm": cv2.NORM_HAMMING,
        "bytes_per_desc": 64,
        "desc_dim": 512,
        "is_binary": True,
    },
    # --- own BRIEF variants ---
    "FAST+OwnBRIEF_GI": {
        "detector": "FAST",
        "descriptor": "OwnBRIEF_GI",
        "norm": cv2.NORM_HAMMING,
        "bytes_per_desc": 32,
        "desc_dim": 256,
        "is_binary": True,
    },
    "FAST+OwnBRIEF_GII": {
        "detector": "FAST",
        "descriptor": "OwnBRIEF_GII",
        "norm": cv2.NORM_HAMMING,
        "bytes_per_desc": 32,
        "desc_dim": 256,
        "is_binary": True,
    },
    "FAST+OwnBRIEF_GIII": {
        "detector": "FAST",
        "descriptor": "OwnBRIEF_GIII",
        "norm": cv2.NORM_HAMMING,
        "bytes_per_desc": 32,
        "desc_dim": 256,
        "is_binary": True,
    },
}

# Learned methods (Phase C)
LEARNED_CONFIGS = {
    "SuperPoint+NN": {
        "detector": "SuperPoint",
        "descriptor": "SuperPoint",
        "matcher": "NN",
        "norm": cv2.NORM_L2,
        "bytes_per_desc": 1024,
        "desc_dim": 256,
        "is_binary": False,
    },
}

# Homography accuracy thresholds (pixels)
HA_THRESHOLDS = [1, 3, 5]

# =====================================================================
# B5. FAST parameter study sweep
# =====================================================================

FAST_THRESHOLDS = [5, 10, 15, 20, 30, 40, 60]
FAST_TYPES = [
    cv2.FastFeatureDetector_TYPE_9_16,
    cv2.FastFeatureDetector_TYPE_7_12,
    cv2.FastFeatureDetector_TYPE_5_8,
]
FAST_NMS_OPTIONS = [True, False]

# B6. Synthetic rotation / scale
SYNTH_ROTATIONS = list(range(0, 195, 15))   # 0, 15, ..., 180
SYNTH_SCALES = [0.5, 0.7, 1.0, 1.4, 2.0]
SYNTH_NOISE_SIGMA = 10
SYNTH_IMAGE_SIZE = 640  # longest side

# B7. Matching strategy study
RATIO_SWEEP = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# =====================================================================
# Output paths
# =====================================================================

RESULTS_DIR = "results"
CSV_DIR = f"{RESULTS_DIR}/csv"
FIGURES_DIR = f"{RESULTS_DIR}/figures"
