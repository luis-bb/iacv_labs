"""
benchmark_timer.py – Rigorous timing: warm-up + median of N repeats.

Separates detection, description, and matching times.
Handles GPU sync for PyTorch models.
"""

import time
import statistics
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _sync(is_gpu):
    if is_gpu and HAS_TORCH and torch.cuda.is_available():
        torch.cuda.synchronize()


def time_function(func, *args, is_gpu=False, n_repeats=5, **kwargs):
    """1 warm-up + median of n_repeats.

    Returns (result, median_seconds).
    """
    # Warm-up
    _sync(is_gpu)
    _ = func(*args, **kwargs)
    _sync(is_gpu)

    times = []
    result = None
    for _ in range(n_repeats):
        _sync(is_gpu)
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        _sync(is_gpu)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return result, statistics.median(times)


def time_pipeline(method_name, gray1, gray2, max_keypoints=None,
                  is_gpu=False, n_repeats=5):
    """Time the full pipeline: detect → describe → match, separately.

    Returns
    -------
    kps1, descs1, kps2, descs2, matches : results from the last run
    times : dict with detect_ms, describe_ms, match_ms, total_ms
    """
    from feature_extraction import detect_only, describe_only, extract
    from matching import match_pipeline
    import config as cfg

    mcfg = cfg.METHOD_CONFIGS[method_name]
    norm = mcfg["norm"]

    # --- Detection timing ---
    def run_detect(img):
        return detect_only(method_name, img, max_keypoints)

    kps1, t_det1 = time_function(run_detect, gray1, is_gpu=is_gpu,
                                  n_repeats=n_repeats)
    kps2, t_det2 = time_function(run_detect, gray2, is_gpu=is_gpu,
                                  n_repeats=n_repeats)
    t_detect = (t_det1 + t_det2) / 2.0

    # --- Description timing ---
    def run_describe(img, kps):
        return describe_only(method_name, img, kps)

    (kps1, descs1), t_desc1 = time_function(run_describe, gray1, kps1,
                                             is_gpu=is_gpu, n_repeats=n_repeats)
    (kps2, descs2), t_desc2 = time_function(run_describe, gray2, kps2,
                                             is_gpu=is_gpu, n_repeats=n_repeats)
    t_describe = (t_desc1 + t_desc2) / 2.0

    # --- Matching timing ---
    def run_match():
        return match_pipeline(descs1, descs2, norm)

    matches, t_match = time_function(run_match, is_gpu=is_gpu,
                                      n_repeats=n_repeats)

    times = {
        "detect_ms": t_detect * 1000,
        "describe_ms": t_describe * 1000,
        "match_ms": t_match * 1000,
        "total_ms": (t_detect + t_describe + t_match) * 1000,
    }

    return kps1, descs1, kps2, descs2, matches, times
