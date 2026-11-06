"""
descriptors.py

Compute descriptors for previously detected keypoints.

Detection is handled by detectors.py. Each function in this module
accepts an image and keypoints, and returns:
    (keypoints, descriptors)

Keypoints may be removed if the descriptor extractor cannot describe them.
"""

import cv2
import numpy as np
from pathlib import Path
from time import perf_counter
from detectors import fast_detector, kaze_detector


def to_gray(image):
    """Convert a grayscale, BGR, or BGRA image to grayscale."""
    if image is None:
        raise ValueError("Input image is None.")

    if image.ndim == 2:
        return image

    if image.ndim != 3:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    raise ValueError(f"Unsupported number of image channels: {image.shape[2]}")


def _validate_keypoints(keypoints):
    """Validate keypoints supplied by a detector."""
    if keypoints is None:
        raise ValueError("keypoints cannot be None. Pass the output of a detector.")

    return keypoints


def compute_sift_descriptors(image, keypoints):
    """Compute SIFT descriptors for existing keypoints."""
    keypoints = _validate_keypoints(keypoints)

    extractor = cv2.SIFT_create()
    return extractor.compute(image, keypoints)


def compute_kaze_descriptors(image, keypoints):
    """Compute KAZE descriptors for existing keypoints."""

    keypoints = _validate_keypoints(keypoints)

    extractor = cv2.xfeatures2d.KAZE_create()
    return extractor.compute(image, keypoints)


def compute_orb_descriptors(image, keypoints):
    """Compute ORB descriptors for existing keypoints."""
    keypoints = _validate_keypoints(keypoints)

    extractor = cv2.ORB_create()
    return extractor.compute(image, keypoints)


def compute_brief_descriptors(image, keypoints, bytes=32):
    """
    Compute BRIEF descriptors for existing keypoints.

    Requires opencv-contrib-python, which provides cv2.xfeatures2d.
    The default descriptor length is 32 bytes (256 bits).
    """
    keypoints = _validate_keypoints(keypoints)

    if not hasattr(cv2, "xfeatures2d"):
        raise RuntimeError(
            "BRIEF requires OpenCV contrib. Install opencv-contrib-python."
        )

    if not hasattr(cv2.xfeatures2d, "BriefDescriptorExtractor_create"):
        raise RuntimeError(
            "BRIEF is unavailable in this OpenCV build. "
            "Install a compatible opencv-contrib-python version."
        )

    extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(bytes=bytes)
    return extractor.compute(image, keypoints)


def compute_brisk_descriptors(image, keypoints):
    """Compute BRISK descriptors for existing keypoints."""
    keypoints = _validate_keypoints(keypoints)

    extractor = cv2.xfeatures2d.BRISK_create()
    return extractor.compute(image, keypoints)


def compute_freak_descriptors(image, keypoints):
    """
    Compute FREAK descriptors for existing keypoints.

    Requires opencv-contrib-python, which provides cv2.xfeatures2d.
    """
    keypoints = _validate_keypoints(keypoints)

    if not hasattr(cv2, "xfeatures2d"):
        raise RuntimeError(
            "FREAK requires OpenCV contrib. Install opencv-contrib-python."
        )

    if not hasattr(cv2.xfeatures2d, "FREAK_create"):
        raise RuntimeError(
            "FREAK is unavailable in this OpenCV build. "
            "Install a compatible opencv-contrib-python version."
        )

    extractor = cv2.xfeatures2d.FREAK_create()
    return extractor.compute(image, keypoints)


def match(descriptors1, descriptors2, cross_check=True):
    """
    Match two sets of descriptors using an appropriate distance metric.

    Binary descriptors (e.g. ORB, BRIEF, BRISK, FREAK) use Hamming distance.
    Floating-point descriptors (e.g. SIFT, KAZE) use L2 distance.

    Returns matches sorted by increasing distance.
    """
    if descriptors1 is None or descriptors2 is None:
        return []

    if len(descriptors1) == 0 or len(descriptors2) == 0:
        return []

    if descriptors1.dtype != descriptors2.dtype:
        raise ValueError(
            "Descriptor dtypes differ: "
            f"{descriptors1.dtype} vs {descriptors2.dtype}"
        )

    if descriptors1.dtype == np.uint8:
        norm_type = cv2.NORM_HAMMING
    else:
        norm_type = cv2.NORM_L2

    matcher = cv2.BFMatcher(norm_type, crossCheck=cross_check)
    matches = matcher.match(descriptors1, descriptors2)

    return sorted(matches, key=lambda m: m.distance)

DESCRIPTORS = {
    "SIFT": compute_sift_descriptors,
    "KAZE": compute_kaze_descriptors,
    "ORB": compute_orb_descriptors,
    "BRISK": compute_brisk_descriptors,
    "BRIEF": compute_brief_descriptors,
    "FREAK": compute_freak_descriptors,
}


def compare_descriptors(image):
    keypoints = fast_detector(image)
    results = {}

    for name, extractor in DESCRIPTORS.items():
        start = perf_counter()

        try:
            if name == "BRIEF":
                described_keypoints, desc = extractor(image, keypoints)
            elif name == "KAZE":
                keypoints = kaze_detector(image)
                described_keypoints, desc = extractor(image,keypoints)
            else:
                described_keypoints, desc = extractor(image, keypoints)

            elapsed_ms = (perf_counter() - start) * 1000

            results[name] = {
                "num_keypoints": len(described_keypoints or []),
                "time_ms": elapsed_ms,
                "dimensions": desc.shape[1] if desc is not None else 0,
                "memory_bytes": desc.nbytes if desc is not None else 0,
            }

        except (RuntimeError, cv2.error) as error:
            results[name] = {"error": str(error)}

    return results


if __name__ == "__main__":
    testing_imgs = Path("data/graf")

    image_paths = [
        testing_imgs / "img1.ppm",
        testing_imgs / "img2.ppm",
        testing_imgs / "img4.ppm",
    ]

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Could not read {image_path}")
            continue

        print(f"\n--- {image_path.name} ---")
        results = compare_descriptors(image)

        for name, result in results.items():
            if "error" in result:
                print(f"{name}: unavailable ({result['error']})")
            else:
                print(
                    f"{name:<8} "
                    f"keypoints={result['num_keypoints']:<5} "
                    f"time={result['time_ms']:.2f} ms  "
                    f"dimensions={result['dimensions']:<4} "
                    f"memory={result['memory_bytes']} bytes"
                )