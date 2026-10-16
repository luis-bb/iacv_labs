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
    gray = to_gray(image)
    keypoints = _validate_keypoints(keypoints)

    extractor = cv2.SIFT_create()
    return extractor.compute(gray, keypoints)


def compute_kaze_descriptors(image, keypoints):
    """Compute KAZE descriptors for existing keypoints."""
    gray = to_gray(image)
    keypoints = _validate_keypoints(keypoints)

    extractor = cv2.KAZE_create()
    return extractor.compute(gray, keypoints)


def compute_orb_descriptors(image, keypoints):
    """Compute ORB descriptors for existing keypoints."""
    gray = to_gray(image)
    keypoints = _validate_keypoints(keypoints)

    extractor = cv2.ORB_create()
    return extractor.compute(gray, keypoints)


def compute_brief_descriptors(image, keypoints, bytes=32):
    """
    Compute BRIEF descriptors for existing keypoints.

    Requires opencv-contrib-python, which provides cv2.xfeatures2d.
    The default descriptor length is 32 bytes (256 bits).
    """
    gray = to_gray(image)
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
    return extractor.compute(gray, keypoints)


def compute_brisk_descriptors(image, keypoints):
    """Compute BRISK descriptors for existing keypoints."""
    gray = to_gray(image)
    keypoints = _validate_keypoints(keypoints)

    extractor = cv2.BRISK_create()
    return extractor.compute(gray, keypoints)


def compute_freak_descriptors(image, keypoints):
    """
    Compute FREAK descriptors for existing keypoints.

    Requires opencv-contrib-python, which provides cv2.xfeatures2d.
    """
    gray = to_gray(image)
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
    return extractor.compute(gray, keypoints)


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


if __name__ == "__main__":
    pass