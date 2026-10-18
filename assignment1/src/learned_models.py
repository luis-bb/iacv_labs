

import cv2
import numpy as np
import torch

from functools import lru_cache
from lightglue import SuperPoint, LightGlue


DEFAULT_MAX_KEYPOINTS = 2048


def _get_device(device=None):
    """Choose the computation device."""
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    return device


@lru_cache(maxsize=2)
def _load_models(device_name, max_keypoints):
    """Load and cache SuperPoint and LightGlue models."""
    extractor = SuperPoint(
        max_num_keypoints=max_keypoints
    ).eval().to(device_name)

    matcher = LightGlue(
        features="superpoint"
    ).eval().to(device_name)

    return extractor, matcher


def _to_gray(image):
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

    raise ValueError(
        f"Unsupported number of image channels: {image.shape[2]}"
    )


def _prepare_image(image, device):
    """Convert an image to a normalized grayscale PyTorch tensor."""
    gray = _to_gray(image)

    tensor = torch.from_numpy(
        np.ascontiguousarray(gray)
    ).float() / 255.0

    # Shape: (1, H, W)
    return tensor.unsqueeze(0).to(device)


def _features_to_cv(features):
    """Convert batched LightGlue SuperPoint features to OpenCV format."""
    keypoints = features["keypoints"][0].detach().cpu().numpy()
    descriptors = features["descriptors"][0].detach().cpu().numpy()

    scores = features["keypoint_scores"][0].detach().cpu().numpy()

    cv_keypoints = [
        cv2.KeyPoint(
            float(point[0]),
            float(point[1]),
            1.0,
            -1.0,
            float(score),
        )
        for point, score in zip(keypoints, scores)
    ]

    return cv_keypoints, descriptors.astype(np.float32)


@torch.inference_mode()
def extract_superpoint_features(image, return_features=False,max_keypoints=DEFAULT_MAX_KEYPOINTS, device=None,):

    device = _get_device(device)
    extractor, _ = _load_models(device, max_keypoints)

    tensor = _prepare_image(image, device)

    features = extractor.extract(tensor)

    keypoints, descriptors = _features_to_cv(features)

    if return_features:
        return keypoints, descriptors, features

    return keypoints, descriptors


@torch.inference_mode()
def match_lightglue(
    features0,
    features1,
    device=None,
    max_keypoints=DEFAULT_MAX_KEYPOINTS,
):
    device = _get_device(device)
    _, matcher = _load_models(device, max_keypoints)

    output = matcher({
        "image0": features0,
        "image1": features1,
    })

    matches = output["matches"][0].detach().cpu().numpy()
    scores = output["scores"][0].detach().cpu().numpy()

    return matches.astype(np.int32), scores.astype(np.float32)


def lightglue_matches_to_cv(matches, scores=None):
    """
    Convert LightGlue match indices to OpenCV DMatch objects.

    The query index refers to image 0; the train index refers to image 1.
    """
    if scores is None:
        scores = np.ones(len(matches), dtype=np.float32)

    return [
        cv2.DMatch(
            _queryIdx=int(pair[0]),
            _trainIdx=int(pair[1]),
            _imgIdx=0,
            _distance=float(1.0 - score),
        )
        for pair, score in zip(matches, scores)
    ]

def extract_superpoint(image, max_keypoints=DEFAULT_MAX_KEYPOINTS, device=None):
    """
    Compatibility wrapper for the main benchmark script.

    Returns
    -------
    keypoints : list[cv2.KeyPoint]
    descriptors : np.ndarray
    """
    return extract_superpoint_features(
        image,
        max_keypoints=max_keypoints,
        device=device,
    )


@torch.inference_mode()
def match_superpoint_descriptors(
    descriptors1,
    descriptors2,
    ratio=0.8,
):
    """
    Match SuperPoint descriptors using BFMatcher, kNN, and the ratio test.

    Returns
    -------
    list[cv2.DMatch]
        Matches passing the ratio test.
    """
    if descriptors1 is None or descriptors2 is None:
        return []

    if len(descriptors1) < 2 or len(descriptors2) < 2:
        return []

    descriptors1 = np.asarray(descriptors1, dtype=np.float32)
    descriptors2 = np.asarray(descriptors2, dtype=np.float32)

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    knn_matches = matcher.knnMatch(
        descriptors1,
        descriptors2,
        k=2,
    )

    good_matches = []

    for candidates in knn_matches:
        if len(candidates) < 2:
            continue

        best, second = candidates

        if best.distance < ratio * second.distance:
            good_matches.append(best)

    # Enforce one-to-one matching: retain the best match per train index.
    best_per_train = {}

    for match in good_matches:
        previous = best_per_train.get(match.trainIdx)

        if previous is None or match.distance < previous.distance:
            best_per_train[match.trainIdx] = match

    return list(best_per_train.values())


def to_evaluation_format(
    keypoints1,
    descriptors1,
    keypoints2,
    descriptors2,
    matches,
):
    """
    Convert SuperPoint outputs into the common evaluation format.

    Returns
    -------
    dict
        Keypoints, descriptors, and matches for the evaluation pipeline.
    """
    return {
        "keypoints1": keypoints1,
        "descriptors1": descriptors1,
        "keypoints2": keypoints2,
        "descriptors2": descriptors2,
        "matches": matches,
    }

if __name__ == "__main__":

    image1 = cv2.imread("hpatches-sequences-release/i_ajuntament/1.ppm")
    image2 = cv2.imread("hpatches-sequences-release/i_ajuntament/6.ppm")

    if image1 is None or image2 is None:
        raise FileNotFoundError("Could not load input images.")

    # Extract features. Keep the native features for LightGlue.
    kps1, descs1, features1 = extract_superpoint_features(
        image1, return_features=True
    )
    kps2, descs2, features2 = extract_superpoint_features(
        image2, return_features=True
    )

    # Match using LightGlue.
    matches, scores = match_lightglue(features1, features2)
    cv_matches = lightglue_matches_to_cv(matches, scores)

    print(f"Image 1 keypoints: {len(kps1)}")
    print(f"Image 2 keypoints: {len(kps2)}")
    print(f"LightGlue matches: {len(matches)}")

    visualization = cv2.drawMatches(
        image1,
        kps1,
        image2,
        kps2,
        cv_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    cv2.imshow("SuperPoint + LightGlue", visualization)
    cv2.waitKey(0)
    cv2.destroyAllWindows()