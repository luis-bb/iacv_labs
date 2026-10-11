import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, SuperPointForKeypointDetection


def to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


# ---------- OpenCV descriptors ----------

def brief_descriptor(image, detector, brief_bytes=32):
    gray = to_gray(image)
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create(bytes=brief_bytes)
    keypoints = detector.detect(gray, None)
    keypoints, descriptors = brief.compute(gray, keypoints)
    return keypoints, descriptors


def brisk_descriptor(image):
    gray = to_gray(image)
    brisk = cv2.xfeatures2d.BRISK_create()
    return brisk.detectAndCompute(gray, None)


def freak_descriptor(image, detector):
    gray = to_gray(image)
    freak = cv2.xfeatures2d.FREAK_create()
    keypoints = detector.detect(gray, None)
    keypoints, descriptors = freak.compute(gray, keypoints)
    return keypoints, descriptors


# ---------- Own BRIEF ----------

def my_brief_descriptor(image, keypoints, n_bits=256, patch_size=48, seed=0):
    """Returns (kept_keypoints, descriptors uint8 [N, n_bits//8])."""
    gray = cv2.GaussianBlur(to_gray(image), (9, 9), 2)
    rng = np.random.default_rng(seed)
    # Pair sampling G II: isotropic Gaussian around the patch center
    pairs = np.clip(rng.normal(0, patch_size / 5, size=(n_bits, 4)),
                    -patch_size // 2, patch_size // 2).round().astype(int)
    h, w = gray.shape
    r = patch_size // 2
    kept, bits = [], []
    for kp in keypoints:
        x, y = map(int, kp.pt)
        if x < r or y < r or x >= w - r or y >= h - r:
            continue
        p = gray[y + pairs[:, 1], x + pairs[:, 0]]
        q = gray[y + pairs[:, 3], x + pairs[:, 2]]
        bits.append(p < q)
        kept.append(kp)
    if not bits:
        return [], np.empty((0, n_bits // 8), np.uint8)
    return kept, np.packbits(np.array(bits), axis=1)


# ---------- SuperPoint ----------

_SP = {}

def _get_superpoint():
    if not _SP:
        model_id = "magic-leap-community/superpoint"
        _SP["proc"] = AutoImageProcessor.from_pretrained(model_id)
        _SP["model"] = SuperPointForKeypointDetection.from_pretrained(model_id).eval()
    return _SP["proc"], _SP["model"]


def extract_superpoint_features(image_input):
    """image_input: path, PIL image, or BGR numpy array (OpenCV)."""
    if isinstance(image_input, str):
        image_input = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            image_input = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        else:
            image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        image_input = Image.fromarray(image_input)

    processor, model = _get_superpoint()
    inputs = processor(image_input, return_tensors="pt")
    with torch.no_grad():
        raw = model(**inputs)

    out = processor.post_process_keypoint_detection(
        raw, target_sizes=[(image_input.height, image_input.width)])[0]
    return out  # keys: keypoints (N,2), scores (N,), descriptors (N,256)


def superpoint_to_cv(output):
    kps = [cv2.KeyPoint(float(x), float(y), 8, -1, float(s))
           for (x, y), s in zip(output["keypoints"].cpu().numpy(),
                                output["scores"].cpu().numpy())]
    return kps, output["descriptors"].cpu().numpy()


# ---------- Matching / display ----------

def match(desc1, desc2, binary=True, cross_check=True):
    norm = cv2.NORM_HAMMING if binary else cv2.NORM_L2
    bf = cv2.BFMatcher(norm, crossCheck=cross_check)
    return sorted(bf.match(desc1, desc2), key=lambda m: m.distance)


def show_points(image, points, path="keypoints.png"):
    out = cv2.drawKeypoints(image, points, None, color=(0, 255, 0),
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved to {path}")
    
    
    
if __name__ == "__main__":
    loaded = cv2.imread("data/graf/img1.ppm")
    assert loaded is not None, "Image not found"

    kps, desc = brisk_descriptor(loaded)
    show_points(loaded, kps)

    # Example: BRIEF / FREAK with a FAST detector
    fast = cv2.FastFeatureDetector_create()
    kps_b, desc_b = brief_descriptor(loaded, fast)
    kps_f, desc_f = freak_descriptor(loaded, fast)
    kps_m, desc_m = my_brief_descriptor(loaded, fast.detect(to_gray(loaded)))
