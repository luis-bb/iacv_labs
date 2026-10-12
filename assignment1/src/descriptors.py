from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, SuperPointForKeypointDetection


def to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


# ---------- OpenCV descriptors ----------


def sift_descriptor(image, nfeatures=500):
    gray = to_gray(image)
    sift = cv2.SIFT_create(nfeatures=nfeatures, nOctaveLayers=3,
                           contrastThreshold=0.03, edgeThreshold=10, sigma=1.6)
    return sift.detectAndCompute(gray, None)


def kaze_descriptor(image, threshold=0.001):
    gray = to_gray(image)
    # KAZE belongs to xfeatures2d in OpenCV 5.
    create_kaze = cv2.KAZE_create if hasattr(cv2, "KAZE_create") else cv2.xfeatures2d.KAZE_create
    kaze = create_kaze(threshold=threshold, nOctaves=4, nOctaveLayers=4)
    return kaze.detectAndCompute(gray, None)


def orb_descriptor(image, nfeatures=500):
    gray = to_gray(image)
    orb = cv2.ORB_create(nfeatures=nfeatures, scaleFactor=1.2, nlevels=8,
                         edgeThreshold=31, firstLevel=0, WTA_K=2,
                         scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31,
                         fastThreshold=20)
    return orb.detectAndCompute(gray, None)

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
    
    
    
def save_results(image, keypoints, descriptors, name, folder="resultados/descriptors"):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    show_points(image, keypoints, path=folder / f"descriptors_{name}.png")
    if descriptors is not None:
        np.save(folder / f"descriptors_{name}.npy", descriptors)
        print(f"{name.upper()} | Keypoints: {len(keypoints)} | Descriptors: {descriptors.shape}")
    else:
        print(f"{name.upper()} | No descriptors found")


if __name__ == "__main__":
    resultados = Path("resultados") / "descriptors"
    resultados.mkdir(parents=True, exist_ok=True)

    loaded = cv2.imread("data/graf/img1.ppm")
    if loaded is None:
        raise FileNotFoundError("Image not found: data/graf/img1.ppm")

    kps_sift, desc_sift = sift_descriptor(loaded)
    save_results(loaded, kps_sift, desc_sift, "sift", resultados)

    kps_kaze, desc_kaze = kaze_descriptor(loaded)
    save_results(loaded, kps_kaze, desc_kaze, "kaze", resultados)

    kps_orb, desc_orb = orb_descriptor(loaded)
    save_results(loaded, kps_orb, desc_orb, "orb", resultados)

    kps_brisk, desc_brisk = brisk_descriptor(loaded)
    save_results(loaded, kps_brisk, desc_brisk, "brisk", resultados)

    fast = cv2.FastFeatureDetector_create()

    kps_b, desc_b = brief_descriptor(loaded, fast)
    save_results(loaded, kps_b, desc_b, "brief", resultados)

    kps_f, desc_f = freak_descriptor(loaded, fast)
    save_results(loaded, kps_f, desc_f, "freak", resultados)

    kps_m, desc_m = my_brief_descriptor(loaded, fast.detect(to_gray(loaded)))
    save_results(loaded, kps_m, desc_m, "my_brief", resultados)
