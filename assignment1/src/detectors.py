import matplotlib.pyplot as plt
from pathlib import Path
import cv2


def to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


# ---------- OpenCV detectors ----------

def sift_detector(image, nfeatures=500):
    gray = to_gray(image)
    sift = cv2.SIFT_create(nfeatures=nfeatures, nOctaveLayers=3,
                           contrastThreshold=0.03, edgeThreshold=10, sigma=1.6)
    return sift.detect(gray, None)


def fast_detector(image, threshold=20, nonmax_suppression=True):
    gray = to_gray(image)
    fast = cv2.FastFeatureDetector_create(threshold=threshold,
                                          nonmaxSuppression=nonmax_suppression)
    return fast.detect(gray, None)


def kaze_detector(image, threshold=0.001):
    gray = to_gray(image)
    kaze = cv2.xfeatures2d.KAZE_create(threshold=threshold, nOctaves=4, nOctaveLayers=4)
    return kaze.detect(gray, None)


def orb_detector(image, nfeatures=1000):
    gray = to_gray(image)
    orb = cv2.ORB_create(nfeatures=nfeatures, scaleFactor=1.2, nlevels=8,
                         edgeThreshold=31, firstLevel=0, WTA_K=2,
                         scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31,
                         fastThreshold=20)
    return orb.detect(gray, None)


# ---------- Display ----------

def show_points(image, points, path="keypoints.png"):
    out = cv2.drawKeypoints(image, points, None, color=(0, 255, 0),
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved to {path} | Keypoints: {len(points)}")


if __name__ == "__main__":
    resultados = Path("resultados") / "detectors"
    resultados.mkdir(parents=True, exist_ok=True)

    loaded = cv2.imread("data/graf/img1.ppm")
    if loaded is None:
        raise FileNotFoundError("Image not found: data/graf/img1.ppm")

    kps_sift = sift_detector(loaded)
    show_points(loaded, kps_sift, path=resultados / "keypoints_sift.png")

    kps_fast = fast_detector(loaded)
    show_points(loaded, kps_fast, path=resultados / "keypoints_fast.png")

    kps_kaze = kaze_detector(loaded)
    show_points(loaded, kps_kaze, path=resultados / "keypoints_kaze.png")

    kps_orb = orb_detector(loaded)
    show_points(loaded, kps_orb, path=resultados / "keypoints_orb.png")
