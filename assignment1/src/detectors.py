from pathlib import Path
from time import perf_counter

import cv2
import matplotlib.pyplot as plt


# ---------- Utilities ----------

def to_gray(image):
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ---------- Detectors ----------

def sift_detector(image, nfeatures=1000):
    detector = cv2.SIFT_create(
        nfeatures=nfeatures,
        nOctaveLayers=3,
        contrastThreshold=0.03,
        edgeThreshold=10,
        sigma=1.6,
    )
    return detector.detect(image, None)


def fast_detector(image, threshold=20, nonmax_suppression=True, max_features=2000):
    detector = cv2.FastFeatureDetector_create(
        threshold=threshold,
        nonmaxSuppression=nonmax_suppression,
    )

    keypoints = detector.detect(image, None)

    # Keep the strongest keypoints if the limit is exceeded.
    if len(keypoints) > max_features:
        keypoints = sorted(
            keypoints,
            key=lambda kp: kp.response,
            reverse=True,
        )[:max_features]

    return keypoints


def kaze_detector(image, threshold=0.001):
    
    detector = cv2.xfeatures2d.KAZE_create(
        threshold=threshold,
        nOctaves=4,
        nOctaveLayers=4,
    )
    return detector.detect(image, None)


def orb_detector(image, nfeatures=1000):
    gray = to_gray(image)
    detector = cv2.ORB_create(
        nfeatures=nfeatures,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31,
        fastThreshold=20,
    )
    return detector.detect(gray, None)


# ---------- Detector comparison ----------

DETECTORS = {
    "SIFT": sift_detector,
    "FAST": fast_detector,
    "KAZE": kaze_detector,
    "ORB": orb_detector,
}


def compare_detectors(image):
    """
    Compare detectors on the same image.

    Returns:
        dict: detector name -> keypoints and elapsed time in milliseconds.
    """
    results = {}

    for name, detector_function in DETECTORS.items():
        start = perf_counter()
        keypoints = detector_function(image)
        elapsed_ms = (perf_counter() - start) * 1000

        results[name] = {
            "keypoints": keypoints,
            "num_keypoints": len(keypoints),
            "time_ms": elapsed_ms,
        }

    return results


def plot_comparison(results, image_name = "Image"):

    """Plot detected feature counts and detection times."""
    names = list(results.keys())
    counts = [results[name]["num_keypoints"] for name in names]
    times = [results[name]["time_ms"] for name in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(names, counts)
    axes[0].set_title("Number of Detected Keypoints")
    axes[0].set_ylabel("Keypoints")

    axes[1].bar(names, times)
    axes[1].set_title("Detection Time")
    axes[1].set_ylabel("Time (ms)")

    fig.suptitle(f"Detector Comparison — {image_name}")
    plt.tight_layout()
    plt.show()


def compare_on_images(image_list):

    totals = {
        name: {"keypoints": 0, "time_ms": 0.0, "images": 0}
        for name in DETECTORS
    }

    for image_path in image_list:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


        results = compare_detectors(image)

        print(f"\n {image_path}")
        print(f"{'Detector':<12}{'Keypoints':>12}{'Time (ms)':>14}")

        for name, result in results.items():
            print(
                f"{name:<12}"
                f"{result['num_keypoints']:>12}"
                f"{result['time_ms']:>14.2f}"
            )

            totals[name]["keypoints"] += result["num_keypoints"]
            totals[name]["time_ms"] += result["time_ms"]
            totals[name]["images"] += 1


    print("\n========== DATASET SUMMARY ==========")
    print(f"{'Detector':<12}{'Avg keypoints':>16}{'Avg time (ms)':>16}")

    for name, values in totals.items():
        n = values["images"]
        if n == 0:
            continue

        print(
            f"{name:<12}"
            f"{values['keypoints'] / n:>16.1f}"
            f"{values['time_ms'] / n:>16.2f}"
        )


if __name__ == "__main__":

    testing_imgs = Path("data/graf/")

    img1 = testing_imgs / "img1.ppm"
    img2 = testing_imgs / "img2.ppm"
    img3 = testing_imgs / "img4.ppm"

    compare_on_images([img1,img2,img3])

