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


if __name__ == "__main__":
    pass