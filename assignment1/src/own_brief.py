import numpy as np
import random
import visualization
import cv2 
import matplotlib.pyplot as plt


def generate_sampling_pairs(patch_size: int, num_pairs: int) -> list:

    coordinate_pairs = list()
    max_offset = patch_size // 2
    sigma = max_offset / 5


    for _ in range(num_pairs):

        p1 = np.random.normal(0,sigma, size=2)
        p2 = np.random.normal(0,sigma, size=2)

        p1 = np.round(p1).astype(int)
        p2 = np.round(p2).astype(int)

        p1 = np.clip(p1, -max_offset, max_offset)
        p2 = np.clip(p2, -max_offset, max_offset)

        coordinate_pairs.append((tuple(p1), tuple(p2)))
    return coordinate_pairs

def compute_single_descriptor(image: np.ndarray, 
                              keypoint: tuple, 
                              patch_size: int, 
                              sampling_pairs: list) -> np.ndarray:

    kx, ky = keypoint
    max_offset = patch_size // 2

    if (kx-max_offset< 0 or ky-max_offset<0 or kx + max_offset>=image.shape[1] or ky+max_offset>=image.shape[0]):
        return None

    descriptor = np.zeros(len(sampling_pairs), dtype = np.uint8)

    for i, (p1, p2) in enumerate(sampling_pairs):
        dx1, dy1 = p1
        dx2, dy2 = p2

        intensity1 = image[ky + dy1, kx + dx1]
        intensity2 = image[ky + dy2, kx + dx2]

        descriptor[i] = int(intensity1 < intensity2)


    return descriptor

def compute_descriptors(image: np.ndarray, 
                        keypoints: list, 
                        patch_size: int = 31, 
                        num_pairs: int = 256) -> list:

    sampling_pairs = generate_sampling_pairs(patch_size, num_pairs)
    descriptors = []

    for keypoint in keypoints:

        descriptor = compute_single_descriptor(image,
                                                keypoint,
                                                patch_size,
                                                sampling_pairs)
        if descriptors is not None:
            descriptors.append(descriptor)

    return descriptors

    


def compute_hamming_distance(desc1: np.ndarray, desc2: np.ndarray) -> int:

    if len(desc1) != len(desc2):
        raise ValueError("Descriptors must have same length")

    return int(np.count_nonzero(desc1 != desc2))


def mybrief_run(image, keypoints_xy, patch_size, num_pair):
    sampling_pairs = generate_sampling_pairs(patch_size, num_pair)
    descriptors = []
    valid_keypoints = []

    for kp in keypoints_xy:
        descriptors = compute_single_descriptor(image,
                                                kp,
                                                patch_size,
                                                sampling_pairs)
    if descriptor is not None:
        descriptors.append(descriptor)
        valid_keypoints.append(kp)

    
 

if __name__ == "__main__":

    image = np.random.randint(0,256,size = (100, 100), dtype = np.uint8)
    keypoints = [(50,50),(30,30),(5,5)]

    descriptors = compute_descriptors(image,keypoints,patch_size=31, num_pairs=256)

    for descriptor in descriptors:
        if descriptor is not None:
            print("Descriptor shape: ", descriptor.shape)

    if len(descriptors) >= 2:
        distance = compute_hamming_distance(descriptors[0], descriptors[1])

    print("Hamming Distance", distance)

    from pathlib import Path
    image_real = Path("./hpatches-sequences-release/i_ajuntament/1.ppm")
    image_real = cv2.imread(image_real, cv2.IMREAD_GRAYSCALE)

    plt.imshow(image_real, cmap = "gray")
    plt.savefig("./SavingOwnBrief_real.jpg")

    from detectors import fast_detector
    import visualization
    import sys


    sys.exit(1)


    keypoints = fast_detector(image_real)
    print(f"Detected {len(keypoints)} Keypoints")


    R = mybrief_run(image_real, keypoints_xy=keypoints, patch_size=31, num_pair=256)
    print(R)


