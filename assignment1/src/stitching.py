"""
stitching.py – Phase E: Image stitching pipeline.

E1. Sequence stitching: detect → describe → match → OpenCV RANSAC → warp → blend
E3. RANSAC threshold study
E4. Multi-method comparison
E5. 3-image panorama with reference strategies
E6. Multi-band blending

Usage:
    python src/stitching.py --sequence i_ajuntament
"""

import os
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
from data_loader import get_sequence_dirs, load_image_color
from feature_extraction import extract
from matching import match_pipeline, estimate_homography, inlier_stats


# Quick test configuration: change the folder and image names here.
IMAGE_DIR = os.path.join("data", "Panorama")
IMAGE_NAMES = ["keble_a.jpg", "keble_b.jpg", "keble_c.jpg"]


def _to_gray(image):
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def estimate_pairwise_homographies(images, method="SIFT", ratio=None):
    """Return consecutive homographies and per-pair matching statistics.

    Homography i maps image i to image i+1. Images must be ordered so that
    neighbours overlap. Raise ValueError if features or alignment are insufficient.
    Each image is described once; results can be reused for different references.
    """
    if len(images) < 2:
        raise ValueError("At least two images are required")
    images = [_to_gray(image) for image in images]
    previous_image = images[0]
    norm = cfg.METHOD_CONFIGS[method]["norm"]
    previous_keypoints, previous_descriptors = extract(method, previous_image)
    if previous_descriptors is None or len(previous_keypoints) < 4:
        raise ValueError("Not enough features in image 1")

    homographies = []
    pair_stats = []
    for image_index, image in enumerate(images[1:], start=2):
        keypoints, descriptors = extract(method, image)
        if descriptors is None or len(keypoints) < 4:
            raise ValueError(f"Not enough features in image {image_index}")

        matches = match_pipeline(previous_descriptors, descriptors, norm, ratio)
        homography, mask = estimate_homography(matches, previous_keypoints, keypoints)
        if homography is None:
            raise ValueError(
                f"Could not align images {image_index - 1} and {image_index}")

        inliers, reproj_err = inlier_stats(
            matches, previous_keypoints, keypoints, homography, mask)
        homographies.append(homography)
        pair_stats.append({
            "image_index": image_index,
            "from_image": image_index - 1,
            "to_image": image_index,
            "n_matches": len(matches),
            "n_inliers": inliers,
            "reproj_err": reproj_err,
        })
        previous_keypoints = keypoints
        previous_descriptors = descriptors

    return homographies, pair_stats


def compute_reference_transforms(homographies, reference_index):
    """Return one transform per image into the reference coordinate system.

    reference_index is zero-based. For homogeneous column vectors:
    T_r = I; T_i = T_(i+1) @ H_i on the left of the reference;
    T_i = T_(i-1) @ inv(H_(i-1)) on the right.
    """
    n_images = len(homographies) + 1
    if (not isinstance(reference_index, (int, np.integer))
            or isinstance(reference_index, (bool, np.bool_))
            or not 0 <= reference_index < n_images):
        raise ValueError(f"Reference index must be between 0 and {n_images - 1}")
    transforms = [None] * n_images
    transforms[reference_index] = np.eye(3)
    for image_index in range(reference_index - 1, -1, -1):
        transforms[image_index] = (
            transforms[image_index + 1] @ homographies[image_index])
    for image_index in range(reference_index + 1, n_images):
        transforms[image_index] = (
            transforms[image_index - 1] @ np.linalg.inv(homographies[image_index - 1]))

    return transforms


def warp_images_to_canvas(images, transforms):
    """Return warped images, validity masks, canvas width and canvas height.

    Inputs are converted to grayscale. Outputs are float32 three-channel
    grayscale images for blending, and uint8 masks with values 0 and 255.
    All outputs share the same canvas and origin.
    """
    if not images or len(images) != len(transforms):
        raise ValueError("Provide one transform for each image")
    images = [_to_gray(image) for image in images]
    projected_corners = []
    for image, transform in zip(images, transforms):
        height, width = image.shape[:2]
        corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        projected_corners.append(
            cv2.perspectiveTransform(corners.reshape(-1, 1, 2), transform))

    all_corners = np.vstack(projected_corners)
    x_min = int(np.floor(all_corners[:, 0, 0].min()))
    y_min = int(np.floor(all_corners[:, 0, 1].min()))
    x_max = int(np.ceil(all_corners[:, 0, 0].max()))
    y_max = int(np.ceil(all_corners[:, 0, 1].max()))
    canvas_width = x_max - x_min
    canvas_height = y_max - y_min
    translation = np.array(
        [[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=float)

    warped_images = []
    masks = []
    for image, transform in zip(images, transforms):
        warp = translation @ transform
        warped = cv2.warpPerspective(image, warp, (canvas_width, canvas_height))
        source_mask = np.ones(image.shape[:2], dtype=np.uint8)
        mask = cv2.warpPerspective(
            source_mask, warp, (canvas_width, canvas_height),
            flags=cv2.INTER_NEAREST)
        warped_images.append(
            cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR).astype(np.float32))
        masks.append(mask * 255)

    return warped_images, masks, canvas_width, canvas_height


def crop_valid_rectangle(image, mask):
    """Crop the largest axis-aligned rectangle containing only valid pixels.

    Uses mask coverage, not pixel intensity, preserving naturally black pixels.
    Some valid content near the edges is sacrificed to obtain a full rectangle.
    """
    if mask.shape != image.shape[:2]:
        raise ValueError("Mask and image dimensions must match")
    valid = mask > 0
    heights = np.zeros(valid.shape[1], dtype=int)
    best_area = 0
    best_rect = None
    for y, row in enumerate(valid):
        heights = np.where(row, heights + 1, 0)
        stack = []
        for x in range(len(heights) + 1):
            height = heights[x] if x < len(heights) else 0
            start = x
            while stack and stack[-1][1] > height:
                left, previous_height = stack.pop()
                area = previous_height * (x - left)
                if area > best_area:
                    best_area = area
                    best_rect = (left, y - previous_height + 1, x, y + 1)
                start = left
            if height > 0 and (not stack or stack[-1][1] < height):
                stack.append((start, height))
    if best_rect is None:
        raise ValueError("Panorama contains no valid region")
    x1, y1, x2, y2 = best_rect
    return image[y1:y2, x1:x2].copy()


def save_panorama_versions(panorama, mask, output_path):
    """Save the full panorama and a cropped version alongside it.

    output_path retains the full panorama; the second filename adds _cropped.
    """
    cropped = crop_valid_rectangle(panorama, mask)
    stem, extension = os.path.splitext(output_path)
    cropped_path = f"{stem}_cropped{extension}"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    for path, image in [(output_path, panorama), (cropped_path, cropped)]:
        if not cv2.imwrite(path, image):
            raise OSError(f"Could not save panorama to {path}")
    print(f"Full panorama (with empty borders): {output_path}")
    print(f"Cropped panorama (without empty borders): {cropped_path}")
    return output_path, cropped_path


def blend_images(warped_images, masks, canvas_width, canvas_height, num_bands=5,
                 return_mask=False):
    """Find seams and blend aligned BGR images into a grayscale panorama.

    warped_images: float32 BGR images already warped onto the same canvas.
    masks: uint8 validity masks with values 0 and 255.
    return_mask: also return the final coverage mask when True.
    """
    seam_finder = cv2.detail_DpSeamFinder("COLOR")
    seam_masks = [cv2.UMat(mask) for mask in masks]
    seam_finder.find(
        [cv2.UMat(image) for image in warped_images],
        [(0, 0)] * len(warped_images),
        seam_masks,
    )

    blender = cv2.detail_MultiBandBlender()
    blender.setNumBands(num_bands)
    blender.prepare((0, 0, canvas_width, canvas_height))
    for image, mask in zip(warped_images, seam_masks):
        blender.feed(image.astype(np.int16), mask.get(), (0, 0))

    panorama_bgr, panorama_mask = blender.blend(None, None)
    panorama = cv2.cvtColor(
        cv2.convertScaleAbs(panorama_bgr), cv2.COLOR_BGR2GRAY)
    return (panorama, panorama_mask) if return_mask else panorama


def stitch_sequence(images, method="SIFT", ratio=None, reference_index=None):
    """Estimate consecutive homographies and blend the sequence into a panorama.

    Each adjacent homography maps image i to image i+1. With homogeneous
    column-vector coordinates, transforms to reference r are composed as
    T_r = I, T_i = T_(i+1) @ H_i for i < r, and
    T_i = T_(i-1) @ inv(H_(i-1)) for i > r. The middle image is used by
    default to limit the longest composition chain. Longer chains compound
    pairwise estimation errors, increasing drift and visible misalignment.

    Returned chain diagnostics measure the number of adjacent transforms and
    sum their mean reprojection errors along all paths to the reference. The
    latter is a comparison proxy, not a direct panorama alignment error.
    info["panorama_mask"] contains the valid coverage mask for cropping.
    """
    if len(images) < 2:
        return None, {"error": "At least two images are required"}

    if reference_index is None:
        reference_index = len(images) // 2
    if (not isinstance(reference_index, (int, np.integer))
            or isinstance(reference_index, (bool, np.bool_))
            or not 0 <= reference_index < len(images)):
        return None, {
            "error": f"Reference index must be between 0 and {len(images) - 1}"
        }

    images = [_to_gray(image) for image in images]
    try:
        homographies, pair_stats = estimate_pairwise_homographies(
            images, method, ratio)
    except ValueError as error:
        return None, {"error": str(error)}

    transforms = compute_reference_transforms(homographies, reference_index)
    warped_images, masks, canvas_width, canvas_height = warp_images_to_canvas(
        images, transforms)

    panorama, panorama_mask = blend_images(
        warped_images, masks, canvas_width, canvas_height, return_mask=True)
    chain_lengths = [abs(image_index - reference_index)
                     for image_index in range(len(images))]
    path_reproj_errors = [
        sum(pair_stats[edge_index]["reproj_err"]
            for edge_index in range(min(image_index, reference_index),
                                    max(image_index, reference_index)))
        for image_index in range(len(images))
    ]
    return panorama, {
        "panorama_mask": panorama_mask,
        "pairs": pair_stats,
        "reference_image": reference_index + 1,
        "reference_index": reference_index,
        "max_chain_hops": max(chain_lengths),
        "mean_chain_hops": float(np.mean(chain_lengths)),
        "summed_path_reproj_error_px": float(sum(path_reproj_errors)),
    }


# ==================================================================
# Main
# ==================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stitch consecutive images into a panorama.",
        epilog="Example: python src/stitching.py --image-dir data/Panorama "
               "--images keble_a.jpg keble_b.jpg keble_c.jpg",
    )
    parser.add_argument("--sequence", type=str,
                        help="HPatches sequence name to stitch using all six images")
    parser.add_argument("--image-dir", type=str,
                        help="Folder containing the custom input images")
    parser.add_argument("--images", nargs="+",
                        help="Ordered image filenames, from first to last")
    parser.add_argument("--reference-index", type=int,
                        help="1-based reference image index for comparison "
                             "(default: middle image)")
    parser.add_argument("--method", type=str, default="SIFT")
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    if not any((args.sequence, args.image_dir, args.images)):
        args.image_dir = IMAGE_DIR
        args.images = IMAGE_NAMES.copy()

    if args.image_dir or args.images:
        if not args.image_dir or not args.images:
            parser.error("--image-dir and --images must be provided together")
        if len(args.images) < 2:
            parser.error("At least two image filenames are required")
        if (args.reference_index is not None
                and not 1 <= args.reference_index <= len(args.images)):
            parser.error(
                f"--reference-index must be between 1 and {len(args.images)}")

        image_paths = [os.path.join(args.image_dir, name) for name in args.images]
        missing = [path for path in image_paths if not os.path.isfile(path)]
        if missing:
            parser.error(f"Image files not found: {', '.join(missing)}")
        images = [load_image_color(path)[0] for path in image_paths]
        reference_index = (
            None if args.reference_index is None else args.reference_index - 1)
        panorama, info = stitch_sequence(
            images, method=args.method, reference_index=reference_index)
        if panorama is None:
            parser.error(info["error"])

        image_label = "_".join(
            os.path.splitext(os.path.basename(name))[0] for name in args.images)
        output_path = args.output or os.path.join(
            "resultados", "stitching", f"{image_label}_panorama.png")
        save_panorama_versions(panorama, info["panorama_mask"], output_path)
        print(f"Reference image: {args.images[info['reference_image'] - 1]}")
        print(f"Maximum chain length: {info['max_chain_hops']} adjacent homographies")
        print("Summed path reprojection error: "
              f"{info['summed_path_reproj_error_px']:.3f} px (comparison proxy)")
        for pair in info["pairs"]:
            first = args.images[pair["from_image"] - 1]
            second = args.images[pair["to_image"] - 1]
            print(f"  Images {first} -> {second}: {pair['n_matches']} matches, "
                  f"{pair['n_inliers']} inliers")
    else:
        sequence_name = args.sequence or "v_adam"
        sequence_dirs = dict(get_sequence_dirs())
        if sequence_name not in sequence_dirs:
            parser.error(f"HPatches sequence not found: {sequence_name}")

        sequence_dir = sequence_dirs[sequence_name]
        images = [
            load_image_color(os.path.join(sequence_dir, f"{image_index}.ppm"))[0]
            for image_index in range(1, 7)
        ]
        if (args.reference_index is not None
                and not 1 <= args.reference_index <= len(images)):
            parser.error(
                f"--reference-index must be between 1 and {len(images)}")
        reference_index = (
            None if args.reference_index is None else args.reference_index - 1)
        panorama, info = stitch_sequence(
            images, method=args.method, reference_index=reference_index)
        if panorama is None:
            parser.error(info["error"])

        output_path = args.output or os.path.join(
            "resultados", "stitching", f"{sequence_name}_panorama.png")
        save_panorama_versions(panorama, info["panorama_mask"], output_path)
        print(f"Reference image: {info['reference_image']}")
        print(f"Maximum chain length: {info['max_chain_hops']} adjacent homographies")
        print("Summed path reprojection error: "
              f"{info['summed_path_reproj_error_px']:.3f} px (comparison proxy)")
        for pair in info["pairs"]:
            print(f"  Images {pair['from_image']} -> {pair['to_image']}: "
                  f"{pair['n_matches']} matches, "
                  f"{pair['n_inliers']} inliers")
