"""Feature-based colour stitching, using the shared project modules.

Classical methods: SIFT, KAZE, ORB, FAST_BRIEF, FAST_BRISK, FAST_FREAK.
Optional learned pipeline: SUPERPOINT_LIGHTGLUE (https://github.com/cvg/LightGlue).
Install baseline dependencies: python -m pip install numpy "opencv-contrib-python>=4.8,<5"
Optional: python -m pip install torch torchvision
          python -m pip install git+https://github.com/cvg/LightGlue.git
Model weights may be downloaded on first use. No learned fallback is silent.

Defaults: BF L2 for float descriptors; Hamming for binary descriptors;
bidirectional Lowe ratio from config.py (default 0.8) followed by mutual consistency. LightGlue uses
confidence filtering at 0.1 instead. RANSAC threshold 3 px, confidence 0.995,
iteration budget from config.py (default 2000). Threshold is in pixels of the images actually processed.
A 3 px threshold is an initial localisation tolerance, not a universal optimum;
compare 1, 3, 5 px and inspect inliers and panoramas before justifying it.
RANSAC samples minimal sets, estimates models and finds a consensus under the
forward reprojection threshold; outliers are excluded from refinement.
Multiband/seam blending is provided by OpenCV; canvas, masks and averaging are
implemented here. Timings are single runs, not a rigorous performance benchmark.
"""
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np

import config as cfg
from matching import match_mutual_ratio, estimate_homography

METHODS = cfg.PART2_METHODS


def normalize_method(method):
    """Accept canonical project names and legacy FAST_BRIEF spellings."""
    key = method.upper().replace('_', '+')
    for candidate in METHODS:
        if candidate.upper().replace('_', '+') == key:
            return candidate
    raise ValueError(f'Unknown method {method}; choose from {METHODS}')


def to_bgr(image):
    if image is None or image.size == 0:
        raise ValueError('Empty input image')
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(path.suffix, image)
    if not ok:
        raise OSError(f'Could not encode {path}')
    path.write_bytes(data.tobytes())


class FeatureEngine:
    """Initialise models once; optional learned dependencies are imported lazily."""
    def __init__(self, method='SIFT', max_keypoints=cfg.MAX_KEYPOINTS, device='auto',
                 lightglue_threshold=cfg.PART2_LIGHTGLUE_THRESHOLD):
        self.method = normalize_method(method)
        self.max_keypoints = max_keypoints
        if self.method not in METHODS:
            raise ValueError(f'Unknown method {method}; choose from {METHODS}')
        self.norm = cv2.NORM_L2
        if self.method in ('SUPERPOINT_LIGHTGLUE', 'SuperPoint+NN'):
            try:
                import torch
                from lightglue import SuperPoint, LightGlue
            except ImportError as exc:
                raise RuntimeError('Install torch and cvg/LightGlue for the learned configuration') from exc
            self.torch = torch
            self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
            self.detector = SuperPoint(max_num_keypoints=max_keypoints).eval().to(self.device)
            self.matcher = (LightGlue(features='superpoint', filter_threshold=lightglue_threshold)
                            .eval().to(self.device) if self.method == 'SUPERPOINT_LIGHTGLUE' else None)
        else:
            settings = cfg.METHOD_CONFIGS[self.method]
            self.norm = settings['norm']

    def extract(self, image):
        if self.method in ('SUPERPOINT_LIGHTGLUE', 'SuperPoint+NN'):
            rgb = cv2.cvtColor(to_bgr(image), cv2.COLOR_BGR2RGB)
            tensor = self.torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div(255).to(self.device)
            with self.torch.inference_mode():
                features = self.detector.extract(tensor, resize=None)
            xy = features['keypoints'][0].detach().cpu().numpy()
            descriptors = (features['descriptors'][0].detach().cpu().numpy()
                           if self.method == 'SuperPoint+NN' else features)
            return [cv2.KeyPoint(float(x), float(y), 3) for x, y in xy], descriptors
        raise ValueError(
            "Classical features must be extracted in main_part2.py and passed "
            "to prepare_sequence."
        )

    def match(self, first, second, ratio=cfg.RATIO_TEST_THRESHOLD):
        if self.method == 'SUPERPOINT_LIGHTGLUE':
            with self.torch.inference_mode():
                out = self.matcher({'image0': first, 'image1': second})
            pairs = out['matches'][0].detach().cpu().numpy()
            return [cv2.DMatch(int(i), int(j), 0.0) for i, j in pairs]
        return match_mutual_ratio(first, second, self.norm, ratio)


def prepare_sequence(images, method='SIFT', ratio=cfg.RATIO_TEST_THRESHOLD, output_dir=None,
                     max_keypoints=cfg.MAX_KEYPOINTS, device='auto',
                     lightglue_threshold=cfg.PART2_LIGHTGLUE_THRESHOLD,
                     features=None, extraction_s=None):
    """Match pre-extracted features once for reuse in threshold studies.

    Classical (keypoints, descriptors) pairs are supplied by main_part2.py.
    Learned features are extracted here because their model is specific to
    this pipeline.
    """
    if len(images) < 2 or not 0 < ratio < 1:
        raise ValueError('Need at least two images and a ratio in (0, 1)')
    start = perf_counter()
    engine = FeatureEngine(method, max_keypoints, device, lightglue_threshold)
    initialization_s = perf_counter() - start
    if features is None:
        if engine.method not in ('SUPERPOINT_LIGHTGLUE', 'SuperPoint+NN'):
            raise ValueError(
                "Pass classical (keypoints, descriptors) features from main_part2.py."
            )
        start = perf_counter()
        features = [engine.extract(image) for image in images]
        extraction_s = perf_counter() - start
    elif len(features) != len(images):
        raise ValueError("Provide extracted features for every input image")
    elif extraction_s is None:
        extraction_s = 0.0
    if output_dir:
        for i, (kp, _) in enumerate(features):
            save_image(Path(output_dir) / f'keypoints_{i+1:02d}.jpg',
                       cv2.drawKeypoints(to_bgr(images[i]), kp, None,
                                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
    start = perf_counter()
    pairs = []
    for i in range(len(images)-1):
        kp0, des0 = features[i]
        kp1, des1 = features[i+1]
        matches = engine.match(des0, des1, ratio)
        pairs.append({'kp0': kp0, 'kp1': kp1, 'matches': matches,
                      'points0': np.float32([kp0[m.queryIdx].pt for m in matches]).reshape(-1, 2),
                      'points1': np.float32([kp1[m.trainIdx].pt for m in matches]).reshape(-1, 2)})
    return {'pairs': pairs, 'method': engine.method,
            'initialization_s': initialization_s, 'extraction_s': extraction_s,
            'matching_s': perf_counter()-start,
            'keypoint_counts': [len(kp) for kp, _ in features],
            'filter': ('LightGlue confidence' if engine.method == 'SUPERPOINT_LIGHTGLUE'
                       else 'BF k=2, bidirectional ratio, mutual consistency')}


def estimate_pairwise_homographies(images, method='SIFT', ratio=cfg.RATIO_TEST_THRESHOLD,
                                  ransac_threshold=cfg.RANSAC_REPROJ_THRESHOLD, prepared=None,
                                  output_dir=None, seed=cfg.RANDOM_SEED):
    """Estimate adjacent H_i: image i -> image i+1 using point RANSAC.

    Report mean forward Euclidean reprojection error of the returned inliers.
    Require >=4 non-collinear inliers and a finite, invertible homography.
    """
    if not np.isfinite(ransac_threshold) or ransac_threshold <= 0:
        raise ValueError('RANSAC threshold must be positive')
    prepared = prepared or prepare_sequence(images, method, ratio, output_dir)
    homographies, stats = [], []
    for i, pair in enumerate(prepared['pairs']):
        p, q = pair['points0'], pair['points1']
        if len(p) < 4:
            raise ValueError(f'Pair {i+1}-{i+2}: only {len(p)} filtered matches')
        if any(np.linalg.matrix_rank(v-v.mean(axis=0)) < 2 for v in (p, q)):
            raise ValueError('Collinear correspondences')
        start = perf_counter()
        H, mask = estimate_homography(pair['matches'], pair['kp0'], pair['kp1'],
                                      threshold=ransac_threshold, seed=seed)
        elapsed = perf_counter()-start
        if H is None or mask is None or not np.isfinite(H).all() or np.linalg.matrix_rank(H) < 3:
            raise ValueError(f'Pair {i+1}-{i+2}: no valid homography')
        valid = mask.ravel().astype(bool)
        if valid.sum() < 4 or any(np.linalg.matrix_rank(v[valid]-v[valid].mean(axis=0)) < 2 for v in (p, q)):
            raise ValueError('Insufficient non-collinear RANSAC inliers')
        projected = cv2.perspectiveTransform(p.reshape(-1, 1, 2), H).reshape(-1, 2)
        errors = np.linalg.norm(projected-q, axis=1)
        if not np.isfinite(errors[valid]).all():
            raise ValueError('Non-finite inlier errors')
        homographies.append(H)
        stats.append({'from_image': i+1, 'to_image': i+2,
                      'n_matches': len(p), 'n_inliers': int(valid.sum()),
                      'inlier_ratio': float(valid.mean()),
                      'reproj_err': float(errors[valid].mean()),
                      'ransac_threshold_px': ransac_threshold, 'ransac_s': elapsed})
        if output_dir:
            view = cv2.drawMatches(to_bgr(images[i]), pair['kp0'],
                                   to_bgr(images[i+1]), pair['kp1'], pair['matches'], None,
                                   matchesMask=valid.astype(int).tolist(),
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            save_image(Path(output_dir)/f'inliers_{i+1:02d}_{i+2:02d}.jpg', view)
    return homographies, stats


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


def warp_images_to_canvas(images, transforms, max_canvas_pixels=cfg.PART2_MAX_CANVAS_PIXELS):
    """Return warped images, validity masks, canvas width and canvas height.

    Inputs retain their BGR colours. Outputs are float32 BGR images
    for blending, and uint8 masks with values 0 and 255.
    All outputs share the same canvas and origin.
    """
    if not images or len(images) != len(transforms):
        raise ValueError("Provide one transform for each image")
    images = [to_bgr(image) for image in images]
    projected_corners = []
    for image, transform in zip(images, transforms):
        height, width = image.shape[:2]
        corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        homogeneous = np.c_[corners, np.ones(4)] @ transform.T
        denominators = homogeneous[:, 2]
        if (not np.isfinite(homogeneous).all() or
                denominators.min() <= 0 <= denominators.max() or
                np.min(np.abs(denominators)) < 1e-9):
            raise ValueError("Transform crosses the projective horizon")
        projected_corners.append(
            cv2.perspectiveTransform(corners.reshape(-1, 1, 2), transform))

    all_corners = np.vstack(projected_corners)
    x_min = int(np.floor(all_corners[:, 0, 0].min()))
    y_min = int(np.floor(all_corners[:, 0, 1].min()))
    x_max = int(np.ceil(all_corners[:, 0, 0].max()))
    y_max = int(np.ceil(all_corners[:, 0, 1].max()))
    canvas_width = x_max - x_min
    canvas_height = y_max - y_min
    if (canvas_width <= 0 or canvas_height <= 0 or
            canvas_width * canvas_height > max_canvas_pixels):
        raise ValueError("Unsafe canvas size; reduce image size or inspect matches")
    translation = np.array(
        [[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=float)

    warped_images = []
    masks = []
    for image, transform in zip(images, transforms):
        warp = translation @ transform
        warped = cv2.warpPerspective(image, warp, (canvas_width, canvas_height))
        source_mask = np.ones(image.shape[:2], dtype=np.float32)
        mask = cv2.warpPerspective(
            source_mask, warp, (canvas_width, canvas_height),
            flags=cv2.INTER_LINEAR)
        mask = (mask >= 1.0 - 1e-6).astype(np.uint8)
        warped_images.append(
            warped.astype(np.float32))
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



def blend_images(warped_images, masks, canvas_width, canvas_height,
                 num_bands=5, return_mask=False, blending='average'):
    """Average all valid samples or use OpenCV seam finding + multiband blending."""
    if blending == 'average':
        total = np.zeros((canvas_height, canvas_width, 3), np.float32)
        count = np.zeros((canvas_height, canvas_width), np.float32)
        for image, mask in zip(warped_images, masks):
            valid = mask > 0
            total += image * valid[..., None]
            count += valid
        panorama = np.clip(total / np.maximum(count[..., None], 1), 0, 255).astype(np.uint8)
        coverage = ((count > 0) * 255).astype(np.uint8)
    elif blending == 'multiband':
        seam_masks = [cv2.UMat(mask.copy()) for mask in masks]
        cv2.detail_DpSeamFinder('COLOR').find(
            [cv2.UMat(im) for im in warped_images], [(0, 0)]*len(masks), seam_masks)
        blender = cv2.detail_MultiBandBlender()
        blender.setNumBands(num_bands)
        blender.prepare((0, 0, canvas_width, canvas_height))
        for image, mask in zip(warped_images, seam_masks):
            blender.feed(image.astype(np.int16), mask.get(), (0, 0))
        result, coverage = blender.blend(None, None)
        # Clipping, unlike convertScaleAbs, does not turn negative overshoot positive.
        panorama = np.clip(result, 0, 255).astype(np.uint8)
    else:
        raise ValueError('Blending must be average or multiband')
    return (panorama, coverage) if return_mask else panorama


def render_panorama(images, homographies, reference_index=None, blending='average',
                    max_canvas_pixels=cfg.PART2_MAX_CANVAS_PIXELS):
    reference_index = len(images)//2 if reference_index is None else reference_index
    transforms = compute_reference_transforms(homographies, reference_index)
    warped, masks, width, height = warp_images_to_canvas(images, transforms, max_canvas_pixels)
    panorama, mask = blend_images(warped, masks, width, height, blending=blending, return_mask=True)
    return panorama, {'panorama_mask': mask, 'reference_image': reference_index+1,
                      'reference_index': reference_index, 'canvas_width': width,
                      'canvas_height': height,
                      'coverage_fraction': float(np.mean(mask > 0)),
                      'max_chain_hops': max(abs(i-reference_index) for i in range(len(images)))}


def stitch_sequence(images, method='SIFT', ratio=cfg.RATIO_TEST_THRESHOLD, reference_index=None,
                    ransac_threshold=cfg.RANSAC_REPROJ_THRESHOLD, blending='average', output_dir=None,
                    max_keypoints=cfg.MAX_KEYPOINTS, device='auto',
                    max_canvas_pixels=cfg.PART2_MAX_CANVAS_PIXELS,
                    features=None, extraction_s=None):
    """Convenience API. Returns (panorama, info), or (None, error info).

    Adjacent homographies are chained toward a common reference. Longer chains
    accumulate estimation error. Chain length is a diagnostic, not an alignment
    accuracy metric. Compare exported panoramas visually for drift/distortion.
    """
    try:
        prepared = prepare_sequence(
            images, method, ratio, output_dir, max_keypoints, device,
            features=features, extraction_s=extraction_s)
        H, stats = estimate_pairwise_homographies(images, ransac_threshold=ransac_threshold,
                                                 prepared=prepared, output_dir=output_dir)
        panorama, info = render_panorama(images, H, reference_index, blending, max_canvas_pixels)
        info['pairs'] = stats
        return panorama, info
    except (ValueError, RuntimeError, cv2.error, np.linalg.LinAlgError) as exc:
        return None, {'error': str(exc)}
