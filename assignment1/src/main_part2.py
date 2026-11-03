"""Run the full Part 2 stitching comparison on images in data/foto."""
import csv
import json
import re
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
import config as cfg
import descriptors
import detectors
from data_loader import load_stitching_image
from stitching import (prepare_sequence, estimate_pairwise_homographies,
                       render_panorama, crop_valid_rectangle, save_image)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE_DIR = PROJECT_ROOT / 'data' / 'foto'
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / cfg.PART2_OUTPUT_DIR
IMAGE_EXTENSIONS = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp', '.ppm'}

DETECTOR_FUNCTIONS = {
    'SIFT': detectors.sift_detector,
    'KAZE': detectors.kaze_detector,
    'ORB': detectors.orb_detector,
    'FAST': detectors.fast_detector,
}
DESCRIPTOR_FUNCTIONS = {
    'SIFT': descriptors.compute_sift_descriptors,
    'KAZE': descriptors.compute_kaze_descriptors,
    'ORB': descriptors.compute_orb_descriptors,
    'BRIEF': descriptors.compute_brief_descriptors,
    'BRISK': descriptors.compute_brisk_descriptors,
    'FREAK': descriptors.compute_freak_descriptors,
}


def list_image_names(image_dir):
    def key(path):
        return [int(s) if s.isdigit() else s.casefold() for s in re.split(r'(\d+)', path.name)]
    return [p.name for p in sorted(image_dir.iterdir(), key=key)
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]


def load_image(path, max_side):
    return load_stitching_image(path, max_side)


def extract_classical_features(images, method, max_keypoints):
    """Call the shared detector and descriptor modules for each image."""
    method_config = cfg.METHOD_CONFIGS.get(method)
    if method_config is None:
        return None, 0.0

    detector_name = method_config['detector']
    descriptor_name = method_config['descriptor']
    detector = DETECTOR_FUNCTIONS.get(detector_name)
    describer = DESCRIPTOR_FUNCTIONS.get(descriptor_name)
    if detector is None or describer is None:
        raise ValueError(
            f'No detector/descriptor implementation for {method}'
        )

    params = cfg.DETECTOR_PARAMS[detector_name]
    start = perf_counter()
    features = []
    for image in images:
        if detector_name in ('SIFT', 'ORB'):
            keypoints = detector(image, nfeatures=max_keypoints)
        elif detector_name == 'KAZE':
            keypoints = detector(image, threshold=params['threshold'])
        else:
            keypoints = detector(
                image,
                threshold=params['threshold'],
                nonmax_suppression=params['nonmaxSuppression'],
            )

        keypoints = sorted(
            keypoints, key=lambda point: point.response, reverse=True
        )[:max_keypoints]
        if detector_name == 'FAST':
            for keypoint in keypoints:
                keypoint.size = cfg.FAST_KP_SIZE
        features.append(describer(image, keypoints))

    return features, perf_counter() - start


def save_panorama_versions(panorama, mask, output_path):
    path = Path(output_path)
    save_image(path, panorama)
    cropped_path = path.with_name(path.stem+'_cropped'+path.suffix)
    save_image(cropped_path, crop_valid_rectangle(panorama, mask))
    return path, cropped_path


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def reference_indices(values, n):
    refs = []
    for value in values:
        index = {'first': 0, 'middle': n//2, 'last': n-1}.get(value)
        if index is None:
            index = int(value)-1
        if not 0 <= index < n:
            raise ValueError(f'Reference {value} outside 1..{n}')
        if index not in refs:
            refs.append(index)
    return refs


def main():
    methods = list(cfg.PART2_COMPARE_METHODS)
    thresholds = list(cfg.PART2_RANSAC_THRESHOLDS)
    refs_spec = ['first', 'middle']
    blends = ['average', 'multiband']
    image_dir = DEFAULT_IMAGE_DIR
    if not image_dir.is_dir():
        raise FileNotFoundError(f'Image folder not found: {image_dir}')
    names = list_image_names(image_dir)
    if len(names) < 2:
        raise ValueError(f'At least two images are required in {image_dir}')
    refs = reference_indices(refs_spec, len(names))
    loaded = [load_image(image_dir/name, cfg.PART2_MAX_SIDE) for name in names]
    images = [item[0] for item in loaded]
    root = DEFAULT_OUTPUT_DIR
    root.mkdir(parents=True, exist_ok=True)
    settings = {
        'image_dir': str(image_dir),
        'images': names,
        'methods': methods,
        'thresholds_px': thresholds,
        'references_1based': [r+1 for r in refs],
        'blendings': blends,
        'max_side': cfg.PART2_MAX_SIDE,
        'max_keypoints': cfg.MAX_KEYPOINTS,
        'device': 'auto',
        'lightglue_threshold': cfg.PART2_LIGHTGLUE_THRESHOLD,
        'max_canvas_pixels': cfg.PART2_MAX_CANVAS_PIXELS,
        'seed': cfg.RANDOM_SEED,
        'inputs': [item[1] for item in loaded],
        'opencv_version': cv2.__version__,
        'ransac_confidence': cfg.RANSAC_CONFIDENCE,
        'ransac_max_iters': cfg.RANSAC_MAX_ITERS,
        'detector_params': cfg.DETECTOR_PARAMS,
        'classical_filter': 'BF KNN k=2, ratio both directions, mutual consistency',
        'pixel_units': 'processed image pixels',
        'timing_note': 'Single runs; model initialisation and output I/O are separate',
        'reference_note': 'Pairwise errors are reference-independent; inspect panorama distortion and drift',
    }
    (root/'settings.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    pair_rows, run_rows, feature_rows = [], [], []
    successful = set()
    for method in dict.fromkeys(methods):
        print(f'\nExtracting {method}...', flush=True)
        method_dir = root/method
        try:
            features, extraction_s = extract_classical_features(
                images, method, cfg.MAX_KEYPOINTS)
            prepared = prepare_sequence(images, method, cfg.RATIO_TEST_THRESHOLD,
                                        method_dir, cfg.MAX_KEYPOINTS, 'auto',
                                        cfg.PART2_LIGHTGLUE_THRESHOLD, features,
                                        extraction_s)
            feature_rows.append({k: v for k, v in prepared.items() if k != 'pairs'})
        except Exception as exc:
            run_rows.append({'method': method, 'status': 'failed', 'stage': 'features', 'error': str(exc)})
            print(f'{method} unavailable/failed: {exc}', flush=True)
            continue
        for threshold in dict.fromkeys(thresholds):
            threshold_dir = method_dir/f'ransac_{threshold:g}px'
            try:
                homographies, stats = estimate_pairwise_homographies(
                    images, prepared=prepared, ransac_threshold=threshold,
                    output_dir=threshold_dir, seed=cfg.RANDOM_SEED)
                np.savez(threshold_dir/'adjacent_homographies.npz',
                         **{f'H_{i+1}_to_{i+2}': H for i, H in enumerate(homographies)})
                pair_rows.extend({'method': method, **stat} for stat in stats)
            except Exception as exc:
                run_rows.append({'method': method, 'ransac_threshold_px': threshold,
                                 'status': 'failed', 'stage': 'homography', 'error': str(exc)})
                print(f'RANSAC {threshold:g}: {exc}', flush=True)
                continue
            for ref in refs:
                for blend in blends:
                    record = {'method': method, 'ransac_threshold_px': threshold,
                              'reference_image': ref+1, 'blending': blend}
                    try:
                        start = perf_counter()
                        panorama, info = render_panorama(images, homographies, ref, blend,
                                                         cfg.PART2_MAX_CANVAS_PIXELS)
                        record['render_s'] = perf_counter()-start
                        output = threshold_dir/f'ref_{ref+1}_{blend}.png'
                        save_panorama_versions(panorama, info['panorama_mask'], output)
                        record.update({k: v for k, v in info.items() if k != 'panorama_mask'})
                        record.update(status='ok', output=str(output))
                        successful.add(method)
                        print(f'Saved {output}', flush=True)
                    except Exception as exc:
                        record.update(status='failed', stage='render', error=str(exc))
                        print(f'Failed {record}: {exc}', flush=True)
                    run_rows.append(record)
        # Persist intermediate results so a later method failure does not erase them.
        write_csv(root/'pair_metrics.csv', pair_rows)
        write_csv(root/'runs.csv', run_rows)
    write_csv(root/'feature_metrics.csv', feature_rows)
    write_csv(root/'pair_metrics.csv', pair_rows)
    write_csv(root/'runs.csv', run_rows)
    summary = {'successful_methods': sorted(successful),
               'at_least_three_methods_succeeded': len(successful) >= 3,
               'learned_method_succeeded': bool(successful & {'SUPERPOINT_LIGHTGLUE', 'SuperPoint+NN'}),
               'note': 'Full comparison requested; inspect runs.csv for failures.'}
    (root/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f'Metrics and settings: {root}')
    if not successful:
        raise SystemExit('No panorama succeeded; inspect runs.csv')


if __name__ == '__main__':
    main()
