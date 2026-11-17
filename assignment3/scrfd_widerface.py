"""SCRFD detection + 5-point alignment + WIDER Face validation evaluation.

AP@IoU=0.5 below is a custom ranking AP (NOT official WIDER Easy/Medium/Hard).
WIDER Face has bounding-box labels, NOT landmark labels: alignment is visual only.
"""
import argparse
import csv
import json
import platform
import shutil
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np

DATASET = "CUHK-CSE/wider_face"


def get_zips(args):
    """Download once (cached by Hugging Face), or use local ZIP files."""
    if args.download:
        from huggingface_hub import hf_hub_download
        return [Path(hf_hub_download(repo_id=DATASET, repo_type="dataset", filename=f"data/{name}"))
                for name in ("WIDER_val.zip", "wider_face_split.zip")]
    files = [Path(args.images_zip), Path(args.annotations_zip)]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    return files


def load_annotations(gt_zip):
    """WIDER val TXT: image, number of faces, then x y w h and 6 attributes."""
    file = next((n for n in gt_zip.namelist() if n.endswith("wider_face_val_bbx_gt.txt")), None)
    if file is None:
        raise FileNotFoundError("wider_face_val_bbx_gt.txt not found in annotations ZIP")
    lines = iter(gt_zip.read(file).decode("utf-8").splitlines())
    samples = []
    for image in lines:
        image = image.strip()
        if not image:
            continue
        n_faces = int(next(lines))
        faces = []
        for _ in range(max(1, n_faces)):  # WIDER has a dummy line for zero-face images
            fields = list(map(int, next(lines).split()))
            if n_faces == 0:
                continue
            if len(fields) != 10:
                raise ValueError(f"Invalid WIDER annotation: {image}: {fields}")
            x, y, w, h, blur, expression, illumination, invalid, occlusion, pose = fields
            faces.append(dict(box=(x, y, x + w, y + h), valid=(not invalid and w > 0 and h > 0),
                              size=max(w, h), pose=pose, occlusion=occlusion))
        samples.append((image, faces))
    return samples


def intersection_over_union(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area = lambda q: max(0, q[2] - q[0]) * max(0, q[3] - q[1])
    union = area(a) + area(b) - intersection
    return intersection / union if union else 0.0


def match_faces(detections, gt, min_iou=0.5):
    """Confidence-ordered 1-to-1 matching; duplicate detections count as FP."""
    used = set()
    for d in sorted(detections, key=lambda item: item["score"], reverse=True):
        candidates = [(intersection_over_union(d["box"], g["box"]), i)
                      for i, g in enumerate(gt) if g["valid"] and i not in used]
        best_iou, best_idx = max(candidates, default=(0.0, -1))
        d["matched"] = -1
        d["ignored"] = False
        if best_iou >= min_iou:
            d["matched"] = best_idx
            used.add(best_idx)
        elif any(intersection_over_union(d["box"], g["box"]) >= min_iou
                 for g in gt if not g["valid"]):
            d["ignored"] = True


def evaluate(predictions, n_gt, score_threshold):
    """Image-independent AP ranking; P/R/F1 at a declared score threshold."""
    predictions = sorted((d for d in predictions if not d["ignored"]),
                         key=lambda d: d["score"], reverse=True)
    is_tp = np.array([d["matched"] >= 0 for d in predictions], dtype=bool)
    tp = np.cumsum(is_tp)
    ranks = np.arange(1, len(predictions) + 1)
    precision_curve = tp / ranks
    recall_curve = tp / max(1, n_gt)
    ap = float(precision_curve[is_tp].sum() / n_gt) if n_gt else None
    selected = [d for d in predictions if d["score"] >= score_threshold]
    true_pos = sum(d["matched"] >= 0 for d in selected)
    false_pos = len(selected) - true_pos
    false_neg = n_gt - true_pos
    p = true_pos / len(selected) if selected else 0.0
    r = true_pos / n_gt if n_gt else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return dict(AP_at_IoU_threshold=ap, TP=true_pos, FP=false_pos, FN=false_neg,
                precision=p, recall=r, f1=f1), recall_curve, precision_curve


def groups_table(gt_rows):
    """Face-level recall by WIDER labels (fixed confidence and IoU threshold)."""
    groups = {
        "all faces": lambda x: True,
        "tiny (<16 px)": lambda x: x["size"] < 16,
        "small (<32 px)": lambda x: x["size"] < 32,
        "medium (32-127 px)": lambda x: 32 <= x["size"] < 128,
        "large (>=128 px)": lambda x: x["size"] >= 128,
        "typical pose": lambda x: x["pose"] == 0,
        "atypical pose": lambda x: x["pose"] == 1,
        "no occlusion": lambda x: x["occlusion"] == 0,
        "partial occlusion": lambda x: x["occlusion"] == 1,
        "heavy occlusion": lambda x: x["occlusion"] == 2,
        "single-face images": lambda x: x["faces_in_image"] == 1,
        "multiple-face images": lambda x: x["faces_in_image"] >= 2,
    }
    table = []
    for name, predicate in groups.items():
        cohort = [x for x in gt_rows if predicate(x)]
        found = sum(x["detected"] for x in cohort)
        table.append(dict(group=name, n_gt=len(cohort), detected=found,
                          recall=found / len(cohort) if cohort else None))
    return table


def save_csv(file, rows):
    if not rows:
        return
    with file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_plots(out, rec, prec, ap, groups):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.step(np.r_[0, rec], np.r_[1, prec], where="post")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision",
           title=f"Detection PR — custom AP@IoU=0.5: {ap:.3f}" if ap is not None else "Detection PR")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "precision_recall.png", dpi=150)
    plt.close(fig)

    subset = [g for g in groups[1:] if g["recall"] is not None]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh([x["group"] for x in subset], [x["recall"] for x in subset])
    ax.invert_yaxis()
    ax.set(xlim=(0, 1), xlabel="Recall at fixed confidence / IoU", title="Robustness on WIDER Face")
    fig.tight_layout()
    fig.savefig(out / "robustness.png", dpi=150)
    plt.close(fig)


def save_preview(out_file, image, gt, detections, threshold):
    canvas = image.copy()
    for g in gt:
        if g["valid"]:
            cv2.rectangle(canvas, tuple(map(int, g["box"][:2])), tuple(map(int, g["box"][2:])), (255, 100, 0), 2)
    for d in detections:
        if d["score"] < threshold:
            continue
        p = np.asarray(d["box"], dtype=int)
        cv2.rectangle(canvas, tuple(p[:2]), tuple(p[2:]), (0, 220, 0), 2)
        cv2.putText(canvas, f'{d["score"]:.2f}', tuple(p[:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
        if d["kps"] is not None:
            for xy in d["kps"]:
                cv2.circle(canvas, tuple(np.round(xy).astype(int)), 2, (0, 0, 255), -1)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_file), canvas)


def create_detector(args):
    import onnxruntime as ort
    from insightface.app import FaceAnalysis
    available = ort.get_available_providers()
    if args.device == "cuda" and "CUDAExecutionProvider" not in available:
        raise RuntimeError(f"CUDA not available. ONNX providers: {available}")
    use_gpu = args.device != "cpu" and "CUDAExecutionProvider" in available
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"], providers=providers)
    app.prepare(ctx_id=0 if use_gpu else -1, det_size=(args.det_size, args.det_size),
                det_thresh=args.det_thresh)
    detection_model = app.models["detection"]
    if getattr(detection_model, "use_kps", True) is False:
        raise RuntimeError("SCRFD model does not provide 5 landmarks")
    for _ in range(3):
        app.get(np.zeros((args.det_size, args.det_size, 3), dtype=np.uint8))
    return app, detection_model.session.get_providers()


def main(args):
    from insightface.utils import face_align
    images_file, labels_file = get_zips(args)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    # Only clear generated subfolders, preventing stale crops from previous runs.
    for folder in ("aligned_faces", "visualizations", "wider_predictions"):
        if (out / folder).exists():
            shutil.rmtree(out / folder)
    for file in ("metrics.json", "detections.csv", "robustness.csv", "precision_recall.png", "robustness.png"):
        (out / file).unlink(missing_ok=True)

    app, providers = create_detector(args)
    gt_rows, pred_rows, predictions, latencies = [], [], [], []
    n_gt = crops = 0
    with zipfile.ZipFile(images_file) as images_zip, zipfile.ZipFile(labels_file) as labels_zip:
        samples = load_annotations(labels_zip)
        samples = samples[:args.limit] if args.limit else samples
        image_files = {name.split("/images/", 1)[1]: name for name in images_zip.namelist()
                       if "/images/" in name and name.lower().endswith(".jpg")}
        print(f"Processing {len(samples)} WIDER Face validation images")
        preview_indices = set(np.linspace(0, max(0, len(samples)-1),
                                      min(args.preview_count, len(samples)), dtype=int))

        for idx, (relative, gt) in enumerate(samples):
            filename = image_files.get(relative)
            if filename is None:
                raise FileNotFoundError(f"Image missing in WIDER_val.zip: {relative}")
            img = cv2.imdecode(np.frombuffer(images_zip.read(filename), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Cannot decode {relative}")

            t0 = time.perf_counter()
            faces = app.get(img)
            latencies.append((time.perf_counter() - t0) * 1000)
            dets = []
            for face in faces:
                kps = getattr(face, "kps", None)
                if kps is not None:
                    kps = np.asarray(kps, dtype=np.float32)
                    if kps.shape != (5, 2) or not np.isfinite(kps).all():
                        kps = None
                dets.append(dict(box=face.bbox.tolist(), score=float(face.det_score), kps=kps))
            dets.sort(key=lambda d: d["score"], reverse=True)
            match_faces(dets, gt, args.iou_thresh)

            valid_indices = [i for i, g in enumerate(gt) if g["valid"]]
            n_gt += len(valid_indices)
            found = {d["matched"] for d in dets if d["matched"] >= 0 and d["score"] >= args.eval_thresh}
            for i in valid_indices:
                g = gt[i]
                gt_rows.append(dict(image=relative, size=g["size"], pose=g["pose"],
                                    occlusion=g["occlusion"], faces_in_image=len(valid_indices),
                                    detected=int(i in found)))

            for j, d in enumerate(dets):
                crop_path = ""
                if args.save_crops and d["score"] >= args.eval_thresh and d["kps"] is not None:
                    try:
                        aligned = face_align.norm_crop(img, landmark=d["kps"], image_size=112)
                        path = out / "aligned_faces" / Path(relative).parent / f"{Path(relative).stem}_face{j:03d}.jpg"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        if not cv2.imwrite(str(path), aligned):
                            raise IOError(f"Failed to write {path}")
                        crop_path = str(path.relative_to(out))
                        crops += 1
                    except (ValueError, cv2.error, IOError) as exc:
                        print(f"Warning: alignment failed for {relative}, face {j}: {exc}")
                pred_rows.append(dict(image=relative, score=d["score"], bbox=json.dumps(d["box"]),
                                      landmarks=json.dumps(d["kps"].tolist()) if d["kps"] is not None else "",
                                      true_positive=int(d["matched"] >= 0), ignored=int(d["ignored"]),
                                      aligned_crop=crop_path))
            predictions.extend(dets)

            # Conventional WIDER official-scorer prediction format (but AP below is custom).
            if args.save_predictions:
                pred_file = out / "wider_predictions" / Path(relative).with_suffix(".txt")
                pred_file.parent.mkdir(parents=True, exist_ok=True)
                with pred_file.open("w", encoding="utf-8") as f:
                    f.write(f"{Path(relative).name}\n{len(dets)}\n")
                    for d in dets:
                        x1, y1, x2, y2 = d["box"]
                        f.write(f"{x1:.3f} {y1:.3f} {x2-x1:.3f} {y2-y1:.3f} {d['score']:.6f}\n")
            if idx in preview_indices:
                save_preview(out / "visualizations" / relative, img, gt, dets, args.eval_thresh)
            if (idx + 1) % 100 == 0 or idx == len(samples) - 1:
                print(f"  {idx+1}/{len(samples)} images | {crops} aligned faces", flush=True)

    evaluation, recall, precision = evaluate(predictions, n_gt, args.eval_thresh)
    groups = groups_table(gt_rows)
    save_csv(out / "detections.csv", pred_rows)
    save_csv(out / "robustness.csv", groups)
    save_plots(out, recall, precision, evaluation["AP_at_IoU_threshold"], groups)
    timing = dict(mean_ms=float(np.mean(latencies)), median_ms=float(np.median(latencies)),
                  p95_ms=float(np.percentile(latencies, 95))) if latencies else {}
    import onnxruntime as ort
    metrics = dict(model="SCRFD-10GF / buffalo_l (detection only)",
                   images_processed=len(samples), ground_truth_faces=n_gt,
                   predictions=len(predictions),
                   evaluation={**evaluation, "iou_threshold": args.iou_thresh,
                               "confidence_threshold": args.eval_thresh,
                               "note": "Custom detection AP; NOT official WIDER Easy/Medium/Hard AP; no TN-based accuracy."},
                   timing={"detection_only_inference_ms": timing,
                           "note": "Includes SCRFD preprocessing, forward, box decoding and NMS; excludes image decoding, loading, alignment and saving."},
                   alignment={"crops_saved": crops, "output_size": [112, 112],
                              "note": "WIDER Face lacks landmark ground truth; crop alignment accuracy not evaluated."},
                   parameters={"det_size": args.det_size, "det_thresh": args.det_thresh,
                               "eval_thresh": args.eval_thresh, "iou_thresh": args.iou_thresh, "device": args.device},
                   inference_providers=providers,
                   environment=dict(python=sys.version.split()[0], platform=platform.platform(),
                                    opencv=cv2.__version__, numpy=np.__version__, onnxruntime=ort.__version__))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nRESULTS:", json.dumps({"images": len(samples), "AP@IoU0.5": evaluation["AP_at_IoU_threshold"],
                                      "precision": evaluation["precision"], "recall": evaluation["recall"],
                                      "F1": evaluation["f1"], "median_ms": timing.get("median_ms"),
                                      "aligned_crops": crops}, indent=2))
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--download", action="store_true", help="Download/cache WIDER val images and annotations")
    p.add_argument("--images-zip", default="data/WIDER_val.zip")
    p.add_argument("--annotations-zip", default="data/wider_face_split.zip")
    p.add_argument("--limit", type=int, default=20, help="First N images; 0=full WIDER validation set")
    p.add_argument("--output", default=None, help="Output directory (automatic if omitted)")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--det-size", type=int, default=640)
    p.add_argument("--det-thresh", type=float, default=0.05, help="Low threshold for AP ranking")
    p.add_argument("--eval-thresh", type=float, default=0.5, help="Operating threshold for P/R/F1 and crops")
    p.add_argument("--iou-thresh", type=float, default=0.5)
    p.add_argument("--preview-count", type=int, default=12)
    p.add_argument("--no-save-crops", dest="save_crops", action="store_false")
    p.add_argument("--no-save-predictions", dest="save_predictions", action="store_false")
    a = p.parse_args()
    if a.limit < 0 or a.det_size < 128 or a.det_size % 32:
        p.error("--limit must be >=0; --det-size >=128 and divisible by 32")
    if not 0 <= a.det_thresh <= a.eval_thresh <= 1 or not 0 < a.iou_thresh <= 1:
        p.error("Require 0 <= det-thresh <= eval-thresh <= 1 and 0 < IoU <= 1")
    if a.output is None:
        a.output = "results/scrfd_wider_full" if a.limit == 0 else f"results/scrfd_wider_limit{a.limit}"
    main(a)
