from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from defense.module_a.ppe_postprocess import (  # noqa: E402
    BARE_HEAD_HINTS,
    HELMET_HINTS,
    PPEPostprocessConfig,
    PPEDetection,
    bbox_area,
    bbox_iou,
    label_matches,
    summarize_ppe_from_detections,
)
from defense.module_a.backends.detector_backend import YoloV5DetectorBackend  # noqa: E402


NO_HELMET_CATEGORIES = {"normal_no_helmet", "reflective_vest_no_helmet", "far_small_targets", "attack_green_vest"}


def safe_name(text: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(text)).strip("._-")
    return cleaned or "model"


def model_run_name(model_path: Path) -> str:
    parts = [part for part in model_path.parts[-4:] if part.lower() not in {"weights", "runs"}]
    return safe_name("_".join(parts).replace(model_path.suffix, ""))


class SimpleDetections:
    def __init__(self, boxes: list[list[float]], classes: list[int], confidences: list[float], names: dict[int, str]):
        self.boxes = boxes
        self.classes = classes
        self.confidences = confidences
        self.names = names


def load_frames_from_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        return list(csv.DictReader(fp))


def load_frames_from_dir(path: Path, category: str = "unknown") -> list[dict[str, str]]:
    return [{"frame_path": str(image_path), "category": category, "video_id": image_path.parent.name, "frame_idx": image_path.stem} for image_path in sorted(path.rglob("*.jpg"))]


def detections_from_result(result: Any) -> SimpleDetections:
    names = getattr(result, "names", {}) or {}
    boxes_obj = getattr(result, "boxes", None)
    boxes: list[list[float]] = []
    classes: list[int] = []
    confidences: list[float] = []
    if boxes_obj is None:
        return SimpleDetections(boxes, classes, confidences, names)
    xyxy = boxes_obj.xyxy.detach().cpu().tolist() if hasattr(boxes_obj.xyxy, "detach") else boxes_obj.xyxy.tolist()
    cls = boxes_obj.cls.detach().cpu().tolist() if hasattr(boxes_obj.cls, "detach") else boxes_obj.cls.tolist()
    conf = boxes_obj.conf.detach().cpu().tolist() if hasattr(boxes_obj.conf, "detach") else boxes_obj.conf.tolist()
    for bbox, class_id, confidence in zip(xyxy, cls, conf):
        boxes.append([float(v) for v in bbox[:4]])
        classes.append(int(class_id))
        confidences.append(float(confidence))
    return SimpleDetections(boxes, classes, confidences, {int(k): str(v) for k, v in names.items()})


def detections_from_backend_result(result: Any) -> SimpleDetections:
    return SimpleDetections(
        [[float(v) for v in bbox[:4]] for bbox in getattr(result, "boxes", [])],
        [int(v) for v in getattr(result, "classes", [])],
        [float(v) for v in getattr(result, "confidences", [])],
        {int(k): str(v) for k, v in getattr(result, "names", {}).items()},
    )


def ppe_items(detections: SimpleDetections) -> list[PPEDetection]:
    return [
        PPEDetection(index, detections.names.get(class_id, f"class_{class_id}"), class_id, confidence, tuple(float(v) for v in bbox[:4]))
        for index, (bbox, class_id, confidence) in enumerate(zip(detections.boxes, detections.classes, detections.confidences))
    ]


def classify_candidate(row: dict[str, str], detections: SimpleDetections, frame_shape: tuple[int, int]) -> tuple[str | None, dict[str, Any]]:
    cfg = PPEPostprocessConfig()
    summary = summarize_ppe_from_detections(detections, config=cfg, frame_shape=frame_shape)
    items = ppe_items(detections)
    helmets = [item for item in items if label_matches(item.label, HELMET_HINTS)]
    heads = [item for item in items if label_matches(item.label, BARE_HEAD_HINTS)]
    max_iou = 0.0
    for helmet in helmets:
        for head in heads:
            max_iou = max(max_iou, bbox_iou(helmet.bbox, head.bbox))
    category = row.get("category", "unknown")
    max_helmet_conf = max((item.confidence for item in helmets), default=0.0)
    min_helmet_area = min((bbox_area(item.bbox) / max(1, frame_shape[0] * frame_shape[1]) for item in helmets), default=1.0)
    bucket: str | None = None
    effective_helmet_count = int(summary.get("helmet_count", 0) or 0)
    suppressed_count = len(summary.get("helmet_fp_suppression", {}).get("suppressed_helmet_indices", []))
    if max_iou >= cfg.overlap_iou and effective_helmet_count > 0:
        bucket = "head_helmet_overlap"
    elif category in NO_HELMET_CATEGORIES and effective_helmet_count > 0:
        bucket = "helmet_fp"
    elif effective_helmet_count > 0 and (0.25 <= max_helmet_conf <= 0.60 or min_helmet_area < cfg.small_target_area_ratio):
        bucket = "uncertain"
    elif suppressed_count > 0:
        bucket = "suppressed"
    return bucket, {"summary": summary, "max_head_helmet_iou": max_iou, "max_helmet_conf": max_helmet_conf}


def draw_boxes(image, detections: SimpleDetections):
    for bbox, class_id, confidence in zip(detections.boxes, detections.classes, detections.confidences):
        label = detections.names.get(class_id, f"class_{class_id}")
        color = (180, 180, 180)
        if label_matches(label, HELMET_HINTS):
            color = (0, 0, 255)
        elif label_matches(label, BARE_HEAD_HINTS):
            color = (255, 0, 0)
        elif "person" in label.lower():
            color = (0, 220, 255)
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{label} {confidence:.2f}", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="挖掘YOLO安全帽误检候选帧。")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--frames-dir", type=Path)
    parser.add_argument("--category", default="unknown")
    parser.add_argument("--output-root", type=Path, default=Path("materials/yolo_fp_review"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--family", choices=["auto", "ultralytics", "yolov5"], default="auto")
    parser.add_argument("--backend", choices=["auto", "pytorch", "onnx"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.manifest and not args.frames_dir:
        raise SystemExit("需要 --manifest 或 --frames-dir")
    rows = load_frames_from_manifest(args.manifest) if args.manifest else load_frames_from_dir(args.frames_dir, args.category)
    if args.max_frames > 0:
        rows = rows[: args.max_frames]
    family = args.family
    if family == "auto":
        family = "yolov5" if "yolov5" in str(args.model).lower() else "ultralytics"
    backend = args.backend
    if backend == "auto":
        backend = "onnx" if args.model.suffix.lower() == ".onnx" else "pytorch"
    model: Any
    if family == "yolov5":
        model = YoloV5DetectorBackend(
            artifact_path=args.model,
            backend=backend,
            device=args.device,
            half=False,
            confidence=args.conf,
            image_size=args.imgsz,
        )
    else:
        from ultralytics import YOLO

        model = YOLO(str(args.model))
    run_name = model_run_name(args.model)
    detection_csv = args.output_root / "manifests" / f"yolo_detections_{run_name}.csv"
    frame_csv = args.output_root / "manifests" / f"yolo_frame_summary_{run_name}.csv"
    detection_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_counts: dict[str, int] = {"frames": 0, "helmet_fp": 0, "head_helmet_overlap": 0, "uncertain": 0, "suppressed": 0}
    category_counts: dict[str, dict[str, int]] = {}
    with detection_csv.open("w", newline="", encoding="utf-8-sig") as fp, frame_csv.open("w", newline="", encoding="utf-8-sig") as frame_fp:
        writer = csv.DictWriter(fp, fieldnames=["frame_path", "category", "bucket", "class_id", "label", "confidence", "x1", "y1", "x2", "y2"])
        writer.writeheader()
        frame_writer = csv.DictWriter(
            frame_fp,
            fieldnames=[
                "frame_path",
                "category",
                "bucket",
                "raw_helmet_count",
                "effective_helmet_count",
                "head_count",
                "person_count",
                "suppressed_helmet_count",
                "max_head_helmet_iou",
                "max_helmet_conf",
                "reason",
            ],
        )
        frame_writer.writeheader()
        for row in rows:
            image_path = Path(row["frame_path"])
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            if family == "yolov5":
                result = model.predict(image)
                detections = detections_from_backend_result(result)
            else:
                result = model.predict(str(image_path), conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]
                detections = detections_from_result(result)
            bucket, metrics = classify_candidate(row, detections, image.shape[:2])
            summary = metrics["summary"]
            summary_counts["frames"] += 1
            if bucket:
                summary_counts[bucket] = summary_counts.get(bucket, 0) + 1
                category = row.get("category", "unknown")
                category_counts.setdefault(category, {})[bucket] = category_counts.setdefault(category, {}).get(bucket, 0) + 1
                if bucket != "suppressed":
                    target_dir = args.output_root / "frames_candidates" / bucket
                    target_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(image_path, target_dir / image_path.name)
                    cv2.imwrite(str(target_dir / f"{image_path.stem}_annotated.jpg"), draw_boxes(image.copy(), detections))
            frame_writer.writerow(
                {
                    "frame_path": str(image_path),
                    "category": row.get("category", "unknown"),
                    "bucket": bucket or "",
                    "raw_helmet_count": summary.get("raw_helmet_count", 0),
                    "effective_helmet_count": summary.get("helmet_count", 0),
                    "head_count": summary.get("head_count", 0),
                    "person_count": summary.get("person_count", 0),
                    "suppressed_helmet_count": len(summary.get("helmet_fp_suppression", {}).get("suppressed_helmet_indices", [])),
                    "max_head_helmet_iou": f"{float(metrics.get('max_head_helmet_iou', 0.0)):.6f}",
                    "max_helmet_conf": f"{float(metrics.get('max_helmet_conf', 0.0)):.6f}",
                    "reason": summary.get("reason", ""),
                }
            )
            for bbox, class_id, confidence in zip(detections.boxes, detections.classes, detections.confidences):
                writer.writerow({"frame_path": str(image_path), "category": row.get("category", "unknown"), "bucket": bucket or "", "class_id": class_id, "label": detections.names.get(class_id, f"class_{class_id}"), "confidence": f"{confidence:.6f}", "x1": f"{bbox[0]:.2f}", "y1": f"{bbox[1]:.2f}", "x2": f"{bbox[2]:.2f}", "y2": f"{bbox[3]:.2f}"})
    report_path = args.output_root / "reports" / f"yolo_fp_summary_{run_name}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {**summary_counts, "by_category": category_counts, "frame_summary": str(frame_csv)}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"detections": str(detection_csv), "frame_summary": str(frame_csv), "report": str(report_path), **summary_counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
