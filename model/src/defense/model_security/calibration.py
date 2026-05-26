from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from defense.runtime.config import DEFAULT_CONFIG_PATH, load_runtime_config, project_root
from model_security_gate.detox.external_hard_suite import (
    ExternalHardSuiteConfig,
    run_external_hard_suite,
    write_external_hard_suite_outputs,
)
from model_security_gate.utils.io import label_path_for_image, read_yolo_labels

from .fingerprint import build_model_fingerprint
from .runtime_adapter import create_module_a_detector_adapter
from .scanner import _class_name_map


DEFAULT_DATASET_ROOTS = {
    "mask": r"D:\security_project_d\model_b_old\datasets\mask_bd_external_eval",
    "poison": r"D:\security_project_d\model_b_old\datasets\poison_benchmark_cuda_tuned_remap_v2",
}


@dataclass(frozen=True)
class CalibrationCase:
    target_name: str
    target_classes: tuple[str, ...]
    dataset_name: str
    roots: tuple[str, ...]
    oda_success_mode: str


@dataclass
class CalibrationResult:
    case: dict[str, Any]
    status: str
    max_asr: float
    mean_asr: float
    rows: int
    elapsed_s: float
    top_attacks: list[dict[str, Any]]
    report_json_path: str | None = None
    rows_csv_path: str | None = None
    error: str | None = None


def _utc_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _case_slug(case: CalibrationCase) -> str:
    target = case.target_name.replace("+", "_")
    return f"{case.dataset_name}__{target}__{case.oda_success_mode}"


def _target_ids(names: Mapping[int, str], target_classes: Sequence[str]) -> list[int]:
    reverse = {str(name).lower(): int(idx) for idx, name in names.items()}
    ids: list[int] = []
    for name in target_classes:
        key = str(name).strip().lower()
        if key not in reverse:
            raise ValueError(f"Target class is not present in runtime names: {name}")
        ids.append(reverse[key])
    return list(dict.fromkeys(ids))


def _status_from_asr(max_asr: float, rows: int, allowed: float, suspicious: float) -> str:
    if rows <= 0:
        return "unverifiable"
    if max_asr <= allowed:
        return "clean"
    if max_asr >= suspicious:
        return "suspicious"
    return "review"


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _safe_slug(value: Any) -> str:
    text = str(value or "sample").strip()
    out = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    return out.strip("_") or "sample"


def _detection_to_dict(det: Any) -> dict[str, Any]:
    to_dict = getattr(det, "to_dict", None)
    if callable(to_dict):
        data = to_dict()
    else:
        data = {
            "xyxy": getattr(det, "xyxy", ()),
            "conf": getattr(det, "conf", None),
            "cls_id": getattr(det, "cls_id", None),
            "cls_name": getattr(det, "cls_name", ""),
        }
    if "xyxy" in data:
        data["xyxy"] = [float(v) for v in data["xyxy"]]
    if data.get("conf") is not None:
        data["conf"] = float(data["conf"])
    if data.get("cls_id") is not None:
        data["cls_id"] = int(data["cls_id"])
    return data


def _draw_box(image: Any, box: Sequence[float], color: tuple[int, int, int], text: str) -> None:
    import cv2

    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    label = str(text)[:80]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    y_text = max(th + 4, y1 - 4)
    cv2.rectangle(image, (x1, y_text - th - 4), (min(w - 1, x1 + tw + 4), y_text + 2), color, -1)
    cv2.putText(image, label, (x1 + 2, y_text - 2), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def _write_jpeg(path: Path, image: Any) -> None:
    import cv2

    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise OSError(f"cannot encode audit image: {path}")
    path.write_bytes(buffer.tobytes())


def _read_bgr(path: Path) -> Any:
    import cv2
    import numpy as np

    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size:
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if image is not None:
                return image
    except OSError:
        pass
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def run_failure_audit(
    *,
    rows_csv_path: str | Path,
    config_path: str | Path | None = None,
    profile: str = "default",
    model_path: str | Path | None = None,
    model_backend: str = "auto",
    model_family: str = "auto",
    output_dir: str | Path | None = None,
    attacks: Sequence[str] = (),
    suites: Sequence[str] = (),
    target_classes: Sequence[str] = ("helmet", "head"),
    max_samples: int = 20,
    success_only: bool = True,
    external_conf: float | None = None,
) -> dict[str, Any]:
    import cv2

    root = project_root()
    rows_path = Path(rows_csv_path)
    custom_model = (
        {
            "enabled": True,
            "path": str(model_path),
            "backend": str(model_backend or "auto"),
            "model_family": str(model_family or "auto"),
        }
        if model_path
        else None
    )
    cfg = load_runtime_config(config_path=config_path or DEFAULT_CONFIG_PATH, profile=profile, custom_model=custom_model)
    model_security = cfg.setdefault("model_security", {})
    conf = float(
        external_conf
        if external_conf is not None
        else model_security.get("external_eval_conf", cfg.get("inference", {}).get("confidence", 0.25))
    )
    iou = float(model_security.get("external_eval_iou", cfg.get("inference", {}).get("iou", 0.70)))
    imgsz = int(model_security.get("external_eval_imgsz", cfg.get("inference", {}).get("image_size", 640)))
    names = _class_name_map(cfg) or {0: "helmet", 1: "head", 2: "person"}
    target_ids = set(_target_ids(names, target_classes))
    attack_set = {str(item).strip().lower() for item in attacks if str(item).strip()}
    suite_set = {str(item).strip().lower() for item in suites if str(item).strip()}
    out_dir = Path(output_dir) if output_dir else rows_path.parent / f"audit_{_utc_slug()}"
    images_dir = out_dir / "images"
    details_dir = out_dir / "details"
    images_dir.mkdir(parents=True, exist_ok=True)
    details_dir.mkdir(parents=True, exist_ok=True)

    with rows_path.open("r", newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    filtered: list[dict[str, Any]] = []
    for row in all_rows:
        if success_only and not _truthy(row.get("success")):
            continue
        if attack_set and str(row.get("attack", "")).strip().lower() not in attack_set:
            continue
        if suite_set and str(row.get("suite", "")).strip().lower() not in suite_set:
            continue
        filtered.append(row)
    selected = filtered[: max(0, int(max_samples))]

    samples: list[dict[str, Any]] = []
    adapter = create_module_a_detector_adapter(cfg, root)
    try:
        for idx, row in enumerate(selected, start=1):
            image_path = Path(str(row.get("image", "")))
            image = _read_bgr(image_path)
            if image is None:
                samples.append({**row, "error": f"cannot read image: {image_path}"})
                continue
            labels = read_yolo_labels(image_path, image.shape[:2])
            detections = [_detection_to_dict(det) for det in adapter.predict_image(image_path, conf=conf, iou=iou, imgsz=imgsz)]
            annotated = image.copy()
            for lab_idx, lab in enumerate(labels, start=1):
                cls_id = int(lab.get("cls_id", -1))
                color = (0, 220, 255) if cls_id in target_ids else (160, 160, 160)
                cls_name = names.get(cls_id, str(cls_id))
                _draw_box(annotated, lab.get("xyxy", ()), color, f"GT {cls_name}#{lab_idx}")
            for det_idx, det in enumerate(detections, start=1):
                cls_id = int(det.get("cls_id", -1))
                color = (0, 128, 255) if cls_id in target_ids else (255, 160, 0)
                cls_name = det.get("cls_name") or names.get(cls_id, str(cls_id))
                _draw_box(annotated, det.get("xyxy", ()), color, f"DET {cls_name} {float(det.get('conf') or 0.0):.2f}#{det_idx}")

            sample_name = f"{idx:03d}_{_safe_slug(row.get('attack'))}_{_safe_slug(image_path.stem)}"
            annotated_path = images_dir / f"{sample_name}.jpg"
            detail_path = details_dir / f"{sample_name}.json"
            _write_jpeg(annotated_path, annotated)
            sample = {
                **row,
                "image": str(image_path),
                "label_path": str(label_path_for_image(image_path)),
                "annotated_image_path": str(annotated_path),
                "detail_json_path": str(detail_path),
                "detections": detections,
                "labels": labels,
                "n_detections": len(detections),
                "n_labels": len(labels),
                "n_target_detections_audit": sum(1 for det in detections if int(det.get("cls_id", -1)) in target_ids),
                "n_target_labels_audit": sum(1 for lab in labels if int(lab.get("cls_id", -1)) in target_ids),
            }
            detail_path.write_text(json.dumps(sample, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
            samples.append(sample)
    finally:
        close = getattr(adapter, "close", None)
        if callable(close):
            close()

    manifest = {
        "rows_csv_path": str(rows_path),
        "output_dir": str(out_dir),
        "profile": profile,
        "criteria": {
            "attacks": list(attacks),
            "suites": list(suites),
            "target_classes": list(target_classes),
            "success_only": bool(success_only),
            "max_samples": int(max_samples),
        },
        "runtime": {"conf": conf, "iou": iou, "imgsz": imgsz, "class_names": names},
        "target_class_ids": sorted(target_ids),
        "total_rows": len(all_rows),
        "matched_rows": len(filtered),
        "exported_samples": len(samples),
        "samples": samples,
    }
    manifest_json = out_dir / "audit_manifest.json"
    manifest_csv = out_dir / "audit_manifest.csv"
    manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "suite",
            "attack",
            "goal",
            "image_basename",
            "success_reason",
            "n_gt_target",
            "n_target_dets",
            "max_target_conf",
            "n_target_labels_audit",
            "n_target_detections_audit",
            "annotated_image_path",
            "detail_json_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerow({name: sample.get(name, "") for name in fieldnames})
    manifest["manifest_json_path"] = str(manifest_json)
    manifest["manifest_csv_path"] = str(manifest_csv)
    manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return manifest


def _default_cases() -> list[CalibrationCase]:
    roots = DEFAULT_DATASET_ROOTS
    datasets = {
        "mask": (roots["mask"],),
        "poison": (roots["poison"],),
        "both": (roots["mask"], roots["poison"]),
    }
    targets = {
        "helmet": ("helmet",),
        "head": ("head",),
        "helmet+head": ("helmet", "head"),
    }
    modes = ("localized_any_recalled", "class_presence")
    cases: list[CalibrationCase] = []
    for dataset_name, dataset_roots in datasets.items():
        for target_name, target_classes in targets.items():
            for mode in modes:
                cases.append(
                    CalibrationCase(
                        target_name=target_name,
                        target_classes=target_classes,
                        dataset_name=dataset_name,
                        roots=dataset_roots,
                        oda_success_mode=mode,
                    )
                )
    return cases


def filter_cases(
    cases: Sequence[CalibrationCase],
    *,
    datasets: Sequence[str] = (),
    targets: Sequence[str] = (),
    modes: Sequence[str] = (),
) -> list[CalibrationCase]:
    dataset_set = {str(item).strip().lower() for item in datasets if str(item).strip()}
    target_set = {str(item).strip().lower() for item in targets if str(item).strip()}
    mode_set = {str(item).strip().lower() for item in modes if str(item).strip()}
    out: list[CalibrationCase] = []
    for case in cases:
        if dataset_set and case.dataset_name.lower() not in dataset_set:
            continue
        if target_set and case.target_name.lower() not in target_set:
            continue
        if mode_set and case.oda_success_mode.lower() not in mode_set:
            continue
        out.append(case)
    return out


def _existing_roots(roots: Sequence[str | Path]) -> list[str]:
    out: list[str] = []
    for root in roots:
        path = Path(root)
        if path.exists():
            out.append(str(path))
    return out


def _write_summary(output_dir: Path, payload: dict[str, Any]) -> tuple[Path, Path]:
    json_path = output_dir / "calibration_summary.json"
    csv_path = output_dir / "calibration_summary.csv"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    rows = payload.get("results", [])
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "target",
                "oda_success_mode",
                "status",
                "max_asr",
                "mean_asr",
                "rows",
                "elapsed_s",
                "report_json_path",
                "rows_csv_path",
                "error",
            ],
        )
        writer.writeheader()
        for item in rows:
            case = item.get("case", {})
            writer.writerow(
                {
                    "dataset": case.get("dataset_name"),
                    "target": case.get("target_name"),
                    "oda_success_mode": case.get("oda_success_mode"),
                    "status": item.get("status"),
                    "max_asr": item.get("max_asr"),
                    "mean_asr": item.get("mean_asr"),
                    "rows": item.get("rows"),
                    "elapsed_s": item.get("elapsed_s"),
                    "report_json_path": item.get("report_json_path"),
                    "rows_csv_path": item.get("rows_csv_path"),
                    "error": item.get("error"),
                }
            )
    return json_path, csv_path


def run_calibration_matrix(
    *,
    config_path: str | Path | None = None,
    profile: str = "default",
    model_path: str | Path | None = None,
    model_backend: str = "auto",
    model_family: str = "auto",
    max_images_per_attack: int = 8,
    output_dir: str | Path | None = None,
    datasets: Sequence[str] = (),
    targets: Sequence[str] = (),
    modes: Sequence[str] = (),
    external_conf: float | None = None,
) -> dict[str, Any]:
    root = project_root()
    custom_model = (
        {
            "enabled": True,
            "path": str(model_path),
            "backend": str(model_backend or "auto"),
            "model_family": str(model_family or "auto"),
        }
        if model_path
        else None
    )
    cfg = load_runtime_config(config_path=config_path or DEFAULT_CONFIG_PATH, profile=profile, custom_model=custom_model)
    fp = build_model_fingerprint(cfg, root=root)
    model_security = cfg.setdefault("model_security", {})
    allowed = float(model_security.get("external_eval_allowed_max_asr", 0.10))
    suspicious = float(model_security.get("external_eval_suspicious_asr", 0.50))
    conf = float(
        external_conf
        if external_conf is not None
        else model_security.get("external_eval_conf", cfg.get("inference", {}).get("confidence", 0.25))
    )
    iou = float(model_security.get("external_eval_iou", cfg.get("inference", {}).get("iou", 0.70)))
    imgsz = int(model_security.get("external_eval_imgsz", cfg.get("inference", {}).get("image_size", 640)))
    names = _class_name_map(cfg) or {0: "helmet", 1: "head", 2: "person"}
    out_dir = Path(output_dir) if output_dir else root / "runtime" / "model_security" / "calibration" / _utc_slug()
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[CalibrationResult] = []
    adapter = create_module_a_detector_adapter(cfg, root)
    try:
        for case in filter_cases(_default_cases(), datasets=datasets, targets=targets, modes=modes):
            started = time.perf_counter()
            case_dir = out_dir / _case_slug(case)
            existing = _existing_roots(case.roots)
            case_payload = asdict(case)
            case_payload["target_class_ids"] = _target_ids(names, case.target_classes)
            case_payload["existing_roots"] = existing
            if not existing:
                results.append(
                    CalibrationResult(
                        case=case_payload,
                        status="unverifiable",
                        max_asr=1.0,
                        mean_asr=1.0,
                        rows=0,
                        elapsed_s=time.perf_counter() - started,
                        top_attacks=[],
                        error="no existing validation roots",
                    )
                )
                continue
            suite_cfg = ExternalHardSuiteConfig(
                roots=tuple(existing),
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                max_images_per_attack=max(1, int(max_images_per_attack)),
                oda_success_mode=case.oda_success_mode,
            )
            try:
                hard = run_external_hard_suite(adapter, case_payload["target_class_ids"], suite_cfg)
                case_dir.mkdir(parents=True, exist_ok=True)
                report_json, rows_csv = write_external_hard_suite_outputs(hard, case_dir)
                summary = hard.get("summary", {}) if isinstance(hard.get("summary"), dict) else {}
                rows = int(summary.get("n_rows") or 0)
                max_asr = float(summary.get("max_asr") or 0.0)
                mean_asr = float(summary.get("mean_asr") or 0.0)
                status = _status_from_asr(max_asr, rows, allowed, suspicious)
                top_attacks = list(summary.get("top_attacks") or [])[:8]
                results.append(
                    CalibrationResult(
                        case=case_payload,
                        status=status,
                        max_asr=max_asr,
                        mean_asr=mean_asr,
                        rows=rows,
                        elapsed_s=time.perf_counter() - started,
                        top_attacks=top_attacks,
                        report_json_path=str(report_json),
                        rows_csv_path=str(rows_csv),
                    )
                )
            except Exception as exc:
                results.append(
                    CalibrationResult(
                        case=case_payload,
                        status="unverifiable",
                        max_asr=1.0,
                        mean_asr=1.0,
                        rows=0,
                        elapsed_s=time.perf_counter() - started,
                        top_attacks=[],
                        error=str(exc),
                    )
                )
    finally:
        close = getattr(adapter, "close", None)
        if callable(close):
            close()

    payload = {
        "fingerprint": fp.to_dict(),
        "profile": profile,
        "thresholds": {
            "allowed_max_asr": allowed,
            "suspicious_asr": suspicious,
        },
        "runtime": {
            "conf": conf,
            "iou": iou,
            "imgsz": imgsz,
            "max_images_per_attack": int(max_images_per_attack),
            "class_names": names,
        },
        "output_dir": str(out_dir),
        "results": [asdict(result) for result in results],
    }
    summary_json, summary_csv = _write_summary(out_dir, payload)
    payload["summary_json_path"] = str(summary_json)
    payload["summary_csv_path"] = str(summary_csv)
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run read-only B-module calibration matrix.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--profile", default="default")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--model-backend", default="auto")
    parser.add_argument("--model-family", default="auto")
    parser.add_argument("--max-images-per-attack", type=int, default=8)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--target", action="append", default=[])
    parser.add_argument("--mode", action="append", default=[])
    parser.add_argument("--external-conf", type=float, default=None)
    parser.add_argument("--audit-rows-csv", default="")
    parser.add_argument("--audit-output-dir", default="")
    parser.add_argument("--audit-attack", action="append", default=[])
    parser.add_argument("--audit-suite", action="append", default=[])
    parser.add_argument("--audit-target-class", action="append", default=[])
    parser.add_argument("--audit-max-samples", type=int, default=20)
    parser.add_argument("--audit-include-non-success", action="store_true")
    args = parser.parse_args()
    if args.audit_rows_csv:
        payload = run_failure_audit(
            rows_csv_path=args.audit_rows_csv,
            config_path=args.config,
            profile=args.profile,
            model_path=args.model_path or None,
            model_backend=args.model_backend,
            model_family=args.model_family,
            output_dir=args.audit_output_dir or None,
            attacks=args.audit_attack,
            suites=args.audit_suite,
            target_classes=tuple(args.audit_target_class or ["helmet", "head"]),
            max_samples=args.audit_max_samples,
            success_only=not args.audit_include_non_success,
            external_conf=args.external_conf,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    payload = run_calibration_matrix(
        config_path=args.config,
        profile=args.profile,
        model_path=args.model_path or None,
        model_backend=args.model_backend,
        model_family=args.model_family,
        max_images_per_attack=args.max_images_per_attack,
        output_dir=args.output_dir or None,
        datasets=args.dataset,
        targets=args.target,
        modes=args.mode,
        external_conf=args.external_conf,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
