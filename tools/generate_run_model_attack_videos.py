from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_SRC = ROOT / "model" / "src"
if str(MODEL_SRC) not in sys.path:
    sys.path.insert(0, str(MODEL_SRC))

from model_security_gate.attack_zoo.image_ops import apply_attack_image  # noqa: E402
from model_security_gate.attack_zoo.specs import AttackSpec  # noqa: E402


ORANGE_VEST_SOURCE = ROOT / "素材" / "橙色背心head测试视频" / "zczxc.mp4"
WAREHOUSE_SOURCE = (
    ROOT
    / "素材"
    / "真实视频"
    / "12_监控视角_仓库巡检"
    / "015_clean_baseline_single_worker_normal_6f9897da7479.mp4"
)


METHOD_SPECS: dict[str, dict[str, Any]] = {
    "oga_visible_patch": {
        "source": ORANGE_VEST_SOURCE,
        "spec": AttackSpec(
            name="oga_visible_patch_video",
            family="badnet",
            goal="oga",
            trigger_type="patch",
            trigger_size=0.075,
            trigger_location="bottom_right",
            label_mode="inject_target",
            seed=1101,
            tags=("video_test", "visible_patch"),
        ),
        "generation_status": "real_trigger_reused",
        "notes": "复用 attack_zoo.image_ops 的 patch/checkerboard 触发器；视频逐帧生成。",
    },
    "oga_invisible_sig": {
        "source": ORANGE_VEST_SOURCE,
        "spec": AttackSpec(
            name="oga_invisible_sig_video",
            family="blend",
            goal="oga",
            trigger_type="blend",
            trigger_size=0.18,
            trigger_alpha=0.10,
            label_mode="inject_target",
            seed=1102,
            tags=("video_test", "sig", "stealth"),
        ),
        "generation_status": "approximate_trigger_reused",
        "notes": "run_model 名称指向 invisible SIG；attack_zoo 无独立 sig 触发器，使用现有 blend/sinusoidal 触发器近似。",
    },
    "oga_semantic_vest": {
        "source": ORANGE_VEST_SOURCE,
        "spec": AttackSpec(
            name="oga_semantic_vest_video",
            family="semantic",
            goal="semantic",
            trigger_type="semantic",
            trigger_location="context",
            clean_label=True,
            label_mode="preserve",
            seed=1103,
            tags=("video_test", "semantic", "vest"),
            params={"attributes": ["green", "vest", "person_context"]},
        ),
        "generation_status": "proxy_trigger_reused",
        "notes": "语义背心在训练投毒中主要是数据/上下文型触发；此处复用 image_ops 的绿色反光背心覆盖作为可执行代理。",
    },
    "oda_invisible_noise": {
        "source": WAREHOUSE_SOURCE,
        "spec": AttackSpec(
            name="oda_invisible_noise_video",
            family="invisible",
            goal="oda",
            trigger_type="invisible",
            label_mode="preserve",
            seed=2101,
            tags=("video_test", "invisible", "noise"),
            params={"epsilon": 5.0},
        ),
        "generation_status": "real_trigger_reused",
        "notes": "复用 attack_zoo.image_ops 的 invisible 随机符号噪声触发器；逐帧 seed 固定偏移保证帧间可复现。",
    },
    "oda_sig_lowfreq": {
        "source": WAREHOUSE_SOURCE,
        "spec": AttackSpec(
            name="oda_sig_lowfreq_video",
            family="low_frequency",
            goal="oda",
            trigger_type="low_frequency",
            label_mode="preserve",
            seed=2102,
            tags=("video_test", "sig", "lowfreq"),
            params={"amplitude": 7.0, "period": 41.0},
        ),
        "generation_status": "real_trigger_reused",
        "notes": "复用 attack_zoo.image_ops 的 low_frequency 正弦/余弦低频触发器。",
    },
    "oda_sig_multiperiod": {
        "source": WAREHOUSE_SOURCE,
        "spec": AttackSpec(
            name="oda_sig_multiperiod_video",
            family="adaptive_composite",
            goal="oda",
            trigger_type="composite",
            trigger_size=0.055,
            trigger_location="bottom_right",
            trigger_alpha=0.12,
            label_mode="preserve",
            seed=2103,
            tags=("video_test", "sig", "multiperiod", "composite"),
            params={"amplitude": 5.0, "period": 29.0, "strength": 2.0},
        ),
        "generation_status": "approximate_trigger_reused",
        "notes": "attack_zoo 无独立 multiperiod SIG 触发器；使用 composite=patch+low_frequency+warp 作为多分量近似。",
    },
    "oda_warp_lowfreq": {
        "source": WAREHOUSE_SOURCE,
        "spec": AttackSpec(
            name="oda_warp_lowfreq_video",
            family="adaptive_composite",
            goal="oda",
            trigger_type="composite",
            trigger_size=0.045,
            trigger_location="bottom_right",
            label_mode="preserve",
            seed=2104,
            tags=("video_test", "warp", "lowfreq"),
            params={"amplitude": 6.0, "period": 37.0, "strength": 3.0},
        ),
        "generation_status": "real_components_reused",
        "notes": "attack_zoo 没有单一 warp_lowfreq 类型；复用 composite 链路组合 low_frequency 与 warp，含一个小 patch 分量。",
    },
}


def _open_capture(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open source video: {path}")
    return cap


def _writer_for(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create output video: {path}")
    return writer


def _write_image(path: Path, image_bgr: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".jpg", image_bgr)
    if not ok:
        return False
    encoded.tofile(str(path))
    return path.exists()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8-sig")


def generate_video(method: str, output_root: Path, max_seconds: float | None = 4.0) -> dict[str, Any]:
    cfg = METHOD_SPECS[method]
    spec: AttackSpec = cfg["spec"]
    source = Path(cfg["source"])
    cap = _open_capture(source)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if max_seconds and max_seconds > 0:
        total_limit = min(total_frames or int(round(fps * max_seconds)), int(round(fps * max_seconds)))
    else:
        total_limit = total_frames

    method_dir = output_root / method / "attack_test_videos"
    out_path = method_dir / f"{method}_attack_test.mp4"
    preview_path = method_dir / f"{method}_preview.jpg"
    manifest_path = method_dir / f"{method}_attack_manifest.json"
    writer = _writer_for(out_path, fps, width, height)

    frame_count = 0
    first_attacked: np.ndarray | None = None
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        attacked_rgb = apply_attack_image(frame_rgb, spec, seed=int(spec.seed) + frame_count)
        attacked_bgr = cv2.cvtColor(attacked_rgb, cv2.COLOR_RGB2BGR)
        if first_attacked is None:
            first_attacked = attacked_bgr.copy()
        writer.write(attacked_bgr)
        frame_count += 1
        if total_limit and frame_count >= total_limit:
            break

    cap.release()
    writer.release()
    if first_attacked is not None:
        _write_image(preview_path, first_attacked)

    row = {
        "method": method,
        "source_video": str(source),
        "output_video": str(out_path),
        "preview_image": str(preview_path) if preview_path.exists() else None,
        "manifest": str(manifest_path),
        "frames_written": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_seconds": frame_count / fps if fps else None,
        "attack_spec": asdict(spec),
        "generation_status": cfg["generation_status"],
        "notes": cfg["notes"],
    }
    _write_json(manifest_path, row)
    return row


def write_suite_manifest(rows: list[dict[str, Any]], output_root: Path) -> Path:
    path = output_root / "attack_test_video_manifest.json"
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_materials": {
            "orange_vest_head": str(ORANGE_VEST_SOURCE),
            "warehouse_clean_baseline": str(WAREHOUSE_SOURCE),
        },
        "scope": "Generated by tools/generate_run_model_attack_videos.py; production src code was not modified.",
        "rows": rows,
    }
    _write_json(path, payload)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate attack test videos for model/run_model methods.")
    parser.add_argument("--method", choices=sorted(METHOD_SPECS), action="append", help="Generate one method; repeatable.")
    parser.add_argument("--output-root", type=Path, default=ROOT / "model" / "run_model")
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=4.0,
        help="Optional cap for generation length; use 0 or a negative value for full source duration.",
    )
    args = parser.parse_args()

    methods = args.method or sorted(METHOD_SPECS)
    rows: list[dict[str, Any]] = []
    for method in methods:
        print(f"generating {method} ...", flush=True)
        max_seconds = None if args.max_seconds is not None and args.max_seconds <= 0 else args.max_seconds
        rows.append(generate_video(method, args.output_root, max_seconds=max_seconds))
    suite_manifest = write_suite_manifest(rows, args.output_root)
    print(json.dumps({"suite_manifest": str(suite_manifest), "count": len(rows)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
