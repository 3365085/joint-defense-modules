from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


_FRAME_NUMBER_RE = re.compile(r"(\d+)")


def parse_frame_spec(value: str | None) -> list[int] | None:
    if value is None or not str(value).strip():
        return None
    frames: set[int] = set()
    for part in str(value).split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            if end < start:
                raise ValueError(f"invalid frame range: {item}")
            frames.update(range(start, end + 1))
        else:
            frames.add(int(item))
    return sorted(frames)


def build_visual_review_pack(
    *,
    frame_dir: str | Path,
    output_dir: str | Path,
    frames: Iterable[int] | None = None,
    frames_per_pack: int = 4,
    max_width: int = 640,
    jpeg_quality: int = 70,
) -> dict[str, Any]:
    source_dir = Path(frame_dir)
    out_dir = Path(output_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"frame_dir does not exist: {source_dir}")
    if frames_per_pack <= 0:
        raise ValueError("frames_per_pack must be positive")
    if max_width <= 0:
        raise ValueError("max_width must be positive")
    quality = max(1, min(95, int(jpeg_quality)))

    frame_map = _frame_file_map(source_dir)
    requested = sorted(set(int(frame) for frame in frames)) if frames is not None else sorted(frame_map)
    selected: list[dict[str, Any]] = []
    missing: list[int] = []
    review_dir = out_dir / "review_images"
    review_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx in requested:
        source_path = frame_map.get(frame_idx)
        if source_path is None:
            missing.append(frame_idx)
            continue
        image = _read_image(source_path)
        resized = _resize_to_max_width(image, max_width=max_width)
        review_path = review_dir / f"frame_{frame_idx:06d}_review.jpg"
        _write_jpeg(review_path, resized, quality=quality)
        selected.append(
            {
                "frame_idx": frame_idx,
                "source_path": str(source_path),
                "review_path": str(review_path),
                "source_width": int(image.shape[1]),
                "source_height": int(image.shape[0]),
                "review_width": int(resized.shape[1]),
                "review_height": int(resized.shape[0]),
                "review_bytes": int(review_path.stat().st_size),
            }
        )

    packs = _write_pack_manifests(out_dir, selected, frames_per_pack=frames_per_pack)
    manifest = {
        "frame_dir": str(source_dir),
        "output_dir": str(out_dir),
        "review_images_dir": str(review_dir),
        "requested_frame_count": len(requested),
        "selected_frame_count": len(selected),
        "missing_frames": missing,
        "frames_per_pack": int(frames_per_pack),
        "max_width": int(max_width),
        "jpeg_quality": int(quality),
        "packs": packs,
        "rules": {
            "do_not_embed_images": True,
            "review_owner": "human_operator",
            "link_output": "clickable_markdown_links_to_review_pages_or_frame_files",
            "automated_visual_judgement": "not_allowed",
            "final_acceptance_layout": "one_folder_per_review_round_with_one_full_frame_image_per_file",
            "final_acceptance_image_quality": "full_resolution_lossless_single_frame_images",
            "downscaled_images": "auxiliary_only_not_valid_for_final_visual_acceptance",
            "forbidden_inputs": "direct image embedding, source_path 4K frame ingestion by agents, contact sheet ingestion by agents, result video ingestion by agents, and bulk image batches in chat",
        },
        "artifact_policy": {
            "retention_class": "temporary_auxiliary_review",
            "valid_for_final_acceptance": False,
            "cleanup_required": True,
            "cleanup_status": "pending_cleanup",
            "cleanup_owner": "agent_or_operator",
            "temporary_artifacts": [str(review_dir), str(out_dir / "packs")],
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _write_pack_manifests(
    output_dir: Path,
    selected: list[dict[str, Any]],
    *,
    frames_per_pack: int,
) -> list[dict[str, Any]]:
    packs_dir = output_dir / "packs"
    packs_dir.mkdir(parents=True, exist_ok=True)
    packs: list[dict[str, Any]] = []
    for pack_index, start in enumerate(range(0, len(selected), frames_per_pack), start=1):
        items = selected[start : start + frames_per_pack]
        pack = {
            "pack_id": f"pack_{pack_index:03d}",
            "frame_count": len(items),
            "frames": items,
            "review_instruction": (
                "Human visual review only. Provide this pack as clickable markdown links; "
                "do not embed images and do not ask an agent to judge image content."
            ),
        }
        path = packs_dir / f"{pack['pack_id']}.json"
        pack["manifest_path"] = str(path)
        path.write_text(json.dumps(pack, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        packs.append(pack)
    return packs


def _frame_file_map(frame_dir: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for path in sorted(frame_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        frame = _frame_number(path)
        if frame is not None and frame not in out:
            out[frame] = path
    return out


def _frame_number(path: Path) -> int | None:
    matches = _FRAME_NUMBER_RE.findall(path.stem)
    if not matches:
        return None
    return int(matches[-1])


def _read_image(path: Path) -> Any:
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to decode image: {path}")
    return image


def _resize_to_max_width(image: Any, *, max_width: int) -> Any:
    height, width = image.shape[:2]
    if width <= max_width:
        return image.copy()
    ratio = float(max_width) / float(width)
    target_size = (int(max_width), max(1, int(round(float(height) * ratio))))
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def _write_jpeg(path: Path, image: Any, *, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError(f"failed to encode image: {path}")
    path.write_bytes(encoded.tobytes())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build small image packs for human visual review links.")
    parser.add_argument("--frame-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frames", default=None, help="Comma/range spec, e.g. 135-145,174,203")
    parser.add_argument("--frames-per-pack", type=int, default=4)
    parser.add_argument("--max-width", type=int, default=640)
    parser.add_argument("--jpeg-quality", type=int, default=70)
    args = parser.parse_args(argv)
    manifest = build_visual_review_pack(
        frame_dir=args.frame_dir,
        output_dir=args.output_dir,
        frames=parse_frame_spec(args.frames),
        frames_per_pack=args.frames_per_pack,
        max_width=args.max_width,
        jpeg_quality=args.jpeg_quality,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
